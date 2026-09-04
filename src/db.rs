//! The database: tables, snapshot reads, single-writer transactions.
//!
//! One revision counter covers the whole `Db`. A commit that changes anything
//! bumps it by one, builds a new [`Root`] and swaps it in behind an `RwLock`
//! held only for the assignment, so readers never see a half-applied commit and
//! never block the writer.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::iter::Peekable;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use crate::tree::{self, Tree};
use crate::watch::{Watch, WatchCell};

pub type Revision = u64;
pub type Key = Box<[u8]>;

/// A stored value with the revision of the commit that wrote it.
struct Object<V> {
    value: Arc<V>,
    revision: Revision,
}

/// Key of the revision indexes: the revision, big-endian, then the primary key.
///
/// `StateDB` gives every object its own revision, so the revision alone is a
/// unique index key (`write_txn.go:130`). Here one revision covers a whole
/// commit, so the primary key is appended to keep the entries of one commit
/// apart. The ordering is unchanged: revision first, ascending.
fn rev_key(revision: Revision, key: &[u8]) -> Key {
    let mut k = Vec::with_capacity(8 + key.len());
    k.extend_from_slice(&revision.to_be_bytes());
    k.extend_from_slice(key);
    k.into()
}

fn revision_of(rev_key: &[u8]) -> Revision {
    let head: [u8; 8] = rev_key[..8].try_into().expect("revision index key");
    Revision::from_be_bytes(head)
}

fn row<V>((key, obj): (Vec<u8>, &Arc<Object<V>>)) -> (Vec<u8>, &V, Revision) {
    (key, obj.value.as_ref(), obj.revision)
}

// ---------------------------------------------------------------- table entry

/// One table's four trees plus its change trackers.
struct TableEntry<V> {
    /// Revision of the last commit that changed this table.
    revision: Revision,
    primary: Tree<Object<V>>,
    /// `rev_key(revision, primary key) -> primary key`.
    rev_index: Tree<Key>,
    /// Deleted objects, by primary key, at their delete revision.
    graveyard: Tree<Object<V>>,
    graveyard_rev: Tree<Key>,
    trackers: Vec<Weak<AtomicU64>>,
    primary_key: fn(&V) -> Key,
}

impl<V> TableEntry<V> {
    fn new(primary_key: fn(&V) -> Key) -> Self {
        Self {
            revision: 0,
            primary: Tree::new(),
            rev_index: Tree::new(),
            graveyard: Tree::new(),
            graveyard_rev: Tree::new(),
            trackers: Vec::new(),
            primary_key,
        }
    }
}

/// The type-erased face of `TableEntry<V>`: what the `Root` can do without
/// knowing the value type.
trait AnyTable: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    /// Drops dead trackers and every graveyard object at or below the
    /// watermark. `None` means nothing changed.
    fn collect(&self, compacted: Revision) -> Option<Arc<dyn AnyTable>>;
}

impl<V: Send + Sync + 'static> AnyTable for TableEntry<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn collect(&self, compacted: Revision) -> Option<Arc<dyn AnyTable>> {
        let mut trackers = Vec::with_capacity(self.trackers.len());
        let mut watermark = Revision::MAX;
        for weak in &self.trackers {
            if let Some(tracker) = weak.upgrade() {
                watermark = watermark.min(tracker.load(Ordering::Relaxed));
                trackers.push(weak.clone());
            }
        }
        let pruned = trackers.len() != self.trackers.len();
        // With no reader left, everything already dead may go.
        let bound = if trackers.is_empty() {
            self.revision
        } else {
            watermark
        }
        .max(compacted);

        let mut graveyard = self.graveyard.txn();
        let mut graveyard_rev = self.graveyard_rev.txn();
        let mut removed = 0usize;
        for (index_key, primary_key) in self.graveyard_rev.iter() {
            if revision_of(&index_key) > bound {
                break;
            }
            graveyard_rev.delete(&index_key);
            graveyard.delete(primary_key);
            removed += 1;
        }
        if removed == 0 && !pruned {
            return None;
        }
        Some(Arc::new(TableEntry {
            revision: self.revision,
            primary: self.primary.clone(),
            rev_index: self.rev_index.clone(),
            // Nothing outside this module watches the graveyard, so its cells
            // are closed here rather than after the root swap.
            graveyard: graveyard.commit_and_notify(),
            graveyard_rev: graveyard_rev.commit_and_notify(),
            trackers,
            primary_key: self.primary_key,
        }))
    }
}

// ----------------------------------------------------------------------- root

/// Everything a snapshot is: swapped as one `Arc`.
#[derive(Clone)]
struct Root {
    revision: Revision,
    /// History below this revision is gone; see [`Db::compact`].
    compacted: Revision,
    tables: Vec<Arc<dyn AnyTable>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic between the two root swaps leaves the root untouched, and one
    // after them leaves it fully applied, so a poisoned lock guards no torn
    // state.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The database. Registration and writes are serialized by one writer lock;
/// reads take an `Arc` of the current root and release the lock at once.
pub struct Db {
    root: RwLock<Arc<Root>>,
    write: Mutex<()>,
    hook: Mutex<Hook>,
}

type Hook = Box<dyn FnMut(Revision, &ReadTxn) + Send>;

impl Default for Db {
    fn default() -> Self {
        Self::new()
    }
}

impl Db {
    #[must_use]
    pub fn new() -> Self {
        Self::with_hook(|_, _| {})
    }

    /// The hook runs after every revision-bumping commit, with the committed
    /// revision and a snapshot taken right after it, while the writer lock is
    /// still held. It must not write to this `Db`; a panic in it propagates out
    /// of `commit`.
    pub fn with_hook(hook: impl FnMut(Revision, &ReadTxn) + Send + 'static) -> Self {
        Self {
            root: RwLock::new(Arc::new(Root {
                revision: 0,
                compacted: 0,
                tables: Vec::new(),
            })),
            write: Mutex::new(()),
            hook: Mutex::new(Box::new(hook)),
        }
    }

    fn snapshot(&self) -> Arc<Root> {
        self.root
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn install(&self, root: Root) {
        *self.root.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(root);
    }

    /// Registers a table. Takes the writer lock and does not bump the revision.
    pub fn table<V: Send + Sync + 'static>(
        &self,
        name: &'static str,
        primary_key: fn(&V) -> Key,
    ) -> Table<V> {
        let _guard = lock(&self.write);
        let mut root = (*self.snapshot()).clone();
        let pos = root.tables.len();
        root.tables.push(Arc::new(TableEntry::new(primary_key)));
        self.install(root);
        Table {
            pos,
            name,
            _v: PhantomData,
        }
    }

    #[must_use]
    pub fn read(&self) -> ReadTxn {
        ReadTxn(self.snapshot())
    }

    /// Opens the write transaction. Blocks until the previous one commits or is
    /// dropped.
    #[must_use]
    pub fn write(&self) -> WriteTxn<'_> {
        let guard = lock(&self.write);
        let root = (*self.snapshot()).clone();
        let pending = std::iter::repeat_with(|| None)
            .take(root.tables.len())
            .collect();
        WriteTxn {
            db: self,
            _guard: guard,
            root,
            pending,
            dirty: false,
        }
    }

    /// Drops graveyard entries at or below `rev` whether or not the change
    /// iterators have read them; those iterators then fail with [`Compacted`].
    ///
    /// This is not history: it does not bump the revision and does not run the
    /// commit hook.
    pub fn compact(&self, rev: Revision) {
        let _guard = lock(&self.write);
        let mut root = (*self.snapshot()).clone();
        root.compacted = rev.min(root.revision).max(root.compacted);
        for table in &mut root.tables {
            if let Some(new) = table.collect(root.compacted) {
                *table = new;
            }
        }
        self.install(root);
    }
}

// --------------------------------------------------------------------- tables

/// A handle to a registered table. Cheap to copy; valid only for the `Db` that
/// returned it.
pub struct Table<V> {
    pos: usize,
    name: &'static str,
    _v: PhantomData<V>,
}

impl<V> Clone for Table<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for Table<V> {}

/// A snapshot. Holding one keeps its version of the data alive and blocks
/// nothing.
pub struct ReadTxn(Arc<Root>);

impl ReadTxn {
    /// The revision of the commit this snapshot was taken after.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.0.revision
    }
}

impl<V: Send + Sync + 'static> Table<V> {
    fn entry<'a>(&self, root: &'a Root) -> &'a TableEntry<V> {
        root.tables
            .get(self.pos)
            .and_then(|t| t.as_any().downcast_ref())
            .unwrap_or_else(|| panic!("table {} belongs to another Db or value type", self.name))
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The revision of the last commit that changed this table.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[must_use]
    pub fn revision(&self, txn: &ReadTxn) -> Revision {
        self.entry(&txn.0).revision
    }

    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[must_use]
    pub fn get<'a>(&self, txn: &'a ReadTxn, key: &[u8]) -> Option<(&'a V, Revision)> {
        self.get_watch(txn, key).0
    }

    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[must_use]
    pub fn get_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        key: &[u8],
    ) -> (Option<(&'a V, Revision)>, Watch) {
        let (obj, watch) = self.entry(&txn.0).primary.get(key);
        (obj.map(|o| (o.value.as_ref(), o.revision)), watch)
    }

    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn prefix<'a>(
        &self,
        txn: &'a ReadTxn,
        prefix: &[u8],
    ) -> impl Iterator<Item = (Vec<u8>, &'a V, Revision)> + use<'a, V> {
        self.prefix_watch(txn, prefix).0
    }

    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn prefix_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        prefix: &[u8],
    ) -> (
        impl Iterator<Item = (Vec<u8>, &'a V, Revision)> + use<'a, V>,
        Watch,
    ) {
        let (iter, watch) = self.entry(&txn.0).primary.prefix(prefix);
        (iter.map(row), watch)
    }

    /// Every entry with a key `>= key`, in order.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn lower_bound<'a>(
        &self,
        txn: &'a ReadTxn,
        key: &[u8],
    ) -> impl Iterator<Item = (Vec<u8>, &'a V, Revision)> + use<'a, V> {
        self.entry(&txn.0).primary.lower_bound(key).map(row)
    }

    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn all<'a>(
        &self,
        txn: &'a ReadTxn,
    ) -> impl Iterator<Item = (Vec<u8>, &'a V, Revision)> + use<'a, V> {
        self.all_watch(txn).0
    }

    /// Every entry, plus a watch that fires on any change to the table.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn all_watch<'a>(
        &self,
        txn: &'a ReadTxn,
    ) -> (
        impl Iterator<Item = (Vec<u8>, &'a V, Revision)> + use<'a, V>,
        Watch,
    ) {
        let entry = self.entry(&txn.0);
        (entry.primary.iter().map(row), entry.primary.root_watch())
    }

    /// How many deleted entries the change readers still hold. Test helper.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[doc(hidden)]
    #[must_use]
    pub fn graveyard_len(&self, txn: &ReadTxn) -> usize {
        self.entry(&txn.0).graveyard.len()
    }

    /// The working copy of this table, opened on first use.
    fn pending<'t>(&self, txn: &'t mut WriteTxn<'_>) -> &'t mut Pending<V> {
        if txn.pending[self.pos].is_none() {
            let entry = self.entry(&txn.root);
            let pending = Pending {
                revision: entry.revision,
                written: false,
                primary: entry.primary.txn(),
                rev_index: entry.rev_index.txn(),
                graveyard: entry.graveyard.txn(),
                graveyard_rev: entry.graveyard_rev.txn(),
                trackers: entry.trackers.clone(),
                new_trackers: Vec::new(),
                primary_key: entry.primary_key,
            };
            txn.pending[self.pos] = Some(Box::new(pending));
        }
        txn.pending[self.pos]
            .as_mut()
            .expect("just opened")
            .as_any_mut()
            .downcast_mut()
            .expect("pending table opened with another value type")
    }

    /// Writes `value` under `primary_key(&value)` and returns what it replaced.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn insert(&self, txn: &mut WriteTxn<'_>, value: V) -> Option<Arc<V>> {
        let revision = txn.root.revision + 1;
        txn.dirty = true;
        let pending = self.pending(txn);
        pending.written = true;
        let key = (pending.primary_key)(&value);
        let object = Object {
            value: Arc::new(value),
            revision,
        };
        let old = pending.primary.insert(&key, object);
        if let Some(old) = &old {
            pending.rev_index.delete(&rev_key(old.revision, &key));
        }
        pending
            .rev_index
            .insert(&rev_key(revision, &key), key.clone());
        // Re-created after a delete: it is live again, so it leaves the graveyard.
        if let Some(dead) = pending.graveyard.delete(&key) {
            pending.graveyard_rev.delete(&rev_key(dead.revision, &key));
        }
        old.map(|o| o.value.clone())
    }

    /// Moves the entry to the graveyard and returns it.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn delete(&self, txn: &mut WriteTxn<'_>, key: &[u8]) -> Option<Arc<V>> {
        let revision = txn.root.revision + 1;
        let pending = self.pending(txn);
        let old = pending.primary.delete(key)?;
        pending.written = true;
        pending.rev_index.delete(&rev_key(old.revision, key));
        pending.graveyard.insert(
            key,
            Object {
                value: old.value.clone(),
                revision,
            },
        );
        pending
            .graveyard_rev
            .insert(&rev_key(revision, key), key.into());
        txn.dirty = true;
        Some(old.value.clone())
    }

    /// Registers a change reader that observes this table as of the commit that
    /// installs it. Until `txn` commits the reader holds nothing back.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn changes(&self, txn: &mut WriteTxn<'_>) -> ChangeIterator<V> {
        let pending = self.pending(txn);
        let tracker = Arc::new(AtomicU64::new(pending.revision));
        pending.new_trackers.push(tracker.clone());
        ChangeIterator {
            table: *self,
            tracker,
        }
    }
}

// ------------------------------------------------------------ write, pending

/// One table's uncommitted trees.
struct Pending<V> {
    /// The table's revision before this transaction.
    revision: Revision,
    written: bool,
    primary: tree::Txn<Object<V>>,
    rev_index: tree::Txn<Key>,
    graveyard: tree::Txn<Object<V>>,
    graveyard_rev: tree::Txn<Key>,
    trackers: Vec<Weak<AtomicU64>>,
    new_trackers: Vec<Arc<AtomicU64>>,
    primary_key: fn(&V) -> Key,
}

trait AnyPending {
    /// Builds the new table entry. `closed` collects the cells of the primary
    /// tree, which the caller closes once the new root is in place.
    fn install(
        self: Box<Self>,
        revision: Revision,
        closed: &mut Vec<WatchCell>,
    ) -> Arc<dyn AnyTable>;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<V: Send + Sync + 'static> AnyPending for Pending<V> {
    fn install(
        self: Box<Self>,
        revision: Revision,
        closed: &mut Vec<WatchCell>,
    ) -> Arc<dyn AnyTable> {
        let this = *self;
        let revision = if this.written {
            revision
        } else {
            this.revision
        };
        let mut trackers = this.trackers;
        for tracker in &this.new_trackers {
            tracker.store(revision, Ordering::Relaxed);
            trackers.push(Arc::downgrade(tracker));
        }
        let (primary, mut cells) = this.primary.commit();
        closed.append(&mut cells);
        Arc::new(TableEntry {
            revision,
            primary,
            // The index trees carry no watch anyone can hold.
            rev_index: this.rev_index.commit_and_notify(),
            graveyard: this.graveyard.commit_and_notify(),
            graveyard_rev: this.graveyard_rev.commit_and_notify(),
            trackers,
            primary_key: this.primary_key,
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The write transaction. Dropping it aborts: nothing was ever visible.
pub struct WriteTxn<'a> {
    db: &'a Db,
    _guard: MutexGuard<'a, ()>,
    root: Root,
    /// One slot per table, `Some` once the table is touched.
    pending: Vec<Option<Box<dyn AnyPending>>>,
    dirty: bool,
}

impl WriteTxn<'_> {
    /// Installs the new root and returns its revision, which is the previous
    /// one if nothing was written.
    ///
    /// # Panics
    ///
    /// Propagates a panic from the commit hook.
    pub fn commit(self) -> Revision {
        let WriteTxn {
            db,
            _guard: guard,
            mut root,
            pending,
            dirty,
        } = self;
        if pending.iter().all(Option::is_none) {
            return root.revision;
        }
        let revision = root.revision + Revision::from(dirty);
        let mut closed = Vec::new();
        for (pos, table) in pending.into_iter().enumerate() {
            if let Some(table) = table {
                root.tables[pos] = table.install(revision, &mut closed);
            }
        }
        // ponytail: every commit walks every table's graveyard, which costs one
        // empty iteration per untouched table. Track the tables with a
        // non-empty graveyard in the root if the table count ever grows.
        for table in &mut root.tables {
            if let Some(new) = table.collect(root.compacted) {
                *table = new;
            }
        }
        root.revision = revision;

        let snapshot = Arc::new(root);
        *db.root.write().unwrap_or_else(PoisonError::into_inner) = snapshot.clone();
        for cell in closed {
            cell.send_replace(true);
        }
        if dirty {
            let txn = ReadTxn(snapshot);
            lock(&db.hook)(revision, &txn);
        }
        drop(guard);
        revision
    }
}

// -------------------------------------------------------------------- changes

/// One committed write, as seen by a [`ChangeIterator`].
pub struct Change<V> {
    pub key: Key,
    pub value: Arc<V>,
    pub revision: Revision,
    pub deleted: bool,
}

/// The history the reader still needed was dropped by [`Db::compact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Compacted {
    pub at: Revision,
}

impl fmt::Display for Compacted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "history up to revision {} was compacted away", self.at)
    }
}

impl Error for Compacted {}

/// A reader of a table's changes. It holds deleted entries in the graveyard
/// until it has seen them; dropping it releases them at the next commit.
pub struct ChangeIterator<V> {
    table: Table<V>,
    tracker: Arc<AtomicU64>,
}

impl<V: Send + Sync + 'static> ChangeIterator<V> {
    /// The changes between the last observed revision and `txn`, in revision
    /// order, plus a watch that fires on the next change to the table.
    ///
    /// The snapshot counts as observed as soon as this returns, whether or not
    /// the iterator is drained.
    ///
    /// # Errors
    ///
    /// [`Compacted`] if [`Db::compact`] dropped changes this reader had not
    /// seen.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn next<'a>(
        &mut self,
        txn: &'a ReadTxn,
    ) -> Result<(impl Iterator<Item = Change<V>> + use<'a, V>, Watch), Compacted> {
        let observed = self.tracker.load(Ordering::Relaxed);
        if observed < txn.0.compacted {
            return Err(Compacted {
                at: txn.0.compacted,
            });
        }
        let entry = self.table.entry(&txn.0);
        let from = observed.saturating_add(1).to_be_bytes();
        let changes = Changes {
            live: entry.rev_index.lower_bound(&from).peekable(),
            dead: entry.graveyard_rev.lower_bound(&from).peekable(),
            entry,
            upper: entry.revision,
        };
        self.tracker.store(entry.revision, Ordering::Relaxed);
        Ok((changes, entry.primary.root_watch()))
    }
}

/// Merges the live and the deleted entries of `(observed, upper]` by revision.
struct Changes<'a, V> {
    live: Peekable<tree::Iter<'a, Key>>,
    dead: Peekable<tree::Iter<'a, Key>>,
    entry: &'a TableEntry<V>,
    upper: Revision,
}

fn peek_revision(iter: &mut Peekable<tree::Iter<'_, Key>>, upper: Revision) -> Option<Revision> {
    let revision = revision_of(&iter.peek()?.0);
    (revision <= upper).then_some(revision)
}

impl<V> Iterator for Changes<'_, V> {
    type Item = Change<V>;

    // ponytail: the primary key is copied into every yielded change. Hand out a
    // borrow of the index entry instead if the copies ever show up.
    fn next(&mut self) -> Option<Self::Item> {
        let live = peek_revision(&mut self.live, self.upper);
        let dead = peek_revision(&mut self.dead, self.upper);
        let take_live = match (live, dead) {
            (None, None) => return None,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(l), Some(d)) => l <= d,
        };
        let (tree, deleted) = if take_live {
            (&self.entry.primary, false)
        } else {
            (&self.entry.graveyard, true)
        };
        let (_, key) = if take_live {
            self.live.next()
        } else {
            self.dead.next()
        }
        .expect("peeked");
        let object = tree
            .get(key)
            .0
            .expect("revision index disagrees with its tree");
        Some(Change {
            key: (**key).clone(),
            value: object.value.clone(),
            revision: object.revision,
            deleted,
        })
    }
}
