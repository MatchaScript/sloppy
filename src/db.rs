//! The database: tables, snapshot reads, single-writer transactions.
//!
//! One revision counter covers the whole `Db`. A commit that changes anything
//! bumps it by one, builds a new root and swaps it in behind an `RwLock`
//! held only for the assignment, so readers never see a half-applied commit and
//! never block the writer.

use std::any::Any;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use crate::tree::{self, Tree};
use crate::watch::{Closed, Watch};

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

fn row<V>(obj: &Arc<Object<V>>) -> (&V, Revision) {
    (obj.value.as_ref(), obj.revision)
}

/// Resolves index hits (the primary keys they list) through the primary tree.
fn resolve<'a, V>(
    entry: &'a TableEntry<V>,
    hits: tree::Iter<'a, Key>,
) -> impl Iterator<Item = (&'a V, Revision)> + use<'a, V> {
    hits.map(move |primary| {
        let object = entry
            .primary
            .value(primary)
            .expect("index disagrees with the primary tree");
        (object.value.as_ref(), object.revision)
    })
}

/// A secondary index: a name and the index keys one value is listed under.
///
/// Non-unique. A value may yield zero, one, or several keys, and several values
/// may share a key. `keys` must be a pure function of the value: a replaced
/// value's entries are removed by calling it again on the old value.
pub struct Index<V> {
    pub name: &'static str,
    pub keys: fn(&V) -> Vec<Key>,
}

impl<V> Clone for Index<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for Index<V> {}

/// What a search for one index key scans: its length big-endian, then the key.
///
/// The length prefix keeps the prefix search exact, so index key `a` does not
/// also match the entries of `ab`.
fn index_prefix(index_key: &[u8]) -> Vec<u8> {
    let len = u32::try_from(index_key.len()).expect("index key longer than 4 GiB");
    let mut k = Vec::with_capacity(4 + index_key.len());
    k.extend_from_slice(&len.to_be_bytes());
    k.extend_from_slice(index_key);
    k
}

/// One value's index keys, ready for `binary_search`.
fn sorted(mut keys: Vec<Key>) -> Vec<Key> {
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Key of an index entry: the search prefix, then the primary key, which keeps
/// the entries of one index key apart.
fn index_entry(index_key: &[u8], primary: &[u8]) -> Key {
    let mut k = index_prefix(index_key);
    k.extend_from_slice(primary);
    k.into()
}

// ---------------------------------------------------------------- table entry

/// One table's trees plus its change trackers.
struct TableEntry<V> {
    /// Revision of the last commit that changed this table.
    revision: Revision,
    primary: Tree<Object<V>>,
    /// `rev_key(revision, primary key) -> primary key`.
    rev_index: Tree<Key>,
    /// Deleted objects, by primary key, at their delete revision.
    graveyard: Tree<Object<V>>,
    graveyard_rev: Tree<Key>,
    /// One tree per registered index, in registration order.
    /// `index_entry(index key, primary key) -> primary key`.
    indexes: Vec<(Index<V>, Tree<Key>)>,
    trackers: Vec<Weak<AtomicU64>>,
    /// Highest delete revision that [`Db::compact`] removed before every
    /// tracker had seen it. A reader below this has lost a change.
    lost: Revision,
    primary_key: fn(&V) -> Key,
}

impl<V> TableEntry<V> {
    fn new(primary_key: fn(&V) -> Key, indexes: Vec<Index<V>>) -> Self {
        Self {
            revision: 0,
            primary: Tree::new(),
            rev_index: Tree::new(),
            graveyard: Tree::new(),
            graveyard_rev: Tree::new(),
            indexes: indexes.into_iter().map(|def| (def, Tree::new())).collect(),
            trackers: Vec::new(),
            lost: 0,
            primary_key,
        }
    }
}

/// The type-erased face of `TableEntry<V>`: what the `Root` can do without
/// knowing the value type.
trait AnyTable: Any + Send + Sync {
    /// Drops dead trackers and every graveyard object at or below the
    /// watermark. `None` means nothing changed.
    fn collect(&self, compacted: Revision) -> Option<Arc<dyn AnyTable>>;
}

impl<V: Send + Sync + 'static> AnyTable for TableEntry<V> {
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
        let watermark = if trackers.is_empty() {
            self.revision
        } else {
            watermark
        };
        let bound = watermark.max(compacted);

        let mut graveyard = self.graveyard.txn();
        let mut graveyard_rev = self.graveyard_rev.txn();
        let mut removed = 0usize;
        let mut lost = self.lost;
        let mut it = self.graveyard_rev.iter();
        while let Some(primary_key) = it.next() {
            let index_key = it.key();
            let revision = revision_of(index_key);
            if revision > bound {
                break;
            }
            if revision > watermark {
                lost = revision;
            }
            graveyard_rev.delete(index_key);
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
            // Nobody takes a watch on the graveyard - this module reads it
            // with `value` - so its cells are closed here rather than after
            // the root swap.
            graveyard: graveyard.commit_and_notify(),
            graveyard_rev: graveyard_rev.commit_and_notify(),
            indexes: self.indexes.clone(),
            trackers,
            lost,
            primary_key: self.primary_key,
        }))
    }
}

// ----------------------------------------------------------------------- root

/// Everything a snapshot is: swapped as one `Arc`.
#[derive(Clone)]
struct Root {
    /// Identifies the `Db`; a [`Table`] handle carries the same id.
    db: u64,
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
///
/// The locks are taken in the order `write`, `queue`, `root`; every path takes
/// a subsequence of that, so none of them cycle.
pub struct Db {
    root: RwLock<Arc<Root>>,
    queue: Mutex<Queue>,
    write: Mutex<()>,
    hook: Mutex<Hook>,
}

/// The roots that are settled but not visible yet, oldest first. [`Db::write`]
/// opens on the newest of them, or on the visible root while it is empty.
struct Queue {
    unpublished: VecDeque<Unpublished>,
}

/// A root this `Db` settled that no reader has seen.
struct Unpublished {
    root: Arc<Root>,
    closed: Closed,
    dirty: bool,
    /// The change readers the preparing transaction registered, which
    /// [`Db::abandon`] puts back to unregistered.
    trackers: Vec<Arc<AtomicU64>>,
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
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let root = Arc::new(Root {
            db: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            revision: 0,
            compacted: 0,
            tables: Vec::new(),
        });
        Self {
            root: RwLock::new(root.clone()),
            queue: Mutex::new(Queue {
                unpublished: VecDeque::new(),
            }),
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

    /// The one place the visible root is replaced. The caller holds the queue
    /// lock, so the roots go visible in the order they were queued.
    fn install(&self, root: Arc<Root>) {
        let old = {
            let mut current = self.root.write().unwrap_or_else(PoisonError::into_inner);
            std::mem::replace(&mut *current, root)
        };
        // A value's destructor may read this Db, so the old root must outlive
        // the root write guard.
        drop(old);
    }

    /// Registers a table. Takes the writer lock and does not bump the revision.
    ///
    /// # Panics
    ///
    /// If two indexes have the same name, or a prepared root is unpublished:
    /// this replaces the visible root, which that one does not build on.
    pub fn table<V: Send + Sync + 'static>(
        &self,
        name: &'static str,
        primary_key: fn(&V) -> Key,
        indexes: &[Index<V>],
    ) -> Table<V> {
        let _guard = lock(&self.write);
        for (at, index) in indexes.iter().enumerate() {
            assert!(
                indexes[..at].iter().all(|other| other.name != index.name),
                "table {name} has duplicate index {}",
                index.name
            );
        }
        let queue = lock(&self.queue);
        assert!(
            queue.unpublished.is_empty(),
            "table registered with a prepared root still unpublished"
        );
        let mut root = (*self.snapshot()).clone();
        let pos = root.tables.len();
        root.tables
            .push(Arc::new(TableEntry::new(primary_key, indexes.to_vec())));
        let db = root.db;
        self.install(Arc::new(root));
        drop(queue);
        Table {
            db,
            pos,
            name,
            _v: PhantomData,
        }
    }

    #[must_use]
    pub fn read(&self) -> ReadTxn {
        ReadTxn(self.snapshot())
    }

    /// Opens the write transaction on the newest prepared root. Blocks until
    /// the previous one commits or is dropped.
    #[must_use]
    pub fn write(&self) -> WriteTxn<'_> {
        let guard = lock(&self.write);
        let root = lock(&self.queue)
            .unpublished
            .back()
            .map_or_else(|| self.snapshot(), |u| u.root.clone());
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
    ///
    /// # Panics
    ///
    /// If a prepared root is unpublished: this replaces the visible root, which
    /// that one does not build on.
    pub fn compact(&self, rev: Revision) {
        let _guard = lock(&self.write);
        let queue = lock(&self.queue);
        assert!(
            queue.unpublished.is_empty(),
            "compact with a prepared root still unpublished"
        );
        let mut root = (*self.snapshot()).clone();
        root.compacted = rev.min(root.revision).max(root.compacted);
        for table in &mut root.tables {
            if let Some(new) = table.collect(root.compacted) {
                *table = new;
            }
        }
        self.install(Arc::new(root));
        drop(queue);
    }

    /// Makes the oldest unpublished root visible and returns its revision. The
    /// queue is the order, so there is none for the caller to get wrong.
    ///
    /// # Panics
    ///
    /// If nothing is prepared. Also propagates a panic from the commit hook.
    // The revision is worth ignoring; making the root visible is the point.
    #[allow(clippy::must_use_candidate)]
    pub fn publish(&self) -> Revision {
        let next = {
            let mut queue = lock(&self.queue);
            let next = queue
                .unpublished
                .pop_front()
                .expect("publish with nothing prepared");
            // Installing under the queue lock puts the roots in front of the
            // readers in the order they were popped in. Closing the cells and
            // running the hook stays outside it, so it holds up no later
            // prepare.
            self.install(next.root.clone());
            drop(queue);
            next
        };
        let revision = next.root.revision;
        next.closed.close();
        if next.dirty {
            let txn = ReadTxn(next.root);
            lock(&self.hook)(revision, &txn);
        }
        revision
    }

    /// Drops every unpublished root, so the next [`Db::write`] opens on the
    /// visible one. Nothing ever saw them, so none of their cells are closed,
    /// and the change readers they registered go back to unregistered.
    ///
    /// The writer lock is not reentrant: an open [`WriteTxn`] must be dropped
    /// first.
    pub fn abandon(&self) {
        let _guard = lock(&self.write);
        let mut queue = lock(&self.queue);
        for dropped in queue.unpublished.drain(..) {
            for tracker in dropped.trackers {
                tracker.store(UNREGISTERED, Ordering::Relaxed);
            }
        }
    }
}

// --------------------------------------------------------------------- tables

/// A handle to a registered table. Cheap to copy; valid only for the `Db` that
/// returned it.
pub struct Table<V> {
    db: u64,
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

/// What one table's reads run against: the root of a [`ReadTxn`], or, for a
/// table a [`WriteTxn`] has touched, that transaction's own pending tree.
///
/// The trait hands back the results rather than the tree they came from, which
/// keeps `tree::Node` and the two tree types out of its signature.
trait Snapshot {
    fn value<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> Option<&Arc<Object<V>>>;

    fn prefix<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        prefix: &[u8],
    ) -> tree::Iter<'_, Object<V>>;

    fn lower_bound<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> tree::Iter<'_, Object<V>>;

    fn all<V: Send + Sync + 'static>(&self, table: &Table<V>) -> tree::Iter<'_, Object<V>>;
}

impl Snapshot for ReadTxn {
    fn value<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> Option<&Arc<Object<V>>> {
        table.entry(&self.0).primary.value(key)
    }

    fn prefix<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        prefix: &[u8],
    ) -> tree::Iter<'_, Object<V>> {
        table.entry(&self.0).primary.prefix(prefix)
    }

    fn lower_bound<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> tree::Iter<'_, Object<V>> {
        table.entry(&self.0).primary.lower_bound(key)
    }

    fn all<V: Send + Sync + 'static>(&self, table: &Table<V>) -> tree::Iter<'_, Object<V>> {
        table.entry(&self.0).primary.iter()
    }
}

/// A table this transaction has written reads from its pending tree, so the
/// writes are there; every other table reads the root the transaction opened
/// on. Reading opens no slot.
impl Snapshot for WriteTxn<'_> {
    fn value<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> Option<&Arc<Object<V>>> {
        match table.opened(self) {
            Some(pending) => pending.primary.get(key),
            None => table.entry(&self.root).primary.value(key),
        }
    }

    fn prefix<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        prefix: &[u8],
    ) -> tree::Iter<'_, Object<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.prefix(prefix),
            None => table.entry(&self.root).primary.prefix(prefix),
        }
    }

    fn lower_bound<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> tree::Iter<'_, Object<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.lower_bound(key),
            None => table.entry(&self.root).primary.lower_bound(key),
        }
    }

    fn all<V: Send + Sync + 'static>(&self, table: &Table<V>) -> tree::Iter<'_, Object<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.iter(),
            None => table.entry(&self.root).primary.iter(),
        }
    }
}

impl<V: Send + Sync + 'static> Table<V> {
    fn entry<'a>(&self, root: &'a Root) -> &'a TableEntry<V> {
        root.tables
            .get(self.pos)
            .filter(|_| root.db == self.db)
            .and_then(|t| (&**t as &dyn Any).downcast_ref())
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

    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    #[must_use]
    pub fn get<'a>(&self, txn: &'a impl Snapshot, key: &[u8]) -> Option<(&'a V, Revision)> {
        let object = txn.value(self, key)?;
        Some((object.value.as_ref(), object.revision))
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

    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    pub fn prefix<'a, S: Snapshot>(
        &self,
        txn: &'a S,
        prefix: &[u8],
    ) -> impl Iterator<Item = (&'a V, Revision)> + use<'a, V, S> {
        txn.prefix(self, prefix).map(row)
    }

    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn prefix_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        prefix: &[u8],
    ) -> (impl Iterator<Item = (&'a V, Revision)> + use<'a, V>, Watch) {
        let (iter, watch) = self.entry(&txn.0).primary.prefix_watch(prefix);
        (iter.map(row), watch)
    }

    /// Every entry with a key `>= key`, in order.
    ///
    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    pub fn lower_bound<'a, S: Snapshot>(
        &self,
        txn: &'a S,
        key: &[u8],
    ) -> impl Iterator<Item = (&'a V, Revision)> + use<'a, V, S> {
        txn.lower_bound(self, key).map(row)
    }

    /// The same, plus a watch that fires on any change to the table: an entry
    /// can enter the range anywhere.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn lower_bound_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        key: &[u8],
    ) -> (impl Iterator<Item = (&'a V, Revision)> + use<'a, V>, Watch) {
        let (iter, watch) = self.entry(&txn.0).primary.lower_bound_watch(key);
        (iter.map(row), watch)
    }

    /// Every entry listed under `key` in the named index, resolved through the
    /// primary tree.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`, or has no such index.
    pub fn by_index<'a>(
        &self,
        txn: &'a ReadTxn,
        index: &'static str,
        key: &[u8],
    ) -> impl Iterator<Item = (&'a V, Revision)> + use<'a, V> {
        let entry = self.entry(&txn.0);
        let hits = self.index_tree(entry, index).prefix(&index_prefix(key));
        resolve(entry, hits)
    }

    /// The same, plus a watch that fires when the primary keys listed under
    /// `key` change. Rewriting a value without changing its index membership
    /// does not fire it.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`, or has no such index.
    pub fn by_index_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        index: &'static str,
        key: &[u8],
    ) -> (impl Iterator<Item = (&'a V, Revision)> + use<'a, V>, Watch) {
        let entry = self.entry(&txn.0);
        let (hits, watch) = self
            .index_tree(entry, index)
            .prefix_watch(&index_prefix(key));
        (resolve(entry, hits), watch)
    }

    fn index_tree<'a>(&self, entry: &'a TableEntry<V>, index: &'static str) -> &'a Tree<Key> {
        entry
            .indexes
            .iter()
            .find_map(|(def, tree)| (def.name == index).then_some(tree))
            .unwrap_or_else(|| panic!("table {} has no index {index}", self.name))
    }

    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    pub fn all<'a, S: Snapshot>(
        &self,
        txn: &'a S,
    ) -> impl Iterator<Item = (&'a V, Revision)> + use<'a, V, S> {
        txn.all(self).map(row)
    }

    /// Every entry, plus a watch that fires on any change to the table.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn all_watch<'a>(
        &self,
        txn: &'a ReadTxn,
    ) -> (impl Iterator<Item = (&'a V, Revision)> + use<'a, V>, Watch) {
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

    /// The working copy of this table, if this transaction has opened one. A
    /// handle from another `Db` reads no slot: it falls through to `entry`,
    /// which is where the mismatch is reported.
    fn opened<'t>(&self, txn: &'t WriteTxn<'_>) -> Option<&'t Pending<V>> {
        let pending = txn
            .pending
            .get(self.pos)
            .filter(|_| txn.root.db == self.db)?
            .as_ref()?;
        Some(
            (&**pending as &dyn Any)
                .downcast_ref()
                .expect("pending table opened with another value type"),
        )
    }

    /// The working copy of this table, opened on first use.
    fn pending<'t>(&self, txn: &'t mut WriteTxn<'_>) -> &'t mut Pending<V> {
        let entry = self.entry(&txn.root);
        let pending = txn.pending[self.pos].get_or_insert_with(|| {
            Box::new(Pending {
                revision: entry.revision,
                written: false,
                primary: entry.primary.txn(),
                rev_index: entry.rev_index.txn(),
                graveyard: entry.graveyard.txn(),
                graveyard_rev: entry.graveyard_rev.txn(),
                indexes: entry
                    .indexes
                    .iter()
                    .map(|(def, tree)| (*def, tree.txn()))
                    .collect(),
                trackers: entry.trackers.clone(),
                new_trackers: Vec::new(),
                lost: entry.lost,
                primary_key: entry.primary_key,
            })
        });
        (&mut **pending as &mut dyn Any)
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
        let value = Arc::new(value);
        let key = (pending.primary_key)(&value);
        let object = Object {
            value: value.clone(),
            revision,
        };
        let old = pending.primary.insert(&key, object);
        for (def, tree) in &mut pending.indexes {
            // Only the difference of the two key sets touches the tree, so an
            // update that keeps a value listed under the same index key leaves
            // that entry, and the watch over it, alone.
            let was = old
                .as_ref()
                .map(|old| sorted((def.keys)(&old.value)))
                .unwrap_or_default();
            let is = sorted((def.keys)(&value));
            for k in was.iter().filter(|k| is.binary_search(k).is_err()) {
                tree.delete(&index_entry(k, &key));
            }
            for k in is.iter().filter(|k| was.binary_search(k).is_err()) {
                tree.insert(&index_entry(k, &key), key.clone());
            }
        }
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
        for (def, tree) in &mut pending.indexes {
            for k in (def.keys)(&old.value) {
                tree.delete(&index_entry(&k, key));
            }
        }
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
    /// installs it. The returned reader may only be used after `txn` commits;
    /// until then it holds nothing back.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn changes(&self, txn: &mut WriteTxn<'_>) -> ChangeIterator<V> {
        let pending = self.pending(txn);
        let tracker = Arc::new(AtomicU64::new(UNREGISTERED));
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
    indexes: Vec<(Index<V>, tree::Txn<Key>)>,
    trackers: Vec<Weak<AtomicU64>>,
    new_trackers: Vec<Arc<AtomicU64>>,
    lost: Revision,
    primary_key: fn(&V) -> Key,
}

trait AnyPending: Any {
    /// Builds the new table entry. `closed` collects the cells of the primary
    /// tree, which the caller closes once the new root is in place, and
    /// `registered` the change readers this table opened, which the caller puts
    /// back to unregistered if the root is abandoned.
    fn install(
        self: Box<Self>,
        revision: Revision,
        closed: &mut Closed,
        registered: &mut Vec<Arc<AtomicU64>>,
    ) -> Arc<dyn AnyTable>;
}

impl<V: Send + Sync + 'static> AnyPending for Pending<V> {
    fn install(
        self: Box<Self>,
        revision: Revision,
        closed: &mut Closed,
        registered: &mut Vec<Arc<AtomicU64>>,
    ) -> Arc<dyn AnyTable> {
        let this = *self;
        let revision = if this.written {
            revision
        } else {
            this.revision
        };
        let mut trackers = this.trackers;
        for tracker in this.new_trackers {
            tracker.store(revision, Ordering::Relaxed);
            trackers.push(Arc::downgrade(&tracker));
            registered.push(tracker);
        }
        let (primary, cells) = this.primary.commit();
        closed.absorb(cells);
        let indexes = this
            .indexes
            .into_iter()
            .map(|(def, txn)| {
                let (tree, cells) = txn.commit();
                closed.absorb(cells);
                (def, tree)
            })
            .collect();
        Arc::new(TableEntry {
            revision,
            primary,
            // Nobody takes a watch on these three; the secondary indexes
            // above, which `by_index_watch` hands out, close after the swap.
            rev_index: this.rev_index.commit_and_notify(),
            graveyard: this.graveyard.commit_and_notify(),
            graveyard_rev: this.graveyard_rev.commit_and_notify(),
            indexes,
            trackers,
            lost: this.lost,
            primary_key: this.primary_key,
        })
    }
}

/// The write transaction. Dropping it aborts: nothing was ever visible.
pub struct WriteTxn<'a> {
    db: &'a Db,
    _guard: MutexGuard<'a, ()>,
    /// The root this transaction opened on; `prepare` clones it to build the
    /// new one.
    root: Arc<Root>,
    /// One slot per table, `Some` once the table is touched.
    pending: Vec<Option<Box<dyn AnyPending>>>,
    dirty: bool,
}

impl<'a> WriteTxn<'a> {
    /// Settles the trees and puts the new root at the end of the queue, keeping
    /// the writer lock: `prepare` drops the guard, `commit` publishes under it.
    ///
    /// A transaction that touched no table queues the root it opened on, so
    /// prepare and publish stay one for one.
    fn settle(self) -> (&'a Db, MutexGuard<'a, ()>, Revision) {
        let WriteTxn {
            db,
            _guard: guard,
            root: parent,
            pending,
            dirty,
        } = self;
        let mut closed = Closed::default();
        let mut trackers = Vec::new();
        let root = if pending.iter().all(Option::is_none) {
            parent
        } else {
            let mut root = (*parent).clone();
            let revision = root.revision + Revision::from(dirty);
            for (pos, table) in pending.into_iter().enumerate() {
                if let Some(table) = table {
                    root.tables[pos] = table.install(revision, &mut closed, &mut trackers);
                }
            }
            // ponytail: every commit walks every table's graveyard, which costs
            // one empty iteration per untouched table. Track the tables with a
            // non-empty graveyard in the root if the table count ever grows.
            for table in &mut root.tables {
                if let Some(new) = table.collect(root.compacted) {
                    *table = new;
                }
            }
            root.revision = revision;
            Arc::new(root)
        };

        let revision = root.revision;
        lock(&db.queue).unpublished.push_back(Unpublished {
            root,
            closed,
            dirty,
            trackers,
        });
        (db, guard, revision)
    }

    /// Settles the trees, queues the new root and releases the writer lock.
    /// Nothing is visible until [`Db::publish`], which the caller owes the `Db`
    /// once for every prepare.
    // The revision is worth ignoring; queueing the root is the point.
    #[allow(clippy::must_use_candidate)]
    pub fn prepare(self) -> Revision {
        let (_, guard, revision) = self.settle();
        drop(guard);
        revision
    }

    /// Settles and publishes under the writer lock, and returns the new
    /// revision, which is the previous one if nothing was written.
    ///
    /// # Panics
    ///
    /// If a prepared root is still unpublished: this transaction opened on it,
    /// not on the visible root. Also propagates a panic from the commit hook.
    // The revision is worth ignoring; the commit itself is the point.
    #[allow(clippy::must_use_candidate)]
    pub fn commit(self) -> Revision {
        assert!(
            lock(&self.db.queue).unpublished.is_empty(),
            "commit with a prepared root still unpublished"
        );
        let (db, guard, _) = self.settle();
        let revision = db.publish();
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

const UNREGISTERED: Revision = Revision::MAX;

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
    /// If the registration transaction has not committed, was aborted, or the
    /// table was not registered in this `Db`.
    pub fn next<'a>(
        &mut self,
        txn: &'a ReadTxn,
    ) -> Result<(impl Iterator<Item = Change<V>> + use<'a, V>, Watch), Compacted> {
        let observed = self.tracker.load(Ordering::Relaxed);
        assert_ne!(
            observed, UNREGISTERED,
            "change reader's registration transaction did not commit"
        );
        let entry = self.table.entry(&txn.0);
        if observed < entry.lost {
            return Err(Compacted { at: entry.lost });
        }
        let from = observed.saturating_add(1).to_be_bytes();
        let changes = Changes {
            live: Side {
                iter: entry.rev_index.lower_bound(&from),
                taken: None,
            },
            dead: Side {
                iter: entry.graveyard_rev.lower_bound(&from),
                taken: None,
            },
            entry,
            upper: entry.revision,
        };
        // A stale snapshot must not rewind what a newer one already observed.
        self.tracker.fetch_max(entry.revision, Ordering::Relaxed);
        Ok((changes, entry.primary.root_watch()))
    }
}

/// One side of the merge: its entries, and the one already taken off it.
struct Side<'a> {
    iter: tree::Iter<'a, Key>,
    /// The next entry's revision, from its index key, and its primary key.
    taken: Option<(Revision, &'a Arc<Key>)>,
}

/// Merges the live and the deleted entries of `(observed, upper]` by revision.
struct Changes<'a, V> {
    live: Side<'a>,
    dead: Side<'a>,
    entry: &'a TableEntry<V>,
    upper: Revision,
}

/// The revision of the side's next entry, if it is in range. The index key
/// lives in the walk, so the entry is taken off the iterator to read it.
fn peek_revision(side: &mut Side<'_>, upper: Revision) -> Option<Revision> {
    if side.taken.is_none() {
        let key = side.iter.next()?;
        side.taken = Some((revision_of(side.iter.key()), key));
    }
    let (revision, _) = side.taken?;
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
            self.live.taken.take()
        } else {
            self.dead.taken.take()
        }
        .expect("peeked");
        let object = tree
            .value(key)
            .expect("revision index disagrees with its tree");
        Some(Change {
            key: (**key).clone(),
            value: object.value.clone(),
            revision: object.revision,
            deleted,
        })
    }
}
