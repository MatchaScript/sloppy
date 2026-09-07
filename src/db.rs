//! The database: tables, snapshot reads, single-writer transactions.
//!
//! One revision counter covers the whole `Db`. A commit that changes anything
//! bumps it by one, builds a new root and swaps it in behind an `RwLock`
//! held only for the assignment, so readers never see a half-applied commit and
//! never block the writer.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use tokio::sync::watch;

use crate::snapshot_tree::{self as tree, Tree};
use crate::watch::{Covered, Watch};

pub type Revision = u64;
pub type Key = Box<[u8]>;

/// One version of one key. A deletion keeps the value it removed, so a reader
/// of the change stream is told what went away.
pub struct Version<V> {
    pub revision: Revision,
    pub value: Arc<V>,
    pub deleted: bool,
}

/// Cloning a version is one `Arc` bump. Derived would ask for `V: Clone`,
/// which no table needs.
impl<V> Clone for Version<V> {
    fn clone(&self) -> Self {
        Self {
            revision: self.revision,
            value: self.value.clone(),
            deleted: self.deleted,
        }
    }
}

/// Every version of one key the database still holds, oldest first, and never
/// empty. The newest version says whether the key is there: a row whose newest
/// version is a tombstone reads as absent and stays until it is collected.
struct Row<V> {
    versions: Arc<[Version<V>]>,
}

/// A path copy shares the version list; a write rebuilds it. Its length is the
/// number of writes to this key since the last compaction.
impl<V> Clone for Row<V> {
    fn clone(&self) -> Self {
        Self {
            versions: self.versions.clone(),
        }
    }
}

impl<V> Row<V> {
    fn newest(&self) -> &Version<V> {
        self.versions.last().expect("a row holds a version")
    }

    /// The value at this key, unless the newest version is a tombstone.
    fn live(&self) -> Option<(&V, Revision)> {
        let newest = self.newest();
        (!newest.deleted).then(|| (newest.value.as_ref(), newest.revision))
    }

    /// The same, as the row holds it: what a write hands back and what the
    /// indexes list.
    fn held(&self) -> Option<&Arc<V>> {
        let newest = self.newest();
        (!newest.deleted).then_some(&newest.value)
    }

    /// The row `version` leaves, and whether it takes a change record.
    ///
    /// A second write of one key in one commit replaces the version the first
    /// left and reuses its record, so one key leaves one record per commit.
    /// Versions at or below `compacted` go, bar the newest: nothing may read
    /// them any more.
    fn written(row: Option<&Self>, version: Version<V>, compacted: Revision) -> (Self, bool) {
        let held: &[Version<V>] = row.map_or(&[], |row| &row.versions);
        let first = held
            .last()
            .is_none_or(|newest| newest.revision < version.revision);
        // A second write of one key in one commit drops the version the first
        // left; the new one takes its place at the end either way.
        let kept = if first { held } else { &held[..held.len() - 1] };
        let keep = kept
            .iter()
            .position(|v| v.revision > compacted)
            .unwrap_or(kept.len());
        let versions = kept[keep..]
            .iter()
            .cloned()
            .chain(std::iter::once(version))
            .collect();
        (Self { versions }, first)
    }
}

// -------------------------------------------------------------- change stream

/// Key of a change record: the table, the revision of the commit, and the place
/// the record took in it, each big-endian.
///
/// The table comes first, so one table's records are a run of their own that a
/// reader scans and the collection drops a prefix of. The revision comes next,
/// so that run is in commit order; the sequence number keeps the records of one
/// commit apart and in the order they were written.
fn stream_key(table: usize, revision: Revision, seq: u32) -> Key {
    let table = u32::try_from(table).expect("a `Db` holds fewer than 4 billion tables");
    let mut k = Vec::with_capacity(16);
    k.extend_from_slice(&table.to_be_bytes());
    k.extend_from_slice(&revision.to_be_bytes());
    k.extend_from_slice(&seq.to_be_bytes());
    k.into()
}

fn table_of(key: &[u8]) -> usize {
    let head: [u8; 4] = key[..4].try_into().expect("change stream key");
    u32::from_be_bytes(head) as usize
}

fn revision_of(key: &[u8]) -> Revision {
    let head: [u8; 8] = key[4..12].try_into().expect("change stream key");
    Revision::from_be_bytes(head)
}

/// Resolves index hits through the primary tree. An index entry carries no
/// value: the primary key is the tail of its key, past the `skip` bytes the
/// search prefix took.
fn resolve<'a, V>(
    entry: &'a TableEntry<V>,
    mut hits: tree::Iter<'a, ()>,
    skip: usize,
) -> impl Iterator<Item = (&'a V, Revision)> + use<'a, V> {
    std::iter::from_fn(move || {
        hits.next()?;
        // A deletion drops the index entries with the value, so every hit
        // resolves to a live row.
        Some(
            entry
                .primary
                .get(&hits.key()[skip..])
                .and_then(Row::live)
                .expect("index disagrees with the primary tree"),
        )
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
/// The length ends the index key whatever bytes it holds, so a search for `a`
/// matches neither the entries of `ab` nor those of an index key that starts
/// with `a` and goes on.
fn index_prefix(index_key: &[u8]) -> Vec<u8> {
    let len = u32::try_from(index_key.len()).expect("an index key is shorter than 4 GiB");
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
/// the entries of one index key apart and is what the hit resolves through.
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
    primary: Tree<Row<V>>,
    /// One tree per registered index, in registration order. The key is
    /// `index_entry(index key, primary key)` and there is no value.
    indexes: Vec<(Index<V>, Tree<()>)>,
    trackers: Vec<Weak<AtomicU64>>,
    /// Highest delete revision that [`Db::compact`] removed before every
    /// tracker had seen it. A reader below this has lost a change.
    lost: Revision,
    /// Lowest revision a tombstoned row is left at, or `Revision::MAX` when
    /// none is left. A sweep is due once a bound reaches it; see
    /// [`AnyTable::collected`]. Rewriting a tombstoned key leaves it low, and
    /// the next sweep puts it right.
    buried: Revision,
    primary_key: fn(&V) -> Key,
}

impl<V> TableEntry<V> {
    fn new(primary_key: fn(&V) -> Key, indexes: Vec<Index<V>>) -> Self {
        Self {
            revision: 0,
            primary: Tree::new(),
            indexes: indexes.into_iter().map(|def| (def, Tree::new())).collect(),
            trackers: Vec::new(),
            lost: 0,
            buried: Revision::MAX,
            primary_key,
        }
    }
}

/// What one table's change readers hold back.
struct Readers {
    trackers: Vec<Weak<AtomicU64>>,
    /// The lowest revision every reader has observed; with no reader left, the
    /// table's own revision, so everything it has may go.
    watermark: Revision,
    /// A tracker whose reader is gone was dropped.
    pruned: bool,
    lost: Revision,
}

/// The type-erased face of `TableEntry<V>`: what the `Root` can do without
/// knowing the value type.
trait AnyTable: Any + Send + Sync {
    fn readers(&self) -> Readers;

    /// A copy with the trackers pruned, `lost` raised, and the rows a tombstone
    /// at or below `dead` ended dropped.
    fn collected(
        &self,
        trackers: Vec<Weak<AtomicU64>>,
        lost: Revision,
        dead: Revision,
    ) -> Arc<dyn AnyTable>;

    /// The lowest revision a tombstoned row is left at.
    fn buried(&self) -> Revision;

    /// The revision of the last commit that changed this table.
    fn revision(&self) -> Revision;
}

impl<V: Send + Sync + 'static> AnyTable for TableEntry<V> {
    fn readers(&self) -> Readers {
        let mut trackers = Vec::with_capacity(self.trackers.len());
        let mut watermark = Revision::MAX;
        for weak in &self.trackers {
            if let Some(tracker) = weak.upgrade() {
                watermark = watermark.min(tracker.load(Ordering::Relaxed));
                trackers.push(weak.clone());
            }
        }
        Readers {
            pruned: trackers.len() != self.trackers.len(),
            watermark: if trackers.is_empty() {
                self.revision
            } else {
                watermark
            },
            trackers,
            lost: self.lost,
        }
    }

    fn collected(
        &self,
        trackers: Vec<Weak<AtomicU64>>,
        lost: Revision,
        dead: Revision,
    ) -> Arc<dyn AnyTable> {
        let (primary, buried) = if dead >= self.buried {
            swept(&self.primary, dead)
        } else {
            (self.primary.clone(), self.buried)
        };
        Arc::new(TableEntry {
            revision: self.revision,
            primary,
            indexes: self.indexes.clone(),
            trackers,
            lost,
            buried,
            primary_key: self.primary_key,
        })
    }

    fn buried(&self) -> Revision {
        self.buried
    }

    fn revision(&self) -> Revision {
        self.revision
    }
}

/// The primary tree without the rows a tombstone at or below `dead` ended, and
/// the lowest revision the tombstoned rows it leaves are at.
///
/// Their deletion has reached every reader and is below the compaction bound,
/// so nothing may ask for the key or its versions again.
// ponytail: one walk of the whole tree, run only when a tombstone is at or
// below the bound. Hold the tombstoned keys in the entry if a workload deletes
// often enough for the walk to show up.
fn swept<V>(primary: &Tree<Row<V>>, dead: Revision) -> (Tree<Row<V>>, Revision) {
    let mut doomed: Vec<Key> = Vec::new();
    let mut buried = Revision::MAX;
    let mut it = primary.iter();
    while let Some(row) = it.next() {
        let newest = row.newest();
        if newest.deleted {
            if newest.revision <= dead {
                doomed.push(it.key().into());
            } else {
                buried = buried.min(newest.revision);
            }
        }
    }
    if doomed.is_empty() {
        return (primary.clone(), buried);
    }
    let mut txn = primary.txn();
    for key in &doomed {
        txn.delete(key);
    }
    (txn.commit(), buried)
}

/// Drops the change records every reader of their table has already seen, and
/// the ones [`Db::compact`] gave up on, then the rows whose deletion both
/// bounds have passed. One run over every table's partition.
fn collect(tables: &mut [Arc<dyn AnyTable>], changes: &mut Tree<Key>, compacted: Revision) {
    let mut txn = changes.txn();
    for (pos, table) in tables.iter_mut().enumerate() {
        let readers = table.readers();
        let bound = readers.watermark.max(compacted);
        let dead = readers.watermark.min(compacted);
        let mut lost = readers.lost;
        let mut doomed: Vec<Key> = Vec::new();
        let mut it = txn.lower_bound(&stream_key(pos, 0, 0));
        while it.next().is_some() {
            let key = it.key();
            if table_of(key) != pos {
                break;
            }
            let revision = revision_of(key);
            if revision > bound {
                break;
            }
            if revision > readers.watermark {
                lost = revision;
            }
            doomed.push(key.into());
        }
        drop(it);
        for key in &doomed {
            txn.delete(key);
        }
        if !doomed.is_empty() || readers.pruned || lost != readers.lost || dead >= table.buried() {
            *table = table.collected(readers.trackers, lost, dead);
        }
    }
    *changes = txn.commit();
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
    /// Every table's change records, one run per table; see [`stream_key`]. A
    /// record is the primary key that was written: the value is in the row.
    changes: Tree<Key>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic between the two root swaps leaves the root untouched, and one
    // after them leaves it fully applied, so a poisoned lock guards no torn
    // state.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a [`Watch`] and a [`ReadTxn`] keep hold of: the visible root, and the
/// receiving end of the revision channel every commit sends on.
///
/// The sending end is the [`Db`]'s, so dropping the database closes the
/// channel and releases every watch parked on it, which is what a watch
/// outliving its database has to wait for.
struct Shared {
    root: RwLock<Arc<Root>>,
    revisions: watch::Receiver<Revision>,
}

impl Shared {
    fn snapshot(&self) -> Arc<Root> {
        self.root
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The one place the visible root is replaced.
    fn install(&self, root: Arc<Root>) {
        let old = {
            let mut current = self.root.write().unwrap_or_else(PoisonError::into_inner);
            std::mem::replace(&mut *current, root)
        };
        // A value's destructor may read this Db, so the old root must outlive
        // the root write guard.
        drop(old);
    }

    /// A watch on `covers`, as of `at`.
    fn cover(self: &Arc<Self>, at: Revision, covers: Covers) -> Watch {
        // The revision this receiver last saw is the one it was cloned at, and
        // what the watch covers is read fresh anyway, so it starts here rather
        // than with every commit since the channel was opened.
        let mut revisions = self.revisions.clone();
        revisions.mark_unchanged();
        Watch::new(
            Arc::new(Cover {
                shared: self.clone(),
                at,
                covers,
            }),
            revisions,
        )
    }
}

/// What one watch covers.
#[derive(Clone, Copy)]
enum Covers {
    /// Any commit that bumps the revision.
    Db,
    /// Any change to one table.
    Table(usize),
}

/// A watch's side of the database: it reads the visible root every time it is
/// asked, so it needs no registration and a watch taken on a snapshot the
/// database has already moved past reports the change straight away.
///
/// Both covers compare a revision, which only rises, so a watch reports every
/// commit to what it covers and never has to read the data.
struct Cover {
    shared: Arc<Shared>,
    /// The revision the watch was taken at: the `Db`'s for [`Covers::Db`], the
    /// table's for [`Covers::Table`].
    at: Revision,
    covers: Covers,
}

impl Covered for Cover {
    fn changed(&self) -> bool {
        let root = self.shared.snapshot();
        match self.covers {
            Covers::Db => root.revision > self.at,
            Covers::Table(table) => root.tables[table].revision() > self.at,
        }
    }
}

/// The database. Registration and writes are serialized by one writer lock;
/// reads take an `Arc` of the current root and release the lock at once.
///
/// The locks are taken in the order `write`, `root`; every path takes a
/// subsequence of that, so none of them cycle.
pub struct Db {
    shared: Arc<Shared>,
    /// The one sender on the revision channel; see [`Shared::revisions`].
    revisions: watch::Sender<Revision>,
    write: Mutex<()>,
}

impl Default for Db {
    fn default() -> Self {
        Self::new()
    }
}

impl Db {
    #[must_use]
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let root = Arc::new(Root {
            db: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            revision: 0,
            compacted: 0,
            tables: Vec::new(),
            changes: Tree::new(),
        });
        let (revisions, receiver) = watch::channel(0);
        Self {
            shared: Arc::new(Shared {
                root: RwLock::new(root),
                revisions: receiver,
            }),
            revisions,
            write: Mutex::new(()),
        }
    }

    fn snapshot(&self) -> Arc<Root> {
        self.shared.snapshot()
    }

    /// Completes on the next commit that bumps the revision. A reader that
    /// persists the change stream waits on this and scans what it finds.
    #[must_use]
    pub fn watch(&self) -> Watch {
        let root = self.snapshot();
        self.shared.cover(root.revision, Covers::Db)
    }

    /// Registers a table. Takes the writer lock and does not bump the revision.
    ///
    /// # Panics
    ///
    /// If two indexes have the same name.
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
        let mut root = (*self.snapshot()).clone();
        let pos = root.tables.len();
        root.tables
            .push(Arc::new(TableEntry::new(primary_key, indexes.to_vec())));
        let db = root.db;
        self.shared.install(Arc::new(root));
        Table {
            db,
            pos,
            name,
            _v: PhantomData,
        }
    }

    #[must_use]
    pub fn read(&self) -> ReadTxn {
        ReadTxn(self.snapshot(), self.shared.clone())
    }

    /// Opens the write transaction on the visible root, at the revision after
    /// it. Blocks until the previous one commits or is dropped.
    #[must_use]
    pub fn write(&self) -> WriteTxn<'_> {
        self.open(None)
    }

    /// The same, at the revision the caller names: what this transaction writes
    /// carries `rev`, and its commit leaves the database there.
    ///
    /// # Panics
    ///
    /// If `rev` is not past the revision of the visible root.
    #[must_use]
    pub fn write_at(&self, rev: Revision) -> WriteTxn<'_> {
        self.open(Some(rev))
    }

    fn open(&self, at: Option<Revision>) -> WriteTxn<'_> {
        let guard = lock(&self.write);
        let root = self.snapshot();
        let revision = at.unwrap_or(root.revision + 1);
        assert!(
            revision > root.revision,
            "revision {revision} is not past the visible {}",
            root.revision
        );
        let pending = std::iter::repeat_with(|| None)
            .take(root.tables.len())
            .collect();
        WriteTxn {
            db: self,
            _guard: guard,
            root,
            revision,
            pending,
            changes: None,
            seq: 0,
            dirty: false,
        }
    }

    /// Drops the change records at or below `rev` whether or not the change
    /// iterators have read them; those iterators then fail with [`Compacted`].
    ///
    /// This is not history: it does not bump the revision.
    pub fn compact(&self, rev: Revision) {
        let _guard = lock(&self.write);
        let mut root = (*self.snapshot()).clone();
        root.compacted = rev.min(root.revision).max(root.compacted);
        let compacted = root.compacted;
        collect(&mut root.tables, &mut root.changes, compacted);
        self.shared.install(Arc::new(root));
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
pub struct ReadTxn(Arc<Root>, Arc<Shared>);

impl ReadTxn {
    /// The revision of the commit this snapshot was taken after.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.0.revision
    }

    /// A watch on `covers`, as this snapshot's table holds it.
    fn cover<V: Send + Sync + 'static>(&self, table: &Table<V>, covers: Covers) -> Watch {
        self.1.cover(table.entry(&self.0).revision, covers)
    }
}

/// What one table's reads run against: the root of a [`ReadTxn`], or, for a
/// table a [`WriteTxn`] has touched, that transaction's own pending tree.
///
/// The trait hands back the results rather than the tree they came from, which
/// keeps `tree::Node` and the two tree types out of its signature.
trait Snapshot {
    fn value<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<&Row<V>>;

    fn prefix<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        prefix: &[u8],
    ) -> tree::Iter<'_, Row<V>>;

    fn lower_bound<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> tree::Iter<'_, Row<V>>;

    fn all<V: Send + Sync + 'static>(&self, table: &Table<V>) -> tree::Iter<'_, Row<V>>;
}

impl Snapshot for ReadTxn {
    fn value<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<&Row<V>> {
        table.entry(&self.0).primary.get(key)
    }

    fn prefix<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        prefix: &[u8],
    ) -> tree::Iter<'_, Row<V>> {
        table.entry(&self.0).primary.prefix(prefix)
    }

    fn lower_bound<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> tree::Iter<'_, Row<V>> {
        table.entry(&self.0).primary.lower_bound(key)
    }

    fn all<V: Send + Sync + 'static>(&self, table: &Table<V>) -> tree::Iter<'_, Row<V>> {
        table.entry(&self.0).primary.iter()
    }
}

/// A table this transaction has written reads from its pending tree, so the
/// writes are there; every other table reads the root the transaction opened
/// on. Reading opens no slot.
impl Snapshot for WriteTxn<'_> {
    fn value<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<&Row<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.get(key),
            None => table.entry(&self.root).primary.get(key),
        }
    }

    fn prefix<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        prefix: &[u8],
    ) -> tree::Iter<'_, Row<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.prefix(prefix),
            None => table.entry(&self.root).primary.prefix(prefix),
        }
    }

    fn lower_bound<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> tree::Iter<'_, Row<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.lower_bound(key),
            None => table.entry(&self.root).primary.lower_bound(key),
        }
    }

    fn all<V: Send + Sync + 'static>(&self, table: &Table<V>) -> tree::Iter<'_, Row<V>> {
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
        txn.value(self, key)?.live()
    }

    /// Every version of `key` this database still holds, oldest first, ending
    /// with a tombstone if the key was deleted. Empty if the key was never
    /// written, or if its row has been collected.
    ///
    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    #[must_use]
    pub fn versions<'a>(&self, txn: &'a impl Snapshot, key: &[u8]) -> &'a [Version<V>] {
        txn.value(self, key).map_or(&[], |row| &row.versions)
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
        txn.prefix(self, prefix).filter_map(Row::live)
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
        txn.lower_bound(self, key).filter_map(Row::live)
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
        let prefix = index_prefix(key);
        let hits = self.index_tree(entry, index).prefix(&prefix);
        resolve(entry, hits, prefix.len())
    }

    fn index_tree<'a>(&self, entry: &'a TableEntry<V>, index: &'static str) -> &'a Tree<()> {
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
        txn.all(self).filter_map(Row::live)
    }

    /// The revisions of the change records this table still holds, in stream
    /// order. Test helper.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[doc(hidden)]
    #[must_use]
    pub fn held_records(&self, txn: &ReadTxn) -> Vec<Revision> {
        let mut it = txn.0.changes.lower_bound(&stream_key(self.pos, 0, 0));
        let mut held = Vec::new();
        while it.next().is_some() {
            if table_of(it.key()) != self.pos {
                break;
            }
            held.push(revision_of(it.key()));
        }
        held
    }

    /// Whether this table's run of the change stream holds a record.
    fn recorded(&self, root: &Root) -> bool {
        let mut it = root.changes.lower_bound(&stream_key(self.pos, 0, 0));
        it.next().is_some() && table_of(it.key()) == self.pos
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
                indexes: entry
                    .indexes
                    .iter()
                    .map(|(def, tree)| (*def, tree.txn()))
                    .collect(),
                trackers: entry.trackers.clone(),
                new_trackers: Vec::new(),
                lost: entry.lost,
                buried: entry.buried,
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
        let (revision, compacted) = (txn.revision, txn.root.compacted);
        txn.dirty = true;
        let value = Arc::new(value);
        let (key, was, first) = {
            let pending = self.pending(txn);
            pending.written = true;
            let key = (pending.primary_key)(&value);
            let was = pending.primary.get(&key).and_then(Row::held).cloned();
            let version = Version {
                revision,
                value: value.clone(),
                deleted: false,
            };
            let (row, first) = Row::written(pending.primary.get(&key), version, compacted);
            pending.primary.insert(&key, row);
            for (def, tree) in &mut pending.indexes {
                // Only the difference of the two key sets touches the tree, so
                // an update that keeps a value listed under the same index key
                // leaves that entry alone.
                let had = was
                    .as_ref()
                    .map(|was| sorted((def.keys)(was)))
                    .unwrap_or_default();
                let is = sorted((def.keys)(&value));
                for k in had.iter().filter(|k| is.binary_search(k).is_err()) {
                    tree.delete(&index_entry(k, &key));
                }
                for k in is.iter().filter(|k| had.binary_search(k).is_err()) {
                    tree.insert(&index_entry(k, &key), ());
                }
            }
            (key, was, first)
        };
        if first {
            let seq = txn.take_seq();
            txn.record(self.pos, revision, seq, key);
        }
        was
    }

    /// Ends the row with a tombstone that keeps the value, and returns it. The
    /// key reads as absent from here on; the row stays until it is collected.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn delete(&self, txn: &mut WriteTxn<'_>, key: &[u8]) -> Option<Arc<V>> {
        let (revision, compacted) = (txn.revision, txn.root.compacted);
        let (old, first) = {
            let pending = self.pending(txn);
            let old = pending.primary.get(key).and_then(Row::held)?.clone();
            let version = Version {
                revision,
                value: old.clone(),
                deleted: true,
            };
            let (row, first) = Row::written(pending.primary.get(key), version, compacted);
            pending.primary.insert(key, row);
            pending.written = true;
            pending.buried = pending.buried.min(revision);
            for (def, tree) in &mut pending.indexes {
                for k in (def.keys)(&old) {
                    tree.delete(&index_entry(&k, key));
                }
            }
            (old, first)
        };
        txn.dirty = true;
        if first {
            let seq = txn.take_seq();
            txn.record(self.pos, revision, seq, key.into());
        }
        Some(old)
    }

    /// Puts `versions` at `key`, in place of whatever is there, and leaves no
    /// change record: this is how a table is rebuilt from what was persisted,
    /// not a write for readers to follow.
    ///
    /// The indexes follow the newest version, so a row loaded as deleted is
    /// listed nowhere.
    ///
    /// The table must hold no change record. A record is resolved through the
    /// row it names, and the loaded versions need not hold the revision it was
    /// written at; a table being rebuilt is one no reader has fallen behind on.
    ///
    /// # Panics
    ///
    /// If `versions` is empty, is not in ascending revision order, or ends past
    /// this transaction's revision, if the table still holds a change record,
    /// or if the table was not registered in this `Db`.
    pub fn load(&self, txn: &mut WriteTxn<'_>, key: &[u8], versions: Vec<Version<V>>) {
        let newest = versions.last().expect("a loaded row holds a version");
        assert!(
            !self.recorded(&txn.root),
            "table {} holds change records a loaded row would strand",
            self.name
        );
        assert!(
            newest.revision <= txn.revision,
            "loaded revision {} is past the transaction's {}",
            newest.revision,
            txn.revision
        );
        assert!(
            versions.iter().is_sorted_by(|a, b| a.revision < b.revision),
            "loaded versions are not in ascending revision order"
        );
        let tombstoned = newest.deleted.then_some(newest.revision);
        txn.dirty = true;
        let pending = self.pending(txn);
        pending.written = true;
        if let Some(revision) = tombstoned {
            pending.buried = pending.buried.min(revision);
        }
        let was = pending.primary.get(key).and_then(Row::held).cloned();
        let row = Row {
            versions: versions.into(),
        };
        let is = row.held().cloned();
        pending.primary.insert(key, row);
        for (def, tree) in &mut pending.indexes {
            for k in was.iter().flat_map(|was| (def.keys)(was)) {
                tree.delete(&index_entry(&k, key));
            }
            for k in is.iter().flat_map(|is| (def.keys)(is)) {
                tree.insert(&index_entry(&k, key), ());
            }
        }
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
    primary: tree::Txn<Row<V>>,
    indexes: Vec<(Index<V>, tree::Txn<()>)>,
    trackers: Vec<Weak<AtomicU64>>,
    new_trackers: Vec<Arc<AtomicU64>>,
    lost: Revision,
    buried: Revision,
    primary_key: fn(&V) -> Key,
}

trait AnyPending: Any {
    /// Builds the new table entry, at `revision` if this table was written.
    fn install(self: Box<Self>, revision: Revision) -> Arc<dyn AnyTable>;
}

impl<V: Send + Sync + 'static> AnyPending for Pending<V> {
    fn install(self: Box<Self>, revision: Revision) -> Arc<dyn AnyTable> {
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
        }
        let indexes = this
            .indexes
            .into_iter()
            .map(|(def, txn)| (def, txn.commit()))
            .collect();
        Arc::new(TableEntry {
            revision,
            primary: this.primary.commit(),
            indexes,
            trackers,
            lost: this.lost,
            buried: this.buried,
            primary_key: this.primary_key,
        })
    }
}

/// The write transaction. Dropping it aborts: nothing was ever visible.
pub struct WriteTxn<'a> {
    db: &'a Db,
    _guard: MutexGuard<'a, ()>,
    /// The root this transaction opened on; the commit clones it to build the
    /// new one.
    root: Arc<Root>,
    /// The revision this transaction writes at, and the one its commit leaves
    /// the database at.
    revision: Revision,
    /// One slot per table, `Some` once the table is touched.
    pending: Vec<Option<Box<dyn AnyPending>>>,
    /// The change stream, opened on the first record.
    changes: Option<tree::Txn<Key>>,
    /// How many records this transaction has written; the next one's place.
    seq: u32,
    dirty: bool,
}

impl WriteTxn<'_> {
    /// The place the next record takes in this transaction.
    fn take_seq(&mut self) -> u32 {
        let seq = self.seq;
        self.seq += 1;
        seq
    }

    fn stream(&mut self) -> &mut tree::Txn<Key> {
        if self.changes.is_none() {
            self.changes = Some(self.root.changes.txn());
        }
        self.changes.as_mut().expect("just opened")
    }

    /// Notes that `key` was written, which is all a record is: the reader
    /// resolves it against the row.
    fn record(&mut self, table: usize, revision: Revision, seq: u32, key: Key) {
        self.stream().insert(&stream_key(table, revision, seq), key);
    }
    /// Settles the trees, swaps the new root in and wakes the watches, all
    /// under the writer lock. Returns the new revision, which is the previous
    /// one if nothing was written.
    // The revision is worth ignoring; the commit itself is the point.
    #[allow(clippy::must_use_candidate)]
    pub fn commit(self) -> Revision {
        let WriteTxn {
            db,
            _guard: guard,
            root: parent,
            revision,
            pending,
            changes,
            seq: _,
            dirty,
        } = self;
        let root = if pending.iter().all(Option::is_none) {
            parent
        } else {
            let mut root = (*parent).clone();
            let revision = if dirty { revision } else { root.revision };
            for (pos, table) in pending.into_iter().enumerate() {
                if let Some(table) = table {
                    root.tables[pos] = table.install(revision);
                }
            }
            if let Some(changes) = changes {
                root.changes = changes.commit();
            }
            // ponytail: every commit walks to the head of every table's run in
            // the stream, which costs one lookup per untouched table. Track the
            // tables with records to collect in the root if the count grows.
            let compacted = root.compacted;
            collect(&mut root.tables, &mut root.changes, compacted);
            root.revision = revision;
            Arc::new(root)
        };

        let revision = root.revision;
        db.shared.install(root);
        if dirty {
            // Readers see the new root before they are told about it, so a
            // watch that wakes here reads what the commit left.
            db.revisions.send_replace(revision);
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

/// A reader of a table's changes. It holds the records it has not seen in the
/// change stream; dropping it releases them at the next commit.
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
        let from = stream_key(self.table.pos, observed.saturating_add(1), 0);
        let changes = Changes {
            iter: txn.0.changes.lower_bound(&from),
            entry,
            table: self.table.pos,
        };
        // A stale snapshot must not rewind what a newer one already observed.
        self.tracker.fetch_max(entry.revision, Ordering::Relaxed);
        Ok((
            changes,
            txn.cover(&self.table, Covers::Table(self.table.pos)),
        ))
    }
}

/// One table's run of the change stream, from where the reader left off. A
/// record names a key; what the reader is told is the version of that key the
/// record's commit left.
struct Changes<'a, V> {
    iter: tree::Iter<'a, Key>,
    entry: &'a TableEntry<V>,
    table: usize,
}

impl<V: Send + Sync + 'static> Iterator for Changes<'_, V> {
    type Item = Change<V>;

    // ponytail: the primary key is copied into every yielded change. Hand out a
    // borrow of the record instead if the copies ever show up.
    fn next(&mut self) -> Option<Self::Item> {
        let key = self.iter.next()?;
        if table_of(self.iter.key()) != self.table {
            return None;
        }
        let revision = revision_of(self.iter.key());
        // A version is dropped no earlier than the record of the commit that
        // wrote it, so both are still here. The versions run oldest first and
        // the record names the newest of them unless a later commit wrote the
        // key again, so the search starts at the end.
        let version = self
            .entry
            .primary
            .get(key)
            .expect("a record outlived its row")
            .versions
            .iter()
            .rev()
            .find(|v| v.revision == revision)
            .expect("a record outlived its version");
        Some(Change {
            key: key.clone(),
            value: version.value.clone(),
            revision,
            deleted: version.deleted,
        })
    }
}
