//! The database: tables, snapshot reads, single-writer transactions.
//!
//! One revision counter covers the whole `Db`. A commit that changes anything
//! bumps it by one, builds a new root and swaps it in behind an `RwLock`
//! held only for the assignment, so readers never see a half-applied commit and
//! never block the writer.

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use tokio::sync::watch;

use crate::tree::{self, Tree};
use crate::watch::{Covered, Watch};

pub type Revision = u64;
pub type Key = Box<[u8]>;

/// A stored value with the revision of the commit that wrote it and the place
/// its change record took in that commit.
struct Object<V> {
    value: Arc<V>,
    revision: Revision,
    seq: u32,
}

/// The tree holds the object inline and clones it on a path copy, which is one
/// `Arc` bump. Derived would ask for `V: Clone`, which no table needs.
impl<V> Clone for Object<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            revision: self.revision,
            seq: self.seq,
        }
    }
}

// -------------------------------------------------------------- change stream

/// One write, as the change stream holds it. The stream carries the records of
/// every table, so the value is erased and the reader of one table casts it
/// back; the key and the flag are the same for all of them.
struct Record {
    key: Key,
    value: Arc<dyn Any + Send + Sync>,
    deleted: bool,
}

/// The tree holds the record behind an `Arc`, so a path copy of a leaf is one
/// bump per record rather than a copy of every key.
type AnyRecord = Arc<Record>;

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

fn row<V>(obj: &Object<V>) -> (&V, Revision) {
    (obj.value.as_ref(), obj.revision)
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
        let object = entry
            .primary
            .get(&hits.key()[skip..])
            .expect("index disagrees with the primary tree");
        Some((object.value.as_ref(), object.revision))
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
    primary: Tree<Object<V>>,
    /// One tree per registered index, in registration order. The key is
    /// `index_entry(index key, primary key)` and there is no value.
    indexes: Vec<(Index<V>, Tree<()>)>,
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
            indexes: indexes.into_iter().map(|def| (def, Tree::new())).collect(),
            trackers: Vec::new(),
            lost: 0,
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

    /// A copy with the trackers pruned and `lost` raised.
    fn collected(&self, trackers: Vec<Weak<AtomicU64>>, lost: Revision) -> Arc<dyn AnyTable>;

    /// The revision of the last commit that changed this table.
    fn revision(&self) -> Revision;

    /// The revision the object at `key` carries, if it is there.
    fn stamp(&self, key: &[u8]) -> Option<Revision>;

    /// How many entries start with `prefix`, and the newest revision among
    /// them. The pair moves whenever an entry under `prefix` is written,
    /// removed, or added: an addition or a rewrite raises the revision, and a
    /// removal on its own lowers the count.
    fn spread(&self, prefix: &[u8]) -> (usize, Revision);

    /// The index entries under `prefix` in the named index.
    fn listed(&self, index: &'static str, prefix: &[u8]) -> Vec<Key>;
}

/// The index entries under `prefix`, which are what a watch on an index key
/// compares.
fn listing(tree: &Tree<()>, prefix: &[u8]) -> Vec<Key> {
    let mut hits = tree.prefix(prefix);
    let mut out = Vec::new();
    while hits.next().is_some() {
        out.push(hits.key().into());
    }
    out
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

    fn collected(&self, trackers: Vec<Weak<AtomicU64>>, lost: Revision) -> Arc<dyn AnyTable> {
        Arc::new(TableEntry {
            revision: self.revision,
            primary: self.primary.clone(),
            indexes: self.indexes.clone(),
            trackers,
            lost,
            primary_key: self.primary_key,
        })
    }

    fn revision(&self) -> Revision {
        self.revision
    }

    fn stamp(&self, key: &[u8]) -> Option<Revision> {
        self.primary.get(key).map(|object| object.revision)
    }

    fn spread(&self, prefix: &[u8]) -> (usize, Revision) {
        let mut count = 0;
        let mut newest = 0;
        for object in self.primary.prefix(prefix) {
            count += 1;
            newest = newest.max(object.revision);
        }
        (count, newest)
    }

    fn listed(&self, index: &'static str, prefix: &[u8]) -> Vec<Key> {
        let tree = self
            .indexes
            .iter()
            .find_map(|(def, tree)| (def.name == index).then_some(tree))
            .expect("the index was resolved when the watch was taken");
        listing(tree, prefix)
    }
}

/// Drops the change records every reader of their table has already seen, and
/// the ones [`Db::compact`] gave up on. One run over every table's partition.
fn collect(tables: &mut [Arc<dyn AnyTable>], changes: &mut Tree<AnyRecord>, compacted: Revision) {
    let mut txn = changes.txn();
    for (pos, table) in tables.iter_mut().enumerate() {
        let readers = table.readers();
        let bound = readers.watermark.max(compacted);
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
        if !doomed.is_empty() || readers.pruned || lost != readers.lost {
            *table = table.collected(readers.trackers, lost);
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
    /// Every table's change records, one run per table; see [`stream_key`].
    changes: Tree<AnyRecord>,
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

/// What one watch covers, with what it held when the watch was taken.
enum Covers {
    /// Any commit that bumps the revision.
    Db,
    /// Any change to one table.
    Table(usize),
    /// One key of a table, and the revision it carried.
    Key(usize, Key, Option<Revision>),
    /// Every key of a table under a prefix, and their count and newest
    /// revision.
    Prefix(usize, Key, (usize, Revision)),
    /// The entries one index key listed.
    Index(usize, &'static str, Key, Vec<Key>),
}

/// A watch's side of the database: it reads the visible root every time it is
/// asked, so it needs no registration and a watch taken on a snapshot the
/// database has already moved past reports the change straight away.
///
/// What it reads is the state of what it covers, not the writes that led
/// there, so it reports the difference between two states: a key, prefix or
/// index key left as it was by the commits since the watch was taken has
/// nothing to report, whatever those commits wrote. [`Covers::Db`] and
/// [`Covers::Table`] compare a revision, which only rises.
struct Cover {
    shared: Arc<Shared>,
    /// The revision the watch was taken at: the `Db`'s for [`Covers::Db`], the
    /// table's otherwise.
    at: Revision,
    covers: Covers,
}

impl Cover {
    /// Whether the table moved at all, and then whether it moved here. The
    /// first test is what keeps a watch on an untouched table free.
    fn moved(
        &self,
        root: &Root,
        table: usize,
        differs: impl FnOnce(&dyn AnyTable) -> bool,
    ) -> bool {
        let entry = &root.tables[table];
        entry.revision() > self.at && differs(&**entry)
    }
}

impl Covered for Cover {
    fn changed(&self) -> bool {
        let root = self.shared.snapshot();
        match &self.covers {
            Covers::Db => root.revision > self.at,
            Covers::Table(table) => root.tables[*table].revision() > self.at,
            Covers::Key(table, key, was) => self.moved(&root, *table, |e| e.stamp(key) != *was),
            Covers::Prefix(table, prefix, was) => {
                self.moved(&root, *table, |e| e.spread(prefix) != *was)
            }
            Covers::Index(table, index, prefix, was) => {
                self.moved(&root, *table, |e| e.listed(index, prefix) != *was)
            }
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

    /// Opens the write transaction on the visible root. Blocks until the
    /// previous one commits or is dropped.
    #[must_use]
    pub fn write(&self) -> WriteTxn<'_> {
        let guard = lock(&self.write);
        let root = self.snapshot();
        let pending = std::iter::repeat_with(|| None)
            .take(root.tables.len())
            .collect();
        WriteTxn {
            db: self,
            _guard: guard,
            root,
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
    fn value<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<&Object<V>>;

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
    fn value<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<&Object<V>> {
        table.entry(&self.0).primary.get(key)
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
    fn value<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<&Object<V>> {
        match table.opened(self) {
            Some(pending) => pending.primary.get(key),
            None => table.entry(&self.root).primary.get(key),
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

    /// The value at `key`, plus a watch on it. The watch compares the key
    /// with what it held here, so a value written and removed again before the
    /// watch is asked leaves it open.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[must_use]
    pub fn get_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        key: &[u8],
    ) -> (Option<(&'a V, Revision)>, Watch) {
        let object = self.entry(&txn.0).primary.get(key);
        let covers = Covers::Key(self.pos, key.into(), object.map(|o| o.revision));
        (
            object.map(|o| (o.value.as_ref(), o.revision)),
            txn.cover(self, covers),
        )
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

    /// Every entry under `prefix`, plus a watch on them. The watch compares
    /// how many there are and the newest revision among them with what it
    /// found here, so an entry added and removed again before the watch is
    /// asked leaves it open.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn prefix_watch<'a>(
        &self,
        txn: &'a ReadTxn,
        prefix: &[u8],
    ) -> (impl Iterator<Item = (&'a V, Revision)> + use<'a, V>, Watch) {
        let entry = self.entry(&txn.0);
        let covers = Covers::Prefix(self.pos, prefix.into(), entry.spread(prefix));
        (
            entry.primary.prefix(prefix).map(row),
            txn.cover(self, covers),
        )
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
        let iter = self.entry(&txn.0).primary.lower_bound(key).map(row);
        (iter, txn.cover(self, Covers::Table(self.pos)))
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

    /// The same, plus a watch that fires when the primary keys listed under
    /// `key` change. Rewriting a value without changing its index membership
    /// does not fire it, and neither does a row listed and unlisted again
    /// before the watch is asked: the watch compares the listing with what it
    /// found here.
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
        let prefix = index_prefix(key);
        let tree = self.index_tree(entry, index);
        let covers = Covers::Index(
            self.pos,
            index,
            prefix.as_slice().into(),
            listing(tree, &prefix),
        );
        let hits = tree.prefix(&prefix);
        (resolve(entry, hits, prefix.len()), txn.cover(self, covers))
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
        let iter = self.entry(&txn.0).primary.iter().map(row);
        (iter, txn.cover(self, Covers::Table(self.pos)))
    }

    /// How many deletions the change readers still hold in the stream. Test
    /// helper.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[doc(hidden)]
    #[must_use]
    pub fn graveyard_len(&self, txn: &ReadTxn) -> usize {
        let mut it = txn.0.changes.lower_bound(&stream_key(self.pos, 0, 0));
        let mut dead = 0;
        while let Some(record) = it.next() {
            if table_of(it.key()) != self.pos {
                break;
            }
            dead += usize::from(record.deleted);
        }
        dead
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
                primary_key: entry.primary_key,
                removed: HashMap::new(),
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
        let seq = txn.take_seq();
        let value = Arc::new(value);
        let (key, old, removed) = {
            let pending = self.pending(txn);
            pending.written = true;
            let key = (pending.primary_key)(&value);
            let object = Object {
                value: value.clone(),
                revision,
                seq,
            };
            let old = pending.primary.insert(&key, object);
            // A key this transaction deleted has no object left to address the
            // record of that deletion, so the write takes its place here.
            let removed = pending.removed.remove(&key);
            for (def, tree) in &mut pending.indexes {
                // Only the difference of the two key sets touches the tree, so
                // an update that keeps a value listed under the same index key
                // leaves that entry, and the watch over it, alone.
                let was = old
                    .as_ref()
                    .map(|old| sorted((def.keys)(&old.value)))
                    .unwrap_or_default();
                let is = sorted((def.keys)(&value));
                for k in was.iter().filter(|k| is.binary_search(k).is_err()) {
                    tree.delete(&index_entry(k, &key));
                }
                for k in is.iter().filter(|k| was.binary_search(k).is_err()) {
                    tree.insert(&index_entry(k, &key), ());
                }
            }
            (key, old, removed)
        };
        if let Some(seq) = removed {
            txn.drop_record(self.pos, revision, seq);
        }
        if let Some(old) = &old {
            txn.supersede(self.pos, old.revision, old.seq, revision);
        }
        txn.record(
            self.pos,
            revision,
            seq,
            Record {
                key: key.clone(),
                value,
                deleted: false,
            },
        );
        old.map(|o| o.value)
    }

    /// Removes the entry, leaves its deletion in the change stream, and returns
    /// the value.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn delete(&self, txn: &mut WriteTxn<'_>, key: &[u8]) -> Option<Arc<V>> {
        let revision = txn.root.revision + 1;
        let old = {
            let pending = self.pending(txn);
            let old = pending.primary.delete(key)?;
            pending.written = true;
            for (def, tree) in &mut pending.indexes {
                for k in (def.keys)(&old.value) {
                    tree.delete(&index_entry(&k, key));
                }
            }
            old
        };
        txn.dirty = true;
        txn.supersede(self.pos, old.revision, old.seq, revision);
        let seq = txn.take_seq();
        txn.record(
            self.pos,
            revision,
            seq,
            Record {
                key: key.into(),
                value: old.value.clone(),
                deleted: true,
            },
        );
        self.pending(txn).removed.insert(key.into(), seq);
        Some(old.value)
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
    indexes: Vec<(Index<V>, tree::Txn<()>)>,
    trackers: Vec<Weak<AtomicU64>>,
    new_trackers: Vec<Arc<AtomicU64>>,
    lost: Revision,
    primary_key: fn(&V) -> Key,
    /// Where the record of each key this transaction deleted sits in the
    /// stream. The primary tree no longer holds these keys, so this is what a
    /// write of one of them again supersedes.
    removed: HashMap<Key, u32>,
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
    /// One slot per table, `Some` once the table is touched.
    pending: Vec<Option<Box<dyn AnyPending>>>,
    /// The change stream, opened on the first record.
    changes: Option<tree::Txn<AnyRecord>>,
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

    fn stream(&mut self) -> &mut tree::Txn<AnyRecord> {
        if self.changes.is_none() {
            self.changes = Some(self.root.changes.txn());
        }
        self.changes.as_mut().expect("just opened")
    }

    fn record(&mut self, table: usize, revision: Revision, seq: u32, record: Record) {
        self.stream()
            .insert(&stream_key(table, revision, seq), Arc::new(record));
    }

    /// Drops the record a write replaces, so one key leaves one record per
    /// commit that touched it. Two writes of one key in one transaction keep
    /// both records, in the order they were written.
    ///
    /// The object in the primary tree is what addresses the record, so a key
    /// re-created in a commit after the one that deleted it leaves the
    /// deletion's record behind until it is collected.
    fn supersede(&mut self, table: usize, revision: Revision, seq: u32, now: Revision) {
        if revision < now {
            self.drop_record(table, revision, seq);
        }
    }

    /// Drops one record by where it sits.
    fn drop_record(&mut self, table: usize, revision: Revision, seq: u32) {
        self.stream().delete(&stream_key(table, revision, seq));
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
            pending,
            changes,
            seq: _,
            dirty,
        } = self;
        let root = if pending.iter().all(Option::is_none) {
            parent
        } else {
            let mut root = (*parent).clone();
            let revision = root.revision + Revision::from(dirty);
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
            table: self.table.pos,
            _v: PhantomData,
        };
        // A stale snapshot must not rewind what a newer one already observed.
        self.tracker.fetch_max(entry.revision, Ordering::Relaxed);
        Ok((
            changes,
            txn.cover(&self.table, Covers::Table(self.table.pos)),
        ))
    }
}

/// One table's run of the change stream, from where the reader left off.
struct Changes<'a, V> {
    iter: tree::Iter<'a, AnyRecord>,
    table: usize,
    _v: PhantomData<V>,
}

impl<V: Send + Sync + 'static> Iterator for Changes<'_, V> {
    type Item = Change<V>;

    // ponytail: the primary key is copied into every yielded change. Hand out a
    // borrow of the record instead if the copies ever show up.
    fn next(&mut self) -> Option<Self::Item> {
        let record = self.iter.next()?;
        let key = self.iter.key();
        if table_of(key) != self.table {
            return None;
        }
        let revision = revision_of(key);
        Some(Change {
            key: record.key.clone(),
            value: record
                .value
                .clone()
                .downcast::<V>()
                .expect("a table's records carry its value type"),
            revision,
            deleted: record.deleted,
        })
    }
}
