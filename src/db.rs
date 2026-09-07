//! The database: tables, versioned rows, single-writer transactions.
//!
//! One revision counter covers the whole `Db`, and a commit that changes
//! anything writes it last. A row is a chain of versions, newest first, so a
//! reader takes the revision the counter holds and follows each chain to the
//! first version at or below it: what a later commit adds carries a higher
//! revision and is passed over until the counter says otherwise. A link is
//! never written again, so a reader that has hold of one keeps what it says.
//!
//! A write transaction buffers its writes per table in the order they were
//! made and applies them under the writer lock, so dropping it touches no tree
//! and no half-applied commit is ever visible.

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use tokio::sync::watch;

use crate::tree::{self, Tree};
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

// -------------------------------------------------------------- version chain

/// One version and everything older than it.
struct Link<V> {
    version: Version<V>,
    prev: Option<Arc<Link<V>>>,
}

/// A row: the head of its version chain. A write is one allocation, the link
/// that points at what was there.
struct Row<V> {
    head: Arc<Link<V>>,
    /// The revision of the oldest version on the chain. A write that finds it
    /// above the compaction bound has nothing to trim and skips the walk.
    tail: Revision,
}

/// A reader clones the head under the leaf's lock and walks the chain once it
/// has let the lock go.
impl<V> Clone for Row<V> {
    fn clone(&self) -> Self {
        Self {
            head: self.head.clone(),
            tail: self.tail,
        }
    }
}

impl<V> Row<V> {
    /// The row `version` leaves on top of this chain.
    fn written(head: Option<Self>, version: Version<V>, compacted: Revision) -> Self {
        let (prev, tail) = match head {
            None => (None, version.revision),
            Some(row) if row.tail > compacted => (Some(row.head), row.tail),
            Some(row) => {
                let (link, tail) = trimmed(&row.head, compacted);
                (Some(link), tail)
            }
        };
        Self {
            head: Arc::new(Link { version, prev }),
            tail,
        }
    }

    /// The chain `versions` makes, which are in ascending revision order.
    fn loaded(versions: &[Version<V>]) -> Self {
        let mut rest = versions.iter();
        let oldest = rest.next().expect("a loaded row holds a version");
        let tail = oldest.revision;
        let mut head = Arc::new(Link {
            version: oldest.clone(),
            prev: None,
        });
        for version in rest {
            head = Arc::new(Link {
                version: version.clone(),
                prev: Some(head),
            });
        }
        Self { head, tail }
    }

    /// The chain from the newest version down.
    fn links(&self) -> impl Iterator<Item = &Link<V>> {
        let mut next = Some(&self.head);
        std::iter::from_fn(move || {
            let link = next?;
            next = link.prev.as_ref();
            Some(&**link)
        })
    }

    /// The newest version at or below `at`. A row whose every version is above
    /// it was written after the reader's revision and is not there yet.
    fn at(&self, at: Revision) -> Option<&Version<V>> {
        self.links()
            .find(|link| link.version.revision <= at)
            .map(|link| &link.version)
    }

    /// The version one commit left, if the row still holds it.
    fn exactly(&self, revision: Revision) -> Option<&Version<V>> {
        self.links()
            .take_while(|link| link.version.revision >= revision)
            .find(|link| link.version.revision == revision)
            .map(|link| &link.version)
    }

    /// The value at `at`, unless the version there is a tombstone.
    fn live(&self, at: Revision) -> Option<(Arc<V>, Revision)> {
        self.at(at)
            .filter(|version| !version.deleted)
            .map(|version| (version.value.clone(), version.revision))
    }

    /// The value the newest version holds: what a write hands back and what the
    /// indexes list.
    fn held(&self) -> Option<&Arc<V>> {
        (!self.head.version.deleted).then_some(&self.head.version.value)
    }

    /// The revision of the tombstone this row ends with, if it ends with one.
    fn tombstoned(&self) -> Option<Revision> {
        self.head
            .version
            .deleted
            .then_some(self.head.version.revision)
    }

    /// Every version at or below `at`, oldest first.
    fn versions(&self, at: Revision) -> Vec<Version<V>> {
        let mut versions: Vec<Version<V>> = self
            .links()
            .skip_while(|link| link.version.revision > at)
            .map(|link| link.version.clone())
            .collect();
        versions.reverse();
        versions
    }
}

/// The chain from `head` down to the first version at or below `compacted`,
/// which ends it: a reader at the compaction bound still reads the value that
/// version holds, and nothing below it may be asked for again. The links above
/// it are rebuilt, one allocation per write since the compaction; a chain that
/// already ends there is passed on as it is. Returns the chain and the
/// revision it now ends at.
fn trimmed<V>(head: &Arc<Link<V>>, compacted: Revision) -> (Arc<Link<V>>, Revision) {
    let mut tail = head;
    while tail.version.revision > compacted {
        match &tail.prev {
            Some(prev) => tail = prev,
            None => return (head.clone(), tail.version.revision),
        }
    }
    let bound = tail.version.revision;
    if tail.prev.is_none() {
        return (head.clone(), bound);
    }
    let mut rebuilt = Arc::new(Link {
        version: tail.version.clone(),
        prev: None,
    });
    let mut above = Vec::new();
    let mut link = head;
    while link.version.revision > bound {
        above.push(&link.version);
        link = link.prev.as_ref().expect("the bound is on the chain");
    }
    for version in above.into_iter().rev() {
        rebuilt = Arc::new(Link {
            version: version.clone(),
            prev: Some(rebuilt),
        });
    }
    (rebuilt, bound)
}

/// Puts `version` at the head of `key`'s chain in `tree`.
fn write_row<V>(tree: &Tree<Row<V>>, key: &[u8], version: Version<V>, compacted: Revision) {
    let row = Row::written(tree.get(key), version, compacted);
    tree.insert(key, row);
}

// -------------------------------------------------------------- change stream

/// Key of a change record: the table, the revision of the commit, and the place
/// the record took in it, each big-endian.
///
/// The table comes first, so one table's records are a run of their own that a
/// reader scans and a compaction drops the front of. The revision comes next,
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

// -------------------------------------------------------------------- indexes

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

/// Moves `key`'s index entries from the value the row held to the one it holds
/// now, at `revision`. An entry both values are listed under is left alone.
/// Returns whether an entry was tombstoned.
fn reindex<V>(
    indexes: &[(Index<V>, Tree<Row<()>>)],
    key: &[u8],
    had: Option<&Arc<V>>,
    is: Option<&Arc<V>>,
    revision: Revision,
    compacted: Revision,
) -> bool {
    let mut tombstoned = false;
    for (def, tree) in indexes {
        let had = had
            .map(|value| sorted((def.keys)(value)))
            .unwrap_or_default();
        let is = is
            .map(|value| sorted((def.keys)(value)))
            .unwrap_or_default();
        for k in had.iter().filter(|k| is.binary_search(k).is_err()) {
            write_row(
                tree,
                &index_entry(k, key),
                listed(revision, true),
                compacted,
            );
            tombstoned = true;
        }
        for k in is.iter().filter(|k| had.binary_search(k).is_err()) {
            write_row(
                tree,
                &index_entry(k, key),
                listed(revision, false),
                compacted,
            );
        }
    }
    tombstoned
}

/// The version an index entry takes. An entry says only that its primary key
/// was listed at that revision; the value it resolves to is in the row.
fn listed(revision: Revision, deleted: bool) -> Version<()> {
    Version {
        revision,
        value: Arc::new(()),
        deleted,
    }
}

// ---------------------------------------------------------------- table entry

/// What a reader of one table reads without taking the writer lock.
struct State {
    /// Revision of the last commit that changed this table.
    revision: AtomicU64,
    /// Highest record revision a compaction dropped. A reader that has observed
    /// less than this has lost a change.
    lost: AtomicU64,
    /// Lowest revision a tombstone in this table sits at, or `Revision::MAX`
    /// when it holds none. A compaction that reaches it sweeps the rows those
    /// tombstones ended.
    buried: AtomicU64,
}

/// One table: the rows, the index entries, and the counters.
struct TableEntry<V> {
    state: State,
    primary: Tree<Row<V>>,
    /// One tree per registered index, in registration order. The key is
    /// `index_entry(index key, primary key)` and there is no value.
    indexes: Vec<(Index<V>, Tree<Row<()>>)>,
    primary_key: fn(&V) -> Key,
}

impl<V> TableEntry<V> {
    fn new(primary_key: fn(&V) -> Key, indexes: Vec<Index<V>>) -> Self {
        Self {
            state: State {
                revision: AtomicU64::new(0),
                lost: AtomicU64::new(0),
                buried: AtomicU64::new(Revision::MAX),
            },
            primary: Tree::new(),
            indexes: indexes.into_iter().map(|def| (def, Tree::new())).collect(),
            primary_key,
        }
    }
}

/// The type-erased face of `TableEntry<V>`: what the `Db` can do to a table
/// without knowing its value type.
trait AnyTable: Any + Send + Sync {
    fn state(&self) -> &State;

    /// Removes the rows and index entries a tombstone at or below `rev` ended,
    /// and leaves `buried` at the lowest tombstone that survives.
    fn sweep(&self, rev: Revision);
}

impl<V: Send + Sync + 'static> AnyTable for TableEntry<V> {
    fn state(&self) -> &State {
        &self.state
    }

    fn sweep(&self, rev: Revision) {
        let mut buried = swept(&self.primary, rev);
        for (_, tree) in &self.indexes {
            buried = buried.min(swept(tree, rev));
        }
        self.state.buried.store(buried, Ordering::Release);
    }
}

/// Removes every row a tombstone at or below `rev` ended and returns the lowest
/// tombstone revision the tree is left with. Their deletion is below the
/// compaction bound, so nothing may ask for the key or its versions again.
// ponytail: one walk of the whole tree, run only when a tombstone has reached
// the bound. Hold the tombstoned keys in the entry if a workload deletes often
// enough for the walk to show up.
fn swept<V>(tree: &Tree<Row<V>>, rev: Revision) -> Revision {
    let mut buried = Revision::MAX;
    let mut doomed = Vec::new();
    for (key, row) in tree.range_from(&[]) {
        if let Some(revision) = row.tombstoned() {
            if revision <= rev {
                doomed.push(key);
            } else {
                buried = buried.min(revision);
            }
        }
    }
    for key in &doomed {
        tree.remove(key);
    }
    buried
}

// ----------------------------------------------------------------- shared, db

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic under the writer lock leaves the trees as the write it was in the
    // middle of left them, and the revision unpublished, so a poisoned lock
    // guards nothing a later writer cannot write over.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a [`Watch`], a [`ReadTxn`] and the writer all reach the data through.
///
/// The sending end of the revision channel is the [`Db`]'s, so dropping the
/// database closes the channel and releases every watch parked on it, which is
/// what a watch outliving its database has to wait for.
struct Shared {
    /// Identifies the `Db`; a [`Table`] handle carries the same id.
    db: u64,
    /// The revision every read runs at: written last by a commit, so a reader
    /// that has it sees everything that commit wrote.
    revision: AtomicU64,
    /// History at or below this revision is gone; see [`Db::compact`].
    compacted: AtomicU64,
    /// Registered tables, in registration order. Only [`Db::table`] writes it,
    /// and only by appending, so a handle keeps its position for good.
    tables: RwLock<Vec<Arc<dyn AnyTable>>>,
    /// Every table's change records, one run per table; see [`stream_key`]. A
    /// record is the primary key that was written: the value is in the row.
    changes: Tree<Key>,
    revisions: watch::Receiver<Revision>,
}

impl Shared {
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

    /// The revision of the last commit that changed one table.
    fn table_revision(&self, pos: usize) -> Revision {
        let tables = self.tables.read().unwrap_or_else(PoisonError::into_inner);
        tables[pos].state().revision.load(Ordering::Acquire)
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

/// A watch's side of the database: it reads the revision it covers every time
/// it is asked, so it needs no registration and a watch taken as of a revision
/// the database has already passed reports the change straight away.
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
        match self.covers {
            Covers::Db => self.shared.revision.load(Ordering::Acquire) > self.at,
            // A table's revision lands before the database's; the watch waits
            // for the latter so the reader it wakes finds the commit visible.
            Covers::Table(table) => {
                self.shared
                    .table_revision(table)
                    .min(self.shared.revision.load(Ordering::Acquire))
                    > self.at
            }
        }
    }
}

/// The database. Registration and writes are serialized by one writer lock;
/// a read takes the revision and holds no lock at all.
///
/// The locks are taken in the order `write`, `tables`, then one tree node at a
/// time; every path takes a subsequence of that, so none of them cycle.
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
        let (revisions, receiver) = watch::channel(0);
        Self {
            shared: Arc::new(Shared {
                db: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                revision: AtomicU64::new(0),
                compacted: AtomicU64::new(0),
                tables: RwLock::new(Vec::new()),
                changes: Tree::new(),
                revisions: receiver,
            }),
            revisions,
            write: Mutex::new(()),
        }
    }

    fn revision(&self) -> Revision {
        self.shared.revision.load(Ordering::Acquire)
    }

    /// Completes on the next commit that bumps the revision. A reader that
    /// persists the change stream waits on this and scans what it finds.
    #[must_use]
    pub fn watch(&self) -> Watch {
        self.shared.cover(self.revision(), Covers::Db)
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
        let mut tables = self
            .shared
            .tables
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let pos = tables.len();
        tables.push(Arc::new(TableEntry::new(primary_key, indexes.to_vec())));
        Table {
            db: self.shared.db,
            pos,
            name,
            _v: PhantomData,
        }
    }

    /// A reader at the revision the last commit left.
    #[must_use]
    pub fn read(&self) -> ReadTxn {
        ReadTxn {
            revision: self.revision(),
            shared: self.shared.clone(),
        }
    }

    /// Opens the write transaction at the revision after the visible one.
    /// Blocks until the previous one commits or is dropped.
    #[must_use]
    pub fn write(&self) -> WriteTxn<'_> {
        self.open(None)
    }

    /// The same, at the revision the caller names: what this transaction writes
    /// carries `rev`, and its commit leaves the database there.
    ///
    /// # Panics
    ///
    /// If `rev` is not past the visible revision.
    #[must_use]
    pub fn write_at(&self, rev: Revision) -> WriteTxn<'_> {
        self.open(Some(rev))
    }

    fn open(&self, at: Option<Revision>) -> WriteTxn<'_> {
        let guard = lock(&self.write);
        let visible = self.revision();
        let revision = at.unwrap_or(visible + 1);
        assert!(
            revision > visible,
            "revision {revision} is not past the visible {visible}"
        );
        let tables = self
            .shared
            .tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
        WriteTxn {
            db: self,
            _guard: guard,
            visible,
            revision,
            buffers: std::iter::repeat_with(|| None).take(tables).collect(),
            dirty: false,
        }
    }

    /// Releases the history at or below `rev`: the versions the next write of
    /// each row drops, and the change records the readers of the change stream
    /// then fail on with [`Compacted`].
    ///
    /// This is not history: it does not bump the revision.
    pub fn compact(&self, rev: Revision) {
        let _guard = lock(&self.write);
        let compacted = rev.max(self.shared.compacted.load(Ordering::Acquire));
        self.shared.compacted.store(compacted, Ordering::Release);
        let tables = self
            .shared
            .tables
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        for (pos, table) in tables.iter().enumerate() {
            let state = table.state();
            self.drop_records(pos, compacted, state);
            if state.buried.load(Ordering::Acquire) <= compacted {
                table.sweep(compacted);
            }
        }
    }

    /// Drops one table's change records at or below `compacted` and notes the
    /// highest of them, which is what a reader that had not read them lost.
    ///
    /// The bound is published before the records go, so a reader walking the
    /// stream while this runs reads it back raised once it has passed over
    /// anything this removed, and is told rather than handed a hole.
    fn drop_records(&self, pos: usize, compacted: Revision, state: &State) {
        let mut lost = 0;
        let mut doomed = Vec::new();
        for (key, _) in self.shared.changes.range_from(&stream_key(pos, 0, 0)) {
            if table_of(&key) != pos || revision_of(&key) > compacted {
                break;
            }
            lost = revision_of(&key);
            doomed.push(key);
        }
        if lost > 0 {
            state.lost.fetch_max(lost, Ordering::AcqRel);
        }
        for key in &doomed {
            self.shared.changes.remove(key);
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

/// A reader: one revision, and the way to the data. Holding one keeps nothing
/// alive but the database and blocks nothing.
pub struct ReadTxn {
    revision: Revision,
    shared: Arc<Shared>,
}

impl ReadTxn {
    /// The revision this reader reads at.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.revision
    }
}

/// What a read runs against: the revision it reads at and, through an open
/// transaction, the writes that transaction has buffered over it.
trait Snapshot {
    /// The revision the trees are read at.
    fn at(&self) -> Revision;

    /// The revision a buffered write carries; the same as `at` for a reader.
    fn revision(&self) -> Revision;

    fn shared(&self) -> &Arc<Shared>;

    /// The write this transaction has buffered for `key`, if any.
    fn buffered<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<Op<V>>;

    /// Everything it has buffered for one table, in key order.
    fn buffer<V: Send + Sync + 'static>(&self, table: &Table<V>) -> Vec<(Key, Op<V>)>;
}

impl Snapshot for ReadTxn {
    fn at(&self) -> Revision {
        self.revision
    }

    fn revision(&self) -> Revision {
        self.revision
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    fn buffered<V: Send + Sync + 'static>(&self, _: &Table<V>, _: &[u8]) -> Option<Op<V>> {
        None
    }

    fn buffer<V: Send + Sync + 'static>(&self, _: &Table<V>) -> Vec<(Key, Op<V>)> {
        Vec::new()
    }
}

/// A transaction reads its own writes: the buffer first, then the trees at the
/// revision it opened on.
impl Snapshot for WriteTxn<'_> {
    fn at(&self) -> Revision {
        self.visible
    }

    fn revision(&self) -> Revision {
        self.revision
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.db.shared
    }

    fn buffered<V: Send + Sync + 'static>(&self, table: &Table<V>, key: &[u8]) -> Option<Op<V>> {
        table.opened(self)?.written(key).cloned()
    }

    fn buffer<V: Send + Sync + 'static>(&self, table: &Table<V>) -> Vec<(Key, Op<V>)> {
        let mut buffered: Vec<(Key, Op<V>)> = table.opened(self).map_or_else(Vec::new, |buffer| {
            buffer
                .pending
                .iter()
                .map(|write| (write.key.clone(), write.op.clone()))
                .collect()
        });
        buffered.sort_by(|(a, _), (b, _)| a.cmp(b));
        buffered
    }
}

/// The rows of one table read at `at`, with the writes a transaction buffered
/// over them at `revision`, in key order. A buffered key stands in for the row.
fn merged<V>(
    rows: tree::Iter<Row<V>>,
    buffered: Vec<(Key, Op<V>)>,
    at: Revision,
    revision: Revision,
) -> impl Iterator<Item = (Key, Arc<V>, Revision)> + use<V> {
    let mut rows = rows.peekable();
    let mut buffered = buffered.into_iter().peekable();
    std::iter::from_fn(move || {
        loop {
            let take_buffered = match (rows.peek(), buffered.peek()) {
                (None, None) => return None,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (Some((row, _)), Some((key, _))) => key <= row,
            };
            let live = if take_buffered {
                let (key, op) = buffered.next().expect("just peeked");
                if rows.peek().is_some_and(|(row, _)| *row == key) {
                    rows.next();
                }
                let version = op.newest(revision);
                (!version.deleted).then_some((key, version.value, version.revision))
            } else {
                let (key, row) = rows.next().expect("just peeked");
                row.live(at).map(|(value, revision)| (key, value, revision))
            };
            if live.is_some() {
                return live;
            }
        }
    })
}

impl<V: Send + Sync + 'static> Table<V> {
    /// Runs `f` on this table's entry, under the read lock that holds the table
    /// list still.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    fn with<R>(&self, shared: &Shared, f: impl FnOnce(&TableEntry<V>) -> R) -> R {
        let tables = shared.tables.read().unwrap_or_else(PoisonError::into_inner);
        let entry = tables
            .get(self.pos)
            .filter(|_| shared.db == self.db)
            .and_then(|table| (&**table as &dyn Any).downcast_ref())
            .unwrap_or_else(|| panic!("table {} belongs to another Db or value type", self.name));
        f(entry)
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The revision of the last commit that changed this table. The table
    /// holds where it stands, not a revision per reader, so a reader that has
    /// fallen behind is told of a commit past its own revision.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[must_use]
    pub fn revision(&self, txn: &ReadTxn) -> Revision {
        self.with(&txn.shared, |entry| {
            entry.state.revision.load(Ordering::Acquire)
        })
    }

    /// The value at `key` and the revision of the commit that wrote it, unless
    /// the version this reader sees is a tombstone.
    ///
    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    #[must_use]
    pub fn get(&self, txn: &impl Snapshot, key: &[u8]) -> Option<(Arc<V>, Revision)> {
        match txn.buffered(self, key) {
            Some(op) => {
                let version = op.newest(txn.revision());
                (!version.deleted).then_some((version.value, version.revision))
            }
            None => self
                .with(txn.shared(), |entry| entry.primary.get(key))
                .and_then(|row| row.live(txn.at())),
        }
    }

    /// Every version of `key` this reader can see, oldest first, ending with a
    /// tombstone if the key was deleted. Empty if the key was never written, or
    /// if its row has been swept.
    ///
    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    #[must_use]
    pub fn versions(&self, txn: &impl Snapshot, key: &[u8]) -> Vec<Version<V>> {
        let stored = || {
            self.with(txn.shared(), |entry| entry.primary.get(key))
                .map(|row| row.versions(txn.at()))
                .unwrap_or_default()
        };
        match txn.buffered(self, key) {
            None => stored(),
            Some(Op::Load(versions)) => versions,
            Some(Op::Write { value, deleted }) => {
                let mut versions = stored();
                versions.push(Version {
                    revision: txn.revision(),
                    value,
                    deleted,
                });
                versions
            }
        }
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
    pub fn lower_bound<S: Snapshot>(
        &self,
        txn: &S,
        key: &[u8],
    ) -> impl Iterator<Item = (Key, Arc<V>, Revision)> + use<V, S> {
        self.scan(txn, key)
    }

    /// Every entry whose key starts with `prefix`, in order.
    ///
    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    pub fn prefix<S: Snapshot>(
        &self,
        txn: &S,
        prefix: &[u8],
    ) -> impl Iterator<Item = (Key, Arc<V>, Revision)> + use<V, S> {
        let prefix: Key = prefix.into();
        self.scan(txn, &prefix)
            .take_while(move |(key, _, _)| key.starts_with(&prefix))
    }

    /// Every entry, in key order.
    ///
    /// `txn` is a [`ReadTxn`], or a [`WriteTxn`], which also sees its own
    /// writes.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    #[allow(private_bounds)]
    pub fn all<S: Snapshot>(
        &self,
        txn: &S,
    ) -> impl Iterator<Item = (Key, Arc<V>, Revision)> + use<V, S> {
        self.scan(txn, &[])
    }

    fn scan<S: Snapshot>(
        &self,
        txn: &S,
        from: &[u8],
    ) -> impl Iterator<Item = (Key, Arc<V>, Revision)> + use<V, S> {
        let rows = self.with(txn.shared(), |entry| entry.primary.range_from(from));
        let buffered = txn
            .buffer(self)
            .into_iter()
            .skip_while(|(key, _)| **key < *from)
            .collect();
        merged(rows, buffered, txn.at(), txn.revision())
    }

    /// Every value listed under `key` in the named index, in primary key order,
    /// resolved through the row each entry names.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`, or has no such index.
    pub fn by_index(
        &self,
        txn: &ReadTxn,
        index: &'static str,
        key: &[u8],
    ) -> impl Iterator<Item = (Arc<V>, Revision)> + use<V> {
        let at = txn.revision;
        let prefix = index_prefix(key);
        let listed = self.with(&txn.shared, |entry| {
            self.index_tree(entry, index)
                .range_from(&prefix)
                .take_while(|(entry_key, _)| entry_key.starts_with(&prefix))
                .filter(|(_, listing)| listing.at(at).is_some_and(|v| !v.deleted))
                .filter_map(|(entry_key, _)| entry.primary.get(&entry_key[prefix.len()..]))
                .filter_map(|row| row.live(at))
                .collect::<Vec<_>>()
        });
        listed.into_iter()
    }

    fn index_tree<'a>(&self, entry: &'a TableEntry<V>, index: &'static str) -> &'a Tree<Row<()>> {
        entry
            .indexes
            .iter()
            .find_map(|(def, tree)| (def.name == index).then_some(tree))
            .unwrap_or_else(|| panic!("table {} has no index {index}", self.name))
    }

    /// Whether this table's run of the change stream holds a record.
    fn recorded(&self, shared: &Shared) -> bool {
        shared
            .changes
            .range_from(&stream_key(self.pos, 0, 0))
            .next()
            .is_some_and(|(key, _)| table_of(&key) == self.pos)
    }

    /// The writes this transaction has buffered for this table, if it has any.
    /// A handle from another `Db` reads no slot: it falls through to `with`,
    /// which is where the mismatch is reported.
    fn opened<'t>(&self, txn: &'t WriteTxn<'_>) -> Option<&'t Buffer<V>> {
        let buffer = txn
            .buffers
            .get(self.pos)
            .filter(|_| txn.db.shared.db == self.db)?
            .as_ref()?;
        Some(
            (&**buffer as &dyn Any)
                .downcast_ref()
                .expect("a table's buffer holds its value type"),
        )
    }

    /// This table's buffer, opened on first use.
    fn buffer<'t>(&self, txn: &'t mut WriteTxn<'_>) -> &'t mut Buffer<V> {
        let buffer = txn.buffers[self.pos].get_or_insert_with(|| Box::new(Buffer::<V>::default()));
        txn.dirty = true;
        (&mut **buffer as &mut dyn Any)
            .downcast_mut()
            .expect("a table's buffer holds its value type")
    }

    /// Writes `value` under `primary_key(&value)` and returns what it replaced.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn insert(&self, txn: &mut WriteTxn<'_>, value: V) -> Option<Arc<V>> {
        let value = Arc::new(value);
        let key = self.with(&txn.db.shared, |entry| (entry.primary_key)(&value));
        let was = self.get(txn, &key).map(|(value, _)| value);
        self.buffer(txn).put(
            key,
            Op::Write {
                value,
                deleted: false,
            },
        );
        was
    }

    /// Ends the row with a tombstone that keeps the value, and returns it. The
    /// key reads as absent from here on; the row stays until it is swept.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn delete(&self, txn: &mut WriteTxn<'_>, key: &[u8]) -> Option<Arc<V>> {
        let old = self.get(txn, key).map(|(value, _)| value)?;
        self.buffer(txn).put(
            key.into(),
            Op::Write {
                value: old.clone(),
                deleted: true,
            },
        );
        Some(old)
    }

    /// Puts `versions` at `key`, in place of whatever is there, and leaves no
    /// change record: this is how a table is rebuilt from what was persisted,
    /// not a write for readers to follow.
    ///
    /// The indexes follow the newest version, so a row loaded as deleted is
    /// listed nowhere.
    ///
    /// A rebuild runs with no reader open. The loaded versions keep the
    /// revisions they were written at, which are at or below what an open
    /// reader reads, so such a reader would see the rows this call puts back.
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
            !self.recorded(&txn.db.shared),
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
        self.buffer(txn).put(key.into(), Op::Load(versions));
    }

    /// A reader of this table's changes, observing it from where it stands now:
    /// what this transaction and every later one write is reported.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn changes(&self, txn: &mut WriteTxn<'_>) -> ChangeIterator<V> {
        let observed = self.with(&txn.db.shared, |entry| {
            entry.state.revision.load(Ordering::Acquire)
        });
        ChangeIterator {
            table: *self,
            observed,
        }
    }
}

// ------------------------------------------------------------- write, buffer

/// One buffered write.
struct Pending<V> {
    key: Key,
    op: Op<V>,
}

/// What a transaction leaves at a key.
enum Op<V> {
    /// An insert or a delete: one version, at the transaction's revision.
    Write { value: Arc<V>, deleted: bool },
    /// A load: the whole chain, in place of what is there.
    Load(Vec<Version<V>>),
}

impl<V> Clone for Op<V> {
    fn clone(&self) -> Self {
        match self {
            Self::Write { value, deleted } => Self::Write {
                value: value.clone(),
                deleted: *deleted,
            },
            Self::Load(versions) => Self::Load(versions.clone()),
        }
    }
}

impl<V> Op<V> {
    /// The version this write leaves at the head of the row.
    fn newest(&self, revision: Revision) -> Version<V> {
        match self {
            Self::Write { value, deleted } => Version {
                revision,
                value: value.clone(),
                deleted: *deleted,
            },
            Self::Load(versions) => versions
                .last()
                .expect("a loaded row holds a version")
                .clone(),
        }
    }
}

/// One table's buffered writes, in the order they were made.
struct Buffer<V> {
    pending: Vec<Pending<V>>,
    /// Where each key's write sits, so a second write of one key replaces the
    /// first where it stands rather than following it.
    at: HashMap<Key, usize>,
}

impl<V> Default for Buffer<V> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            at: HashMap::new(),
        }
    }
}

impl<V> Buffer<V> {
    fn put(&mut self, key: Key, op: Op<V>) {
        if let Some(&at) = self.at.get(&key) {
            self.pending[at].op = op;
        } else {
            self.at.insert(key.clone(), self.pending.len());
            self.pending.push(Pending { key, op });
        }
    }

    fn written(&self, key: &[u8]) -> Option<&Op<V>> {
        self.at.get(key).map(|&at| &self.pending[at].op)
    }
}

/// What a commit is applying, as it walks the tables in order.
struct Applying<'a> {
    revision: Revision,
    compacted: Revision,
    /// The table the current buffer belongs to.
    pos: usize,
    /// The place the next record takes in this commit.
    seq: u32,
    changes: &'a Tree<Key>,
}

/// The type-erased face of `Buffer<V>`: what a commit does with one without
/// knowing the value type.
trait AnyBuffer: Any {
    /// Writes everything this buffer holds to `table`, in order.
    fn apply(self: Box<Self>, table: &dyn AnyTable, at: &mut Applying<'_>);
}

impl<V: Send + Sync + 'static> AnyBuffer for Buffer<V> {
    fn apply(self: Box<Self>, table: &dyn AnyTable, at: &mut Applying<'_>) {
        let entry: &TableEntry<V> = (table as &dyn Any)
            .downcast_ref()
            .expect("a table's buffer holds its value type");
        let mut buried = Revision::MAX;
        for Pending { key, op } in self.pending {
            let row = entry.primary.get(&key);
            let had = row.as_ref().and_then(Row::held).cloned();
            match op {
                Op::Write { value, deleted } => {
                    let is = (!deleted).then(|| value.clone());
                    if deleted {
                        buried = buried.min(at.revision);
                    }
                    if reindex(
                        &entry.indexes,
                        &key,
                        had.as_ref(),
                        is.as_ref(),
                        at.revision,
                        at.compacted,
                    ) {
                        buried = buried.min(at.revision);
                    }
                    let version = Version {
                        revision: at.revision,
                        value,
                        deleted,
                    };
                    entry
                        .primary
                        .insert(&key, Row::written(row, version, at.compacted));
                    at.changes
                        .insert(&stream_key(at.pos, at.revision, at.seq), key);
                    at.seq += 1;
                }
                Op::Load(versions) => {
                    let loaded = Row::loaded(&versions);
                    if let Some(revision) = loaded.tombstoned() {
                        buried = buried.min(revision);
                    }
                    let revision = loaded.head.version.revision;
                    if reindex(
                        &entry.indexes,
                        &key,
                        had.as_ref(),
                        loaded.held(),
                        revision,
                        at.compacted,
                    ) {
                        buried = buried.min(revision);
                    }
                    entry.primary.insert(&key, loaded);
                }
            }
        }
        entry.state.revision.store(at.revision, Ordering::Release);
        if buried < Revision::MAX {
            entry.state.buried.fetch_min(buried, Ordering::AcqRel);
        }
    }
}

/// The write transaction. Dropping it aborts: the buffer goes and no tree was
/// ever touched.
pub struct WriteTxn<'a> {
    db: &'a Db,
    _guard: MutexGuard<'a, ()>,
    /// The revision the data was at when this transaction opened: what its own
    /// reads see under the buffer.
    visible: Revision,
    /// The revision this transaction writes at, and the one its commit leaves
    /// the database at.
    revision: Revision,
    /// One slot per table, `Some` once the table is written.
    buffers: Vec<Option<Box<dyn AnyBuffer>>>,
    dirty: bool,
}

impl WriteTxn<'_> {
    /// The revision this transaction writes at.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.revision
    }

    /// Writes the buffer to the trees, publishes the revision and wakes the
    /// watches, all under the writer lock. Returns the new revision, which is
    /// the previous one if nothing was written.
    // The revision is worth ignoring; the commit itself is the point.
    #[allow(clippy::must_use_candidate)]
    pub fn commit(self) -> Revision {
        let WriteTxn {
            db,
            _guard: guard,
            visible,
            revision,
            buffers,
            dirty,
        } = self;
        if !dirty {
            return visible;
        }
        let shared = &db.shared;
        {
            let tables = shared.tables.read().unwrap_or_else(PoisonError::into_inner);
            let mut at = Applying {
                revision,
                compacted: shared.compacted.load(Ordering::Acquire),
                pos: 0,
                seq: 0,
                changes: &shared.changes,
            };
            for (pos, buffer) in buffers.into_iter().enumerate() {
                if let Some(buffer) = buffer {
                    at.pos = pos;
                    buffer.apply(&*tables[pos], &mut at);
                }
            }
        }
        // The revision comes last: a reader that has it has everything this
        // commit wrote, and one that read the revision before it passes over
        // every version this commit left.
        shared.revision.store(revision, Ordering::Release);
        db.revisions.send_replace(revision);
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

/// A reader of one table's changes. It holds the revision it has read up to and
/// nothing else: the records wait in the change stream until a compaction drops
/// them, whether or not anyone has read them.
pub struct ChangeIterator<V> {
    table: Table<V>,
    observed: Revision,
}

impl<V: Send + Sync + 'static> ChangeIterator<V> {
    /// The changes between the last observed revision and `txn`, in revision
    /// order, plus a watch that fires on the next change to the table.
    ///
    /// The revision counts as observed as soon as this returns, whether or not
    /// the iterator is drained. A reader handed an older revision than one it
    /// has already read reads nothing and keeps the revision it had.
    ///
    /// # Errors
    ///
    /// [`Compacted`] if [`Db::compact`] dropped changes this reader had not
    /// seen.
    ///
    /// # Panics
    ///
    /// If the table was not registered in this `Db`.
    pub fn next(
        &mut self,
        txn: &ReadTxn,
    ) -> Result<(impl Iterator<Item = Change<V>> + use<V>, Watch), Compacted> {
        let at = txn.revision;
        let pos = self.table.pos;
        let observed = self.observed;
        let (changes, taken) = self.table.with(&txn.shared, |entry| {
            let mut changes = Vec::new();
            for (record, key) in
                txn.shared
                    .changes
                    .range_from(&stream_key(pos, observed.saturating_add(1), 0))
            {
                if table_of(&record) != pos {
                    break;
                }
                let revision = revision_of(&record);
                if revision > at {
                    break;
                }
                // A record and the version it names are dropped together, so a
                // compaction running alongside this walk takes both, and the
                // bound below reports what it took.
                if let Some(version) = entry
                    .primary
                    .get(&key)
                    .as_ref()
                    .and_then(|row| row.exactly(revision))
                {
                    changes.push(Change {
                        key,
                        value: version.value.clone(),
                        revision,
                        deleted: version.deleted,
                    });
                }
            }
            // The walk reads the stream as it stands rather than a snapshot
            // of it, so the bound is read after it: a compaction that ran
            // alongside published the bound before it took the records, and
            // anything it took from under this walk is at or below it.
            let lost = entry.state.lost.load(Ordering::Acquire);
            if observed < lost {
                return Err(Compacted { at: lost });
            }
            Ok((
                changes,
                entry.state.revision.load(Ordering::Acquire).min(at),
            ))
        })?;
        self.observed = observed.max(at);
        Ok((
            changes.into_iter(),
            txn.shared.cover(taken, Covers::Table(pos)),
        ))
    }
}
