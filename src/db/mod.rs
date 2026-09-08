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

mod row;
mod stream;
mod table;
mod write;

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use tokio::sync::watch;

use crate::tree::Tree;
use crate::watch::{Covered, Watch};

use stream::{revision_of, stream_key, table_of};
use table::{AnyTable, Pending, State, TableEntry};

pub use row::Version;
pub use stream::{Change, ChangeIterator, Compacted};
pub use table::{Index, Table};
pub use write::WriteTxn;

pub type Revision = u64;
pub type Key = Box<[u8]>;

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
    /// The revision a commit panicked partway through, or 0. See [`Db::lock`].
    wedged: AtomicU64,
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
                wedged: AtomicU64::new(0),
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

    /// Takes the writer lock, on a database that can still be written to.
    ///
    /// A commit that panicked partway through left the writes it had applied
    /// in the trees, under a revision it never published. There is no undo:
    /// the next commit would take that revision and publish those writes along
    /// with its own, so it is refused, and so is every writer after it. A read
    /// runs on, at the last revision a commit did publish.
    ///
    /// # Panics
    ///
    /// If a commit panicked partway through.
    fn lock(&self) -> MutexGuard<'_, ()> {
        let guard = self.write.lock().unwrap_or_else(PoisonError::into_inner);
        let wedged = self.shared.wedged.load(Ordering::Acquire);
        assert_eq!(
            wedged, 0,
            "a commit panicked partway through revision {wedged}, which it never published"
        );
        guard
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
        let _guard = self.lock();
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
        let guard = self.lock();
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
        let _guard = self.lock();
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

    fn shared(&self) -> &Arc<Shared>;

    /// The write this transaction has buffered for `key`, if any.
    fn buffered<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> Option<Pending<V>>;

    /// Everything it has buffered for one table, in key order.
    fn buffer<V: Send + Sync + 'static>(&self, table: &Table<V>) -> Vec<Pending<V>>;
}

impl Snapshot for ReadTxn {
    fn at(&self) -> Revision {
        self.revision
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    fn buffered<V: Send + Sync + 'static>(&self, _: &Table<V>, _: &[u8]) -> Option<Pending<V>> {
        None
    }

    fn buffer<V: Send + Sync + 'static>(&self, _: &Table<V>) -> Vec<Pending<V>> {
        Vec::new()
    }
}

/// A transaction reads its own writes: the buffer first, then the trees at the
/// revision it opened on.
impl Snapshot for WriteTxn<'_> {
    fn at(&self) -> Revision {
        self.visible
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.db.shared
    }

    fn buffered<V: Send + Sync + 'static>(
        &self,
        table: &Table<V>,
        key: &[u8],
    ) -> Option<Pending<V>> {
        table.opened(self)?.written(key).cloned()
    }

    fn buffer<V: Send + Sync + 'static>(&self, table: &Table<V>) -> Vec<Pending<V>> {
        let mut buffered: Vec<Pending<V>> = table
            .opened(self)
            .map_or_else(Vec::new, |buffer| buffer.pending.clone());
        buffered.sort_by(|a, b| a.key.cmp(&b.key));
        buffered
    }
}
