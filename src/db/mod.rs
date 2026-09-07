//! The database: tables, snapshot reads, single-writer transactions.
//!
//! One revision counter covers the whole `Db`. A commit that changes anything
//! bumps it by one, builds a new root and swaps it in behind an `RwLock`
//! held only for the assignment, so readers never see a half-applied commit and
//! never block the writer.

mod row;
mod stream;
mod table;
mod write;

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use tokio::sync::watch;

use crate::tree::{self, Tree};
use crate::watch::{Covered, Watch};

use row::Row;
use stream::collect;
use table::{AnyTable, TableEntry};

pub use row::Version;
pub use stream::{Change, ChangeIterator, Compacted};
pub use table::{Index, Table};
pub use write::WriteTxn;

pub type Revision = u64;
pub type Key = Box<[u8]>;

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
