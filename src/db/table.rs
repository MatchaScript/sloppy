use std::any::Any;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crate::tree::{self, Tree};

use super::row::Row;
use super::stream::{ChangeIterator, UNREGISTERED, revision_of, stream_key, table_of};
use super::write::WriteTxn;
use super::{Key, ReadTxn, Revision, Root, Snapshot, Version};

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

/// One table's trees plus its change trackers.
pub(super) struct TableEntry<V> {
    /// Revision of the last commit that changed this table.
    pub(super) revision: Revision,
    pub(super) primary: Tree<Row<V>>,
    /// One tree per registered index, in registration order. The key is
    /// `index_entry(index key, primary key)` and there is no value.
    indexes: Vec<(Index<V>, Tree<()>)>,
    trackers: Vec<Weak<AtomicU64>>,
    /// Highest delete revision that [`Db::compact`] removed before every
    /// tracker had seen it. A reader below this has lost a change.
    pub(super) lost: Revision,
    /// Lowest revision a tombstoned row is left at, or `Revision::MAX` when
    /// none is left. A sweep is due once a bound reaches it; see
    /// [`AnyTable::collected`]. Rewriting a tombstoned key leaves it low, and
    /// the next sweep puts it right.
    buried: Revision,
    primary_key: fn(&V) -> Key,
}

impl<V> TableEntry<V> {
    pub(super) fn new(primary_key: fn(&V) -> Key, indexes: Vec<Index<V>>) -> Self {
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
pub(super) struct Readers {
    pub(super) trackers: Vec<Weak<AtomicU64>>,
    /// The lowest revision every reader has observed; with no reader left, the
    /// table's own revision, so everything it has may go.
    pub(super) watermark: Revision,
    /// A tracker whose reader is gone was dropped.
    pub(super) pruned: bool,
    pub(super) lost: Revision,
}

/// The type-erased face of `TableEntry<V>`: what the `Root` can do without
/// knowing the value type.
pub(super) trait AnyTable: Any + Send + Sync {
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

/// A handle to a registered table. Cheap to copy; valid only for the `Db` that
/// returned it.
pub struct Table<V> {
    pub(super) db: u64,
    pub(super) pos: usize,
    pub(super) name: &'static str,
    pub(super) _v: PhantomData<V>,
}

impl<V> Clone for Table<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for Table<V> {}

impl<V: Send + Sync + 'static> Table<V> {
    pub(super) fn entry<'a>(&self, root: &'a Root) -> &'a TableEntry<V> {
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
    pub(super) fn opened<'t>(&self, txn: &'t WriteTxn<'_>) -> Option<&'t Pending<V>> {
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

/// One table's uncommitted trees.
pub(super) struct Pending<V> {
    /// The table's revision before this transaction.
    revision: Revision,
    written: bool,
    pub(super) primary: tree::Txn<Row<V>>,
    indexes: Vec<(Index<V>, tree::Txn<()>)>,
    trackers: Vec<Weak<AtomicU64>>,
    new_trackers: Vec<Arc<AtomicU64>>,
    lost: Revision,
    buried: Revision,
    primary_key: fn(&V) -> Key,
}

pub(super) trait AnyPending: Any {
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
