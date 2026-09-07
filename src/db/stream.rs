use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tree::{self, Tree};
use crate::watch::Watch;

use super::table::{AnyTable, Table, TableEntry};
use super::{Covers, Key, ReadTxn, Revision};

/// Key of a change record: the table, the revision of the commit, and the place
/// the record took in it, each big-endian.
///
/// The table comes first, so one table's records are a run of their own that a
/// reader scans and the collection drops a prefix of. The revision comes next,
/// so that run is in commit order; the sequence number keeps the records of one
/// commit apart and in the order they were written.
pub(super) fn stream_key(table: usize, revision: Revision, seq: u32) -> Key {
    let table = u32::try_from(table).expect("a `Db` holds fewer than 4 billion tables");
    let mut k = Vec::with_capacity(16);
    k.extend_from_slice(&table.to_be_bytes());
    k.extend_from_slice(&revision.to_be_bytes());
    k.extend_from_slice(&seq.to_be_bytes());
    k.into()
}

pub(super) fn table_of(key: &[u8]) -> usize {
    let head: [u8; 4] = key[..4].try_into().expect("change stream key");
    u32::from_be_bytes(head) as usize
}

pub(super) fn revision_of(key: &[u8]) -> Revision {
    let head: [u8; 8] = key[4..12].try_into().expect("change stream key");
    Revision::from_be_bytes(head)
}

/// Drops the change records every reader of their table has already seen, and
/// the ones [`Db::compact`] gave up on, then the rows whose deletion both
/// bounds have passed. One run over every table's partition.
pub(super) fn collect(
    tables: &mut [Arc<dyn AnyTable>],
    changes: &mut Tree<Key>,
    compacted: Revision,
) {
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

/// One committed write, as seen by a [`ChangeIterator`].
pub struct Change<V> {
    pub key: Key,
    pub value: Arc<V>,
    pub revision: Revision,
    pub deleted: bool,
}

/// The history the reader still needed was dropped by [`Db::compact`](super::Db::compact).
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
    pub(super) table: Table<V>,
    pub(super) tracker: Arc<AtomicU64>,
}

pub(super) const UNREGISTERED: Revision = Revision::MAX;

impl<V: Send + Sync + 'static> ChangeIterator<V> {
    /// The changes between the last observed revision and `txn`, in revision
    /// order, plus a watch that fires on the next change to the table.
    ///
    /// The snapshot counts as observed as soon as this returns, whether or not
    /// the iterator is drained.
    ///
    /// # Errors
    ///
    /// [`Compacted`] if [`Db::compact`](super::Db::compact) dropped changes this reader had not
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
