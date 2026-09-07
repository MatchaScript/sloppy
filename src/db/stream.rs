use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::watch::Watch;

use super::table::Table;
use super::{Covers, Key, ReadTxn, Revision};

/// Key of a change record: the table, the revision of the commit, and the place
/// the record took in it, each big-endian.
///
/// The table comes first, so one table's records are a run of their own that a
/// reader scans and a compaction drops the front of. The revision comes next,
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
    pub(super) table: Table<V>,
    pub(super) observed: Revision,
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
