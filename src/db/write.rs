use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::Ordering;

use super::table::{AnyBuffer, Applying};
use super::{Db, Revision};

/// The write transaction. Dropping it aborts: the buffer goes and no tree was
/// ever touched.
pub struct WriteTxn<'a> {
    pub(super) db: &'a Db,
    pub(super) _guard: MutexGuard<'a, ()>,
    /// The revision the data was at when this transaction opened: what its own
    /// reads see under the buffer.
    pub(super) visible: Revision,
    /// The revision this transaction writes at, and the one its commit leaves
    /// the database at.
    pub(super) revision: Revision,
    /// One slot per table, `Some` once the table is written.
    pub(super) buffers: Vec<Option<Box<dyn AnyBuffer>>>,
    pub(super) dirty: bool,
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
