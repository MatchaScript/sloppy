use std::sync::{Arc, MutexGuard};

use crate::tree;

use super::stream::{collect, stream_key};
use super::table::AnyPending;
use super::{Db, Key, Revision, Root};

/// The write transaction. Dropping it aborts: nothing was ever visible.
pub struct WriteTxn<'a> {
    pub(super) db: &'a Db,
    pub(super) _guard: MutexGuard<'a, ()>,
    /// The root this transaction opened on; the commit clones it to build the
    /// new one.
    pub(super) root: Arc<Root>,
    /// The revision this transaction writes at, and the one its commit leaves
    /// the database at.
    pub(super) revision: Revision,
    /// One slot per table, `Some` once the table is touched.
    pub(super) pending: Vec<Option<Box<dyn AnyPending>>>,
    /// The change stream, opened on the first record.
    pub(super) changes: Option<tree::Txn<Key>>,
    /// How many records this transaction has written; the next one's place.
    pub(super) seq: u32,
    pub(super) dirty: bool,
}

impl WriteTxn<'_> {
    /// The place the next record takes in this transaction.
    pub(super) fn take_seq(&mut self) -> u32 {
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
    pub(super) fn record(&mut self, table: usize, revision: Revision, seq: u32, key: Key) {
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
