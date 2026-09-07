use std::sync::Arc;

use super::Revision;

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
pub(super) struct Row<V> {
    pub(super) versions: Arc<[Version<V>]>,
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
    pub(super) fn newest(&self) -> &Version<V> {
        self.versions.last().expect("a row holds a version")
    }

    /// The value at this key, unless the newest version is a tombstone.
    pub(super) fn live(&self) -> Option<(&V, Revision)> {
        let newest = self.newest();
        (!newest.deleted).then(|| (newest.value.as_ref(), newest.revision))
    }

    /// The same, as the row holds it: what a write hands back and what the
    /// indexes list.
    pub(super) fn held(&self) -> Option<&Arc<V>> {
        let newest = self.newest();
        (!newest.deleted).then_some(&newest.value)
    }

    /// The row `version` leaves, and whether it takes a change record.
    ///
    /// A second write of one key in one commit replaces the version the first
    /// left and reuses its record, so one key leaves one record per commit.
    /// Versions at or below `compacted` go, bar the newest: nothing may read
    /// them any more.
    pub(super) fn written(
        row: Option<&Self>,
        version: Version<V>,
        compacted: Revision,
    ) -> (Self, bool) {
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
