use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError};

use crate::tree::{self, Tree};

use super::row::Row;
use super::stream::{ChangeIterator, stream_key, table_of};
use super::write::WriteTxn;
use super::{Key, ReadTxn, Revision, Shared, Snapshot, Version};

/// Puts `version` at the head of `key`'s chain in `tree`.
fn write_row<V>(tree: &Tree<Row<V>>, key: &[u8], version: Version<V>, compacted: Revision) {
    let row = Row::written(tree.get(key), version, compacted);
    tree.insert(key, row);
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

/// What a reader of one table reads without taking the writer lock.
pub(super) struct State {
    /// Revision of the last commit that changed this table.
    pub(super) revision: AtomicU64,
    /// Highest record revision a compaction dropped. A reader that has observed
    /// less than this has lost a change.
    pub(super) lost: AtomicU64,
    /// Lowest revision a tombstone in this table sits at, or `Revision::MAX`
    /// when it holds none. A compaction that reaches it sweeps the rows those
    /// tombstones ended.
    pub(super) buried: AtomicU64,
}

/// One table: the rows, the index entries, and the counters.
pub(super) struct TableEntry<V> {
    pub(super) state: State,
    pub(super) primary: Tree<Row<V>>,
    /// One tree per registered index, in registration order. The key is
    /// `index_entry(index key, primary key)` and there is no value.
    pub(super) indexes: Vec<(Index<V>, Tree<Row<()>>)>,
    pub(super) primary_key: fn(&V) -> Key,
}

impl<V> TableEntry<V> {
    pub(super) fn new(primary_key: fn(&V) -> Key, indexes: Vec<Index<V>>) -> Self {
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
pub(super) trait AnyTable: Any + Send + Sync {
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

/// The rows of one table read at `at`, with the writes a transaction buffered
/// over them, in key order. A buffered key stands in for the row.
fn merged<V>(
    rows: tree::Iter<Row<V>>,
    buffered: Vec<Pending<V>>,
    at: Revision,
) -> impl Iterator<Item = (Key, Arc<V>, Revision)> + use<V> {
    let mut rows = rows.peekable();
    let mut buffered = buffered.into_iter().peekable();
    std::iter::from_fn(move || {
        loop {
            let take_buffered = match (rows.peek(), buffered.peek()) {
                (None, None) => return None,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (Some((row, _)), Some(pending)) => pending.key <= *row,
            };
            let live = if take_buffered {
                let pending = buffered.next().expect("just peeked");
                if rows.peek().is_some_and(|(row, _)| *row == pending.key) {
                    rows.next();
                }
                let version = pending.newest().clone();
                (!version.deleted).then_some((pending.key, version.value, version.revision))
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
    pub(super) fn with<R>(&self, shared: &Shared, f: impl FnOnce(&TableEntry<V>) -> R) -> R {
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
            Some(pending) => {
                let version = pending.newest();
                (!version.deleted).then_some((version.value.clone(), version.revision))
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
            Some(pending) => pending.versions(stored),
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
            .skip_while(|pending| *pending.key < *from)
            .collect();
        merged(rows, buffered, txn.at())
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
    pub(super) fn opened<'t>(&self, txn: &'t WriteTxn<'_>) -> Option<&'t Buffer<V>> {
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
        let version = Version {
            revision: txn.revision,
            value,
            deleted: false,
        };
        self.buffer(txn).wrote(key, version);
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
        let version = Version {
            revision: txn.revision,
            value: old.clone(),
            deleted: true,
        };
        self.buffer(txn).wrote(key.into(), version);
        Some(old)
    }

    /// Puts `versions` at `key`, in place of whatever is there, and leaves no
    /// change record: this is how a table is rebuilt from what was persisted,
    /// not a write for readers to follow. In place of whatever is there covers
    /// a write this transaction has already made at `key`; an `insert` or a
    /// `delete` after this one goes on the end of the chain loaded here.
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
        self.buffer(txn).loaded(key.into(), versions);
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

/// What a transaction leaves at one key: the chain a load put in place of the
/// row, then the version an insert or a delete left on top of it. A load with a
/// write over it is a row rebuilt and then written, and the write follows the
/// chain the load put there.
pub(super) struct Pending<V> {
    pub(super) key: Key,
    /// The whole chain, in place of what the row holds. No change record.
    pub(super) loaded: Option<Vec<Version<V>>>,
    /// One version, at the transaction's revision, over what is below it.
    pub(super) wrote: Option<Version<V>>,
}

impl<V> Clone for Pending<V> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            loaded: self.loaded.clone(),
            wrote: self.wrote.clone(),
        }
    }
}

impl<V> Pending<V> {
    /// The version this leaves at the head of the row.
    pub(super) fn newest(&self) -> &Version<V> {
        self.wrote
            .as_ref()
            .or_else(|| self.loaded.as_ref().and_then(|versions| versions.last()))
            .expect("a pending write holds a version")
    }

    /// The versions a reader of this key sees, oldest first: the loaded chain,
    /// or `stored` when this key was not loaded, and then the write.
    pub(super) fn versions(self, stored: impl FnOnce() -> Vec<Version<V>>) -> Vec<Version<V>> {
        let mut versions = self.loaded.unwrap_or_else(stored);
        versions.extend(self.wrote);
        versions
    }
}

/// One table's buffered writes, in the order they were made.
pub(super) struct Buffer<V> {
    pub(super) pending: Vec<Pending<V>>,
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
    /// This key's place in the buffer, opened on first use.
    fn entry(&mut self, key: Key) -> &mut Pending<V> {
        let at = *self.at.entry(key.clone()).or_insert(self.pending.len());
        if at == self.pending.len() {
            self.pending.push(Pending {
                key,
                loaded: None,
                wrote: None,
            });
        }
        &mut self.pending[at]
    }

    /// Leaves `version` on top of the key, in place of an earlier write of it.
    fn wrote(&mut self, key: Key, version: Version<V>) {
        self.entry(key).wrote = Some(version);
    }

    /// Puts `versions` in place of the row, and of anything written over it.
    fn loaded(&mut self, key: Key, versions: Vec<Version<V>>) {
        let pending = self.entry(key);
        pending.loaded = Some(versions);
        pending.wrote = None;
    }

    pub(super) fn written(&self, key: &[u8]) -> Option<&Pending<V>> {
        self.at.get(key).map(|&at| &self.pending[at])
    }
}

/// What a commit is applying, as it walks the tables in order.
pub(super) struct Applying<'a> {
    pub(super) revision: Revision,
    pub(super) compacted: Revision,
    /// The table the current buffer belongs to.
    pub(super) pos: usize,
    /// The place the next record takes in this commit.
    pub(super) seq: u32,
    pub(super) changes: &'a Tree<Key>,
}

/// The type-erased face of `Buffer<V>`: what a commit does with one without
/// knowing the value type.
pub(super) trait AnyBuffer: Any {
    /// Writes everything this buffer holds to `table`, in order.
    fn apply(self: Box<Self>, table: &dyn AnyTable, at: &mut Applying<'_>);
}

impl<V: Send + Sync + 'static> AnyBuffer for Buffer<V> {
    fn apply(self: Box<Self>, table: &dyn AnyTable, at: &mut Applying<'_>) {
        let entry: &TableEntry<V> = (table as &dyn Any)
            .downcast_ref()
            .expect("a table's buffer holds its value type");
        let mut buried = Revision::MAX;
        for Pending { key, loaded, wrote } in self.pending {
            let mut row = entry.primary.get(&key);
            if let Some(versions) = loaded {
                let had = row.as_ref().and_then(Row::held).cloned();
                let loaded = Row::loaded(&versions);
                let revision = loaded.newest().revision;
                if let Some(revision) = loaded.tombstoned() {
                    buried = buried.min(revision);
                }
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
                row = Some(loaded);
            }
            if let Some(version) = wrote {
                let had = row.as_ref().and_then(Row::held).cloned();
                let is = (!version.deleted).then(|| version.value.clone());
                if version.deleted {
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
                row = Some(Row::written(row, version, at.compacted));
                at.changes
                    .insert(&stream_key(at.pos, at.revision, at.seq), key.clone());
                at.seq += 1;
            }
            entry
                .primary
                .insert(&key, row.expect("a pending write leaves a row"));
        }
        entry.state.revision.store(at.revision, Ordering::Release);
        if buried < Revision::MAX {
            entry.state.buried.fetch_min(buried, Ordering::AcqRel);
        }
    }
}
