//! Row 2 and row 3 of the plan: root cell, transactions, revision, hook,
//! revision index, graveyard, `changes`, watermark, `compact`.

use std::sync::{Arc, Mutex};

use sloppy::db::{Change, ChangeIterator, Db, Index, Key, ReadTxn, Revision, Table};

#[derive(Debug)]
struct Item {
    key: &'static str,
    val: u32,
}

fn pk(item: &Item) -> Key {
    item.key.as_bytes().into()
}

fn item(key: &'static str, val: u32) -> Item {
    Item { key, val }
}

/// `(key, revision, deleted)` of every change, in the order yielded.
fn drain<I: Iterator<Item = Change<Item>>>(changes: I) -> Vec<(String, Revision, bool)> {
    changes
        .map(|c| {
            (
                String::from_utf8(c.key.to_vec()).unwrap(),
                c.revision,
                c.deleted,
            )
        })
        .collect()
}

#[test]
fn snapshots_hold_their_version() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let before = db.read();
    let mut w = db.write();
    // Taken while the writer is open, and not blocked by it.
    let during = db.read();
    items.insert(&mut w, item("a", 2));
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 2);

    for old in [&before, &during] {
        assert_eq!(old.revision(), 1);
        assert_eq!(items.get(old, b"a").unwrap().0.val, 1);
        assert_eq!(items.get(old, b"a").unwrap().1, 1);
        assert_eq!(items.all(old).count(), 1);
    }

    let after = db.read();
    assert_eq!(after.revision(), 2);
    assert_eq!(items.get(&after, b"a").unwrap().0.val, 2);
    assert_eq!(items.get(&after, b"a").unwrap().1, 2);
    assert_eq!(items.all(&after).count(), 2);
    assert_eq!(
        items
            .prefix(&after, b"b")
            .map(|(k, _, _)| k)
            .collect::<Vec<_>>(),
        vec![b"b".to_vec()]
    );
    assert_eq!(items.lower_bound(&after, b"b").count(), 1);
}

#[test]
fn abort_leaves_nothing() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen = calls.clone();
    let db = Db::with_hook(move |rev: Revision, _: &ReadTxn| seen.lock().unwrap().push(rev));
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let mut w = db.write();
    items.insert(&mut w, item("a", 2));
    items.insert(&mut w, item("b", 1));
    drop(w);

    let r = db.read();
    assert_eq!(r.revision(), 1);
    assert_eq!(items.get(&r, b"a").unwrap().0.val, 1);
    assert_eq!(items.all(&r).count(), 1);

    // An empty transaction commits to the same revision.
    assert_eq!(db.write().commit(), 1);
    assert_eq!(*calls.lock().unwrap(), vec![1]);
}

#[test]
#[should_panic(expected = "registration transaction did not commit")]
fn a_change_reader_rejects_an_aborted_registration() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut w = db.write();
    let mut reader = items.changes(&mut w);
    drop(w);

    let _ = reader.next(&db.read());
}

#[test]
fn hook_sees_each_commit_once() {
    type Log = Vec<(Revision, Vec<(String, Revision, bool)>)>;
    let log: Arc<Mutex<Log>> = Arc::new(Mutex::new(Vec::new()));
    let iter: Arc<Mutex<Option<ChangeIterator<Item>>>> = Arc::new(Mutex::new(None));

    let (log_in, iter_in) = (log.clone(), iter.clone());
    let db = Db::with_hook(move |rev: Revision, txn: &ReadTxn| {
        let mut slot = iter_in.lock().unwrap();
        let changes = slot
            .as_mut()
            .map(|it| drain(it.next(txn).unwrap().0))
            .unwrap_or_default();
        log_in.lock().unwrap().push((rev, changes));
    });
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    // Registering costs a commit but no revision: the hook is not called.
    let mut w = db.write();
    *iter.lock().unwrap() = Some(items.changes(&mut w));
    assert_eq!(w.commit(), 1);

    let mut w = db.write();
    items.insert(&mut w, item("b", 7));
    assert_eq!(w.commit(), 2);

    let mut w = db.write();
    items.delete(&mut w, b"a");
    items.insert(&mut w, item("c", 9));
    assert_eq!(w.commit(), 3);

    assert_eq!(
        *log.lock().unwrap(),
        vec![
            (1, vec![]),
            (2, vec![("b".into(), 2, false)]),
            (3, vec![("c".into(), 3, false), ("a".into(), 3, true)]),
        ]
    );
}

#[test]
fn table_revision_tracks_its_own_writes() {
    let db = Db::new();
    let one = db.table("one", pk as fn(&Item) -> Key, &[]);
    let two = db.table("two", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    one.insert(&mut w, item("a", 1));
    two.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let r = db.read();
    assert_eq!(one.revision(&r), 1);
    assert_eq!(two.revision(&r), 1);

    let mut w = db.write();
    one.insert(&mut w, item("a", 2));
    assert_eq!(w.commit(), 2);

    let r = db.read();
    assert_eq!(r.revision(), 2);
    assert_eq!(one.revision(&r), 2);
    assert_eq!(two.revision(&r), 1);
}

/// Registers a reader over `items` and commits, so the tracker takes effect.
fn observe(db: &Db, items: Table<Item>) -> ChangeIterator<Item> {
    let mut w = db.write();
    let it = items.changes(&mut w);
    w.commit();
    it
}

#[test]
fn changes_report_the_last_state_of_each_key() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let mut reader = observe(&db, items);

    let mut w = db.write();
    assert_eq!(items.insert(&mut w, item("a", 2)).unwrap().val, 1);
    assert_eq!(w.commit(), 2);

    let mut w = db.write();
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 3);

    let mut w = db.write();
    assert_eq!(items.delete(&mut w, b"a").unwrap().val, 2);
    assert_eq!(w.commit(), 4);

    let r = db.read();
    let (changes, _watch) = reader.next(&r).unwrap();
    // No update of "a": its live entry is gone, only the deletion remains.
    assert_eq!(
        drain(changes),
        vec![("b".into(), 3, false), ("a".into(), 4, true)]
    );

    // Create and delete between two `next` calls: only the deletion is seen.
    let mut w = db.write();
    items.insert(&mut w, item("c", 1));
    assert_eq!(w.commit(), 5);
    // The reader has read up to 4, so this commit clears the graveyard.
    assert_eq!(items.graveyard_len(&db.read()), 0);
    let mut w = db.write();
    items.delete(&mut w, b"c");
    assert_eq!(w.commit(), 6);

    let r = db.read();
    let (changes, _watch) = reader.next(&r).unwrap();
    assert_eq!(drain(changes), vec![("c".into(), 6, true)]);
}

#[test]
fn graveyard_holds_until_every_reader_has_read() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let mut fast = observe(&db, items);
    let mut slow = observe(&db, items);

    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 2);
    assert_eq!(items.graveyard_len(&db.read()), 1);

    let r = db.read();
    assert_eq!(drain(fast.next(&r).unwrap().0), vec![("a".into(), 2, true)]);

    // The slow reader has not seen it, so the next commit keeps it.
    let mut w = db.write();
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 3);
    assert_eq!(items.graveyard_len(&db.read()), 1);

    let r = db.read();
    assert_eq!(
        drain(slow.next(&r).unwrap().0),
        vec![("a".into(), 2, true), ("b".into(), 3, false)]
    );
    let mut w = db.write();
    items.insert(&mut w, item("c", 1));
    assert_eq!(w.commit(), 4);
    assert_eq!(items.graveyard_len(&db.read()), 0);
}

#[test]
fn dropping_a_reader_releases_the_graveyard() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let reader = observe(&db, items);
    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 2);
    assert_eq!(items.graveyard_len(&db.read()), 1);

    drop(reader);
    let mut w = db.write();
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 3);
    assert_eq!(items.graveyard_len(&db.read()), 0);
}

#[test]
fn no_reader_means_no_graveyard() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    w.commit();
    let mut w = db.write();
    items.delete(&mut w, b"a");
    w.commit();

    assert_eq!(items.graveyard_len(&db.read()), 0);
}

#[test]
fn compact_drops_history_the_reader_needed() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let mut slow = observe(&db, items);

    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 2);
    assert_eq!(items.graveyard_len(&db.read()), 1);

    db.compact(2);
    let r = db.read();
    assert_eq!(items.graveyard_len(&r), 0);
    // Compaction is not history.
    assert_eq!(r.revision(), 2);

    assert_eq!(slow.next(&r).err().map(|e| e.at), Some(2));
}

#[test]
fn compact_spares_a_reader_that_lost_nothing() {
    let db = Db::new();
    let cold = db.table("cold", pk as fn(&Item) -> Key, &[]);
    let hot = db.table("hot", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    cold.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);
    let mut reader = observe(&db, cold);

    for i in 0..10 {
        let mut w = db.write();
        hot.insert(&mut w, item("h", i));
        w.commit();
    }
    // Nothing in cold's graveyard was dropped: the reader is intact.
    db.compact(11);
    let r = db.read();
    assert_eq!(reader.next(&r).map(|(c, _)| c.count()).ok(), Some(0));

    // A tombstone the reader has not seen is compacted away: now it has lost.
    let mut w = db.write();
    cold.delete(&mut w, b"a");
    let rev = w.commit();
    db.compact(rev);
    assert_eq!(reader.next(&db.read()).err().map(|e| e.at), Some(rev));
}

struct ReadsOnDrop(std::sync::Weak<Db>);

impl Drop for ReadsOnDrop {
    fn drop(&mut self) {
        if let Some(db) = self.0.upgrade() {
            drop(db.read());
        }
    }
}

fn reads_on_drop_key(_: &ReadsOnDrop) -> Key {
    b"key".as_slice().into()
}

#[test]
fn compact_drops_values_outside_the_root_lock() {
    let db = Arc::new(Db::new());
    let items = db.table("items", reads_on_drop_key as fn(&ReadsOnDrop) -> Key, &[]);

    let mut w = db.write();
    let _reader = items.changes(&mut w);
    w.commit();

    let mut w = db.write();
    items.insert(&mut w, ReadsOnDrop(Arc::downgrade(&db)));
    w.commit();

    let mut w = db.write();
    let deleted = items.delete(&mut w, b"key").unwrap();
    let revision = w.commit();
    drop(deleted);

    let (done, completed) = std::sync::mpsc::channel();
    let compacting = db.clone();
    let thread = std::thread::spawn(move || {
        compacting.compact(revision);
        done.send(()).unwrap();
    });
    completed
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("compact deadlocked while dropping a value");
    thread.join().unwrap();
}

#[test]
fn stale_snapshot_does_not_rewind_the_tracker() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut reader = observe(&db, items);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    w.commit();
    let old = db.read();
    let mut w = db.write();
    items.insert(&mut w, item("b", 2));
    w.commit();

    assert_eq!(
        reader.next(&db.read()).map(|(c, _)| c.count()).ok(),
        Some(2)
    );
    assert_eq!(reader.next(&old).map(|(c, _)| c.count()).ok(), Some(0));
    assert_eq!(
        reader.next(&db.read()).map(|(c, _)| c.count()).ok(),
        Some(0)
    );
}

#[test]
fn watches_close_on_the_keys_they_cover() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    for (k, v) in [("p/1", 1), ("p/2", 2), ("q/1", 3)] {
        items.insert(&mut w, item(k, v));
    }
    assert_eq!(w.commit(), 1);

    let r = db.read();
    let (_, touched) = items.get_watch(&r, b"p/1");
    let (_, elsewhere) = items.get_watch(&r, b"q/1");
    let (_, covering) = items.prefix_watch(&r, b"p/");
    let (_, whole_table) = items.all_watch(&r);

    let mut w = db.write();
    items.insert(&mut w, item("p/1", 9));
    assert_eq!(w.commit(), 2);

    assert!(touched.is_closed());
    assert!(covering.is_closed());
    assert!(whole_table.is_closed());
    assert!(!elsewhere.is_closed());

    // A delete closes the same watches an insert does.
    let r = db.read();
    let (_, removed) = items.get_watch(&r, b"q/1");
    let (_, over_removed) = items.prefix_watch(&r, b"q/");
    let mut w = db.write();
    items.delete(&mut w, b"q/1");
    assert_eq!(w.commit(), 3);
    assert!(removed.is_closed());
    assert!(over_removed.is_closed());

    // A key gaining a descendant is not a change to that key's value.
    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    items.insert(&mut w, item("ab", 2));
    assert_eq!(w.commit(), 4);

    let r = db.read();
    let (_, at_a) = items.get_watch(&r, b"a");
    let (_, at_abc) = items.get_watch(&r, b"abc");
    let (_, under_a) = items.prefix_watch(&r, b"a");
    let mut w = db.write();
    items.insert(&mut w, item("abc", 3));
    assert_eq!(w.commit(), 5);
    assert!(!at_a.is_closed());
    assert!(at_abc.is_closed(), "the missing key appeared");
    assert!(under_a.is_closed(), "an entry appeared under the prefix");

    // Updating and deleting the key itself do close it.
    let r = db.read();
    let (_, at_a) = items.get_watch(&r, b"a");
    let mut w = db.write();
    items.insert(&mut w, item("a", 9));
    assert_eq!(w.commit(), 6);
    assert!(at_a.is_closed());

    let r = db.read();
    let (_, at_a) = items.get_watch(&r, b"a");
    let (_, at_ab) = items.get_watch(&r, b"ab");
    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 7);
    assert!(at_a.is_closed());
    assert!(!at_ab.is_closed());
}

/// A watch registered before a commit still completes after it: the cell is
/// closed, not dropped, so nothing is missed between registering and awaiting.
#[tokio::test]
async fn awaiting_a_closed_watch_returns_at_once() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let before = db.read();
    let (_, mut taken_before) = items.get_watch(&before, b"a");

    let mut w = db.write();
    items.insert(&mut w, item("a", 2));
    assert_eq!(w.commit(), 2);

    taken_before.changed().await;
    // The same cell, subscribed to after the commit that closed it.
    let (_, mut taken_after) = items.get_watch(&before, b"a");
    taken_after.changed().await;

    // On the new snapshot the key has an open watch again.
    let (_, fresh) = items.get_watch(&db.read(), b"a");
    assert!(!fresh.is_closed());
}

/// The same when nobody watched the key before the commit: the cell had no
/// channel to close, and the reader that comes late is still told.
#[tokio::test]
async fn a_watch_taken_after_the_commit_is_already_closed() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 1);

    let before = db.read();
    let mut w = db.write();
    items.insert(&mut w, item("a", 2));
    assert_eq!(w.commit(), 2);

    let (value, mut late) = items.get_watch(&before, b"a");
    assert_eq!(value.unwrap().0.val, 1, "the old snapshot still reads 1");
    assert!(late.is_closed());
    late.changed().await;

    let (_, elsewhere) = items.get_watch(&before, b"b");
    assert!(!elsewhere.is_closed());
}

#[tokio::test]
async fn a_parked_task_wakes_on_a_commit_under_its_prefix() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("p/1", 1));
    assert_eq!(w.commit(), 1);

    let r = db.read();
    let (_, mut watch) = items.prefix_watch(&r, b"p/");
    let waiter = tokio::spawn(async move {
        watch.changed().await;
    });
    // Let the task reach the await before anything is committed.
    tokio::task::yield_now().await;

    let mut w = db.write();
    items.insert(&mut w, item("p/2", 2));
    assert_eq!(w.commit(), 2);

    waiter.await.unwrap();
}

// ------------------------------------------------------------ secondary index

#[derive(Debug)]
struct Row {
    key: &'static str,
    tenants: &'static [&'static str],
}

fn row_pk(r: &Row) -> Key {
    r.key.as_bytes().into()
}

fn row_tenants(r: &Row) -> Vec<Key> {
    r.tenants.iter().map(|t| t.as_bytes().into()).collect()
}

const BY_TENANT: Index<Row> = Index {
    name: "tenant",
    keys: row_tenants,
};

fn row(key: &'static str, tenants: &'static [&'static str]) -> Row {
    Row { key, tenants }
}

/// The primary keys listed under `tenant`.
fn by_tenant(db: &Db, rows: Table<Row>, tenant: &str) -> Vec<String> {
    rows.by_index(&db.read(), "tenant", tenant.as_bytes())
        .map(|(k, _, _)| String::from_utf8(k).unwrap())
        .collect()
}

/// The first byte of the key as a second index, so one table carries two.
fn row_first_byte(r: &Row) -> Vec<Key> {
    vec![r.key.as_bytes()[..1].into()]
}

#[test]
fn two_indexes_on_one_table_stay_independent() {
    let db = Db::new();
    let by_first = Index {
        name: "first",
        keys: row_first_byte,
    };
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT, by_first]);

    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    rows.insert(&mut w, row("s1", &["a"]));
    rows.insert(&mut w, row("s2", &["b"]));
    w.commit();
    let list = |index: &'static str, key: &str| -> Vec<String> {
        rows.by_index(&db.read(), index, key.as_bytes())
            .map(|(k, _, _)| String::from_utf8(k).unwrap())
            .collect()
    };
    assert_eq!(list("tenant", "a"), ["r1", "s1"]);
    assert_eq!(list("first", "s"), ["s1", "s2"]);

    // Deleting a tenant's rows through one index updates the other.
    let mut w = db.write();
    for key in by_tenant(&db, rows, "a") {
        rows.delete(&mut w, key.as_bytes());
    }
    w.commit();
    assert!(list("tenant", "a").is_empty());
    assert_eq!(list("first", "s"), ["s2"]);
    assert_eq!(list("first", "r"), Vec::<String>::new());
}

#[test]
fn an_index_follows_the_values_it_covers() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    rows.insert(&mut w, row("r2", &["a"]));
    rows.insert(&mut w, row("r3", &["ab"]));
    rows.insert(&mut w, row("r4", &["a", "b"]));
    rows.insert(&mut w, row("r5", &[]));
    assert_eq!(w.commit(), 1);

    // "a" and "ab" are different keys, not a prefix of one another.
    assert_eq!(by_tenant(&db, rows, "a"), ["r1", "r2", "r4"]);
    assert_eq!(by_tenant(&db, rows, "ab"), ["r3"]);
    // A value under two keys is listed under both; one under none is nowhere.
    assert_eq!(by_tenant(&db, rows, "b"), ["r4"]);
    assert!(by_tenant(&db, rows, "none").is_empty());

    // An update moves the entry to its new tenant.
    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["b"]));
    assert_eq!(w.commit(), 2);
    assert_eq!(by_tenant(&db, rows, "a"), ["r2", "r4"]);
    assert_eq!(by_tenant(&db, rows, "b"), ["r1", "r4"]);

    // A delete drops every entry of the value.
    let mut w = db.write();
    rows.delete(&mut w, b"r4");
    assert_eq!(w.commit(), 3);
    assert_eq!(by_tenant(&db, rows, "a"), ["r2"]);
    assert_eq!(by_tenant(&db, rows, "b"), ["r1"]);

    // The rows come back resolved through the primary tree.
    let r = db.read();
    let listed: Vec<_> = rows
        .by_index(&r, "tenant", b"a")
        .map(|(_, v, rev)| (v.key, rev))
        .collect();
    assert_eq!(listed, [("r2", 1)]);
}

#[test]
fn an_index_watch_covers_one_index_key() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    rows.insert(&mut w, row("r9", &["b"]));
    assert_eq!(w.commit(), 1);

    let r = db.read();
    let (_, watched) = rows.by_index_watch(&r, "tenant", b"a");
    let (_, elsewhere) = rows.by_index_watch(&r, "tenant", b"b");

    let mut w = db.write();
    rows.insert(&mut w, row("r2", &["a"]));
    assert_eq!(w.commit(), 2);

    assert!(watched.is_closed());
    assert!(!elsewhere.is_closed());
}

/// The index tree only sees the difference of the two key sets, so a row that
/// keeps its index key keeps its entry, and the watch over that key stays open.
#[test]
fn an_index_watch_ignores_an_update_that_keeps_its_key() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    assert_eq!(w.commit(), 1);

    let r = db.read();
    let (_, same_tenant) = rows.by_index_watch(&r, "tenant", b"a");
    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    assert_eq!(w.commit(), 2);
    assert_eq!(
        rows.get(&db.read(), b"r1").unwrap().1,
        2,
        "the row itself was rewritten"
    );
    assert!(!same_tenant.is_closed());

    // Changing the tenant moves the entry, which both keys see.
    let r = db.read();
    let (_, left) = rows.by_index_watch(&r, "tenant", b"a");
    let (_, joined) = rows.by_index_watch(&r, "tenant", b"b");
    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["b"]));
    assert_eq!(w.commit(), 3);
    assert!(left.is_closed());
    assert!(joined.is_closed());
    assert!(by_tenant(&db, rows, "a").is_empty());
    assert_eq!(by_tenant(&db, rows, "b"), ["r1"]);
}

#[test]
#[should_panic(expected = "table rows has no index colour")]
fn an_unknown_index_is_a_programming_error() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);
    let _ = rows.by_index(&db.read(), "colour", b"a").count();
}

#[test]
#[should_panic(expected = "table rows has duplicate index tenant")]
fn duplicate_index_names_are_rejected() {
    let db = Db::new();
    let _ = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT, BY_TENANT]);
}

#[test]
#[should_panic(expected = "belongs to another Db")]
fn table_handle_rejects_another_db() {
    let a = Db::new();
    let b = Db::new();
    let ta = a.table("t", pk as fn(&Item) -> Key, &[]);
    let _tb = b.table("t", pk as fn(&Item) -> Key, &[]);
    let _ = ta.get(&b.read(), b"k");
}

#[test]
#[should_panic(expected = "belongs to another Db")]
fn table_handle_rejects_another_db_after_its_slot_was_opened() {
    let a = Db::new();
    let b = Db::new();
    let ta = a.table("t", pk as fn(&Item) -> Key, &[]);
    let tb = b.table("t", pk as fn(&Item) -> Key, &[]);

    let mut w = b.write();
    tb.insert(&mut w, item("local", 1));
    ta.insert(&mut w, item("foreign", 2));
}
