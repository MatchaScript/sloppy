//! Row 2 and row 3 of the plan: root cell, transactions, revision, hook,
//! revision index, graveyard, `changes`, watermark, `compact`.

use std::sync::{Arc, Mutex};

use sloppy::db::{Change, ChangeIterator, Db, Key, ReadTxn, Revision, Table};

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let one = db.table("one", pk as fn(&Item) -> Key);
    let two = db.table("two", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
    let cold = db.table("cold", pk as fn(&Item) -> Key);
    let hot = db.table("hot", pk as fn(&Item) -> Key);

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

#[test]
fn stale_snapshot_does_not_rewind_the_tracker() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key);
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
    let items = db.table("items", pk as fn(&Item) -> Key);

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
}
