//! Row 2 and row 3 of the plan: root cell, transactions, revision, the change
//! stream, watches, `compact`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use sloppy::db::{Change, ChangeIterator, Db, Index, Key, Revision, Table, Version};

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

/// The same with the value each change carries.
fn drain_values<I: Iterator<Item = Change<Item>>>(
    changes: I,
) -> Vec<(String, u32, Revision, bool)> {
    changes
        .map(|c| {
            (
                String::from_utf8(c.key.to_vec()).unwrap(),
                c.value.val,
                c.revision,
                c.deleted,
            )
        })
        .collect()
}

/// `(revision, value, deleted)` of every version of `key`.
fn versions(
    items: Table<Item>,
    txn: &sloppy::db::ReadTxn,
    key: &[u8],
) -> Vec<(Revision, u32, bool)> {
    items
        .versions(txn, key)
        .iter()
        .map(|v| (v.revision, v.value.val, v.deleted))
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
            .map(|(_, v, _)| v.key)
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    assert_eq!(items.lower_bound(&after, b"b").count(), 1);
}

/// The four plain reads through the open transaction see what it has written,
/// at the revision its commit will carry, while everything else reads the root.
#[test]
fn a_write_txn_reads_its_own_writes() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let other = db.table("other", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    for key in ["a", "b", "d"] {
        items.insert(&mut w, item(key, 1));
    }
    other.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let mut w = db.write();
    items.insert(&mut w, item("b", 2));
    items.insert(&mut w, item("c", 2));
    assert_eq!(items.delete(&mut w, b"a").unwrap().val, 1);
    // Taken while the writer is open, after its writes.
    let concurrent = db.read();

    // What this transaction wrote is at the revision it will commit; what it
    // left alone keeps the revision of the commit that wrote it.
    assert_eq!(items.get(&w, b"b").map(|(v, r)| (v.val, r)), Some((2, 2)));
    assert_eq!(items.get(&w, b"c").map(|(v, r)| (v.val, r)), Some((2, 2)));
    assert_eq!(items.get(&w, b"d").map(|(v, r)| (v.val, r)), Some((1, 1)));
    assert!(items.get(&w, b"a").is_none(), "deleted in this transaction");
    assert_eq!(
        items
            .lower_bound(&w, b"a")
            .map(|(_, v, r)| (v.key, v.val, r))
            .collect::<Vec<_>>(),
        vec![("b", 2, 2), ("c", 2, 2), ("d", 1, 1)]
    );
    assert_eq!(items.prefix(&w, b"c").count(), 1);
    assert_eq!(items.all(&w).count(), 3);

    // A table this transaction never touched reads the root it opened on.
    assert_eq!(other.get(&w, b"a").map(|(v, r)| (v.val, r)), Some((1, 1)));

    // None of it is visible to a reader until the commit.
    assert_eq!(concurrent.revision(), 1);
    assert_eq!(items.get(&concurrent, b"a").map(|(v, _)| v.val), Some(1));
    assert_eq!(items.get(&concurrent, b"b").map(|(v, _)| v.val), Some(1));
    assert!(items.get(&concurrent, b"c").is_none());
    assert_eq!(items.all(&concurrent).count(), 3);

    assert_eq!(w.commit(), 2);
    let after = db.read();
    assert_eq!(items.all(&after).count(), 3);
    assert_eq!(
        items.get(&after, b"c").map(|(v, r)| (v.val, r)),
        Some((2, 2))
    );
}

#[test]
fn abort_leaves_nothing() {
    let db = Db::new();
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
}

/// A reader observes the table from where it stood when it was taken, so one
/// taken in a transaction that never commits reads the commits that follow it.
#[test]
fn a_change_reader_from_an_aborted_registration_reads_on() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut w = db.write();
    let mut reader = items.changes(&mut w);
    drop(w);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    assert_eq!(
        drain(reader.next(&db.read()).unwrap().0),
        vec![("a".into(), 1, false)]
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

/// Takes a reader over `items`, observing the table where it stands.
fn observe(db: &Db, items: Table<Item>) -> ChangeIterator<Item> {
    let mut w = db.write();
    let it = items.changes(&mut w);
    w.commit();
    it
}

#[test]
fn changes_report_one_record_per_key_and_commit() {
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
    // Both commits that touched "a" are there: the update and the deletion,
    // which carries the value it removed.
    assert_eq!(
        drain_values(changes),
        vec![
            ("a".into(), 2, 2, false),
            ("b".into(), 1, 3, false),
            ("a".into(), 2, 4, true),
        ]
    );

    // Create in one commit and delete in the next: one record each.
    let mut w = db.write();
    items.insert(&mut w, item("c", 1));
    assert_eq!(w.commit(), 5);
    let mut w = db.write();
    items.delete(&mut w, b"c");
    assert_eq!(w.commit(), 6);

    let r = db.read();
    let (changes, _watch) = reader.next(&r).unwrap();
    assert_eq!(
        drain(changes),
        vec![("c".into(), 5, false), ("c".into(), 6, true)]
    );
}

/// One key written twice in one commit leaves one version and one record: the
/// second write replaces what the first left. Written again in a later commit
/// it leaves a second version and a second record, and both reach the reader.
#[test]
fn a_key_written_twice_in_one_commit_leaves_one_record() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut reader = observe(&db, items);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    items.insert(&mut w, item("a", 2));
    assert_eq!(w.commit(), 1);

    let mut w = db.write();
    items.insert(&mut w, item("a", 3));
    assert_eq!(w.commit(), 2);

    let r = db.read();
    assert_eq!(versions(items, &r, b"a"), [(1, 2, false), (2, 3, false)]);
    assert_eq!(
        drain(reader.next(&r).unwrap().0),
        vec![("a".into(), 1, false), ("a".into(), 2, false)]
    );
}

/// Each record hands over the version its own commit left, so a key rewritten
/// in a later commit reports both values rather than the newest twice.
#[test]
fn changes_carry_the_value_each_commit_left() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut reader = observe(&db, items);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    items.insert(&mut w, item("a", 2));
    assert_eq!(w.commit(), 1);

    let mut w = db.write();
    items.insert(&mut w, item("a", 3));
    assert_eq!(w.commit(), 2);

    assert_eq!(
        drain_values(reader.next(&db.read()).unwrap().0),
        vec![("a".into(), 2, 1, false), ("a".into(), 3, 2, false)]
    );
}

/// Two readers of one table read it apart: each is told of the deletion once,
/// with the value it removed, whenever it gets round to reading.
#[test]
fn every_reader_is_told_of_a_deletion() {
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
    let r = db.read();
    // The key is gone, and the tombstone that ends its row holds the value.
    assert!(items.get(&r, b"a").is_none());
    assert_eq!(versions(items, &r, b"a"), [(1, 1, false), (2, 1, true)]);

    assert_eq!(drain(fast.next(&r).unwrap().0), vec![("a".into(), 2, true)]);

    // The slow reader has not read it, and the record is still there for it.
    let mut w = db.write();
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 3);

    let r = db.read();
    assert_eq!(
        drain(slow.next(&r).unwrap().0),
        vec![("a".into(), 2, true), ("b".into(), 3, false)]
    );
    // Each reader is told once: the fast one gets what came after its read.
    let mut w = db.write();
    items.insert(&mut w, item("c", 1));
    assert_eq!(w.commit(), 4);
    let r = db.read();
    assert_eq!(
        drain(fast.next(&r).unwrap().0),
        vec![("b".into(), 3, false), ("c".into(), 4, false)]
    );
    assert_eq!(
        drain(slow.next(&r).unwrap().0),
        vec![("c".into(), 4, false)]
    );
}

/// Rebuilding a table by deleting every key and writing it again leaves one
/// record per key and per round, and no tombstone: the write replaces the
/// version the deletion left and reuses its record.
#[test]
fn re_creating_a_key_in_the_same_commit_replaces_its_deletion() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    for key in ["a", "b"] {
        items.insert(&mut w, item(key, 1));
    }
    assert_eq!(w.commit(), 1);

    let mut stalled = observe(&db, items);
    for round in 2..5 {
        let mut w = db.write();
        for key in ["a", "b"] {
            items.delete(&mut w, key.as_bytes());
        }
        for key in ["a", "b"] {
            items.insert(&mut w, item(key, round));
        }
        w.commit();

        let r = db.read();
        let round = Revision::from(round);
        assert_eq!(
            items
                .versions(&r, b"a")
                .last()
                .map(|v| (v.revision, v.deleted)),
            Some((round, false)),
            "round {round} ends on the value, not the deletion"
        );
    }

    assert_eq!(
        drain_values(stalled.next(&db.read()).unwrap().0),
        vec![
            ("a".into(), 2, 2, false),
            ("b".into(), 2, 2, false),
            ("a".into(), 3, 3, false),
            ("b".into(), 3, 3, false),
            ("a".into(), 4, 4, false),
            ("b".into(), 4, 4, false),
        ]
    );
}

/// Reading a change does not release its record: what one reader has taken is
/// still there for the next, and only a compaction drops it.
#[test]
fn records_live_until_a_compaction() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut eager = observe(&db, items);
    let mut lazy = observe(&db, items);
    let mut stalled = observe(&db, items);

    for val in 1..=3 {
        let mut w = db.write();
        items.insert(&mut w, item("a", val));
        assert_eq!(w.commit(), Revision::from(val));
        assert_eq!(eager.next(&db.read()).unwrap().0.count(), 1);
    }

    // The reader that waited is told of every commit, one record each.
    assert_eq!(
        drain(lazy.next(&db.read()).unwrap().0),
        vec![
            ("a".into(), 1, false),
            ("a".into(), 2, false),
            ("a".into(), 3, false),
        ]
    );

    db.compact(3);
    assert_eq!(stalled.next(&db.read()).err().map(|e| e.at), Some(3));
    assert_eq!(eager.next(&db.read()).unwrap().0.count(), 0);
}

/// The versions a compaction released go when the key is next written; until
/// then the row keeps them, so a reader below the bound still finds them.
#[test]
fn a_write_trims_the_versions_a_compaction_released() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    for val in 1..=3 {
        let mut w = db.write();
        items.insert(&mut w, item("a", val));
        assert_eq!(w.commit(), Revision::from(val));
    }
    assert_eq!(
        versions(items, &db.read(), b"a"),
        [(1, 1, false), (2, 2, false), (3, 3, false)]
    );

    db.compact(2);
    assert_eq!(
        versions(items, &db.read(), b"a"),
        [(1, 1, false), (2, 2, false), (3, 3, false)],
        "compaction alone does not touch the row"
    );

    let mut w = db.write();
    items.insert(&mut w, item("a", 4));
    assert_eq!(w.commit(), 4);
    assert_eq!(
        versions(items, &db.read(), b"a"),
        [(2, 2, false), (3, 3, false), (4, 4, false)],
        "the version at the bound ends the row, so a reader there still reads it"
    );

    // The same over a longer row: what the write leaves is every version past
    // the bound, the one at the bound and the one it just wrote included.
    for val in 5..=7 {
        let mut w = db.write();
        items.insert(&mut w, item("a", val));
        assert_eq!(w.commit(), Revision::from(val));
    }
    db.compact(5);
    let mut w = db.write();
    items.insert(&mut w, item("a", 8));
    assert_eq!(w.commit(), 8);
    assert_eq!(
        versions(items, &db.read(), b"a"),
        [(5, 5, false), (6, 6, false), (7, 7, false), (8, 8, false)]
    );
}

/// The other half of the reclamation: a row whose deletion both bounds have
/// passed leaves the primary tree with its versions.
#[test]
fn a_compaction_past_a_deletion_sweeps_the_row() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 1);
    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 2);
    assert_eq!(
        versions(items, &db.read(), b"a"),
        [(1, 1, false), (2, 1, true)]
    );

    db.compact(2);
    let r = db.read();
    assert!(items.get(&r, b"a").is_none());
    assert!(versions(items, &r, b"a").is_empty(), "the row went with it");
    assert_eq!(
        versions(items, &r, b"b"),
        [(1, 1, false)],
        "a live row stays"
    );
}

/// The compaction bound is the only bound: a change reader that has not seen
/// the deletion loses it with the row, and says so when it next reads.
#[test]
fn a_reader_below_a_deletion_loses_it() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut reader = observe(&db, items);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);
    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 2);

    db.compact(2);
    assert!(versions(items, &db.read(), b"a").is_empty());
    assert_eq!(reader.next(&db.read()).err().map(|e| e.at), Some(2));
}

/// The caller's revision is what the transaction writes at and what its commit
/// leaves the database at.
#[test]
fn write_at_takes_the_revision_it_is_given() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write_at(10);
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 10);

    let r = db.read();
    assert_eq!(r.revision(), 10);
    assert_eq!(
        items.get(&r, b"a").map(|(v, rev)| (v.val, rev)),
        Some((1, 10))
    );

    // A transaction that writes nothing leaves the revision where it was.
    assert_eq!(db.write_at(20).commit(), 10);
    // And `write` carries on from what is visible.
    let mut w = db.write();
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 11);
}

#[test]
#[should_panic(expected = "is not past the visible")]
fn write_at_rejects_a_revision_that_goes_back() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let _ = db.write_at(1);
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

    db.compact(2);
    let r = db.read();
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
    // None of cold's records was dropped: the reader is intact.
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
fn compact_drops_values_outside_the_writer_lock() {
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
fn stale_snapshot_does_not_rewind_the_reader() {
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

/// The `Db` watch completes on the next commit that bumps the revision, which
/// is where a reader that persists the change stream wakes up.
#[tokio::test]
async fn a_db_watch_completes_on_the_next_revision() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut watch = db.watch();
    assert!(!watch.is_closed());
    // A transaction that writes nothing takes no revision, so it is not a
    // change to wake on.
    assert_eq!(db.write().commit(), 0);
    assert!(!watch.is_closed());

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    assert!(watch.is_closed());
    watch.changed().await;
}

/// A watch outlives the database it came from, and there is no commit left to
/// wake it, so dropping the `Db` has to release it.
#[tokio::test]
async fn a_watch_completes_when_its_db_is_dropped() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let mut watch = db.watch();
    assert!(!watch.is_closed());

    drop(db);
    assert!(watch.is_closed());
    watch.changed().await;
}

/// A table watch takes its baseline from the snapshot it was handed, so one
/// taken on a snapshot the database has already moved past reports the change
/// it missed instead of parking on a commit that has been and gone.
#[tokio::test]
async fn a_table_watch_on_a_stale_snapshot_is_already_closed() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let mut reader = observe(&db, items);

    let old = db.read();
    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let (changes, mut watch) = reader.next(&old).unwrap();
    assert_eq!(changes.count(), 0, "the snapshot is from before the commit");
    assert!(watch.is_closed());
    watch.changed().await;
}

/// The other watch: a change reader's, which completes on the next commit to
/// its table. A task parked on it wakes without being asked again.
#[tokio::test]
async fn a_parked_task_wakes_on_a_commit_to_its_table() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let other = db.table("other", pk as fn(&Item) -> Key, &[]);
    let mut reader = observe(&db, items);

    let r = db.read();
    let (changes, mut watch) = reader.next(&r).unwrap();
    assert_eq!(changes.count(), 0);
    assert!(!watch.is_closed());

    // A commit to another table is not a change to this one.
    let mut w = db.write();
    other.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);
    assert!(!watch.is_closed());

    let waiter = tokio::spawn(async move {
        watch.changed().await;
    });
    // Let the task reach the await before anything is committed.
    tokio::task::yield_now().await;

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
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
        .map(|(v, _)| v.key.to_string())
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
            .map(|(v, _)| v.key.to_string())
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
        .map(|(v, rev)| (v.key, rev))
        .collect();
    assert_eq!(listed, [("r2", 1)]);
}

/// An index entry carries versions like the row it lists, so a listing is read
/// at the reader's revision: what a later commit deleted or moved is still
/// listed for a reader taken before it.
#[test]
fn an_index_lists_what_its_reader_can_see() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    rows.insert(&mut w, row("r2", &["a"]));
    assert_eq!(w.commit(), 1);
    let before = db.read();

    let mut w = db.write();
    rows.delete(&mut w, b"r1");
    rows.insert(&mut w, row("r2", &["b"]));
    assert_eq!(w.commit(), 2);

    let listed: Vec<_> = rows
        .by_index(&before, "tenant", b"a")
        .map(|(v, _)| v.key)
        .collect();
    assert_eq!(listed, ["r1", "r2"], "the reader is before both writes");
    assert!(by_tenant(&db, rows, "a").is_empty());
    assert_eq!(by_tenant(&db, rows, "b"), ["r2"]);
}

/// An index key is arbitrary bytes. The entry keeps it apart from the primary
/// key that follows it, so a key holding the byte the two used to be joined
/// with is still only found by itself.
#[test]
fn an_index_key_holding_a_zero_byte_lists_only_its_own_rows() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    rows.insert(&mut w, row("p", &["a\0b"]));
    rows.insert(&mut w, row("b\0p", &["z"]));
    assert_eq!(w.commit(), 1);

    assert_eq!(by_tenant(&db, rows, "a\0b"), ["p"]);
    assert!(by_tenant(&db, rows, "a").is_empty());
}

fn version(revision: Revision, value: Row, deleted: bool) -> Version<Row> {
    Version {
        revision,
        value: Arc::new(value),
        deleted,
    }
}

/// Recovery puts a row back with the versions it had. The revisions come from
/// what was persisted, the indexes follow the newest version, and the change
/// stream stays empty: this is not a write for readers to follow.
#[test]
fn load_puts_a_row_back_without_a_record() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    let mut reader = rows.changes(&mut w);
    w.commit();

    let mut w = db.write_at(5);
    rows.load(
        &mut w,
        b"r1",
        vec![
            version(3, row("r1", &["a"]), false),
            version(4, row("r1", &["b"]), false),
        ],
    );
    rows.load(&mut w, b"r2", vec![version(4, row("r2", &["a"]), true)]);
    assert_eq!(w.commit(), 5);

    let r = db.read();
    assert_eq!(
        rows.get(&r, b"r1").map(|(v, rev)| (v.key, rev)),
        Some(("r1", 4))
    );
    assert!(rows.get(&r, b"r2").is_none(), "loaded as deleted");
    assert_eq!(
        rows.versions(&r, b"r1")
            .iter()
            .map(|v| (v.revision, v.value.key))
            .collect::<Vec<_>>(),
        [(3, "r1"), (4, "r1")]
    );
    // The newest version is what the indexes list.
    assert_eq!(by_tenant(&db, rows, "b"), ["r1"]);
    assert!(by_tenant(&db, rows, "a").is_empty());

    assert_eq!(reader.next(&r).map(|(c, _)| c.count()).ok(), Some(0));
}

/// A record is resolved through the row it names, so a row a reader has not
/// caught up with must not be replaced.
#[test]
#[should_panic(expected = "table rows holds change records a loaded row would strand")]
fn loading_over_a_held_record_is_a_programming_error() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write();
    let _reader = rows.changes(&mut w);
    w.commit();

    let mut w = db.write();
    rows.insert(&mut w, row("r1", &["a"]));
    assert_eq!(w.commit(), 1);

    let mut w = db.write();
    rows.load(&mut w, b"r1", vec![version(2, row("r1", &["b"]), false)]);
    w.commit();
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

// ---------------------------------------------------------- commit ordering

/// Two writers running flat out: the writer lock covers the whole commit, so
/// the revisions come out in order, once each, and neither writer is lost.
#[test]
fn commits_from_two_threads_serialize() {
    const ROUNDS: u32 = 5_000;

    let db = Arc::new(Db::new());
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    let seen: Vec<Vec<Revision>> = ["a", "b"]
        .map(|key| {
            let db = db.clone();
            std::thread::spawn(move || {
                (0..ROUNDS)
                    .map(|n| {
                        let mut w = db.write();
                        items.insert(&mut w, item(key, n));
                        w.commit()
                    })
                    .collect()
            })
        })
        .map(|thread| thread.join().expect("a writer panicked"))
        .into();

    for revisions in &seen {
        assert!(
            revisions.windows(2).all(|pair| pair[0] < pair[1]),
            "one thread's revisions went backwards"
        );
    }
    let mut every = seen.concat();
    every.sort_unstable();
    assert_eq!(
        every,
        (1..=Revision::from(ROUNDS) * 2).collect::<Vec<_>>(),
        "every commit took one revision of its own"
    );

    let r = db.read();
    assert_eq!(items.all(&r).count(), 2, "both writers' keys are there");
    assert_eq!(items.get(&r, b"a").unwrap().0.val, ROUNDS - 1);
    assert_eq!(items.get(&r, b"b").unwrap().0.val, ROUNDS - 1);
}

/// `table` and `compact` run between transactions without moving the revision,
/// and the next `write` carries on from where they left the database.
#[test]
fn table_and_compact_leave_the_revision_where_it_was() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);

    let mut w = db.write();
    items.insert(&mut w, item("a", 1));
    assert_eq!(w.commit(), 1);

    let late = db.table("late", pk as fn(&Item) -> Key, &[]);
    let mut w = db.write();
    late.insert(&mut w, item("x", 1));
    assert_eq!(w.commit(), 2);
    assert_eq!(late.get(&db.read(), b"x").unwrap().0.val, 1);

    let mut slow = observe(&db, items);
    let mut w = db.write();
    items.delete(&mut w, b"a");
    assert_eq!(w.commit(), 3);

    db.compact(3);
    assert_eq!(db.read().revision(), 3, "a compaction is not a commit");
    let mut w = db.write();
    items.insert(&mut w, item("b", 1));
    assert_eq!(w.commit(), 4);
    // The reader lost the deletion the compaction dropped.
    assert_eq!(slow.next(&db.read()).err().map(|e| e.at), Some(3));
}

/// A commit that panics partway through has already written part of itself to
/// the trees under a revision it never published. There is no undo, so the
/// database takes no further writes rather than letting the next commit take
/// that revision and publish them.
#[test]
fn a_panicked_commit_stops_the_writer() {
    /// Panics on one row, so a commit fails between two keys.
    fn tenants_or_panic(r: &Row) -> Vec<Key> {
        assert_ne!(r.key, "boom", "the index panics on this row");
        row_tenants(r)
    }
    let by_tenant = Index {
        name: "tenant",
        keys: tenants_or_panic,
    };

    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[by_tenant]);

    let commit = catch_unwind(AssertUnwindSafe(|| {
        let mut w = db.write();
        rows.insert(&mut w, row("r1", &["a"]));
        rows.insert(&mut w, row("boom", &["a"]));
        w.commit()
    }));
    assert!(commit.is_err(), "the index panicked while applying");

    // A read runs on, at the revision the last whole commit published.
    let r = db.read();
    assert_eq!(r.revision(), 0);
    assert!(rows.get(&r, b"r1").is_none(), "never published");

    let refused = catch_unwind(AssertUnwindSafe(|| db.write()))
        .err()
        .expect("the next write is refused");
    let message = *refused.downcast::<String>().expect("a panic message");
    assert!(
        message.contains("panicked partway through revision 1"),
        "{message}"
    );
}

/// A row written often enough carries a chain as long as its history, and the
/// chain is let go one link at a time rather than one stack frame at a time.
#[test]
fn a_long_version_chain_is_dropped_without_recursion() {
    let db = Db::new();
    let items = db.table("items", pk as fn(&Item) -> Key, &[]);
    for val in 0..200_000 {
        let mut w = db.write();
        items.insert(&mut w, item("a", val));
        w.commit();
    }
    drop(db);
}

/// A row loaded and then written in one transaction: the write follows the
/// chain the load put there, and leaves the record a write leaves.
#[test]
fn a_write_over_a_loaded_row_keeps_its_versions() {
    let db = Db::new();
    let rows = db.table("rows", row_pk as fn(&Row) -> Key, &[BY_TENANT]);

    let mut w = db.write_at(5);
    rows.load(
        &mut w,
        b"r1",
        vec![
            version(3, row("r1", &["a"]), false),
            version(4, row("r1", &["b"]), false),
        ],
    );
    rows.insert(&mut w, row("r1", &["c"]));
    assert_eq!(
        rows.versions(&w, b"r1")
            .iter()
            .map(|v| (v.revision, v.value.tenants))
            .collect::<Vec<_>>(),
        [(3, &["a"][..]), (4, &["b"][..]), (5, &["c"][..])],
        "the transaction reads the load under its own write"
    );
    assert_eq!(w.commit(), 5);

    let r = db.read();
    assert_eq!(
        rows.versions(&r, b"r1")
            .iter()
            .map(|v| (v.revision, v.value.tenants))
            .collect::<Vec<_>>(),
        [(3, &["a"][..]), (4, &["b"][..]), (5, &["c"][..])],
    );
    // The indexes follow the newest version, whichever put it there.
    assert_eq!(by_tenant(&db, rows, "c"), ["r1"]);
    assert!(by_tenant(&db, rows, "b").is_empty());
}
