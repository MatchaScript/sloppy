//! The tree against a `BTreeMap` over a fixed-seed random operation stream.

use std::collections::BTreeMap;

use sloppy::tree::Tree;

/// Numerical Recipes LCG. Fixed seed, so a failure repeats.
struct Lcg(u64);

impl Lcg {
    fn step(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn below(&mut self, n: u64) -> u64 {
        self.step() % n
    }

    /// A key of length 0..8 over a three-letter alphabet, so prefixes collide.
    fn key(&mut self) -> Vec<u8> {
        let len = self.below(9);
        (0..len)
            .map(|_| b'a' + u8::try_from(self.below(3)).unwrap())
            .collect()
    }
}

fn entries(tree: &Tree<u64>) -> Vec<(Vec<u8>, u64)> {
    tree.iter().map(|(k, v)| (k, **v)).collect()
}

fn model_entries<'a>(it: impl Iterator<Item = (&'a Vec<u8>, &'a u64)>) -> Vec<(Vec<u8>, u64)> {
    it.map(|(k, v)| (k.clone(), *v)).collect()
}

#[test]
fn matches_btreemap() {
    let mut rng = Lcg(0x2026_0904);
    let mut tree: Tree<u64> = Tree::new();
    let mut model: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    let mut ops = 0;
    let mut stamp = 0;
    let mut hit_deletes = 0;

    while ops < 20_000 {
        let batch = 1 + rng.below(16);
        let mut txn = tree.txn();
        for _ in 0..batch {
            let key = rng.key();
            if rng.below(3) == 0 {
                let gone = txn.delete(&key).map(|v| *v);
                assert_eq!(gone, model.remove(&key), "delete {key:?}");
                hit_deletes += usize::from(gone.is_some());
            } else {
                stamp += 1;
                assert_eq!(
                    txn.insert(&key, stamp).map(|v| *v),
                    model.insert(key.clone(), stamp),
                    "insert {key:?}"
                );
            }
            // The txn must see its own earlier writes.
            assert_eq!(txn.get(&key).map(|v| **v), model.get(&key).copied());
            ops += 1;
        }
        tree = txn.commit_and_notify();

        tree.assert_invariants();
        assert_eq!(tree.len(), model.len());
        assert_eq!(tree.is_empty(), model.is_empty());
        assert_eq!(entries(&tree), model_entries(model.iter()));

        for _ in 0..4 {
            let key = rng.key();
            assert_eq!(
                tree.get(&key).0.map(|v| **v),
                model.get(&key).copied(),
                "get {key:?}"
            );
        }
        for _ in 0..2 {
            let p = rng.key();
            let want = model_entries(model.iter().filter(|(k, _)| k.starts_with(&p)));
            let got: Vec<_> = tree.prefix(&p).map(|(k, v)| (k, **v)).collect();
            assert_eq!(got, want, "prefix {p:?}");
        }
        for _ in 0..2 {
            let key = rng.key();
            let want = model_entries(model.range(key.clone()..));
            let got: Vec<_> = tree.lower_bound(&key).map(|(k, v)| (k, **v)).collect();
            assert_eq!(got, want, "lower_bound {key:?}");
        }
    }
    assert!(!tree.is_empty());
    // The stream has to exercise removal and merging, not just misses.
    assert!(hit_deletes > 1_000, "only {hit_deletes} deletes hit");
}

#[test]
fn watches_close_on_the_changed_path() {
    let mut txn = Tree::new().txn();
    txn.insert(b"a", 1);
    txn.insert(b"b", 2);
    let t0 = txn.commit_and_notify();

    // A commit on an unrelated key leaves the key's own node alone.
    let (_, wa) = t0.get(b"a");
    let (_, wb) = t0.get(b"b");
    let root = t0.root_watch();
    let mut txn = t0.txn();
    txn.insert(b"b", 22);
    let t1 = txn.commit_and_notify();
    assert!(!wa.is_closed());
    assert!(wb.is_closed());
    assert!(root.is_closed(), "root fires on any commit");

    // A commit on the key itself closes its watch.
    let (_, wa) = t1.get(b"a");
    let mut txn = t1.txn();
    txn.delete(b"a");
    let t2 = txn.commit_and_notify();
    assert!(wa.is_closed());

    // A watch on a missing key closes when the key appears.
    let (missing, wz) = t2.get(b"zz");
    assert!(missing.is_none());
    let mut txn = t2.txn();
    txn.insert(b"zz", 3);
    let t3 = txn.commit_and_notify();
    assert!(wz.is_closed());

    // A prefix watch closes when an entry appears under it.
    let (empty, wp) = t3.prefix_watch(b"b/");
    assert_eq!(empty.count(), 0);
    let mut txn = t3.txn();
    txn.insert(b"b/x", 4);
    let t4 = txn.commit_and_notify();
    assert!(wp.is_closed());

    let (found, wp) = t4.prefix_watch(b"b/");
    assert_eq!(found.count(), 1);
    let mut txn = t4.txn();
    txn.insert(b"zz", 5);
    let t5 = txn.commit_and_notify();
    assert!(!wp.is_closed());
    assert_eq!(t5.len(), 3);
}

#[test]
fn a_value_watch_ignores_the_rest_of_the_subtree() {
    let mut txn = Tree::new().txn();
    txn.insert(b"a", 1);
    txn.insert(b"ab", 2);
    let t0 = txn.commit_and_notify();

    // A key gaining a descendant is not a change to that key's value.
    let (_, wa) = t0.get(b"a");
    let (_, wab) = t0.get(b"ab");
    let (_, wabc) = t0.get(b"abc");
    let mut txn = t0.txn();
    txn.insert(b"abc", 3);
    let t1 = txn.commit_and_notify();
    assert!(!wa.is_closed());
    assert!(!wab.is_closed());
    assert!(wabc.is_closed(), "the missing key appeared");

    // Replacing a value closes that value's watch alone.
    let (_, wa) = t1.get(b"a");
    let (_, wab) = t1.get(b"ab");
    let mut txn = t1.txn();
    txn.insert(b"a", 11);
    let t2 = txn.commit_and_notify();
    assert!(wa.is_closed());
    assert!(!wab.is_closed());

    // Deleting "a" merges its node into "ab", which keeps its own cell.
    let (_, wa) = t2.get(b"a");
    let (_, wab) = t2.get(b"ab");
    let mut txn = t2.txn();
    txn.delete(b"a");
    let t3 = txn.commit_and_notify();
    assert!(wa.is_closed());
    assert!(!wab.is_closed());
    assert_eq!(t3.len(), 2);
}

/// One node through every kind: 4 -> 16 -> 48 -> 256 on the way up and back
/// down again, checked against the model after every single write.
#[test]
fn one_node_grows_and_shrinks_through_every_kind() {
    let mut tree: Tree<u64> = Tree::new();
    let mut model: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    // The prefix carries a value of its own, so the node survives losing all
    // but one child instead of being merged away.
    let mut keys = vec![b"k".to_vec()];
    keys.extend((0..=255u8).map(|byte| vec![b'k', byte]));

    for (i, key) in keys.iter().enumerate() {
        let stamp = u64::try_from(i).unwrap();
        let mut txn = tree.txn();
        assert_eq!(txn.insert(key, stamp), None);
        tree = txn.commit_and_notify();
        model.insert(key.clone(), stamp);
        tree.assert_invariants();
        assert_eq!(tree.len(), model.len());
        assert_eq!(entries(&tree), model_entries(model.iter()));
    }
    assert_eq!(tree.len(), 257, "the node holds all 256 children");

    // Back down to one child, from the middle outwards so the removals do not
    // all fall at one end of the node.
    let mut order: Vec<&Vec<u8>> = keys[1..].iter().collect();
    order.sort_by_key(|key| key[1].wrapping_sub(128));
    for key in order.iter().take(255) {
        let mut txn = tree.txn();
        assert!(txn.delete(key).is_some());
        tree = txn.commit_and_notify();
        model.remove(*key);
        tree.assert_invariants();
        assert_eq!(tree.len(), model.len());
        assert_eq!(entries(&tree), model_entries(model.iter()));
    }
    assert_eq!(tree.len(), 2, "the prefix and its last child");
}

/// A cell has no channel until somebody watches it, so a commit can replace a
/// node before any watch exists. The reader that subscribes afterwards must
/// still be told, because no later commit will close that cell for it.
#[tokio::test]
async fn a_watch_taken_after_the_change_is_already_closed() {
    let mut txn = Tree::new().txn();
    txn.insert(b"a", 1);
    txn.insert(b"b", 2);
    let t0 = txn.commit_and_notify();

    // Nobody watched anything before this commit.
    let mut txn = t0.txn();
    txn.insert(b"a", 11);
    let t1 = txn.commit_and_notify();

    let (value, mut late) = t0.get(b"a");
    assert_eq!(
        value.map(|v| **v),
        Some(1),
        "the old snapshot still reads 1"
    );
    assert!(late.is_closed());
    late.changed().await;

    // The key nobody wrote is untouched, on either snapshot.
    let (_, elsewhere) = t0.get(b"b");
    assert!(!elsewhere.is_closed());
    let (_, fresh) = t1.get(b"a");
    assert!(!fresh.is_closed());
}

/// One transaction over overlapping keys must land where the same operations
/// applied one commit at a time land, and must leave an older snapshot alone.
#[test]
fn a_batched_txn_matches_separate_txns() {
    let mut txn = Tree::new().txn();
    for key in [b"aa".as_slice(), b"ab", b"b"] {
        txn.insert(key, 0);
    }
    let base = txn.commit_and_notify();
    let before = base.clone();

    // Insert, update and delete, over keys that split, merge and split again.
    let ops: [(&[u8], Option<u64>); 8] = [
        (b"aa", Some(1)),
        (b"aac", Some(2)),
        (b"aa", Some(3)),
        (b"ab", None),
        (b"aac", None),
        (b"abc", Some(4)),
        (b"b", None),
        (b"aa", None),
    ];

    let mut txn = base.txn();
    for (key, value) in ops {
        match value {
            Some(v) => drop(txn.insert(key, v)),
            None => drop(txn.delete(key)),
        }
    }
    let batched = txn.commit_and_notify();
    batched.assert_invariants();

    let mut separate = base.clone();
    for (key, value) in ops {
        let mut txn = separate.txn();
        match value {
            Some(v) => drop(txn.insert(key, v)),
            None => drop(txn.delete(key)),
        }
        separate = txn.commit_and_notify();
    }

    assert_eq!(entries(&batched), entries(&separate));
    assert_eq!(batched.len(), separate.len());

    // The snapshot taken before the batch still reads as it did.
    assert_eq!(
        entries(&before),
        vec![(b"aa".to_vec(), 0), (b"ab".to_vec(), 0), (b"b".to_vec(), 0)]
    );
    assert_eq!(before.len(), 3);
    before.assert_invariants();
}

/// `Cell::watch` and the commit that closes the cell must not miss each other.
/// Both go through the cell's lock, so whichever runs second sees the work of
/// the first and a watch handed out around a closing commit is complete once
/// both have returned. The two-atomic version this replaced could let each side
/// read the other's stale value on a weakly ordered machine, leaving a watch on
/// a node already out of the tree that no later commit would ever close.
#[test]
fn a_watch_racing_the_commit_that_closes_it_still_closes() {
    use std::sync::{Arc, Barrier};

    for _ in 0..2000 {
        let mut txn = Tree::new().txn();
        txn.insert(b"a", 1);
        let t0 = Arc::new(txn.commit_and_notify());
        let gate = Arc::new(Barrier::new(2));

        let reader = {
            let (t0, gate) = (t0.clone(), gate.clone());
            std::thread::spawn(move || {
                gate.wait();
                t0.get(b"a").1
            })
        };
        let writer = {
            let (t0, gate) = (t0.clone(), gate.clone());
            std::thread::spawn(move || {
                gate.wait();
                let mut txn = t0.txn();
                txn.insert(b"a", 2);
                drop(txn.commit_and_notify());
            })
        };

        writer.join().unwrap();
        let watch = reader.join().unwrap();
        assert!(
            watch.is_closed(),
            "the commit closed the cell this watch was taken on"
        );
    }
}
