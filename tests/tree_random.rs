//! The tree against a `BTreeMap` over a fixed-seed random operation stream.

use std::collections::BTreeMap;

use sloppy::tree::{Iter, Tree};

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

fn walked(mut it: Iter<'_, u64>) -> Vec<(Vec<u8>, u64)> {
    let mut out = Vec::new();
    while let Some(v) = it.next() {
        out.push((it.key().to_vec(), *v));
    }
    out
}

fn entries(tree: &Tree<u64>) -> Vec<(Vec<u8>, u64)> {
    walked(tree.iter())
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
        let mut touched: Vec<Vec<u8>> = Vec::new();
        for _ in 0..batch {
            let key = rng.key();
            touched.push(key.clone());
            if rng.below(3) == 0 {
                let gone = txn.delete(&key);
                assert_eq!(gone, model.remove(&key), "delete {key:?}");
                hit_deletes += usize::from(gone.is_some());
            } else {
                stamp += 1;
                assert_eq!(
                    txn.insert(&key, stamp),
                    model.insert(key.clone(), stamp),
                    "insert {key:?}"
                );
            }
            // The txn must see its own earlier writes.
            assert_eq!(txn.get(&key).copied(), model.get(&key).copied());
            ops += 1;
        }

        // The same reads on the open txn, which the model already agrees with,
        // before anything is committed.
        assert_eq!(walked(txn.iter()), model_entries(model.iter()));
        for _ in 0..2 {
            let p = rng.key();
            let want = model_entries(model.iter().filter(|(k, _)| k.starts_with(&p)));
            assert_eq!(walked(txn.prefix(&p)), want, "txn prefix {p:?}");
        }
        for _ in 0..2 {
            let key = rng.key();
            let want = model_entries(model.range(key.clone()..));
            assert_eq!(
                walked(txn.lower_bound(&key)),
                want,
                "txn lower_bound {key:?}"
            );
            // A key this batch left alone reads the same on the snapshot the
            // txn was opened on.
            if !touched.contains(&key) {
                assert_eq!(
                    txn.get(&key).copied(),
                    tree.get(&key).copied(),
                    "untouched {key:?}"
                );
            }
        }

        tree = txn.commit();

        tree.assert_invariants();
        assert_eq!(tree.len(), model.len());
        assert_eq!(tree.is_empty(), model.is_empty());
        assert_eq!(entries(&tree), model_entries(model.iter()));

        for _ in 0..4 {
            let key = rng.key();
            assert_eq!(
                tree.get(&key).copied(),
                model.get(&key).copied(),
                "get {key:?}"
            );
        }
        for _ in 0..2 {
            let p = rng.key();
            let want = model_entries(model.iter().filter(|(k, _)| k.starts_with(&p)));
            assert_eq!(walked(tree.prefix(&p)), want, "prefix {p:?}");
        }
        for _ in 0..2 {
            let key = rng.key();
            let want = model_entries(model.range(key.clone()..));
            assert_eq!(walked(tree.lower_bound(&key)), want, "lower_bound {key:?}");
        }
    }
    assert!(!tree.is_empty());
    // The stream has to exercise removal and merging, not just misses.
    assert!(hit_deletes > 1_000, "only {hit_deletes} deletes hit");
}

/// 257 keys under one prefix and back down to two, which drives the leaves
/// through split, borrow and merge, checked against the model after every
/// single write.
#[test]
fn a_node_grows_and_shrinks_through_every_split_and_merge() {
    let mut tree: Tree<u64> = Tree::new();
    let mut model: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    let mut keys = vec![b"k".to_vec()];
    keys.extend((0..=255u8).map(|byte| vec![b'k', byte]));

    for (i, key) in keys.iter().enumerate() {
        let stamp = u64::try_from(i).unwrap();
        let mut txn = tree.txn();
        assert_eq!(txn.insert(key, stamp), None);
        tree = txn.commit();
        model.insert(key.clone(), stamp);
        tree.assert_invariants();
        assert_eq!(tree.len(), model.len());
        assert_eq!(entries(&tree), model_entries(model.iter()));
    }
    assert_eq!(tree.len(), 257, "every key is there");

    // Back down to two, from the middle outwards so the removals do not all
    // fall at one end.
    let mut order: Vec<&Vec<u8>> = keys[1..].iter().collect();
    order.sort_by_key(|key| key[1].wrapping_sub(128));
    for key in order.iter().take(255) {
        let mut txn = tree.txn();
        assert!(txn.delete(key).is_some());
        tree = txn.commit();
        model.remove(*key);
        tree.assert_invariants();
        assert_eq!(tree.len(), model.len());
        assert_eq!(entries(&tree), model_entries(model.iter()));
    }
    assert_eq!(tree.len(), 2, "the prefix and the last key under it");
}

/// One transaction over overlapping keys must land where the same operations
/// applied one commit at a time land, and must leave an older snapshot alone.
#[test]
fn a_batched_txn_matches_separate_txns() {
    let mut txn = Tree::new().txn();
    for key in [b"aa".as_slice(), b"ab", b"b"] {
        txn.insert(key, 0);
    }
    let base = txn.commit();
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
    let batched = txn.commit();
    batched.assert_invariants();

    let mut separate = base.clone();
    for (key, value) in ops {
        let mut txn = separate.txn();
        match value {
            Some(v) => drop(txn.insert(key, v)),
            None => drop(txn.delete(key)),
        }
        separate = txn.commit();
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
