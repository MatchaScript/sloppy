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
            let got: Vec<_> = tree.prefix(&p).0.map(|(k, v)| (k, **v)).collect();
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
    let (empty, wp) = t3.prefix(b"b/");
    assert_eq!(empty.count(), 0);
    let mut txn = t3.txn();
    txn.insert(b"b/x", 4);
    let t4 = txn.commit_and_notify();
    assert!(wp.is_closed());

    let (found, wp) = t4.prefix(b"b/");
    assert_eq!(found.count(), 1);
    let mut txn = t4.txn();
    txn.insert(b"zz", 5);
    let t5 = txn.commit_and_notify();
    assert!(!wp.is_closed());
    assert_eq!(t5.len(), 3);
}
