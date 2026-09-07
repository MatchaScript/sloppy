//! The tree against a `BTreeMap` over a fixed-seed random operation stream,
//! and against readers running beside the writer.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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

fn walked(it: Iter<u64>) -> Vec<(Vec<u8>, u64)> {
    it.map(|(key, value)| (key.into_vec(), value)).collect()
}

fn entries(tree: &Tree<u64>) -> Vec<(Vec<u8>, u64)> {
    walked(tree.range_from(&[]))
}

fn model_entries<'a>(it: impl Iterator<Item = (&'a Vec<u8>, &'a u64)>) -> Vec<(Vec<u8>, u64)> {
    it.map(|(k, v)| (k.clone(), *v)).collect()
}

#[test]
fn matches_btreemap() {
    let mut rng = Lcg(0x2026_0904);
    let tree: Tree<u64> = Tree::new();
    let mut model: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    let mut ops = 0;
    let mut stamp = 0;
    let mut hit_deletes = 0;

    while ops < 20_000 {
        let batch = 1 + rng.below(16);
        for _ in 0..batch {
            let key = rng.key();
            match rng.below(6) {
                0 | 1 => {
                    let gone = tree.remove(&key);
                    assert_eq!(gone, model.remove(&key), "remove {key:?}");
                    hit_deletes += usize::from(gone.is_some());
                }
                2 => {
                    tree.update(&key, |value| *value += 1_000_000);
                    if let Some(value) = model.get_mut(&key) {
                        *value += 1_000_000;
                    }
                }
                _ => {
                    stamp += 1;
                    assert_eq!(
                        tree.insert(&key, stamp),
                        model.insert(key.clone(), stamp),
                        "insert {key:?}"
                    );
                }
            }
            assert_eq!(
                tree.get(&key),
                model.get(&key).copied(),
                "read back {key:?}"
            );
            ops += 1;
        }

        tree.assert_invariants();
        assert_eq!(entries(&tree), model_entries(model.iter()));

        for _ in 0..4 {
            let key = rng.key();
            assert_eq!(tree.get(&key), model.get(&key).copied(), "get {key:?}");
        }
        for _ in 0..2 {
            let key = rng.key();
            let want = model_entries(model.range(key.clone()..));
            assert_eq!(walked(tree.range_from(&key)), want, "range_from {key:?}");
        }
    }
    assert!(!entries(&tree).is_empty());
    // The stream has to exercise removal and merging, not just misses.
    assert!(hit_deletes > 1_000, "only {hit_deletes} deletes hit");
}

/// 257 keys under one prefix and back down to two, which drives the leaves
/// through split, borrow and merge, checked against the model after every
/// single write.
#[test]
fn a_node_grows_and_shrinks_through_every_split_and_merge() {
    let tree: Tree<u64> = Tree::new();
    let mut model: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

    let mut keys = vec![b"k".to_vec()];
    keys.extend((0..=255u8).map(|byte| vec![b'k', byte]));

    for (i, key) in keys.iter().enumerate() {
        let stamp = u64::try_from(i).unwrap();
        assert_eq!(tree.insert(key, stamp), None);
        model.insert(key.clone(), stamp);
        tree.assert_invariants();
        assert_eq!(entries(&tree), model_entries(model.iter()));
    }
    assert_eq!(entries(&tree).len(), 257, "every key is there");

    // Back down to two, from the middle outwards so the removals do not all
    // fall at one end.
    let mut order: Vec<&Vec<u8>> = keys[1..].iter().collect();
    order.sort_by_key(|key| key[1].wrapping_sub(128));
    for key in order.iter().take(255) {
        assert!(tree.remove(key).is_some());
        model.remove(*key);
        tree.assert_invariants();
        assert_eq!(entries(&tree), model_entries(model.iter()));
    }
    assert_eq!(
        entries(&tree).len(),
        2,
        "the prefix and the last key under it"
    );
}

#[test]
fn a_walk_keeps_keys_that_extend_the_last_one_with_zero_bytes() {
    // The walk seeks the next leaf at `last ++ 0x00`, so a key that is exactly
    // that, or extends it, must still come out. Enough keys to cross leaves.
    let tree: Tree<u64> = Tree::new();
    let mut model = BTreeMap::new();
    for i in 0..=255u8 {
        for key in [
            vec![i],
            vec![i, 0],
            vec![i, 0, 0],
            vec![i, 0, 1],
            vec![i, 1],
        ] {
            tree.insert(&key, u64::from(i));
            model.insert(key, u64::from(i));
        }
    }
    assert_eq!(entries(&tree), model_entries(model.iter()));
}

/// The value the writer keeps at a key. A reader that sees anything else at
/// that key read something no write ever put there.
fn stamped(key: &[u8]) -> u64 {
    key.iter()
        .fold(1, |acc: u64, byte| acc.wrapping_mul(131) + u64::from(*byte))
}

/// Anchors sit under a byte the churn never writes, so they interleave with it
/// without ever colliding.
const ANCHOR: &[u8] = b"abc\0";
const ANCHORS: u8 = 255;

/// A key the writer puts down once and never touches again. Every walk has to
/// hand back all of them and every lookup has to find them, whatever the churn
/// does to the leaves they sit in.
fn anchor(at: u8) -> Vec<u8> {
    let mut key = ANCHOR.to_vec();
    key.push(at);
    key
}

/// Three readers walk and look up keys while the writer splits and merges the
/// tree under them. A reader may see any key the writer has written and miss
/// any it has removed, but every pair it hands back must be one the writer
/// wrote, and a walk must be sorted and free of repeats.
#[test]
fn readers_walk_a_tree_the_writer_is_reshaping() {
    let tree: Tree<u64> = Tree::new();
    for at in 0..ANCHORS {
        tree.insert(&anchor(at), stamped(&anchor(at)));
    }
    let done = AtomicBool::new(false);
    let seen = AtomicUsize::new(0);
    let walks = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for reader in 0..3 {
            let (tree, done, seen, walks) = (&tree, &done, &seen, &walks);
            scope.spawn(move || {
                let mut rng = Lcg(0x1000 + reader);
                while !done.load(Ordering::Relaxed) {
                    let mut last: Option<Vec<u8>> = None;
                    let mut held = 0;
                    for (key, value) in tree.range_from(&[]) {
                        let key = key.into_vec();
                        assert_eq!(value, stamped(&key), "a value no write left at {key:?}");
                        assert!(
                            last.as_ref().is_none_or(|last| *last < key),
                            "walk out of order at {key:?}"
                        );
                        held += u8::from(key.starts_with(ANCHOR));
                        last = Some(key);
                        seen.fetch_add(1, Ordering::Relaxed);
                    }
                    assert_eq!(held, ANCHORS, "a walk lost a key no write removed");
                    walks.fetch_add(1, Ordering::Relaxed);
                    for _ in 0..4 {
                        let key = anchor(u8::try_from(rng.below(u64::from(ANCHORS))).unwrap());
                        assert_eq!(
                            tree.get(&key),
                            Some(stamped(&key)),
                            "a lookup lost a key no write removed"
                        );
                    }
                }
            });
        }

        // The writer keeps going until the readers have walked the tree often
        // enough for the walks to have run beside the reshaping, not after it.
        let mut rng = Lcg(0x2026_0907);
        let mut model: BTreeMap<Vec<u8>, u64> = (0..ANCHORS)
            .map(|at| (anchor(at), stamped(&anchor(at))))
            .collect();
        let mut ops = 0;
        while ops < 20_000 || walks.load(Ordering::Relaxed) < 500 {
            ops += 1;
            let key = rng.key();
            if rng.below(3) == 0 {
                assert_eq!(tree.remove(&key), model.remove(&key), "remove {key:?}");
            } else {
                let value = stamped(&key);
                assert_eq!(
                    tree.insert(&key, value),
                    model.insert(key.clone(), value),
                    "insert {key:?}"
                );
            }
        }
        done.store(true, Ordering::Relaxed);
        assert!(
            model.len() > 1_000,
            "the tree stayed too small to have depth"
        );
        assert_eq!(entries(&tree), model_entries(model.iter()));
    });

    tree.assert_invariants();
    assert!(
        seen.load(Ordering::Relaxed) > 1_000,
        "the readers walked past almost nothing"
    );
}
