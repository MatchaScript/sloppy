//! Per-object cost of the basic operations, one table, 8-byte big-endian
//! `u64` keys, no secondary index: single and batched commits, random point
//! gets over 1000 objects, and a full walk over 100000.
//!
//! `cargo run --release --example micro_bench`. Prints ns per object and
//! resident bytes per stored object, so runs compare across commits and with
//! other engines measured the same way.

use std::time::{Duration, Instant};

use sloppy::db::{Db, Key, Table};

#[derive(Clone)]
struct Obj {
    id: u64,
}

fn key_of(id: u64) -> Key {
    id.to_be_bytes().into()
}

fn table(db: &Db) -> Table<Obj> {
    db.table("objects", |o| key_of(o.id), &[])
}

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .expect("status")
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|l| l.trim().strip_suffix(" kB"))
        .and_then(|l| l.parse().ok())
        .expect("VmRSS")
}

fn ns(d: Duration, per: u64) -> u64 {
    u64::try_from(d.as_nanos()).expect("fits") / per
}

/// Numerical Recipes LCG, so the shuffle repeats without a rand crate.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) % n
    }
}

fn shuffled(n: u64) -> Vec<u64> {
    let mut ids: Vec<u64> = (0..n).collect();
    let mut rng = Lcg(0x2026_0907);
    for i in (1..ids.len()).rev() {
        let j = usize::try_from(rng.below(u64::try_from(i).expect("fits") + 1)).expect("fits");
        ids.swap(i, j);
    }
    ids
}

const N: u64 = 1_000;
const N_ITER: u64 = 100_000;

#[allow(clippy::too_many_lines)]
fn main() {
    // WriteTxn_1: one replace per commit on a one-object table.
    let db = Db::new();
    let t = table(&db);
    let iters = 1_000_000;
    let clock = Instant::now();
    for _ in 0..iters {
        let mut txn = db.write();
        t.insert(&mut txn, Obj { id: 123 });
        txn.commit();
    }
    println!("WriteTxn_1 {} ns/op", ns(clock.elapsed(), iters));

    // WriteTxn_100 / _1000: batches of inserts of ids 0..batch per commit.
    for batch in [100u64, 1_000] {
        let db = Db::new();
        let t = table(&db);
        let commits = 2_000;
        let clock = Instant::now();
        for _ in 0..commits {
            let mut txn = db.write();
            for id in 0..batch {
                t.insert(&mut txn, Obj { id });
            }
            txn.commit();
        }
        println!(
            "WriteTxn_{batch} {} ns/op",
            ns(clock.elapsed(), commits * batch)
        );
    }

    // RandomInsert: 1000 shuffled ids per commit, same ids every commit.
    let db = Db::new();
    let t = table(&db);
    let ids = shuffled(N);
    let commits = 2_000;
    let clock = Instant::now();
    for _ in 0..commits {
        let mut txn = db.write();
        for &id in &ids {
            t.insert(&mut txn, Obj { id });
        }
        txn.commit();
    }
    println!("RandomInsert {} ns/op", ns(clock.elapsed(), commits * N));

    // RandomLookup: 1000 objects, shuffled point gets on one snapshot per round.
    let db = Db::new();
    let t = table(&db);
    let mut txn = db.write();
    for id in 0..N {
        t.insert(&mut txn, Obj { id });
    }
    txn.commit();
    let keys: Vec<Key> = shuffled(N).into_iter().map(key_of).collect();
    let rounds = 20_000;
    let clock = Instant::now();
    let mut sum = 0u64;
    for _ in 0..rounds {
        let r = db.read();
        for k in &keys {
            sum += t.get(&r, k).expect("present").0.id;
        }
    }
    println!("RandomLookup {} ns/op", ns(clock.elapsed(), rounds * N));
    assert!(sum > 0);

    // FullIteration_All: 100000 objects, walk them all.
    let before = rss_kb();
    let db = Db::new();
    let t = table(&db);
    let mut txn = db.write();
    for id in 0..N_ITER {
        t.insert(&mut txn, Obj { id });
    }
    txn.commit();
    println!("resident {} B/object", (rss_kb() - before) * 1024 / N_ITER);
    let rounds = 200;
    let clock = Instant::now();
    for _ in 0..rounds {
        let r = db.read();
        let mut i = 0u64;
        for (o, _) in t.all(&r) {
            assert_eq!(o.id, i);
            i += 1;
        }
        assert_eq!(i, N_ITER);
    }
    println!(
        "FullIteration_All {} ns/op",
        ns(clock.elapsed(), rounds * N_ITER)
    );

    // Tree_*: the bare tree, without the table layer above it.
    {
        let before = rss_kb();
        let tree = sloppy::tree::Tree::<Obj>::new();
        let mut txn = tree.txn();
        for id in 0..N_ITER {
            txn.insert(&id.to_be_bytes(), Obj { id });
        }
        let _tree = txn.commit_and_notify();
        println!(
            "Tree_resident {} B/object",
            (rss_kb() - before) * 1024 / N_ITER
        );

        let commits = 2_000;
        let batch = 1_000u64;
        let clock = Instant::now();
        let mut tree = sloppy::tree::Tree::<Obj>::new();
        for _ in 0..commits {
            let mut txn = tree.txn();
            for id in 0..batch {
                txn.insert(&id.to_be_bytes(), Obj { id });
            }
            tree = txn.commit_and_notify();
        }
        println!(
            "Tree_WriteTxn_1000 {} ns/op",
            ns(clock.elapsed(), commits * batch)
        );

        let clock = Instant::now();
        let rounds = 100_000;
        for i in 0..rounds {
            let id = i % batch;
            assert_eq!(tree.value(&id.to_be_bytes()).map(|o| o.id), Some(id));
        }
        println!("Tree_RandomLookup {} ns/op", ns(clock.elapsed(), rounds));
    }
}
