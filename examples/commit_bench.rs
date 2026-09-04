//! Commit cost for row 7 of the keyspace plan.
//!
//! `cargo run --release --example commit_bench`. No arguments: the record
//! counts, the seed and the iteration counts are fixed so two runs compare.

use std::time::{Duration, Instant};

use sloppy::db::{Db, Index, Key, Table};

/// Numerical Recipes LCG, so the key stream repeats without a rand crate.
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

/// 16-24 bytes with a shared per-tenant prefix.
struct Route {
    key: Key,
    tenant: Key,
    hop: u32,
}

const TENANTS: u64 = 64;

fn key_of(id: u64) -> Key {
    format!("tenant/{}/route/{id}", id % TENANTS)
        .into_bytes()
        .into()
}

fn route(id: u64, hop: u32) -> Route {
    Route {
        key: key_of(id),
        tenant: (id % TENANTS).to_string().into_bytes().into(),
        hop,
    }
}

fn ns(elapsed: Duration, ops: u128) -> u128 {
    elapsed.as_nanos() / ops
}

/// Resident set in KiB, from the second field of `/proc/self/statm`.
fn rss_kb() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    pages * 4
}

fn main() {
    for n in [1_000u64, 50_000] {
        bench(n);
    }
}

fn bench(n: u64) {
    let mut rng = Lcg(0x2026_0904 ^ n);
    let before = rss_kb();
    let db = Db::new();
    let table: Table<Route> = db.table(
        "routes",
        |r| r.key.clone(),
        &[Index {
            name: "tenant",
            keys: |r| vec![r.tenant.clone()],
        }],
    );

    for start in (0..n).step_by(100) {
        let mut txn = db.write();
        for id in start..(start + 100).min(n) {
            table.insert(&mut txn, route(id, 1));
        }
        txn.commit();
    }
    println!("N={n} rss_kb {}", rss_kb().saturating_sub(before));

    // (a) One update of an existing record per commit: the plan's single
    // writer critical section.
    let batch: Vec<Route> = (0..10_000).map(|_| route(rng.below(n), 2)).collect();
    let clock = Instant::now();
    for value in batch {
        let mut txn = db.write();
        table.insert(&mut txn, value);
        txn.commit();
    }
    println!("N={n} update_commit {}", ns(clock.elapsed(), 10_000));

    // (b) A fresh record created and removed in one commit: graveyard plus the
    // collection the next commit runs.
    let batch: Vec<Route> = (0..10_000).map(|i| route(n + i, 3)).collect();
    let keys: Vec<Key> = batch.iter().map(|r| r.key.clone()).collect();
    let clock = Instant::now();
    for (value, key) in batch.into_iter().zip(keys) {
        let mut txn = db.write();
        table.insert(&mut txn, value);
        table.delete(&mut txn, &key);
        txn.commit();
    }
    println!("N={n} insert_delete_commit {}", ns(clock.elapsed(), 10_000));

    // (c) Point gets on one snapshot.
    let probes: Vec<Key> = (0..100_000).map(|_| key_of(rng.below(n))).collect();
    let rtxn = db.read();
    let clock = Instant::now();
    let mut hops = 0u64;
    for key in &probes {
        let (route, _) = table.get(&rtxn, key).expect("every probed key exists");
        hops += u64::from(route.hop);
    }
    println!("N={n} point_get {}", ns(clock.elapsed(), 100_000));
    assert!(hops > 0);

    // (d) One scan of 100 entries from a random position.
    let starts: Vec<Key> = (0..1_000).map(|_| key_of(rng.below(n))).collect();
    let clock = Instant::now();
    let mut scanned = 0usize;
    for key in &starts {
        scanned += table.lower_bound(&rtxn, key).take(100).count();
    }
    println!("N={n} lower_bound_100 {}", ns(clock.elapsed(), 1_000));
    assert!(scanned > 0);
    drop(rtxn);

    // (e) Draining the change iterator over 100 commits.
    let mut txn = db.write();
    let mut changes = table.changes(&mut txn);
    txn.commit();
    let mut spent = Duration::ZERO;
    for _ in 0..10 {
        for _ in 0..100 {
            let mut txn = db.write();
            table.insert(&mut txn, route(rng.below(n), 4));
            txn.commit();
        }
        let rtxn = db.read();
        let clock = Instant::now();
        let (drain, _watch) = changes.next(&rtxn).expect("nothing was compacted");
        let seen = drain.count();
        spent += clock.elapsed();
        assert!(seen > 0);
    }
    println!("N={n} changes_drain {}", ns(spent, 10));
}
