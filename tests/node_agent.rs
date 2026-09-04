//! Row 6: the node agent's R1-R5, with one writer and three readers running
//! side by side.
//!
//! R1 readers never block on an open write transaction.
//! R2/R3 the differ rebuilds the table from `changes` alone, deletions
//! included, and ends up with what `all()` holds.
//! R4 the differ sleeps on a `Watch` between rounds instead of spinning.
//! R5 the tenant index lists what a scan of `all()` would.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use sloppy::db::{Change, Db, Index, Key};

const KEYS: u64 = 200;
const COMMITS: usize = 2_000;
const TENANTS: [&str; 4] = ["red", "green", "blue", "violet"];
/// The writer's last commit, which also wakes a differ parked on the watch.
const SENTINEL: &[u8] = b"k/done";
/// Value of `finish` while the writer still has commits to make.
const RUNNING: u64 = 0;

#[derive(Debug)]
struct Rec {
    key: Key,
    tenant: &'static str,
    n: u64,
}

fn rec_key(r: &Rec) -> Key {
    r.key.clone()
}

fn rec_tenant(r: &Rec) -> Vec<Key> {
    vec![r.tenant.as_bytes().into()]
}

const BY_TENANT: Index<Rec> = Index {
    name: "tenant",
    keys: rec_tenant,
};

/// Numerical Recipes LCG. Fixed seed, so a failure repeats.
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

fn key_of(i: u64) -> Key {
    format!("k/{i:04}").into_bytes().into()
}

fn apply(map: &mut BTreeMap<Vec<u8>, u64>, changes: impl Iterator<Item = Change<Rec>>) {
    for c in changes {
        if c.deleted {
            map.remove(&*c.key);
        } else {
            map.insert(c.key.into_vec(), c.value.n);
        }
    }
}

/// What one reader thread measured.
struct Reads {
    name: &'static str,
    loops: usize,
}

#[test]
fn a_read_does_not_wait_for_an_open_write_transaction() {
    let db = Arc::new(Db::new());
    let table = db.table("recs", rec_key as fn(&Rec) -> Key, &[]);
    let reading = db.clone();
    let write = db.write();

    let completed = thread::scope(|scope| {
        let (ready, started) = std::sync::mpsc::channel();
        let (done, finished) = std::sync::mpsc::channel();
        scope.spawn(move || {
            ready.send(()).unwrap();
            done.send(table.all(&reading.read()).count()).unwrap();
        });
        started.recv().unwrap();
        let completed = finished.recv_timeout(Duration::from_secs(1)).is_ok();
        drop(write);
        completed
    });
    assert!(completed, "read waited for the open write transaction");
}

#[test]
// One acceptance run: splitting the four threads apart would only move the
// shared setup into arguments.
#[allow(clippy::too_many_lines)]
fn readers_keep_up_with_a_busy_writer() {
    let db = Arc::new(Db::new());
    let table = db.table("recs", rec_key as fn(&Rec) -> Key, &[BY_TENANT]);
    // The revision of the writer's last commit, announced just before it is
    // made: a reader that reads `RUNNING`, or a lower revision, still has that
    // commit coming to close the watch it holds.
    let finish = Arc::new(AtomicU64::new(RUNNING));

    let mut w = db.write();
    let mut differ_iter = table.changes(&mut w);
    w.commit();

    let writer = {
        let (db, finish) = (db.clone(), finish.clone());
        thread::spawn(move || {
            let mut rng = Lcg(0x2026_0904);
            let start = Instant::now();
            for round in 0..COMMITS {
                let mut w = db.write();
                for _ in 0..=rng.below(4) {
                    let key = key_of(rng.below(KEYS));
                    if rng.below(4) == 0 {
                        table.delete(&mut w, &key);
                    } else {
                        let tenant = TENANTS[usize::try_from(rng.below(4)).unwrap()];
                        let n = round as u64;
                        table.insert(&mut w, Rec { key, tenant, n });
                    }
                }
                w.commit();
            }
            let last = db.read().revision() + 1;
            finish.store(last, Ordering::SeqCst);
            let mut w = db.write();
            table.insert(
                &mut w,
                Rec {
                    key: SENTINEL.into(),
                    tenant: TENANTS[0],
                    n: 0,
                },
            );
            assert_eq!(w.commit(), last, "the sentinel commit is the announced one");
            start.elapsed()
        })
    };

    let differ = {
        let (db, finish) = (db.clone(), finish.clone());
        thread::spawn(move || {
            // ponytail: one current-thread runtime just to park on the watch.
            // The reader is otherwise plain blocking code.
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime");
            let mut map = BTreeMap::new();
            let mut reads = Reads {
                name: "differ",
                loops: 0,
            };
            loop {
                let snapshot = db.read();
                let (changes, mut watch) = differ_iter.next(&snapshot).expect("nothing compacted");
                apply(&mut map, changes);
                reads.loops += 1;
                let last = finish.load(Ordering::SeqCst);
                if last != RUNNING && snapshot.revision() >= last {
                    break;
                }
                rt.block_on(watch.changed());
            }
            (map, reads)
        })
    };

    let scanner = {
        let (db, finish) = (db.clone(), finish.clone());
        thread::spawn(move || {
            let mut reads = Reads {
                name: "scanner",
                loops: 0,
            };
            while finish.load(Ordering::SeqCst) == RUNNING {
                let snapshot = db.read();
                let seen = table.all(&snapshot).count();
                reads.loops += 1;
                assert!(seen <= usize::try_from(KEYS).unwrap() + 1);
            }
            reads
        })
    };

    let by_tenant = {
        let (db, finish) = (db.clone(), finish.clone());
        thread::spawn(move || {
            let mut reads = Reads {
                name: "index reader",
                loops: 0,
            };
            while finish.load(Ordering::SeqCst) == RUNNING {
                let snapshot = db.read();
                let listed: Vec<_> = table
                    .by_index(&snapshot, "tenant", TENANTS[0].as_bytes())
                    .map(|(k, v, _)| (k, v.n))
                    .collect();
                reads.loops += 1;
                assert!(listed.len() <= usize::try_from(KEYS).unwrap() + 1);
            }
            reads
        })
    };

    let writing = writer.join().expect("writer");
    let (differed, differ_reads) = differ.join().expect("differ");
    let measured = [
        differ_reads,
        scanner.join().unwrap(),
        by_tenant.join().unwrap(),
    ];

    println!("writer: {writing:?} for {COMMITS} commits");
    for r in &measured {
        println!("{}: {} reads", r.name, r.loops);
    }
    // R4: a round happens per commit at most, so the differ waited, not spun.
    assert!(measured[0].loops <= COMMITS + 2, "the differ spun");

    // R2 and R3: the changes alone rebuild the table, deletions included.
    let snapshot = db.read();
    let live: BTreeMap<Vec<u8>, u64> = table.all(&snapshot).map(|(k, v, _)| (k, v.n)).collect();
    assert_eq!(differed, live);
    assert!(live.contains_key(SENTINEL));

    // R5: each index listing is what a scan would have found.
    for tenant in TENANTS {
        let want: BTreeSet<Vec<u8>> = table
            .all(&snapshot)
            .filter(|(_, v, _)| v.tenant == tenant)
            .map(|(k, _, _)| k)
            .collect();
        let listed: BTreeSet<Vec<u8>> = table
            .by_index(&snapshot, "tenant", tenant.as_bytes())
            .map(|(k, _, _)| k)
            .collect();
        assert_eq!(listed, want, "index listing for {tenant}");
    }
}
