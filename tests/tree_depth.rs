//! A chain of keys that are each a prefix of the next is one node per key, so
//! its length is the depth of every write to its end. The keys' owner picks it,
//! so a write must not spend stack on it. Runs on a thread with the stack of a
//! tokio worker.

use sloppy::tree::Tree;

#[test]
fn a_deep_chain_costs_no_stack() {
    const N: usize = 32_768;
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| {
            let key = vec![b'a'; N + 1];
            let mut tree = Tree::<usize>::new();
            let mut txn = tree.txn();
            // Longest first: each insert splits the root, so the chain builds
            // in linear time. The last one descends the whole chain.
            for n in (1..=N).rev() {
                txn.insert(&key[..n], n);
            }
            txn.insert(&key, N + 1);
            tree = txn.commit_and_notify();
            assert_eq!(tree.len(), N + 1);
            tree.assert_invariants();

            let mut txn = tree.txn();
            assert_eq!(*txn.delete(&key).unwrap(), N + 1);
            assert_eq!(*txn.delete(&key[..N]).unwrap(), N);
            tree = txn.commit_and_notify();
            assert_eq!(tree.len(), N - 1);
            tree.assert_invariants();
            assert!(tree.get(&key[..N]).0.is_none());
            assert_eq!(**tree.get(&key[..N - 1]).0.unwrap(), N - 1);
        })
        .unwrap()
        .join()
        .unwrap();
}
