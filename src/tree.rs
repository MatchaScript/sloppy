//! Persistent radix tree keyed by byte strings.
//!
//! A write path-copies: every node from the root down to the change is rebuilt,
//! everything else stays shared through `Arc`. The full key of a node is the
//! concatenation of the prefixes on the path from the root.

use std::cmp::Ordering;
use std::sync::Arc;

use crate::watch::{Watch, WatchCell, cell};

// ponytail: one node kind, children in a sorted Vec searched by binary search.
// Split into node4/16/48/256 when lookup or memory shows up in a measurement.
struct Node<V> {
    prefix: Box<[u8]>,
    value: Option<Arc<V>>,
    /// Sorted and unique by the first byte of the child's prefix, which the
    /// `u8` duplicates so the search does not chase the `Arc`.
    children: Vec<(u8, Arc<Node<V>>)>,
    /// Closed by every commit that rebuilds this node, so it covers the whole
    /// subtree.
    watch: WatchCell,
    /// Closed only by a commit that writes or removes this node's own value.
    value_watch: WatchCell,
}

impl<V> Node<V> {
    /// The value is new here, so its cell starts fresh.
    fn new(prefix: &[u8], value: Option<Arc<V>>, children: Vec<(u8, Arc<Node<V>>)>) -> Arc<Self> {
        Self::carrying(prefix, value, cell(), children)
    }

    /// The value comes over unchanged from another node and keeps that node's
    /// cell, so a `Watch` taken there still tracks it.
    fn carrying(
        prefix: &[u8],
        value: Option<Arc<V>>,
        value_watch: WatchCell,
        children: Vec<(u8, Arc<Node<V>>)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            prefix: prefix.into(),
            value,
            children,
            watch: cell(),
            value_watch,
        })
    }

    /// Path copy of `self`: the value and its cell carry over.
    fn with_children(&self, prefix: &[u8], children: Vec<(u8, Arc<Node<V>>)>) -> Arc<Self> {
        Self::carrying(
            prefix,
            self.value.clone(),
            self.value_watch.clone(),
            children,
        )
    }

    fn child(&self, byte: u8) -> Result<usize, usize> {
        self.children.binary_search_by_key(&byte, |(b, _)| *b)
    }
}

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Returns the value at `key` and the deepest node the descent reached, which
/// is where an insert of a missing `key` would attach.
fn descend<'a, V>(root: &'a Arc<Node<V>>, key: &[u8]) -> (Option<&'a Arc<V>>, &'a Node<V>) {
    let mut node: &Node<V> = root;
    let mut rest = key;
    loop {
        if !rest.starts_with(&node.prefix) {
            return (None, node);
        }
        rest = &rest[node.prefix.len()..];
        if rest.is_empty() {
            return (node.value.as_ref(), node);
        }
        match node.child(rest[0]) {
            Ok(i) => node = &node.children[i].1,
            Err(_) => return (None, node),
        }
    }
}

/// An immutable snapshot. `clone` is one `Arc` bump.
pub struct Tree<V> {
    root: Arc<Node<V>>,
    len: usize,
}

impl<V> Clone for Tree<V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            len: self.len,
        }
    }
}

impl<V> Default for Tree<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> Tree<V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: Node::new(&[], None, Vec::new()),
            len: 0,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Fires on any change to the tree.
    #[must_use]
    pub fn root_watch(&self) -> Watch {
        Watch::new(&self.root.watch)
    }

    /// The value at `key`, plus a watch on that value alone - or, when `key`
    /// is absent, on the whole subtree of the deepest node the descent reached,
    /// which is where the key would appear.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> (Option<&Arc<V>>, Watch) {
        let (value, node) = descend(&self.root, key);
        let watch = if value.is_some() {
            &node.value_watch
        } else {
            &node.watch
        };
        (value, Watch::new(watch))
    }

    /// Every entry whose key starts with `p`, plus the watch of the node the
    /// descent along `p` reaches.
    #[must_use]
    pub fn prefix(&self, p: &[u8]) -> (Iter<'_, V>, Watch) {
        let mut node: &Node<V> = &self.root;
        let mut key = Vec::new();
        let mut rest = p;
        loop {
            if node.prefix.len() >= rest.len() {
                let stack = if node.prefix.starts_with(rest) {
                    vec![Frame { key, node }]
                } else {
                    Vec::new()
                };
                return (Iter { stack }, Watch::new(&node.watch));
            }
            if !rest.starts_with(&node.prefix) {
                break;
            }
            key.extend_from_slice(&node.prefix);
            rest = &rest[node.prefix.len()..];
            match node.child(rest[0]) {
                Ok(i) => node = &node.children[i].1,
                Err(_) => break,
            }
        }
        (Iter { stack: Vec::new() }, Watch::new(&node.watch))
    }

    /// Every entry with a key `>= key`, in order. Descends along `key` and
    /// seeds the stack with the subtrees that are entirely `>= key`.
    #[must_use]
    pub fn lower_bound(&self, key: &[u8]) -> Iter<'_, V> {
        self.lower_bound_watch(key).0
    }

    /// The same, plus the root watch: an entry can enter the range anywhere.
    #[must_use]
    pub fn lower_bound_watch(&self, key: &[u8]) -> (Iter<'_, V>, Watch) {
        let mut stack = Vec::new();
        let mut node: &Node<V> = &self.root;
        let mut acc: Vec<u8> = Vec::new();
        let mut rest = key;
        loop {
            let n = node.prefix.len().min(rest.len());
            match node.prefix[..n].cmp(&rest[..n]) {
                // The whole subtree sorts below `key`.
                Ordering::Less => break,
                // The whole subtree sorts above `key`.
                Ordering::Greater => {
                    stack.push(Frame { key: acc, node });
                    break;
                }
                Ordering::Equal => {}
            }
            if node.prefix.len() >= rest.len() {
                // `key` runs out inside this node, so its whole subtree is >= key.
                stack.push(Frame { key: acc, node });
                break;
            }
            // This node's own key is a strict prefix of `key`, so it sorts below
            // `key` and is skipped, as are the children before `rest[0]`.
            acc.extend_from_slice(&node.prefix);
            rest = &rest[node.prefix.len()..];
            let (down, greater) = match node.child(rest[0]) {
                Ok(i) => (Some(&node.children[i].1), i + 1),
                Err(i) => (None, i),
            };
            for (_, c) in node.children[greater..].iter().rev() {
                stack.push(Frame {
                    key: acc.clone(),
                    node: c,
                });
            }
            match down {
                Some(c) => node = c,
                None => break,
            }
        }
        (Iter { stack }, self.root_watch())
    }

    /// Every entry, in ascending key order.
    #[allow(clippy::iter_without_into_iter)] // `IntoIterator` for `&Tree` has no user yet.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, V> {
        Iter {
            stack: vec![Frame {
                key: Vec::new(),
                node: &self.root,
            }],
        }
    }

    #[must_use]
    pub fn txn(&self) -> Txn<V> {
        Txn {
            root: self.root.clone(),
            len: self.len,
            closed: Vec::new(),
        }
    }

    /// Checks the shape invariants. Test helper.
    ///
    /// # Panics
    ///
    /// If a non-root node is uncompressed, or a children list is unsorted or
    /// disagrees with the child's own prefix.
    #[doc(hidden)]
    pub fn assert_invariants(&self) {
        fn walk<V>(node: &Node<V>, root: bool) {
            assert!(
                root || !node.prefix.is_empty(),
                "empty prefix below the root"
            );
            assert!(
                root || node.value.is_some() || node.children.len() >= 2,
                "node with no value and fewer than two children"
            );
            for w in node.children.windows(2) {
                assert!(w[0].0 < w[1].0, "children not sorted and unique");
            }
            for (b, c) in &node.children {
                assert_eq!(
                    *b, c.prefix[0],
                    "child byte disagrees with the child prefix"
                );
                walk(c, false);
            }
        }
        walk(&self.root, true);
    }
}

struct Frame<'a, V> {
    /// Full key of the node's parent path, without the node's own prefix.
    key: Vec<u8>,
    node: &'a Node<V>,
}

/// Yields `(key, value)` in ascending byte-lexicographic order.
pub struct Iter<'a, V> {
    stack: Vec<Frame<'a, V>>,
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = (Vec<u8>, &'a Arc<V>);

    // ponytail: one key `Vec` built per visited node. Hand out a cursor instead
    // if the allocations ever matter.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(frame) = self.stack.pop() {
            let mut key = frame.key;
            key.extend_from_slice(&frame.node.prefix);
            for (_, c) in frame.node.children.iter().rev() {
                self.stack.push(Frame {
                    key: key.clone(),
                    node: c,
                });
            }
            if let Some(value) = frame.node.value.as_ref() {
                return Some((key, value));
            }
        }
        None
    }
}

/// A batch of writes over one snapshot. Dropping it aborts.
pub struct Txn<V> {
    root: Arc<Node<V>>,
    len: usize,
    closed: Vec<WatchCell>,
}

impl<V> Txn<V> {
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&Arc<V>> {
        descend(&self.root, key).0
    }

    /// Returns the replaced value.
    pub fn insert(&mut self, key: &[u8], value: V) -> Option<Arc<V>> {
        let (root, old) = insert(&self.root, key, Arc::new(value), &mut self.closed);
        self.root = root;
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub fn delete(&mut self, key: &[u8]) -> Option<Arc<V>> {
        let (root, old) = delete(&self.root, key, true, &mut self.closed);
        match (root, old) {
            (Some(root), Some(old)) => {
                self.root = root;
                self.len -= 1;
                Some(old)
            }
            _ => None,
        }
    }

    /// The new tree and the cells the caller must close.
    #[must_use]
    pub fn commit(self) -> (Tree<V>, Vec<WatchCell>) {
        (
            Tree {
                root: self.root,
                len: self.len,
            },
            self.closed,
        )
    }

    #[must_use]
    pub fn commit_and_notify(self) -> Tree<V> {
        let (tree, closed) = self.commit();
        for c in closed {
            c.send_replace(true);
        }
        tree
    }
}

/// Path-copies `node` with `key` (relative to `node`) set to `value`.
fn insert<V>(
    node: &Arc<Node<V>>,
    key: &[u8],
    value: Arc<V>,
    closed: &mut Vec<WatchCell>,
) -> (Arc<Node<V>>, Option<Arc<V>>) {
    let common = lcp(&node.prefix, key);
    closed.push(node.watch.clone());

    if common < node.prefix.len() {
        // The key diverges inside the prefix: split, the tail keeps the subtree.
        let tail = node.with_children(&node.prefix[common..], node.children.clone());
        if common == key.len() {
            let children = vec![(tail.prefix[0], tail)];
            return (Node::new(&key[..common], Some(value), children), None);
        }
        let leaf = Node::new(&key[common..], Some(value), Vec::new());
        let mut children = vec![(tail.prefix[0], tail), (leaf.prefix[0], leaf)];
        children.sort_unstable_by_key(|(b, _)| *b);
        return (Node::new(&key[..common], None, children), None);
    }

    if common == key.len() {
        let old = node.value.clone();
        // On a valueless node this closes a cell nobody holds: `get` hands out
        // the value cell only for a key that exists, and the watch on the
        // missing key is the subtree cell closed above.
        closed.push(node.value_watch.clone());
        let new = Node::new(&node.prefix, Some(value), node.children.clone());
        return (new, old);
    }

    let rest = &key[common..];
    let mut children = node.children.clone();
    let old = match node.child(rest[0]) {
        Ok(i) => {
            let (child, old) = insert(&children[i].1, rest, value, closed);
            children[i].1 = child;
            old
        }
        Err(i) => {
            children.insert(i, (rest[0], Node::new(rest, Some(value), Vec::new())));
            None
        }
    };
    (node.with_children(&node.prefix, children), old)
}

/// Path-copies `node` with `key` (relative to `node`) removed. A `None` node
/// with a `Some` value means the subtree is gone; a `None` value means `key`
/// was absent and nothing changed.
fn delete<V>(
    node: &Arc<Node<V>>,
    key: &[u8],
    is_root: bool,
    closed: &mut Vec<WatchCell>,
) -> (Option<Arc<Node<V>>>, Option<Arc<V>>) {
    if !key.starts_with(&node.prefix) {
        return (None, None);
    }
    let rest = &key[node.prefix.len()..];

    if rest.is_empty() {
        let Some(old) = node.value.clone() else {
            return (None, None);
        };
        closed.push(node.watch.clone());
        closed.push(node.value_watch.clone());
        let children = node.children.clone();
        return (
            shrink(&node.prefix, None, children, is_root, closed),
            Some(old),
        );
    }

    let Ok(i) = node.child(rest[0]) else {
        return (None, None);
    };
    let (child, old) = delete(&node.children[i].1, rest, false, closed);
    let Some(old) = old else {
        return (None, None);
    };
    closed.push(node.watch.clone());
    let mut children = node.children.clone();
    match child {
        Some(c) => children[i] = (c.prefix[0], c),
        None => drop(children.remove(i)),
    }
    let value = node.value.clone().map(|v| (v, node.value_watch.clone()));
    (
        shrink(&node.prefix, value, children, is_root, closed),
        Some(old),
    )
}

/// Rebuilds a node that just lost its value or a child: a node with no value
/// and no children is dropped, one with no value and a single child is merged
/// into that child. The root always stays, with an empty prefix.
fn shrink<V>(
    prefix: &[u8],
    value: Option<(Arc<V>, WatchCell)>,
    children: Vec<(u8, Arc<Node<V>>)>,
    is_root: bool,
    closed: &mut Vec<WatchCell>,
) -> Option<Arc<Node<V>>> {
    if value.is_none() && !is_root {
        if children.is_empty() {
            return None;
        }
        if let [(_, only)] = children.as_slice() {
            closed.push(only.watch.clone());
            let mut merged = prefix.to_vec();
            merged.extend_from_slice(&only.prefix);
            return Some(only.with_children(&merged, only.children.clone()));
        }
    }
    let (value, watch) = value.unzip();
    Some(Node::carrying(
        prefix,
        value,
        watch.unwrap_or_else(cell),
        children,
    ))
}
