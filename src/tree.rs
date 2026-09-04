//! Persistent radix tree keyed by byte strings.
//!
//! A write path-copies: every node from the root down to the change is rebuilt,
//! everything else stays shared through `Arc`. The full key of a node is the
//! concatenation of the prefixes on the path from the root.

use std::array;
use std::cmp::Ordering;
use std::iter;
use std::sync::Arc;

use crate::watch::{Watch, WatchCell, cell};

// ponytail: four node kinds, and nothing below them - no SIMD key compare, no
// separate leaf node. The kind follows from the child count, so a path copy
// picks it and growth and shrink need no code of their own.
struct Node<V> {
    prefix: Box<[u8]>,
    value: Option<Arc<V>>,
    children: Children<V>,
    /// Closed by every commit that rebuilds this node, so it covers the whole
    /// subtree.
    watch: WatchCell,
    /// Closed only by a commit that writes or removes this node's own value.
    value_watch: WatchCell,
}

impl<V> Node<V> {
    /// The value is new here, so its cell starts fresh.
    fn new(prefix: &[u8], value: Option<Arc<V>>, children: Children<V>) -> Arc<Self> {
        Self::carrying(prefix, value, cell(), children)
    }

    /// The value comes over unchanged from another node and keeps that node's
    /// cell, so a `Watch` taken there still tracks it.
    fn carrying(
        prefix: &[u8],
        value: Option<Arc<V>>,
        value_watch: WatchCell,
        children: Children<V>,
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
    fn with_children(&self, prefix: &[u8], children: Children<V>) -> Arc<Self> {
        Self::carrying(
            prefix,
            self.value.clone(),
            self.value_watch.clone(),
            children,
        )
    }

    /// The byte a parent files this node under.
    fn key(&self) -> u8 {
        self.prefix[0]
    }
}

// -------------------------------------------------------------- node kinds

type Slot<V> = Option<Arc<Node<V>>>;

/// The children of one node, in the four sizes of `StateDB`'s `part`
/// (`part/node.go`). Every kind holds its occupied slots in ascending key
/// order, so an in-order walk is a slice walk whatever the kind is.
///
/// The two large kinds are boxed, so a node with few children stays small.
enum Children<V> {
    N4(Sorted<V, 4>),
    N16(Box<Sorted<V, 16>>),
    N48(Box<Node48<V>>),
    N256(Box<Node256<V>>),
}

/// Up to `K` children, with the keys beside the slots so a lookup does not
/// chase the `Arc`s.
struct Sorted<V, const K: usize> {
    keys: [u8; K],
    slots: [Slot<V>; K],
    len: u8,
}

/// Sorted slots plus a key index: `0` is absent, anything else is the slot
/// number plus one.
struct Node48<V> {
    index: [u8; 256],
    slots: [Slot<V>; 48],
    len: u8,
}

/// One slot per key byte, so the array is the index.
struct Node256<V> {
    slots: [Slot<V>; 256],
    len: u16,
}

fn key_of<V>(slot: Option<&Arc<Node<V>>>) -> u8 {
    slot.expect("occupied slot").key()
}

impl<V, const K: usize> Sorted<V, K> {
    fn fill(children: impl Iterator<Item = Arc<Node<V>>>) -> Self {
        let mut this = Self {
            keys: [0; K],
            slots: array::from_fn(|_| None),
            len: 0,
        };
        for child in children {
            let at = usize::from(this.len);
            this.keys[at] = child.key();
            this.slots[at] = Some(child);
            this.len += 1;
        }
        this
    }

    fn occupied(&self) -> &[Slot<V>] {
        &self.slots[..usize::from(self.len)]
    }

    fn get(&self, byte: u8) -> Option<&Arc<Node<V>>> {
        let at = self.keys[..usize::from(self.len)]
            .iter()
            .position(|k| *k == byte)?;
        self.slots[at].as_ref()
    }
}

impl<V> Clone for Children<V> {
    fn clone(&self) -> Self {
        Self::build(self.len(), self.iter().cloned())
    }
}

impl<V> Children<V> {
    /// Builds the kind that fits `len` children, taken in ascending key order.
    ///
    /// The kind follows from the count alone, so growth and shrink are one rule
    /// read in two directions, at the boundaries `part/txn.go` uses: a node4
    /// full at 4 promotes on the 5th child, and a node16 back down to 4 demotes.
    fn build(len: usize, children: impl Iterator<Item = Arc<Node<V>>>) -> Self {
        match len {
            0..=4 => Self::N4(Sorted::fill(children)),
            5..=16 => Self::N16(Box::new(Sorted::fill(children))),
            17..=48 => {
                let mut this = Node48 {
                    index: [0; 256],
                    slots: array::from_fn(|_| None),
                    len: 0,
                };
                for child in children {
                    this.index[usize::from(child.key())] = this.len + 1;
                    this.slots[usize::from(this.len)] = Some(child);
                    this.len += 1;
                }
                Self::N48(Box::new(this))
            }
            _ => {
                let mut this = Node256 {
                    slots: array::from_fn(|_| None),
                    len: 0,
                };
                for child in children {
                    let at = usize::from(child.key());
                    this.slots[at] = Some(child);
                    this.len += 1;
                }
                Self::N256(Box::new(this))
            }
        }
    }

    fn empty() -> Self {
        Self::build(0, iter::empty())
    }

    fn one(child: Arc<Node<V>>) -> Self {
        Self::build(1, iter::once(child))
    }

    /// Two children, given in either order.
    fn pair(a: Arc<Node<V>>, b: Arc<Node<V>>) -> Self {
        let ordered = if a.key() < b.key() { [a, b] } else { [b, a] };
        Self::build(2, ordered.into_iter())
    }

    fn len(&self) -> usize {
        match self {
            Self::N4(s) => usize::from(s.len),
            Self::N16(s) => usize::from(s.len),
            Self::N48(n) => usize::from(n.len),
            Self::N256(n) => usize::from(n.len),
        }
    }

    /// The slots in ascending key order. Only `N256` has empty ones.
    fn slots(&self) -> &[Slot<V>] {
        match self {
            Self::N4(s) => s.occupied(),
            Self::N16(s) => s.occupied(),
            Self::N48(n) => &n.slots[..usize::from(n.len)],
            Self::N256(n) => &n.slots[..],
        }
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = &Arc<Node<V>>> {
        self.slots().iter().flatten()
    }

    fn get(&self, byte: u8) -> Option<&Arc<Node<V>>> {
        match self {
            Self::N4(s) => s.get(byte),
            Self::N16(s) => s.get(byte),
            Self::N48(n) => match n.index[usize::from(byte)] {
                0 => None,
                slot => n.slots[usize::from(slot - 1)].as_ref(),
            },
            Self::N256(n) => n.slots[usize::from(byte)].as_ref(),
        }
    }

    /// The slots below `byte` and the slots above it. The slot at `byte`, if
    /// there is one, is in neither.
    fn split(&self, byte: u8) -> (&[Slot<V>], &[Slot<V>]) {
        let at = usize::from(byte);
        if let Self::N256(n) = self {
            return (&n.slots[..at], &n.slots[at + 1..]);
        }
        let slots = self.slots();
        let below = slots.partition_point(|s| key_of(s.as_ref()) < byte);
        let above =
            below + usize::from(slots.get(below).is_some_and(|s| key_of(s.as_ref()) == byte));
        (&slots[..below], &slots[above..])
    }

    /// Adds `child` under its own key, replacing whatever sits there.
    fn with(&self, child: Arc<Node<V>>) -> Self {
        let len = self.len() + usize::from(self.get(child.key()).is_none());
        let (below, above) = self.split(child.key());
        Self::build(
            len,
            below
                .iter()
                .flatten()
                .cloned()
                .chain(iter::once(child))
                .chain(above.iter().flatten().cloned()),
        )
    }

    /// Removes the child at `byte`, which must be there.
    fn without(&self, byte: u8) -> Self {
        let (below, above) = self.split(byte);
        Self::build(
            self.len() - 1,
            below
                .iter()
                .flatten()
                .chain(above.iter().flatten())
                .cloned(),
        )
    }

    /// The one child, when that is all there is.
    fn only(&self) -> Option<&Arc<Node<V>>> {
        (self.len() == 1).then(|| self.iter().next().expect("one child"))
    }

    /// Checks that the kind matches the child count and that the lookup
    /// structures agree with the slots. Test helper.
    fn assert_ok(&self) {
        let len = self.len();
        let fits = match self {
            Self::N4(s) => {
                s.assert_keys();
                len <= 4
            }
            Self::N16(s) => {
                s.assert_keys();
                (5..=16).contains(&len)
            }
            Self::N48(n) => {
                let listed = n.index.iter().filter(|slot| **slot != 0).count();
                assert_eq!(listed, len, "node48 index lists {listed} of {len} children");
                for (byte, slot) in n.index.iter().enumerate() {
                    if *slot == 0 {
                        continue;
                    }
                    let child = n.slots[usize::from(slot - 1)]
                        .as_ref()
                        .expect("node48 index points at an empty slot");
                    assert_eq!(
                        usize::from(child.key()),
                        byte,
                        "node48 index points at another key"
                    );
                }
                (17..=48).contains(&len)
            }
            Self::N256(n) => {
                for (byte, slot) in n.slots.iter().enumerate() {
                    if let Some(child) = slot {
                        assert_eq!(usize::from(child.key()), byte, "node256 child out of place");
                    }
                }
                (49..=256).contains(&len)
            }
        };
        assert!(fits, "node kind does not match {len} children");
        for pair in self.slots().windows(2) {
            if let [Some(a), Some(b)] = pair {
                assert!(a.key() < b.key(), "children not sorted and unique");
            }
        }
    }
}

impl<V, const K: usize> Sorted<V, K> {
    fn assert_keys(&self) {
        for (key, slot) in self.keys.iter().zip(self.occupied()) {
            assert_eq!(
                *key,
                key_of(slot.as_ref()),
                "key byte disagrees with the child prefix"
            );
        }
    }
}

// ------------------------------------------------------------------- tree

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
        match node.children.get(rest[0]) {
            Some(child) => node = child,
            None => return (None, node),
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
            root: Node::new(&[], None, Children::empty()),
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
            match node.children.get(rest[0]) {
                Some(child) => node = child,
                None => break,
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
            for child in node.children.split(rest[0]).1.iter().rev().flatten() {
                stack.push(Frame {
                    key: acc.clone(),
                    node: child,
                });
            }
            match node.children.get(rest[0]) {
                Some(child) => node = child,
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
    /// If a non-root node is uncompressed, or a node's children disagree with
    /// its kind, its order, or its own lookup structures.
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
            node.children.assert_ok();
            for child in node.children.iter() {
                walk(child, false);
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
            for child in frame.node.children.iter().rev() {
                self.stack.push(Frame {
                    key: key.clone(),
                    node: child,
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
            return (
                Node::new(&key[..common], Some(value), Children::one(tail)),
                None,
            );
        }
        let leaf = Node::new(&key[common..], Some(value), Children::empty());
        return (
            Node::new(&key[..common], None, Children::pair(tail, leaf)),
            None,
        );
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
    let (child, old) = match node.children.get(rest[0]) {
        Some(child) => insert(child, rest, value, closed),
        None => (Node::new(rest, Some(value), Children::empty()), None),
    };
    (
        node.with_children(&node.prefix, node.children.with(child)),
        old,
    )
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

    let Some(down) = node.children.get(rest[0]) else {
        return (None, None);
    };
    let (child, old) = delete(down, rest, false, closed);
    let Some(old) = old else {
        return (None, None);
    };
    closed.push(node.watch.clone());
    let children = match child {
        Some(child) => node.children.with(child),
        None => node.children.without(rest[0]),
    };
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
    children: Children<V>,
    is_root: bool,
    closed: &mut Vec<WatchCell>,
) -> Option<Arc<Node<V>>> {
    if value.is_none() && !is_root {
        if children.len() == 0 {
            return None;
        }
        if let Some(only) = children.only() {
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
