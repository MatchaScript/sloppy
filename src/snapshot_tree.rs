//! Persistent B+tree keyed by byte strings.
//!
//! A write path-copies: every node from the root down to the change is rebuilt,
//! everything else stays shared through `Arc`. A node this transaction already
//! rebuilt is mutated in place instead, which is what the `txn` stamp is for.
//!
//! Values live in the leaves; a branch holds separators and children. The keys
//! of one node sit end to end in a single buffer with their offsets beside it,
//! so rebuilding the node copies two allocations rather than one per key. A
//! leaf keeps the common prefix of its keys once and the remainders in that
//! buffer, so a descent matches the prefix and then binary-searches remainders
//! that no longer repeat it.
//!
//! The depth is `log` of the entry count, so every walk down and back up is a
//! recursion the keys' owner cannot make deep.
//!
//! The tree says nothing about notification: a node stands for a key range that
//! the range's owner never named, so what a reader watches is the database's
//! to decide.

use std::cmp::Ordering;
use std::sync::Arc;

/// Identifies the transaction that built a node. `0` is "no transaction".
type TxnId = u64;

/// Entries in a leaf, children in a branch. A node over this splits.
const ORDER: usize = 32;

/// Under this a node borrows from a sibling or merges with it. The root is
/// exempt: it may hold anything from nothing to a full node.
const MIN: usize = ORDER / 2;

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The keys of one node, in ascending order, end to end in one buffer.
///
/// A path copy clones this with two allocations whatever the key count is,
/// which is what the layout is for.
#[derive(Clone, Default)]
struct Keys {
    bytes: Vec<u8>,
    /// Where each key ends in `bytes`; the previous end is where it starts.
    ends: Vec<u32>,
}

fn offset(at: usize) -> u32 {
    u32::try_from(at).expect("one node's keys are shorter than 4 GiB together")
}

impl Keys {
    fn len(&self) -> usize {
        self.ends.len()
    }

    fn start(&self, at: usize) -> usize {
        if at == 0 {
            0
        } else {
            self.ends[at - 1] as usize
        }
    }

    fn get(&self, at: usize) -> &[u8] {
        &self.bytes[self.start(at)..self.ends[at] as usize]
    }

    fn owned(&self, at: usize) -> Box<[u8]> {
        self.get(at).into()
    }

    fn search(&self, key: &[u8]) -> Result<usize, usize> {
        let (mut low, mut high) = (0, self.len());
        while low < high {
            let mid = usize::midpoint(low, high);
            match self.get(mid).cmp(key) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(low)
    }

    /// How many keys are `<= key`.
    fn upper_bound(&self, key: &[u8]) -> usize {
        match self.search(key) {
            Ok(at) => at + 1,
            Err(at) => at,
        }
    }

    /// Appends `head ++ tail` as one more key, which must sort after the rest.
    fn push(&mut self, head: &[u8], tail: &[u8]) {
        self.bytes.extend_from_slice(head);
        self.bytes.extend_from_slice(tail);
        self.ends.push(offset(self.bytes.len()));
    }

    fn insert(&mut self, at: usize, key: &[u8]) {
        let start = self.start(at);
        self.bytes.splice(start..start, key.iter().copied());
        let grew = offset(key.len());
        self.ends.insert(at, offset(start) + grew);
        for end in &mut self.ends[at + 1..] {
            *end += grew;
        }
    }

    fn remove(&mut self, at: usize) {
        let start = self.start(at);
        let end = self.ends[at] as usize;
        let shrank = offset(end - start);
        self.bytes.drain(start..end);
        self.ends.remove(at);
        for end in &mut self.ends[at..] {
            *end -= shrank;
        }
    }

    /// Takes the keys from `at` onwards.
    fn split_off(&mut self, at: usize) -> Self {
        let start = offset(self.start(at));
        Self {
            bytes: self.bytes.split_off(start as usize),
            ends: self.ends.split_off(at).iter().map(|e| e - start).collect(),
        }
    }

    /// Appends every key of `other`, which all sort after these.
    fn append(&mut self, other: &Self) {
        let base = offset(self.bytes.len());
        self.bytes.extend_from_slice(&other.bytes);
        self.ends.extend(other.ends.iter().map(|end| end + base));
    }

    /// Rebuilds every key as `head` followed by the key with `strip` bytes cut
    /// off its front.
    fn rewrite(&mut self, head: &[u8], strip: usize) {
        let mut bytes = Vec::with_capacity(self.bytes.len() + head.len() * self.len());
        let mut ends = Vec::with_capacity(self.len());
        for at in 0..self.len() {
            bytes.extend_from_slice(head);
            bytes.extend_from_slice(&self.get(at)[strip..]);
            ends.push(offset(bytes.len()));
        }
        self.bytes = bytes;
        self.ends = ends;
    }
}

/// The values of one key range, in ascending key order.
struct Leaf<V> {
    /// A prefix of every key here. Not necessarily the longest one: a delete
    /// may leave it shorter than it could be, which costs space, not meaning.
    prefix: Box<[u8]>,
    /// Each key with `prefix` cut off the front.
    rest: Keys,
    values: Vec<V>,
    txn: TxnId,
}

/// `separators[i]` is the first key of `children[i + 1]`.
struct Branch<V> {
    separators: Keys,
    children: Vec<Arc<Node<V>>>,
    txn: TxnId,
}

enum Node<V> {
    Leaf(Leaf<V>),
    Branch(Branch<V>),
}

/// A node that came out of a split: its first key, and the node.
type Split<V> = Option<(Box<[u8]>, Arc<Node<V>>)>;

impl<V> Leaf<V> {
    fn empty(txn: TxnId) -> Self {
        Self {
            prefix: Box::default(),
            rest: Keys::default(),
            values: Vec::new(),
            txn,
        }
    }

    /// Where `key` sits, or where it would go.
    fn locate(&self, key: &[u8]) -> Result<usize, usize> {
        let common = self.prefix.len().min(key.len());
        match key[..common].cmp(&self.prefix[..common]) {
            Ordering::Less => return Err(0),
            Ordering::Greater => return Err(self.rest.len()),
            Ordering::Equal => {}
        }
        if key.len() < self.prefix.len() {
            // `key` is a strict prefix of the shared prefix, so it sorts below
            // everything here.
            return Err(0);
        }
        self.rest.search(&key[self.prefix.len()..])
    }

    fn full_key(&self, at: usize) -> Box<[u8]> {
        let rest = self.rest.get(at);
        let mut key = Vec::with_capacity(self.prefix.len() + rest.len());
        key.extend_from_slice(&self.prefix);
        key.extend_from_slice(rest);
        key.into()
    }

    /// Cuts the shared prefix back to `keep` bytes, giving what it loses back
    /// to every remainder.
    fn shrink(&mut self, keep: usize) {
        if keep == self.prefix.len() {
            return;
        }
        let prefix = std::mem::take(&mut self.prefix);
        self.rest.rewrite(&prefix[keep..], 0);
        self.prefix = prefix[..keep].into();
    }

    /// Cuts the shared prefix back to what `key` also starts with, so `key` can
    /// be stored here as a remainder.
    fn adopt(&mut self, key: &[u8]) {
        self.shrink(lcp(&self.prefix, key));
    }

    /// Takes the longest common prefix of the remainders into the shared
    /// prefix. Run after a split, where each half usually shares more than the
    /// whole did.
    fn compress(&mut self) {
        if self.rest.len() == 0 {
            return;
        }
        let mut common = self.rest.get(0).len();
        for at in 1..self.rest.len() {
            common = lcp(&self.rest.get(0)[..common], self.rest.get(at));
        }
        if common == 0 {
            return;
        }
        let mut prefix = Vec::with_capacity(self.prefix.len() + common);
        prefix.extend_from_slice(&self.prefix);
        prefix.extend_from_slice(&self.rest.get(0)[..common]);
        self.prefix = prefix.into();
        self.rest.rewrite(&[], common);
    }

    fn split(&mut self, txn: TxnId) -> Split<V> {
        if self.rest.len() <= ORDER {
            return None;
        }
        let at = self.rest.len() / 2;
        let mut right = Self {
            prefix: self.prefix.clone(),
            rest: self.rest.split_off(at),
            values: self.values.split_off(at),
            txn,
        };
        let first = right.full_key(0);
        self.compress();
        right.compress();
        Some((first, Arc::new(Node::Leaf(right))))
    }
}

impl<V: Clone> Leaf<V> {
    /// Appends every entry of `other`, which sorts after this one.
    fn absorb(&mut self, other: &Self) {
        self.shrink(lcp(&self.prefix, &other.prefix));
        let carried = &other.prefix[self.prefix.len()..];
        for at in 0..other.rest.len() {
            self.rest.push(carried, other.rest.get(at));
        }
        self.values.extend(other.values.iter().cloned());
    }

    /// Moves one entry from the front of `other` onto the end of this leaf.
    fn take_first(&mut self, other: &mut Self) {
        let key = other.full_key(0);
        other.rest.remove(0);
        let value = other.values.remove(0);
        self.adopt(&key);
        self.rest.push(&key[self.prefix.len()..], &[]);
        self.values.push(value);
    }

    /// Moves one entry from the end of `other` onto the front of this leaf.
    fn take_last(&mut self, other: &mut Self) {
        let last = other.rest.len() - 1;
        let key = other.full_key(last);
        other.rest.remove(last);
        let value = other.values.pop().expect("the leaf held it");
        self.adopt(&key);
        self.rest.insert(0, &key[self.prefix.len()..]);
        self.values.insert(0, value);
    }
}

impl<V> Branch<V> {
    /// The child `key` belongs to.
    fn child_of(&self, key: &[u8]) -> usize {
        self.separators.upper_bound(key)
    }

    fn split(&mut self, txn: TxnId) -> Split<V> {
        if self.children.len() <= ORDER {
            return None;
        }
        let at = self.children.len() / 2;
        let children = self.children.split_off(at);
        let mut separators = self.separators.split_off(at - 1);
        // The separator between the halves is the first key of the right one,
        // which is what the parent files it under.
        let first = separators.owned(0);
        separators.remove(0);
        Some((
            first,
            Arc::new(Node::Branch(Self {
                separators,
                children,
                txn,
            })),
        ))
    }

    /// Replaces the separator at `at`, which the two children beside it moved.
    fn refile(&mut self, at: usize, key: &[u8]) {
        self.separators.remove(at);
        self.separators.insert(at, key);
    }
}

impl<V> Node<V> {
    fn txn(&self) -> TxnId {
        match self {
            Self::Leaf(leaf) => leaf.txn,
            Self::Branch(branch) => branch.txn,
        }
    }

    /// Entries for a leaf, children for a branch.
    fn len(&self) -> usize {
        match self {
            Self::Leaf(leaf) => leaf.values.len(),
            Self::Branch(branch) => branch.children.len(),
        }
    }
}

impl<V: Clone> Node<V> {
    fn copy(&self, txn: TxnId) -> Self {
        match self {
            Self::Leaf(leaf) => Self::Leaf(Leaf {
                prefix: leaf.prefix.clone(),
                rest: leaf.rest.clone(),
                values: leaf.values.clone(),
                txn,
            }),
            Self::Branch(branch) => Self::Branch(Branch {
                separators: branch.separators.clone(),
                children: branch.children.clone(),
                txn,
            }),
        }
    }

    /// The node this transaction may mutate: `*node` itself when this
    /// transaction built it, a path copy of it otherwise. A node this
    /// transaction built has never been published, so nobody can be reading it.
    fn own(node: &mut Arc<Self>, txn: TxnId) -> &mut Self {
        if node.txn() != txn {
            *node = Arc::new(node.copy(txn));
        }
        Arc::get_mut(node).expect("a node stamped with this transaction is not shared")
    }
}

impl<V: Clone> Branch<V> {
    /// Restores the entry count of child `at` after a delete took it below
    /// `MIN`, by merging it with a neighbour or moving one entry across.
    fn rebalance(&mut self, at: usize, txn: TxnId) {
        if self.children[at].len() >= MIN || self.children.len() < 2 {
            return;
        }
        let (left, right) = if at + 1 < self.children.len() {
            (at, at + 1)
        } else {
            (at - 1, at)
        };
        if self.children[left].len() + self.children[right].len() <= ORDER {
            self.merge(left, txn);
        } else {
            self.shift(left, at == left, txn);
        }
    }

    /// Folds `left + 1` into `left`.
    fn merge(&mut self, left: usize, txn: TxnId) {
        let separator = self.separators.owned(left);
        self.separators.remove(left);
        let right = self.children.remove(left + 1);
        match (Node::own(&mut self.children[left], txn), &*right) {
            (Node::Leaf(a), Node::Leaf(b)) => a.absorb(b),
            (Node::Branch(a), Node::Branch(b)) => {
                a.separators.push(&separator, &[]);
                a.separators.append(&b.separators);
                a.children.extend(b.children.iter().cloned());
            }
            _ => unreachable!("every leaf is at the same depth"),
        }
    }

    /// Moves one entry between `left` and `left + 1`, towards `left` when
    /// `to_left`, and files the right neighbour under its new first key.
    fn shift(&mut self, left: usize, to_left: bool, txn: TxnId) {
        let separator = self.separators.owned(left);
        let (below, above) = self.children.split_at_mut(left + 1);
        let raised = match (
            Node::own(&mut below[left], txn),
            Node::own(&mut above[0], txn),
        ) {
            (Node::Leaf(a), Node::Leaf(b)) => {
                if to_left {
                    a.take_first(b);
                } else {
                    b.take_last(a);
                }
                b.full_key(0)
            }
            (Node::Branch(a), Node::Branch(b)) => {
                if to_left {
                    a.separators.push(&separator, &[]);
                    a.children.push(b.children.remove(0));
                    let raised = b.separators.owned(0);
                    b.separators.remove(0);
                    raised
                } else {
                    b.children.insert(0, a.children.pop().expect("held it"));
                    b.separators.insert(0, &separator);
                    let last = a.separators.len() - 1;
                    let raised = a.separators.owned(last);
                    a.separators.remove(last);
                    raised
                }
            }
            _ => unreachable!("every leaf is at the same depth"),
        };
        self.refile(left, &raised);
    }
}

// ------------------------------------------------------------------- lookup

/// The value at `key`.
fn find<'a, V>(root: &'a Arc<Node<V>>, key: &[u8]) -> Option<&'a V> {
    let mut node: &Node<V> = root;
    loop {
        match node {
            Node::Leaf(leaf) => return leaf.locate(key).ok().map(|at| &leaf.values[at]),
            Node::Branch(branch) => node = &branch.children[branch.child_of(key)],
        }
    }
}

// --------------------------------------------------------------------- tree

/// An immutable snapshot. `clone` is one `Arc` bump.
pub struct Tree<V> {
    root: Arc<Node<V>>,
    len: usize,
    /// The transaction that produced this tree; the next one gets `last_txn + 1`.
    last_txn: TxnId,
}

impl<V> Clone for Tree<V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            len: self.len,
            last_txn: self.last_txn,
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
            root: Arc::new(Node::Leaf(Leaf::empty(0))),
            len: 0,
            last_txn: 0,
        }
    }

    /// The value at `key`.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&V> {
        find(&self.root, key)
    }

    /// Every entry whose key starts with `p`.
    #[must_use]
    pub fn prefix(&self, p: &[u8]) -> Iter<'_, V> {
        Iter::under(&self.root, p)
    }

    /// Every entry with a key `>= key`, in order.
    #[must_use]
    pub fn lower_bound(&self, key: &[u8]) -> Iter<'_, V> {
        Iter::from(&self.root, key)
    }

    /// Every entry, in ascending key order.
    #[allow(clippy::iter_without_into_iter)] // `IntoIterator` for `&Tree` has no user yet.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, V> {
        Iter::whole(&self.root)
    }

    #[must_use]
    pub fn txn(&self) -> Txn<V> {
        Txn {
            root: self.root.clone(),
            len: self.len,
            id: self.last_txn + 1,
        }
    }
}

// --------------------------------------------------------------------- walk

/// Yields values in ascending byte-lexicographic key order.
pub struct Iter<'a, V> {
    /// The branches on the path down, each with the child to descend into next.
    stack: Vec<(&'a Branch<V>, usize)>,
    /// The leaf the walk is in, and the entry to yield next.
    leaf: Option<(&'a Leaf<V>, usize)>,
    /// The key of the entry the last `next` returned.
    path: Vec<u8>,
    /// Stops the walk at the first key that does not start with this.
    stop: Option<Box<[u8]>>,
}

impl<'a, V> Iter<'a, V> {
    fn empty() -> Self {
        Self {
            stack: Vec::new(),
            leaf: None,
            path: Vec::new(),
            stop: None,
        }
    }

    /// The whole tree, in order.
    fn whole(root: &'a Arc<Node<V>>) -> Self {
        let mut iter = Self::empty();
        iter.descend(root);
        iter
    }

    /// Every entry with a key `>= key`, in order.
    fn from(root: &'a Arc<Node<V>>, key: &[u8]) -> Self {
        let mut iter = Self::empty();
        let mut node: &'a Node<V> = root;
        loop {
            match node {
                Node::Leaf(leaf) => {
                    iter.leaf = Some((leaf, leaf.locate(key).unwrap_or_else(|at| at)));
                    return iter;
                }
                Node::Branch(branch) => {
                    let at = branch.child_of(key);
                    iter.stack.push((branch, at + 1));
                    node = &branch.children[at];
                }
            }
        }
    }

    /// Every entry whose key starts with `p`.
    fn under(root: &'a Arc<Node<V>>, p: &[u8]) -> Self {
        let mut iter = Self::from(root, p);
        iter.stop = Some(p.into());
        iter
    }

    /// Enters the leftmost leaf of `node`.
    fn descend(&mut self, node: &'a Node<V>) {
        let mut node = node;
        loop {
            match node {
                Node::Leaf(leaf) => {
                    self.leaf = Some((leaf, 0));
                    return;
                }
                Node::Branch(branch) => {
                    self.stack.push((branch, 1));
                    node = &branch.children[0];
                }
            }
        }
    }

    /// Moves on to the next leaf. `false` when there is none.
    fn advance(&mut self) -> bool {
        while let Some(&(branch, at)) = self.stack.last() {
            if at < branch.children.len() {
                self.stack.last_mut().expect("just read").1 = at + 1;
                self.descend(&branch.children[at]);
                return true;
            }
            self.stack.pop();
        }
        false
    }

    /// The key of the entry the last [`Iterator::next`] returned. Meaningless
    /// before the first `next`, and after one that returned `None`.
    #[must_use]
    pub fn key(&self) -> &[u8] {
        &self.path
    }
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = &'a V;

    // ponytail: no allocation per entry; the key is rebuilt in the walk's own
    // buffer, which `key` hands out a borrow of.
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (leaf, at) = self.leaf?;
            if at == leaf.values.len() {
                self.leaf = None;
                if self.advance() {
                    continue;
                }
                return None;
            }
            self.leaf = Some((leaf, at + 1));
            self.path.clear();
            self.path.extend_from_slice(&leaf.prefix);
            self.path.extend_from_slice(leaf.rest.get(at));
            if self
                .stop
                .as_ref()
                .is_some_and(|p| !self.path.starts_with(p))
            {
                self.leaf = None;
                self.stack.clear();
                return None;
            }
            return Some(&leaf.values[at]);
        }
    }
}

// -------------------------------------------------------------------- write

/// A batch of writes over one snapshot. Dropping it aborts.
pub struct Txn<V> {
    root: Arc<Node<V>>,
    len: usize,
    /// This transaction's stamp: the node it built, it may write in place.
    id: TxnId,
}

impl<V: Clone> Txn<V> {
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&V> {
        find(&self.root, key)
    }

    /// Every entry, in ascending key order.
    #[allow(clippy::iter_without_into_iter)] // `IntoIterator` for `&Txn` has no user yet.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, V> {
        Iter::whole(&self.root)
    }

    /// Every entry whose key starts with `p`.
    #[must_use]
    pub fn prefix(&self, p: &[u8]) -> Iter<'_, V> {
        Iter::under(&self.root, p)
    }

    /// Every entry with a key `>= key`, in order.
    #[must_use]
    pub fn lower_bound(&self, key: &[u8]) -> Iter<'_, V> {
        Iter::from(&self.root, key)
    }

    /// Returns the replaced value.
    pub fn insert(&mut self, key: &[u8], value: V) -> Option<V> {
        let (old, split) = insert(&mut self.root, key, value, self.id);
        if let Some((separator, right)) = split {
            let left = std::mem::replace(&mut self.root, Arc::new(Node::Leaf(Leaf::empty(0))));
            let mut separators = Keys::default();
            separators.push(&separator, &[]);
            self.root = Arc::new(Node::Branch(Branch {
                separators,
                children: vec![left, right],
                txn: self.id,
            }));
        }
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub fn delete(&mut self, key: &[u8]) -> Option<V> {
        // The descent below rebuilds as it goes, so the key is looked up first:
        // a miss must leave the tree alone.
        find(&self.root, key)?;
        let old = remove(&mut self.root, key, self.id);
        // A branch left with a single child is a level that fences nothing.
        while let Node::Branch(branch) = &*self.root {
            if branch.children.len() != 1 {
                break;
            }
            let only = branch.children[0].clone();
            self.root = only;
        }
        self.len -= 1;
        Some(old)
    }

    /// The tree this transaction leaves behind.
    #[must_use]
    pub fn commit(self) -> Tree<V> {
        Tree {
            root: self.root,
            len: self.len,
            last_txn: self.id,
        }
    }
}

/// Writes `value` at `key` below `*node`, replacing `*node` with the node that
/// takes its place and returning what that node split off, if anything.
fn insert<V: Clone>(
    node: &mut Arc<Node<V>>,
    key: &[u8],
    value: V,
    txn: TxnId,
) -> (Option<V>, Split<V>) {
    match Node::own(node, txn) {
        Node::Leaf(leaf) => {
            leaf.adopt(key);
            match leaf.locate(key) {
                Ok(at) => (Some(std::mem::replace(&mut leaf.values[at], value)), None),
                Err(at) => {
                    leaf.rest.insert(at, &key[leaf.prefix.len()..]);
                    leaf.values.insert(at, value);
                    (None, leaf.split(txn))
                }
            }
        }
        Node::Branch(branch) => {
            let at = branch.child_of(key);
            let (old, split) = insert(&mut branch.children[at], key, value, txn);
            if let Some((separator, right)) = split {
                branch.separators.insert(at, &separator);
                branch.children.insert(at + 1, right);
            }
            (old, branch.split(txn))
        }
    }
}

/// Removes `key`, which must be present, replacing `*node` with the node that
/// takes its place.
fn remove<V: Clone>(node: &mut Arc<Node<V>>, key: &[u8], txn: TxnId) -> V {
    match Node::own(node, txn) {
        Node::Leaf(leaf) => {
            let at = leaf.locate(key).expect("the key was looked up first");
            leaf.rest.remove(at);
            leaf.values.remove(at)
        }
        Node::Branch(branch) => {
            let at = branch.child_of(key);
            let old = remove(&mut branch.children[at], key, txn);
            branch.rebalance(at, txn);
            old
        }
    }
}
