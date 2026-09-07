//! Mutable B+tree keyed by byte strings.
//!
//! Every node sits behind its own `RwLock` inside an `Arc`. A reader walks from
//! the root to a leaf with lock coupling: it takes the child's read lock before
//! it lets go of the parent's, so it can never be routed by a branch that the
//! writer is in the middle of restructuring. The single writer takes the leaf's
//! write lock and writes the entry in place; a write that would overfill or
//! empty out a leaf goes down again holding the write lock of every node on the
//! path, and that second descent is where splits, borrows and merges happen.
//!
//! Values live in the leaves; a branch holds separators and children. The keys
//! of one node sit end to end in a single buffer with their offsets beside it,
//! so moving keys between nodes copies runs of bytes rather than one allocation
//! per key. A leaf keeps the common prefix of its keys once and the remainders
//! in that buffer, so a descent matches the prefix and then binary-searches
//! remainders that no longer repeat it.
//!
//! A walk holds no lock between leaves: it takes one leaf's worth of entries as
//! owned values and seeks again from the root for the next leaf. Nodes carry no
//! sibling pointers, so a split has nothing to keep straight but the parent's
//! separators, and a key that moved to the new sibling is found by the seek
//! that follows.
//!
//! The depth is `log` of the entry count, so every walk down and back up is a
//! recursion the keys' owner cannot make deep.
//!
//! The tree says nothing about notification: a node stands for a key range that
//! the range's owner never named, so what a reader watches is the database's
//! to decide.

use std::cmp::Ordering;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Entries in a leaf, children in a branch. A node over this splits.
const ORDER: usize = 32;

/// Under this a node borrows from a sibling or merges with it. The root is
/// exempt: it may hold anything from nothing to a full node.
const MIN: usize = ORDER / 2;

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A node's lock, taken as it stands. A writer that panicked under it left one
/// node's arrays as they were mid-write, not the tree's shape, so the poison
/// flag says nothing the tree acts on.
fn read<V>(node: &RwLock<Node<V>>) -> RwLockReadGuard<'_, Node<V>> {
    node.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<V>(node: &RwLock<Node<V>>) -> RwLockWriteGuard<'_, Node<V>> {
    node.write().unwrap_or_else(PoisonError::into_inner)
}

/// The keys of one node, in ascending order, end to end in one buffer.
#[derive(Default)]
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
}

/// `separators[i]` is the first key of `children[i + 1]`.
struct Branch<V> {
    separators: Keys,
    children: Vec<Link<V>>,
}

enum Node<V> {
    Leaf(Leaf<V>),
    Branch(Branch<V>),
}

/// A node as its parent holds it: one lock of its own, shared with whoever is
/// reading it.
type Link<V> = Arc<RwLock<Node<V>>>;

/// A node that came out of a split: its first key, and the node.
type Split<V> = Option<(Box<[u8]>, Link<V>)>;

/// One entry as a walk hands it out: the whole key, and a copy of the value.
type Entry<V> = (Box<[u8]>, V);

/// One leaf's worth of entries, and the key the next leaf starts at or above.
type Batch<V> = (Vec<Entry<V>>, Option<Box<[u8]>>);

fn link<V>(node: Node<V>) -> Link<V> {
    Arc::new(RwLock::new(node))
}

impl<V> Leaf<V> {
    fn empty() -> Self {
        Self {
            prefix: Box::default(),
            rest: Keys::default(),
            values: Vec::new(),
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

    /// Puts `value` at `at`, where `key` was found not to be.
    fn place(&mut self, at: usize, key: &[u8], value: V) {
        self.rest.insert(at, &key[self.prefix.len()..]);
        self.values.insert(at, value);
    }

    fn take(&mut self, at: usize) -> V {
        self.rest.remove(at);
        self.values.remove(at)
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

    fn split(&mut self) -> Split<V> {
        if self.rest.len() <= ORDER {
            return None;
        }
        let at = self.rest.len() / 2;
        let mut right = Self {
            prefix: self.prefix.clone(),
            rest: self.rest.split_off(at),
            values: self.values.split_off(at),
        };
        let first = right.full_key(0);
        self.compress();
        right.compress();
        Some((first, link(Node::Leaf(right))))
    }

    /// Takes every entry of `other`, which sorts after this one.
    fn absorb(&mut self, other: &mut Self) {
        self.shrink(lcp(&self.prefix, &other.prefix));
        let carried = &other.prefix[self.prefix.len()..];
        for at in 0..other.rest.len() {
            self.rest.push(carried, other.rest.get(at));
        }
        self.values.append(&mut other.values);
    }

    /// Moves one entry from the front of `other` onto the end of this leaf.
    fn take_first(&mut self, other: &mut Self) {
        let key = other.full_key(0);
        let value = other.take(0);
        self.adopt(&key);
        self.rest.push(&key[self.prefix.len()..], &[]);
        self.values.push(value);
    }

    /// Moves one entry from the end of `other` onto the front of this leaf.
    fn take_last(&mut self, other: &mut Self) {
        let last = other.rest.len() - 1;
        let key = other.full_key(last);
        let value = other.take(last);
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

    /// Entries for a leaf child, children for a branch one.
    fn child_len(&self, at: usize) -> usize {
        read(&self.children[at]).len()
    }

    fn split(&mut self) -> Split<V> {
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
            link(Node::Branch(Self {
                separators,
                children,
            })),
        ))
    }

    /// Replaces the separator at `at`, which the two children beside it moved.
    fn refile(&mut self, at: usize, key: &[u8]) {
        self.separators.remove(at);
        self.separators.insert(at, key);
    }

    /// Restores the entry count of child `at` after a delete took it below
    /// `MIN`, by merging it with a neighbour or moving one entry across.
    ///
    /// Both children are write-locked while their entries move, and this
    /// branch's own lock is held throughout, so a reader is either already
    /// inside one of them and holding it up, or has yet to be told which of
    /// them its key is in.
    fn rebalance(&mut self, at: usize) {
        if self.child_len(at) >= MIN || self.children.len() < 2 {
            return;
        }
        let (left, right) = if at + 1 < self.children.len() {
            (at, at + 1)
        } else {
            (at - 1, at)
        };
        if self.child_len(left) + self.child_len(right) <= ORDER {
            self.merge(left);
        } else {
            self.shift(left, at == left);
        }
    }

    /// Folds `left + 1` into `left`.
    fn merge(&mut self, left: usize) {
        let separator = self.separators.owned(left);
        self.separators.remove(left);
        let right = self.children.remove(left + 1);
        let mut into = write(&self.children[left]);
        let mut from = write(&right);
        match (&mut *into, &mut *from) {
            (Node::Leaf(a), Node::Leaf(b)) => a.absorb(b),
            (Node::Branch(a), Node::Branch(b)) => {
                a.separators.push(&separator, &[]);
                a.separators.append(&b.separators);
                a.children.append(&mut b.children);
            }
            _ => unreachable!("every leaf is at the same depth"),
        }
    }

    /// Moves one entry between `left` and `left + 1`, towards `left` when
    /// `to_left`, and files the right neighbour under its new first key.
    fn shift(&mut self, left: usize, to_left: bool) {
        let separator = self.separators.owned(left);
        let raised = {
            let mut below = write(&self.children[left]);
            let mut above = write(&self.children[left + 1]);
            match (&mut *below, &mut *above) {
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
            }
        };
        self.refile(left, &raised);
    }
}

impl<V> Node<V> {
    /// Entries for a leaf, children for a branch.
    fn len(&self) -> usize {
        match self {
            Self::Leaf(leaf) => leaf.values.len(),
            Self::Branch(branch) => branch.children.len(),
        }
    }

    fn leaf(&mut self) -> &mut Leaf<V> {
        match self {
            Self::Leaf(leaf) => leaf,
            Self::Branch(_) => unreachable!("the descent stopped at a leaf"),
        }
    }
}

// ------------------------------------------------------------------- descent

/// Walks to the leaf `key` belongs to and hands it to `f`, along with the
/// separator that fences the leaf on the right if it has one.
///
/// Each child's read lock is taken before the parent's is let go, so the writer
/// cannot move `key` out of a leaf between the branch that named the leaf and
/// the leaf itself: to restructure either, it must take a lock this walk is
/// holding.
fn descend<V, R, F: FnOnce(&Leaf<V>, Option<&[u8]>) -> R>(
    guard: RwLockReadGuard<'_, Node<V>>,
    key: &[u8],
    hi: Option<&[u8]>,
    f: F,
) -> R {
    let (child, fence) = match &*guard {
        Node::Leaf(leaf) => return f(leaf, hi),
        Node::Branch(branch) => {
            let at = branch.child_of(key);
            let fence = (at < branch.separators.len()).then(|| branch.separators.owned(at));
            (branch.children[at].clone(), fence)
        }
    };
    let below = read(&child);
    drop(guard);
    descend(below, key, fence.as_deref().or(hi), f)
}

/// The leaf `key` belongs to. Only the writer walks this way: it is the one
/// thread that changes the shape, so it can let go of a branch before it reads
/// the child the branch named.
fn leaf_of<V>(root: &Link<V>, key: &[u8]) -> Link<V> {
    let mut node = root.clone();
    loop {
        let below = match &*read(&node) {
            Node::Leaf(_) => None,
            Node::Branch(branch) => Some(branch.children[branch.child_of(key)].clone()),
        };
        match below {
            Some(child) => node = child,
            None => return node,
        }
    }
}

/// The entries of the leaf `from` belongs to, from `from` onwards, and the key
/// the next leaf starts at or above.
fn batch<V: Clone>(root: &Link<V>, from: &[u8]) -> Batch<V> {
    descend(read(root), from, None, |leaf, hi| {
        let at = leaf.locate(from).unwrap_or_else(|at| at);
        let taken = (at..leaf.rest.len())
            .map(|at| (leaf.full_key(at), leaf.values[at].clone()))
            .collect();
        (taken, hi.map(Box::from))
    })
}

/// The first key that sorts after `key`: every byte string above `key` is at or
/// above it.
fn after(key: &[u8]) -> Box<[u8]> {
    let mut next = Vec::with_capacity(key.len() + 1);
    next.extend_from_slice(key);
    next.push(0);
    next.into()
}

// --------------------------------------------------------------------- tree

/// A B+tree one thread writes and any number read.
pub struct Tree<V> {
    /// The root keeps its identity for the life of the tree: a split moves its
    /// contents into a new child and a collapse pulls the last child's back up,
    /// so a walk that is already inside the root stays on the tree.
    root: Link<V>,
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
            root: link(Node::Leaf(Leaf::empty())),
        }
    }

    /// Writes `value` at `key` and returns what it replaced.
    pub fn insert(&self, key: &[u8], value: V) -> Option<V> {
        let leaf = leaf_of(&self.root, key);
        {
            let mut guard = write(&leaf);
            let held = guard.leaf();
            held.adopt(key);
            match held.locate(key) {
                Ok(at) => return Some(std::mem::replace(&mut held.values[at], value)),
                Err(at) if held.rest.len() < ORDER => {
                    held.place(at, key, value);
                    return None;
                }
                Err(_) => {}
            }
        }
        // The leaf is full, so the write has to be able to split it and file
        // the half that comes off in the parent.
        self.insert_deep(key, value);
        None
    }

    /// Removes `key` and returns what was there.
    // The removed value is worth ignoring; the removal itself is the point.
    #[allow(clippy::must_use_candidate)]
    pub fn remove(&self, key: &[u8]) -> Option<V> {
        let leaf = leaf_of(&self.root, key);
        {
            let mut guard = write(&leaf);
            let held = guard.leaf();
            let at = held.locate(key).ok()?;
            if held.rest.len() > MIN || Arc::ptr_eq(&leaf, &self.root) {
                return Some(held.take(at));
            }
        }
        // The leaf is down to `MIN`, so the delete has to be able to rebalance
        // it against a sibling.
        Some(self.remove_deep(key))
    }

    /// Hands the value at `key` to `f`, under the leaf's write lock. Does
    /// nothing if the key is not there.
    pub fn update(&self, key: &[u8], f: impl FnOnce(&mut V)) {
        let leaf = leaf_of(&self.root, key);
        let mut guard = write(&leaf);
        let held = guard.leaf();
        if let Ok(at) = held.locate(key) {
            f(&mut held.values[at]);
        }
    }

    /// Writes a key the leaf it belongs to had no room for, holding the write
    /// lock of every node from the root down so no reader is routed by a branch
    /// a split has yet to reach.
    fn insert_deep(&self, key: &[u8], value: V) {
        let mut root = write(&self.root);
        if let Some((separator, right)) = insert_at(&mut root, key, value) {
            let left = std::mem::replace(
                &mut *root,
                Node::Branch(Branch {
                    separators: Keys::default(),
                    children: Vec::new(),
                }),
            );
            let Node::Branch(branch) = &mut *root else {
                unreachable!("just put a branch there")
            };
            branch.separators.push(&separator, &[]);
            branch.children.push(link(left));
            branch.children.push(right);
        }
    }

    /// Removes a key whose leaf would underflow, under the same locks.
    fn remove_deep(&self, key: &[u8]) -> V {
        let mut root = write(&self.root);
        let old = remove_at(&mut root, key);
        // A branch left with a single child is a level that fences nothing. The
        // child's contents come up into the root, and the child is left holding
        // the emptied branch, which points at nothing.
        if let Node::Branch(branch) = &mut *root
            && branch.children.len() == 1
        {
            let only = branch.children.pop().expect("just counted it");
            std::mem::swap(&mut *root, &mut *write(&only));
        }
        old
    }

    /// Checks the shape invariants. Test helper.
    ///
    /// # Panics
    ///
    /// If the keys are out of order, a leaf's remainders disagree with its
    /// prefix, a separator does not fence the children it sits between, the
    /// leaves are at different depths, or a node is over- or under-filled.
    #[doc(hidden)]
    pub fn assert_invariants(&self) {
        let mut depth = None;
        check(&read(&self.root), 0, true, None, None, &mut depth);
    }
}

impl<V: Clone> Tree<V> {
    /// The value at `key`.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<V> {
        descend(read(&self.root), key, None, |leaf, _| {
            leaf.locate(key).ok().map(|at| leaf.values[at].clone())
        })
    }

    /// Every entry with a key `>= key`, in ascending key order.
    #[must_use]
    pub fn range_from(&self, key: &[u8]) -> Iter<V> {
        Iter {
            root: self.root.clone(),
            from: Some(key.into()),
            taken: Vec::new().into_iter(),
        }
    }

    /// Removes every entry with a key below `key`, from the left edge.
    pub fn remove_below(&self, key: &[u8]) {
        for (found, _) in self.range_from(&[]) {
            if *found >= *key {
                break;
            }
            self.remove(&found);
        }
    }
}

/// Writes `value` at `key` below `node`, which is write-locked, and returns
/// what `node` split off, if anything.
fn insert_at<V>(node: &mut Node<V>, key: &[u8], value: V) -> Split<V> {
    match node {
        Node::Leaf(leaf) => {
            leaf.adopt(key);
            match leaf.locate(key) {
                Ok(at) => {
                    leaf.values[at] = value;
                    None
                }
                Err(at) => {
                    leaf.place(at, key, value);
                    leaf.split()
                }
            }
        }
        Node::Branch(branch) => {
            let at = branch.child_of(key);
            let split = insert_at(&mut write(&branch.children[at]), key, value);
            if let Some((separator, right)) = split {
                branch.separators.insert(at, &separator);
                branch.children.insert(at + 1, right);
            }
            branch.split()
        }
    }
}

/// Removes `key`, which must be present, below the write-locked `node`.
fn remove_at<V>(node: &mut Node<V>, key: &[u8]) -> V {
    match node {
        Node::Leaf(leaf) => {
            let at = leaf
                .locate(key)
                .expect("the key was found before the descent");
            leaf.take(at)
        }
        Node::Branch(branch) => {
            let at = branch.child_of(key);
            let old = remove_at(&mut write(&branch.children[at]), key);
            branch.rebalance(at);
            old
        }
    }
}

/// Walks the subtree checking every invariant: `lo` and `hi` are the bounds the
/// parent's separators put on it.
fn check<V>(
    node: &Node<V>,
    depth: usize,
    is_root: bool,
    lo: Option<&[u8]>,
    hi: Option<&[u8]>,
    leaf_depth: &mut Option<usize>,
) {
    assert!(
        is_root || node.len() >= MIN,
        "a node below the root holds {} of {MIN} entries",
        node.len()
    );
    assert!(
        node.len() <= ORDER,
        "a node holds more than {ORDER} entries"
    );
    match node {
        Node::Leaf(leaf) => {
            assert_eq!(
                *leaf_depth.get_or_insert(depth),
                depth,
                "the leaves are at different depths"
            );
            assert_eq!(leaf.rest.len(), leaf.values.len(), "leaf arrays disagree");
            let keys: Vec<Box<[u8]>> = (0..leaf.rest.len()).map(|at| leaf.full_key(at)).collect();
            for pair in keys.windows(2) {
                assert!(pair[0] < pair[1], "leaf keys out of order");
            }
            for key in &keys {
                assert!(key.starts_with(&leaf.prefix), "key without the leaf prefix");
                assert!(lo.is_none_or(|lo| **key >= *lo), "key below its separator");
                assert!(hi.is_none_or(|hi| **key < *hi), "key above its separator");
            }
        }
        Node::Branch(branch) => {
            assert_eq!(
                branch.separators.len() + 1,
                branch.children.len(),
                "a branch's separators do not fence its children"
            );
            for (at, child) in branch.children.iter().enumerate() {
                let below = if at == 0 {
                    lo
                } else {
                    Some(branch.separators.get(at - 1))
                };
                let above = if at < branch.separators.len() {
                    Some(branch.separators.get(at))
                } else {
                    hi
                };
                check(&read(child), depth + 1, false, below, above, leaf_depth);
            }
        }
    }
}

// --------------------------------------------------------------------- walk

/// Yields entries in ascending byte-lexicographic key order.
///
/// One leaf's worth of entries is taken at a time and handed out as owned
/// values, so the walk holds no lock between them and the writer is free to
/// reshape the tree behind it. The next leaf is found by seeking again from the
/// root, past the last key that was yielded.
pub struct Iter<V> {
    root: Link<V>,
    /// Where the next seek starts. `None` once the walk has run off the end.
    from: Option<Box<[u8]>>,
    taken: std::vec::IntoIter<Entry<V>>,
}

impl<V: Clone> Iterator for Iter<V> {
    type Item = Entry<V>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entry) = self.taken.next() {
                return Some(entry);
            }
            // A seek that lands past the end of its leaf takes nothing, and
            // the leaf's right fence says where to look instead.
            let (taken, fence) = batch(&self.root, &self.from.take()?);
            self.from = taken.last().map_or(fence, |(key, _)| Some(after(key)));
            self.taken = taken.into_iter();
        }
    }
}
