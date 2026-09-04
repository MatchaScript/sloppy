//! Persistent radix tree keyed by byte strings.
//!
//! A write path-copies: every node from the root down to the change is rebuilt,
//! everything else stays shared through `Arc`. The full key of a node is the
//! concatenation of the prefixes on the path from the root. A node this
//! transaction already rebuilt is mutated in place instead, which is what the
//! `txn` stamp is for.

use std::array;
use std::cmp::Ordering;
use std::iter;
use std::ops::Deref;
use std::sync::Arc;

use crate::watch::{Cell, Closed, Closes, Watch};

/// Identifies the transaction that built a node. `0` is "no transaction".
type TxnId = u64;

/// How much prefix fits in a node. The heap variant is a 16-byte fat pointer
/// beside a tag, so everything up to 22 bytes rides along for free.
const INLINE: usize = 22;

/// One node's compressed prefix, kept in the node while it fits, which for real
/// keys is nearly always. A descent then compares the prefix without following
/// a pointer out of the node.
#[derive(Clone)]
enum Prefix {
    Inline { len: u8, bytes: [u8; INLINE] },
    Heap(Box<[u8]>),
}

impl Prefix {
    fn new(prefix: &[u8]) -> Self {
        match u8::try_from(prefix.len()) {
            Ok(len) if prefix.len() <= INLINE => {
                let mut bytes = [0; INLINE];
                bytes[..prefix.len()].copy_from_slice(prefix);
                Self::Inline { len, bytes }
            }
            _ => Self::Heap(prefix.into()),
        }
    }
}

impl Deref for Prefix {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Inline { len, bytes } => &bytes[..usize::from(*len)],
            Self::Heap(prefix) => prefix,
        }
    }
}

// ponytail: four node kinds, and nothing below them - no SIMD key compare, no
// separate leaf node. The kind follows from the child count, so a path copy
// picks it and growth and shrink need no code of their own.
struct Node<V> {
    prefix: Prefix,
    value: Option<Arc<V>>,
    children: Children<V>,
    /// Closed by every commit that rebuilds this node, so it covers the whole
    /// subtree. It sits in the node, and goes with it.
    subtree: Cell,
    /// Closed only by a commit that writes or removes this node's own value, so
    /// it outlives the node and is shared by every copy that keeps the value.
    value_cell: Arc<Cell>,
    /// The transaction that created this node. While that transaction runs, the
    /// node is reachable only from its root, so it can be mutated in place.
    txn: TxnId,
}

impl<V> Node<V> {
    /// The value is new here, so its cell starts fresh.
    fn new(prefix: &[u8], value: Option<Arc<V>>, children: Children<V>, txn: TxnId) -> Arc<Self> {
        Arc::new(Self {
            prefix: Prefix::new(prefix),
            value,
            children,
            subtree: Cell::default(),
            value_cell: Arc::default(),
            txn,
        })
    }

    /// Path copy: the value comes over unchanged and keeps its cell, so a
    /// `Watch` taken on the old node still tracks it.
    fn copy(&self, txn: TxnId) -> Arc<Self> {
        Arc::new(Self {
            prefix: self.prefix.clone(),
            value: self.value.clone(),
            children: self.children.clone(),
            subtree: Cell::default(),
            value_cell: self.value_cell.clone(),
            txn,
        })
    }
}

/// Mutation keeps a replaced node alive until the commit closes its cell, which
/// is why these need `V` to outlive the transaction.
impl<V: 'static> Node<V> {
    /// The node this transaction may mutate: `*node` itself when this
    /// transaction built it, a path copy of it otherwise.
    ///
    /// The subtree cell that a reader can be holding is the one of the node
    /// that was there before this transaction started, and it is closed here,
    /// once, when that node is copied. A node this transaction built has never
    /// been published, so nobody can hold its cell and mutating it in place
    /// loses no notification.
    fn own<'a>(node: &'a mut Arc<Self>, w: &mut Writing) -> &'a mut Self {
        if node.txn != w.id {
            w.closed.push(node.clone());
            *node = node.copy(w.id);
        }
        Arc::get_mut(node).expect("a node stamped with this transaction is not shared")
    }
}

impl<V> Node<V> {
    /// The byte a parent files this node under.
    fn key(&self) -> u8 {
        self.prefix[0]
    }
}

/// Dropping the last reference to a deep chain must not recurse down it: the
/// depth is the keys' owner's to pick. Children this node held alone are
/// unlinked here and dropped one at a time, so each of their drops is shallow.
impl<V> Drop for Node<V> {
    fn drop(&mut self) {
        let mut stack = Vec::new();
        std::mem::replace(&mut self.children, Children::empty()).drain_into(&mut stack);
        while let Some(mut child) = stack.pop() {
            if let Some(child) = Arc::get_mut(&mut child) {
                std::mem::replace(&mut child.children, Children::empty()).drain_into(&mut stack);
            }
        }
    }
}

/// A replaced node closes its subtree cell, and holds it alive until then.
impl<V> Closes for Node<V> {
    fn close(&self) {
        self.subtree.close();
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

impl<V, const K: usize> Clone for Sorted<V, K> {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys,
            slots: array::from_fn(|i| self.slots[i].clone()),
            len: self.len,
        }
    }
}

impl<V> Clone for Node48<V> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            slots: array::from_fn(|i| self.slots[i].clone()),
            len: self.len,
        }
    }
}

impl<V> Clone for Node256<V> {
    fn clone(&self) -> Self {
        Self {
            slots: array::from_fn(|i| self.slots[i].clone()),
            len: self.len,
        }
    }
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

    fn keys(&self) -> &[u8] {
        &self.keys[..usize::from(self.len)]
    }

    fn get(&self, byte: u8) -> Option<&Arc<Node<V>>> {
        let at = self.keys().iter().position(|k| *k == byte)?;
        self.slots[at].as_ref()
    }

    fn slot_mut(&mut self, byte: u8) -> Option<&mut Slot<V>> {
        let at = self.keys().iter().position(|k| *k == byte)?;
        Some(&mut self.slots[at])
    }

    /// Where `byte` sits, or where it would go.
    fn find(&self, byte: u8) -> Result<usize, usize> {
        let at = self.keys().partition_point(|k| *k < byte);
        if self.keys().get(at) == Some(&byte) {
            Ok(at)
        } else {
            Err(at)
        }
    }

    /// A copy with `child` filed under `byte`, which must be absent. There
    /// must be room for it.
    fn with(&self, byte: u8, child: Arc<Node<V>>) -> Self {
        let mut this = self.clone();
        let at = this.find(byte).expect_err("the byte is absent");
        let end = usize::from(this.len);
        this.keys[at..=end].rotate_right(1);
        this.slots[at..=end].rotate_right(1);
        this.keys[at] = byte;
        this.slots[at] = Some(child);
        this.len += 1;
        this
    }

    /// A copy without the child at slot `at`.
    fn without(&self, at: usize) -> Self {
        let mut this = self.clone();
        let end = usize::from(this.len);
        this.keys[at..end].rotate_left(1);
        this.slots[at..end].rotate_left(1);
        this.keys[end - 1] = 0;
        this.slots[end - 1] = None;
        this.len -= 1;
        this
    }
}

impl<V> Node48<V> {
    /// A copy with `child` filed under `byte`, which must be absent. There
    /// must be room for it.
    fn with(&self, byte: u8, child: Arc<Node<V>>) -> Self {
        let mut this = self.clone();
        // The slots stay in key order, so the new one lands after every child
        // below `byte` and the index entries behind it move up.
        let at = this.index[..usize::from(byte)]
            .iter()
            .filter(|slot| **slot != 0)
            .count();
        let end = usize::from(this.len);
        this.slots[at..=end].rotate_right(1);
        for slot in &mut this.index {
            if usize::from(*slot) > at {
                *slot += 1;
            }
        }
        this.index[usize::from(byte)] = u8::try_from(at + 1).expect("node48 holds 48 slots");
        this.slots[at] = Some(child);
        this.len += 1;
        this
    }

    /// A copy without the child at `byte`, which must be there.
    fn without(&self, byte: u8) -> Self {
        let mut this = self.clone();
        let at = usize::from(this.index[usize::from(byte)] - 1);
        let end = usize::from(this.len);
        this.slots[at..end].rotate_left(1);
        this.slots[end - 1] = None;
        this.index[usize::from(byte)] = 0;
        for slot in &mut this.index {
            if usize::from(*slot) > at + 1 {
                *slot -= 1;
            }
        }
        this.len -= 1;
        this
    }
}

impl<V> Node256<V> {
    /// A copy with `child` filed under `byte`, which must be absent.
    fn with(&self, byte: u8, child: Arc<Node<V>>) -> Self {
        let mut this = self.clone();
        this.slots[usize::from(byte)] = Some(child);
        this.len += 1;
        this
    }

    fn without(&self, byte: u8) -> Self {
        let mut this = self.clone();
        this.slots[usize::from(byte)] = None;
        this.len -= 1;
        this
    }
}

impl<V> Clone for Children<V> {
    fn clone(&self) -> Self {
        match self {
            Self::N4(s) => Self::N4(s.clone()),
            Self::N16(s) => Self::N16(s.clone()),
            Self::N48(n) => Self::N48(n.clone()),
            Self::N256(n) => Self::N256(n.clone()),
        }
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

    /// How many children this kind holds before the next one takes over.
    fn cap(&self) -> usize {
        match self {
            Self::N4(_) => 4,
            Self::N16(_) => 16,
            Self::N48(_) => 48,
            Self::N256(_) => 256,
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

    /// The slot filed under `byte`. Only `N256` has one for an absent child.
    fn slot_mut(&mut self, byte: u8) -> Option<&mut Slot<V>> {
        match self {
            Self::N4(s) => s.slot_mut(byte),
            Self::N16(s) => s.slot_mut(byte),
            Self::N48(n) => match n.index[usize::from(byte)] {
                0 => None,
                slot => Some(&mut n.slots[usize::from(slot - 1)]),
            },
            Self::N256(n) => Some(&mut n.slots[usize::from(byte)]),
        }
    }

    fn get_mut(&mut self, byte: u8) -> Option<&mut Arc<Node<V>>> {
        self.slot_mut(byte).and_then(Option::as_mut)
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

    /// Adds `child` under its own key, which must be absent.
    ///
    /// Inside one kind this copies the arrays and shifts one position: the keys
    /// are a memcpy and the slots one `Arc` bump per child, with no child
    /// dereferenced. Only a count that crosses a kind boundary rebuilds.
    fn with(&self, child: Arc<Node<V>>) -> Self {
        let byte = child.key();
        debug_assert!(self.get(byte).is_none(), "the key is absent");
        if self.len() == self.cap() {
            let (below, above) = self.split(byte);
            return Self::build(
                self.len() + 1,
                below
                    .iter()
                    .flatten()
                    .cloned()
                    .chain(iter::once(child))
                    .chain(above.iter().flatten().cloned()),
            );
        }
        match self {
            Self::N4(s) => Self::N4(s.with(byte, child)),
            Self::N16(s) => Self::N16(Box::new(s.with(byte, child))),
            Self::N48(n) => Self::N48(Box::new(n.with(byte, child))),
            Self::N256(n) => Self::N256(Box::new(n.with(byte, child))),
        }
    }

    /// Removes the child at `byte`, which must be there.
    fn without(&self, byte: u8) -> Self {
        let len = self.len();
        let demotes = match self {
            Self::N4(_) => false,
            Self::N16(_) => len == 5,
            Self::N48(_) => len == 17,
            Self::N256(_) => len == 49,
        };
        if demotes {
            let (below, above) = self.split(byte);
            return Self::build(
                len - 1,
                below
                    .iter()
                    .flatten()
                    .chain(above.iter().flatten())
                    .cloned(),
            );
        }
        match self {
            Self::N4(s) => Self::N4(s.without(s.find(byte).expect("the child is there"))),
            Self::N16(s) => Self::N16(Box::new(
                s.without(s.find(byte).expect("the child is there")),
            )),
            Self::N48(n) => Self::N48(Box::new(n.without(byte))),
            Self::N256(n) => Self::N256(Box::new(n.without(byte))),
        }
    }

    /// Moves every child into `out`.
    fn drain_into(self, out: &mut Vec<Arc<Node<V>>>) {
        match self {
            Self::N4(s) => out.extend(s.slots.into_iter().flatten()),
            Self::N16(s) => out.extend(s.slots.into_iter().flatten()),
            Self::N48(n) => out.extend(n.slots.into_iter().flatten()),
            Self::N256(n) => out.extend(n.slots.into_iter().flatten()),
        }
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
                let listed = n.slots.iter().flatten().count();
                assert_eq!(listed, len, "node256 holds {listed} of {len} children");
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
        assert!(
            self.slots[usize::from(self.len)..]
                .iter()
                .all(Option::is_none),
            "sorted node has children past its len"
        );
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
            root: Node::new(&[], None, Children::empty(), 0),
            len: 0,
            last_txn: 0,
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
        self.root.subtree.watch()
    }

    /// The value at `key`, without watching anything: a reader that would drop
    /// the `Watch` should not make the cell allocate a channel.
    #[must_use]
    pub fn value(&self, key: &[u8]) -> Option<&Arc<V>> {
        descend(&self.root, key).0
    }

    /// The value at `key`, plus a watch on that value alone - or, when `key`
    /// is absent, on the whole subtree of the deepest node the descent reached,
    /// which is where the key would appear.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> (Option<&Arc<V>>, Watch) {
        let (value, node) = descend(&self.root, key);
        let watch = if value.is_some() {
            node.value_cell.watch()
        } else {
            node.subtree.watch()
        };
        (value, watch)
    }

    /// Every entry whose key starts with `p`, plus the watch of the node the
    /// descent along `p` reaches.
    #[must_use]
    pub fn prefix(&self, p: &[u8]) -> Iter<'_, V> {
        self.prefix_covering(p).0
    }

    /// The same, plus the watch of the node the descent along `p` reaches.
    #[must_use]
    pub fn prefix_watch(&self, p: &[u8]) -> (Iter<'_, V>, Watch) {
        let (iter, node) = self.prefix_covering(p);
        (iter, node.subtree.watch())
    }

    /// The entries under `p` and the node covering them, without a watch.
    fn prefix_covering(&self, p: &[u8]) -> (Iter<'_, V>, &Node<V>) {
        let mut node: &Node<V> = &self.root;
        let mut key = Vec::new();
        let mut rest = p;
        loop {
            if node.prefix.len() >= rest.len() {
                let stack = if node.prefix.starts_with(rest) {
                    vec![Frame {
                        node,
                        len: key.len(),
                    }]
                } else {
                    Vec::new()
                };
                return (Iter { path: key, stack }, node);
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
        (
            Iter {
                path: Vec::new(),
                stack: Vec::new(),
            },
            node,
        )
    }

    /// Every entry with a key `>= key`, in order. Descends along `key` and
    /// seeds the stack with the subtrees that are entirely `>= key`.
    #[must_use]
    pub fn lower_bound(&self, key: &[u8]) -> Iter<'_, V> {
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
                    stack.push(Frame {
                        node,
                        len: acc.len(),
                    });
                    break;
                }
                Ordering::Equal => {}
            }
            if node.prefix.len() >= rest.len() {
                // `key` runs out inside this node, so its whole subtree is >= key.
                stack.push(Frame {
                    node,
                    len: acc.len(),
                });
                break;
            }
            // This node's own key is a strict prefix of `key`, so it sorts below
            // `key` and is skipped, as are the children before `rest[0]`.
            acc.extend_from_slice(&node.prefix);
            rest = &rest[node.prefix.len()..];
            for child in node.children.split(rest[0]).1.iter().rev().flatten() {
                stack.push(Frame {
                    node: child,
                    len: acc.len(),
                });
            }
            match node.children.get(rest[0]) {
                Some(child) => node = child,
                None => break,
            }
        }
        Iter { path: acc, stack }
    }

    /// The same, plus the root watch: an entry can enter the range anywhere.
    #[must_use]
    pub fn lower_bound_watch(&self, key: &[u8]) -> (Iter<'_, V>, Watch) {
        (self.lower_bound(key), self.root_watch())
    }

    /// Every entry, in ascending key order.
    #[allow(clippy::iter_without_into_iter)] // `IntoIterator` for `&Tree` has no user yet.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, V> {
        Iter {
            path: Vec::new(),
            stack: vec![Frame {
                node: &self.root,
                len: 0,
            }],
        }
    }

    #[must_use]
    pub fn txn(&self) -> Txn<V> {
        Txn {
            root: self.root.clone(),
            len: self.len,
            w: Writing {
                id: self.last_txn + 1,
                closed: Closed::default(),
            },
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
        assert!(self.root.prefix.is_empty(), "non-empty root prefix");
        let mut values = 0;
        let mut stack = vec![self.root.as_ref()];
        while let Some(node) = stack.pop() {
            node.children.assert_ok();
            values += usize::from(node.value.is_some());
            for child in node.children.iter() {
                assert!(!child.prefix.is_empty(), "empty prefix below the root");
                assert!(
                    child.value.is_some() || child.children.len() >= 2,
                    "node with no value and fewer than two children"
                );
                stack.push(child);
            }
        }
        assert_eq!(values, self.len, "tree len disagrees with its values");
    }
}

struct Frame<'a, V> {
    node: &'a Node<V>,
    /// Length of the walk's path buffer up to this node's parent, so the node's
    /// own key is that much of the buffer plus its prefix.
    len: usize,
}

/// Yields `(key, value)` in ascending byte-lexicographic order.
pub struct Iter<'a, V> {
    /// The key of the node the walk is at, rewound to each frame's `len`.
    path: Vec<u8>,
    stack: Vec<Frame<'a, V>>,
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = (Vec<u8>, &'a Arc<V>);

    // ponytail: one key `Vec` per yielded entry, none per visited node. Hand out
    // a borrow of the path instead if even that ever matters.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(frame) = self.stack.pop() {
            self.path.truncate(frame.len);
            self.path.extend_from_slice(&frame.node.prefix);
            let len = self.path.len();
            for child in frame.node.children.iter().rev() {
                self.stack.push(Frame { node: child, len });
            }
            if let Some(value) = frame.node.value.as_ref() {
                return Some((self.path.clone(), value));
            }
        }
        None
    }
}

/// What one transaction carries down the tree: its id and the cells it owes the
/// caller.
struct Writing {
    id: TxnId,
    closed: Closed,
}

/// A batch of writes over one snapshot. Dropping it aborts.
pub struct Txn<V> {
    root: Arc<Node<V>>,
    len: usize,
    w: Writing,
}

impl<V: 'static> Txn<V> {
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&Arc<V>> {
        descend(&self.root, key).0
    }

    /// Returns the replaced value.
    pub fn insert(&mut self, key: &[u8], value: V) -> Option<Arc<V>> {
        let old = insert(&mut self.root, key, Arc::new(value), &mut self.w);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub fn delete(&mut self, key: &[u8]) -> Option<Arc<V>> {
        // The descent below rebuilds as it goes, so the key is looked up first:
        // a miss must leave the tree alone.
        descend(&self.root, key).0?;
        let old = delete(&mut self.root, key, &mut self.w);
        self.len -= 1;
        Some(old)
    }

    /// The new tree and the cells the caller must close.
    #[must_use]
    pub fn commit(self) -> (Tree<V>, Closed) {
        (
            Tree {
                root: self.root,
                len: self.len,
                last_txn: self.w.id,
            },
            self.w.closed,
        )
    }

    #[must_use]
    pub fn commit_and_notify(self) -> Tree<V> {
        let (tree, closed) = self.commit();
        closed.close();
        tree
    }
}

/// Writes `value` at `key`, taken relative to `*node`, and replaces `*node`
/// with the node that takes its place.
///
/// A loop, not a recursion: the depth is the length of the longest chain of
/// keys that are each a prefix of the next, which the keys' owner controls.
fn insert<V: 'static>(
    mut node: &mut Arc<Node<V>>,
    mut key: &[u8],
    value: Arc<V>,
    w: &mut Writing,
) -> Option<Arc<V>> {
    loop {
        let common = lcp(&node.prefix, key);

        if common < node.prefix.len() {
            // The key diverges inside the prefix: this node keeps its subtree as
            // the tail of a split, and a new node takes its place above.
            let tail = {
                let n = Node::own(node, w);
                n.prefix = Prefix::new(&n.prefix[common..]);
                node.clone()
            };
            *node = if common == key.len() {
                Node::new(&key[..common], Some(value), Children::one(tail), w.id)
            } else {
                let leaf = Node::new(&key[common..], Some(value), Children::empty(), w.id);
                Node::new(&key[..common], None, Children::pair(tail, leaf), w.id)
            };
            return None;
        }

        if common == key.len() {
            // On a valueless node this closes a cell nobody holds: `get` hands out
            // the value cell only for a key that exists, and the watch on the
            // missing key is the subtree cell that `own` closes.
            w.closed.push(node.value_cell.clone());
            let n = Node::own(node, w);
            n.value_cell = Arc::default();
            return n.value.replace(value);
        }

        let rest = &key[common..];
        let n = Node::own(node, w);
        // Looked up twice: the loop hands `node` on to the child's slot, which
        // a borrow held across the miss branch would not allow.
        if n.children.get(rest[0]).is_none() {
            n.children = n
                .children
                .with(Node::new(rest, Some(value), Children::empty(), w.id));
            return None;
        }
        node = n.children.get_mut(rest[0]).expect("checked above");
        key = rest;
    }
}

/// Removes `key`, which must be present, and replaces `*root` with the root
/// that takes its place.
///
/// The way down owns every node on the path and takes each next node out of
/// its parent's slot, so the way back up holds one node at a time: it puts the
/// child back, drops it if it is gone, and shrinks the parent.
fn delete<V: 'static>(root: &mut Arc<Node<V>>, key: &[u8], w: &mut Writing) -> Arc<V> {
    let mut node = std::mem::replace(root, Node::new(&[], None, Children::empty(), 0));
    let mut rest = key;
    let mut path: Vec<(Arc<Node<V>>, u8)> = Vec::new();
    loop {
        rest = &rest[node.prefix.len()..];
        let Some(&byte) = rest.first() else { break };
        let child = Node::own(&mut node, w)
            .children
            .slot_mut(byte)
            .and_then(Option::take)
            .expect("the key was looked up first");
        path.push((node, byte));
        node = child;
    }

    w.closed.push(node.value_cell.clone());
    let n = Node::own(&mut node, w);
    n.value_cell = Arc::default();
    let old = n.value.take().expect("the key was looked up first");

    let mut gone = shrink(&mut node, path.is_empty(), w);
    while let Some((mut parent, byte)) = path.pop() {
        let p = Arc::get_mut(&mut parent).expect("owned on the way down");
        *p.children.slot_mut(byte).expect("taken on the way down") = Some(node);
        if gone {
            p.children = p.children.without(byte);
        }
        node = parent;
        gone = shrink(&mut node, path.is_empty(), w);
    }
    *root = node;
    old
}

/// Finishes a node that just lost its value or a child: a node with no value
/// and no children is gone, one with no value and a single child merges into
/// that child. The root always stays, with an empty prefix.
///
/// Returns whether the node is gone.
fn shrink<V: 'static>(node: &mut Arc<Node<V>>, is_root: bool, w: &mut Writing) -> bool {
    if is_root || node.value.is_some() {
        return false;
    }
    let Some(only) = node.children.only().cloned() else {
        return node.children.len() == 0;
    };
    let mut merged = node.prefix.to_vec();
    merged.extend_from_slice(&only.prefix);
    // The parent goes away with the assignment, which leaves the child
    // unshared, so `own` mutates it in place when this transaction built it.
    *node = only;
    Node::own(node, w).prefix = Prefix::new(&merged);
    false
}

#[cfg(test)]
mod layout {
    use super::{Cell, Children, Node, Prefix};

    /// `cargo test -- --nocapture node_layout` prints the sizes this file is
    /// laid out around.
    #[test]
    fn node_layout() {
        println!(
            "Node<u64> {}, Prefix {}, Children<u64> {}, Cell {}",
            size_of::<Node<u64>>(),
            size_of::<Prefix>(),
            size_of::<Children<u64>>(),
            size_of::<Cell>(),
        );
        assert_eq!(
            size_of::<Prefix>(),
            24,
            "the inline prefix should fill the heap variant, no more"
        );
        assert!(size_of::<Node<u64>>() <= 120, "the node grew a word");
    }
}

#[cfg(test)]
mod lookup {
    use super::{Tree, descend};

    /// The read paths that drop the watch go through `value`, which must leave
    /// the cell without a channel: one allocation per row otherwise, kept for
    /// as long as the cell rides across node rebuilds.
    #[test]
    fn a_plain_lookup_allocates_no_channel() {
        let mut txn = Tree::new().txn();
        txn.insert(b"a", 1u64);
        let tree = txn.commit_and_notify();

        assert_eq!(tree.value(b"a").map(|v| **v), Some(1));
        let node = descend(&tree.root, b"a").1;
        assert!(!node.value_cell.has_channel(), "`value` watches nothing");

        let (_, watch) = tree.get(b"a");
        assert!(node.value_cell.has_channel(), "`get` hands out a watch");
        drop(watch);
    }
}
