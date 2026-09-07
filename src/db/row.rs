use std::sync::Arc;

use super::Revision;

/// One version of one key. A deletion keeps the value it removed, so a reader
/// of the change stream is told what went away.
pub struct Version<V> {
    pub revision: Revision,
    pub value: Arc<V>,
    pub deleted: bool,
}

/// Cloning a version is one `Arc` bump. Derived would ask for `V: Clone`,
/// which no table needs.
impl<V> Clone for Version<V> {
    fn clone(&self) -> Self {
        Self {
            revision: self.revision,
            value: self.value.clone(),
            deleted: self.deleted,
        }
    }
}

/// One version and everything older than it.
struct Link<V> {
    version: Version<V>,
    prev: Option<Arc<Link<V>>>,
}

/// A row: the head of its version chain. A write is one allocation, the link
/// that points at what was there.
pub(super) struct Row<V> {
    head: Arc<Link<V>>,
    /// The revision of the oldest version on the chain. A write that finds it
    /// above the compaction bound has nothing to trim and skips the walk.
    tail: Revision,
}

/// A reader clones the head under the leaf's lock and walks the chain once it
/// has let the lock go.
impl<V> Clone for Row<V> {
    fn clone(&self) -> Self {
        Self {
            head: self.head.clone(),
            tail: self.tail,
        }
    }
}

impl<V> Row<V> {
    /// The row `version` leaves on top of this chain.
    pub(super) fn written(head: Option<Self>, version: Version<V>, compacted: Revision) -> Self {
        let (prev, tail) = match head {
            None => (None, version.revision),
            Some(row) if row.tail > compacted => (Some(row.head), row.tail),
            Some(row) => {
                let (link, tail) = trimmed(&row.head, compacted);
                (Some(link), tail)
            }
        };
        Self {
            head: Arc::new(Link { version, prev }),
            tail,
        }
    }

    /// The chain `versions` makes, which are in ascending revision order.
    pub(super) fn loaded(versions: &[Version<V>]) -> Self {
        let mut rest = versions.iter();
        let oldest = rest.next().expect("a loaded row holds a version");
        let tail = oldest.revision;
        let mut head = Arc::new(Link {
            version: oldest.clone(),
            prev: None,
        });
        for version in rest {
            head = Arc::new(Link {
                version: version.clone(),
                prev: Some(head),
            });
        }
        Self { head, tail }
    }

    /// The version at the head of the chain.
    pub(super) fn newest(&self) -> &Version<V> {
        &self.head.version
    }

    /// The chain from the newest version down.
    fn links(&self) -> impl Iterator<Item = &Link<V>> {
        let mut next = Some(&self.head);
        std::iter::from_fn(move || {
            let link = next?;
            next = link.prev.as_ref();
            Some(&**link)
        })
    }

    /// The newest version at or below `at`. A row whose every version is above
    /// it was written after the reader's revision and is not there yet.
    pub(super) fn at(&self, at: Revision) -> Option<&Version<V>> {
        self.links()
            .find(|link| link.version.revision <= at)
            .map(|link| &link.version)
    }

    /// The version one commit left, if the row still holds it.
    pub(super) fn exactly(&self, revision: Revision) -> Option<&Version<V>> {
        self.links()
            .take_while(|link| link.version.revision >= revision)
            .find(|link| link.version.revision == revision)
            .map(|link| &link.version)
    }

    /// The value at `at`, unless the version there is a tombstone.
    pub(super) fn live(&self, at: Revision) -> Option<(Arc<V>, Revision)> {
        self.at(at)
            .filter(|version| !version.deleted)
            .map(|version| (version.value.clone(), version.revision))
    }

    /// The value the newest version holds: what a write hands back and what the
    /// indexes list.
    pub(super) fn held(&self) -> Option<&Arc<V>> {
        (!self.head.version.deleted).then_some(&self.head.version.value)
    }

    /// The revision of the tombstone this row ends with, if it ends with one.
    pub(super) fn tombstoned(&self) -> Option<Revision> {
        self.head
            .version
            .deleted
            .then_some(self.head.version.revision)
    }

    /// Every version at or below `at`, oldest first.
    pub(super) fn versions(&self, at: Revision) -> Vec<Version<V>> {
        let mut versions: Vec<Version<V>> = self
            .links()
            .skip_while(|link| link.version.revision > at)
            .map(|link| link.version.clone())
            .collect();
        versions.reverse();
        versions
    }
}

/// The chain from `head` down to the first version at or below `compacted`,
/// which ends it: a reader at the compaction bound still reads the value that
/// version holds, and nothing below it may be asked for again. The links above
/// it are rebuilt, one allocation per write since the compaction; a chain that
/// already ends there is passed on as it is. Returns the chain and the
/// revision it now ends at.
fn trimmed<V>(head: &Arc<Link<V>>, compacted: Revision) -> (Arc<Link<V>>, Revision) {
    let mut tail = head;
    while tail.version.revision > compacted {
        match &tail.prev {
            Some(prev) => tail = prev,
            None => return (head.clone(), tail.version.revision),
        }
    }
    let bound = tail.version.revision;
    if tail.prev.is_none() {
        return (head.clone(), bound);
    }
    let mut rebuilt = Arc::new(Link {
        version: tail.version.clone(),
        prev: None,
    });
    let mut above = Vec::new();
    let mut link = head;
    while link.version.revision > bound {
        above.push(&link.version);
        link = link.prev.as_ref().expect("the bound is on the chain");
    }
    for version in above.into_iter().rev() {
        rebuilt = Arc::new(Link {
            version: version.clone(),
            prev: Some(rebuilt),
        });
    }
    (rebuilt, bound)
}
