//! Change notification for one tree node.
//!
//! A node holds a cell; a commit that replaces the node closes it. A `Watch`
//! taken before the commit then reports the node as changed, so a reader that
//! registers and only later awaits never misses the change.
//!
//! The channel behind a cell is allocated on the first `Watch`, so a tree
//! nobody watches allocates none. A cell that is closed before it ever had a
//! channel keeps the fact in `replaced`: a reader on an old snapshot that
//! subscribes only afterwards gets a `Watch` that is already complete, which no
//! later commit would ever close for it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::watch;

/// One node's notification cell.
#[derive(Debug, Default)]
pub(crate) struct Cell {
    replaced: AtomicBool,
    sender: OnceLock<watch::Sender<bool>>,
}

impl Cell {
    /// A watch on this cell, already complete if the node is gone.
    pub(crate) fn watch(&self) -> Watch {
        let sender = self.sender.get_or_init(|| watch::channel(false).0);
        let watch = Watch(sender.subscribe());
        // Read after subscribing, against a `close` that writes the flag before
        // it looks for the channel: whichever of the two runs second sees the
        // work of the first.
        if self.replaced.load(Ordering::Acquire) {
            sender.send_replace(true);
        }
        watch
    }
}

/// What a commit completes once the new root is in place: the cell of a node it
/// replaced, or the cell of a value it overwrote.
pub(crate) trait Closes {
    fn close(&self);
}

/// The handle a commit holds until then.
pub(crate) type Closing = std::sync::Arc<dyn Closes>;

/// The cells one commit owes, closed once the new root is in place.
#[derive(Default)]
pub struct Closed(Vec<Closing>);

impl Closed {
    pub(crate) fn push(&mut self, cell: Closing) {
        self.0.push(cell);
    }

    /// Takes over another batch: one commit spans several trees.
    pub fn absorb(&mut self, other: Self) {
        self.0.extend(other.0);
    }

    /// Completes every watch taken on them.
    pub fn close(self) {
        for cell in self.0 {
            cell.close();
        }
    }
}

impl Closes for Cell {
    fn close(&self) {
        self.replaced.store(true, Ordering::Release);
        if let Some(sender) = self.sender.get() {
            sender.send_replace(true);
        }
    }
}

/// Completes once the watched node is replaced.
#[derive(Clone, Debug)]
pub struct Watch(watch::Receiver<bool>);

impl Watch {
    /// Waits until the watched node is replaced. Returns at once if it already was.
    pub async fn changed(&mut self) {
        // A dropped cell counts as changed: the node is gone.
        let _ = self.0.wait_for(|closed| *closed).await;
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        *self.0.borrow() || self.0.has_changed().is_err()
    }
}
