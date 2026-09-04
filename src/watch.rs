//! Change notification for one tree node.
//!
//! A node holds a cell; a commit that replaces the node closes it. A `Watch`
//! taken before the commit then reports the node as changed, so a reader that
//! registers and only later awaits never misses the change.
//!
//! The channel behind a cell is allocated on the first `Watch`, so a tree
//! nobody watches allocates none. A cell that is closed before it ever had a
//! channel keeps the fact in its state: a reader on an old snapshot that
//! subscribes only afterwards gets a `Watch` that is already complete, which no
//! later commit would ever close for it.

use std::sync::Mutex;

use tokio::sync::watch;

/// One node's notification cell.
#[derive(Debug, Default)]
pub(crate) struct Cell {
    state: Mutex<State>,
}

/// A cell holds no channel until the first watcher asks for one, and none again
/// once it is closed.
#[derive(Debug, Default)]
enum State {
    #[default]
    Fresh,
    Watched(watch::Sender<bool>),
    Closed,
}

impl Cell {
    /// A watch on this cell, already complete if the node is gone.
    pub(crate) fn watch(&self) -> Watch {
        // One lock covers both the channel and the closed flag, so a `watch`
        // and a `close` that race cannot each miss the other.
        let mut state = self.state.lock().expect("a cell holds no panicking code");
        match &*state {
            State::Watched(sender) => Watch(sender.subscribe()),
            // The sender is dropped with the channel it was made for: the
            // receiver already carries the closed value.
            State::Closed => Watch(watch::channel(true).1),
            State::Fresh => {
                let (sender, receiver) = watch::channel(false);
                *state = State::Watched(sender);
                Watch(receiver)
            }
        }
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
        let mut state = self.state.lock().expect("a cell holds no panicking code");
        if let State::Watched(sender) = std::mem::replace(&mut *state, State::Closed) {
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
