//! Change notification for one tree node.
//!
//! A node holds a cell; a commit that replaces the node closes it. A `Watch`
//! taken before the commit then reports the node as changed, so a reader that
//! registers and only later awaits never misses the change.

use tokio::sync::watch;

/// One node's notification cell. Closing it is `send_replace(true)`.
pub(crate) type WatchCell = watch::Sender<bool>;

pub(crate) fn cell() -> WatchCell {
    watch::channel(false).0
}

/// Completes once the watched node is replaced.
#[derive(Clone, Debug)]
pub struct Watch(watch::Receiver<bool>);

impl Watch {
    pub(crate) fn new(cell: &WatchCell) -> Self {
        Self(cell.subscribe())
    }

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
