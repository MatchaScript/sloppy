//! Change notification.
//!
//! One channel carries the revision of the visible root, so every commit wakes
//! every waiting reader once. A `Watch` then asks what it covers whether it has
//! changed since the watch was taken, and waits again if it has not.
//!
//! Nothing is registered anywhere, so a watch taken on a snapshot the database
//! has already moved past reports the change it missed rather than waiting for
//! a commit that will not come.

use std::sync::Arc;

use tokio::sync::watch;

/// What one `Watch` covers, asked again on every revision. The database
/// implements it: a revision means nothing here.
pub(crate) trait Covered: Send + Sync {
    /// Whether what this covers differs from what it held when the watch was
    /// taken.
    fn changed(&self) -> bool;
}

/// Completes once what it covers is written.
#[derive(Clone)]
pub struct Watch {
    covered: Arc<dyn Covered>,
    revisions: watch::Receiver<u64>,
}

impl Watch {
    pub(crate) fn new(covered: Arc<dyn Covered>, revisions: watch::Receiver<u64>) -> Self {
        Self { covered, revisions }
    }

    /// Waits until what this covers is written. Returns at once if it already
    /// was, and when the database it came from is gone.
    pub async fn changed(&mut self) {
        while !self.covered.changed() {
            if self.revisions.changed().await.is_err() {
                return;
            }
        }
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.covered.changed() || self.revisions.has_changed().is_err()
    }
}
