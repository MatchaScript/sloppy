//! In-memory keyspace: single writer, non-blocking snapshot reads,
//! revision-indexed change enumeration, and watches on a table or the
//! database.

pub mod db;
/// The snapshot tree `db` still reads and writes, until its rows carry their
/// own versions and it moves onto [`tree`].
mod snapshot_tree;
pub mod tree;
pub mod watch;
