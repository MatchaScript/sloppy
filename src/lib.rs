//! In-memory keyspace: single writer, non-blocking snapshot reads,
//! revision-indexed change enumeration, and per-prefix watches.

pub mod tree;
pub mod watch;
