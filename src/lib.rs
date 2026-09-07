//! In-memory keyspace: single writer, non-blocking snapshot reads,
//! revision-indexed change enumeration, and watches on a table or the
//! database.

pub mod db;
pub mod tree;
pub mod watch;
