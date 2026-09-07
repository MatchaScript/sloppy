# sloppy

In-memory keyspace with snapshot reads, revision-indexed changes, and watches.
Tables hold values behind a persistent adaptive radix tree; a write transaction
builds a new root and publishes it, readers keep the root they opened.

## Notice

Design follows [cilium/statedb](https://github.com/cilium/statedb) (Apache-2.0)
and [cockroachdb/pebble](https://github.com/cockroachdb/pebble) (BSD-3-Clause).
No code is copied from either.
