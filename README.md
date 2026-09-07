# sloppy

In-memory keyspace with snapshot reads, revision-indexed changes, and watches.
Tables hold values behind a persistent adaptive radix tree; a write transaction
builds a new root and publishes it, readers keep the root they opened.

## Notice

The table, revision index, graveyard, and watch design follows
[cilium/statedb](https://github.com/cilium/statedb) (Apache-2.0), reimplemented
here in safe Rust. The split of a commit into prepare and publish follows the
commit pipeline of [cockroachdb/pebble](https://github.com/cockroachdb/pebble)
(BSD-3-Clause). No code is copied from either.
