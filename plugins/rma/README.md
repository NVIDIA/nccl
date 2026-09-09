# RMA Developer Guide

## Architecture

The RMA plugin allows one-sided, host-driven network operations to be customized. It serves as
the network layer for GIN CPU Proxy and for one-sided NCCL operations (e.g. `ncclPutSignal`).

This document uses `ncclImpl` to refer to NCCL's internal (non-customized) implementations of these APIs.
References to `ncclImpl` are just examples, custom implementations may make other design decisions.

## Plugin architecture

### Plugin name and supporting multiple RMA plugins

When NCCL is initialized, it will look for a `libnccl-rma.so` library and
dynamically load it, then look for symbols inside the library.

The `NCCL_RMA_PLUGIN` environment variable allows users to select one or more
RMA plugins. If set to a bare plugin name such as `myrma`, NCCL can load a
library named `libnccl-rma-myrma.so`. `NCCL_RMA_PLUGIN` can also be set to a
shared library file name or an absolute path to the plugin file. Multiple
plugins can be specified as a comma-separated list.

For example, any of the following can be used to load a plugin:

```shell
export NCCL_RMA_PLUGIN=myrma
export NCCL_RMA_PLUGIN=libnccl-rma-myrma.so
export NCCL_RMA_PLUGIN=/path/to/your/plugin/libnccl-rma-myrma.so
```

Set the `LD_LIBRARY_PATH` to include the plugin directory when using a bare name
or shared library file name:

```shell
export LD_LIBRARY_PATH=/path/to/your/plugin:$LD_LIBRARY_PATH
```

If none of the listed plugins can be loaded, NCCL looks for the RMA symbols in
the GIN plugin and in the NET plugin, and finally falls back to its internal RMA
implementation over InfiniBand.

Only one RMA implementation is used: NCCL picks the first one that both succeeds
`init` and reports at least one device from `devices`, and disables the remaining
external plugins.

### Struct versioning

Once a library is found, NCCL will look for a symbol named `ncclRmaPlugin_vX`,
with `X` increasing over time. The versioning ensures that the plugin and NCCL
core are compatible.

Plugins are encouraged to provide multiple of those symbols, implementing
multiple versions of the NCCL RMA API, so that the same plugin can be compiled
and support a wide range of NCCL versions.

## RMA Concepts

### Core Concepts

**RMA connections and contexts**

A connection is a (src, dst) network device pair. An RMA context is a sub-resource of an
RMA connection; the plugin manages one or more *connections*, each of which has one or
more *contexts*. RMA contexts are allocated in blocks of varying sizes.

The primary goal of RMA contexts is to increase network performance by increasing
parallelism, with separate, parallel communication channels, which are independent
from each other ordering-wise. RMA contexts are typically used by different CUDA CTAs
processing data independently. In `ncclImpl`, each RMA context corresponds to a different
queue pair.

Each context manages a fixed number of RMA resources (e.g. counters, signals, barriers).
RMA ordering guarantees apply within a single context. For example, the completion
of a signal guarantees the completion of all previous puts *on the same context*.
For the remainder of this document, you may assume order requirements are per-context.

The number of connections is expected to be very low. As of this writing, the max is
4 and multiple connections are only created if a rank has affinity to multiple network
devices (e.g. dual port NICs). The typical number of contexts is \~4-256, although
nothing prevents applications from creating more or less.

### Operations

**Put**
Put moves data from a local source buffer to a (likely remote) target buffer. A single call to `Put`
may include a signal operation. See below for more details.

**Get**
Get moves data from a (likely remote) source buffer to a local target buffer. Gets are
not required to be completed in the order they are requested.

**Signal**
Signal increments a target memory address by a fixed value. There's 2 types of visibility guarantees:
strong and weak. The visibility of a *strong* signal guarantees the completion/visibility of all previous
puts and signals, including the bundled put in the case of put+signal. The visibility of a *weak*
signal guarantees the completion/visibility of the bundled put (in the case of put+signal), but makes
no guarantees on previous puts or previous signals.

**Flush**
Flush flushes the PCIe path from a NIC to a GPU. The completion of a flush must guarantee that the data
from all previous *completed* gets is visible on the GPU. Gets that are not complete at the time flush
is issued need not have data visible.

## `ncclRma` API

### Connection Setup

One `collComm` is created per RMA connection. A `collComm` manages connections to all
peers. A `collComm` is initialized in several steps:

1) NCCL calls `listen` on all ranks in the communicator. `listen` returns an opaque `listenComm` and `handle`.
2) NCCL exchanges handles among all peers.
3) NCCL calls `connect` on all ranks and includes a list of all peers' handles. `connect` returns an opaque `collComm`.

`ncclImpl` initializes a TCP socket in `listen` and stores it in `listenComm`. The address of the socket is included in `handle` so
that all peers can connect to the socket.

### Context Setup

RMA contexts are allocated in blocks via `createContext`. Each block of RMA contexts returns a
corresponding opaque `rmaCtx`.

### Memory registration

The RMA functions for memory registration are `regMrSym` and `regMrSymDmaBuf`. These functions
are called once per memory region, per connection.

RMA memory registration is symmetric. If a call to `regMrSym` is made on one rank, the plugin may
assume `regMrSym` is called on all ranks. The returned handles should have enough information
for the custom implementation to execute RMA operations on the local buffer and the
remote buffers of all peers.

Memory registration also accepts an optional `mrFlags` argument. If memory is registered
with `NCCL_NET_MR_FLAG_FORCE_SO`, the memory region may be used for signal operations. If
memory is registered with `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET`, the memory region is used
as a signal that starts at 0 and is never reset. Some implementations use the "never reset" flag
as a performance optimization as it guarantees the RMA plugin is the only entity that ever writes
to that MR.

### Data operations

RMA data operations are submitted via plugin functions (`iput`, `iputSignal`, etc.).
Refer to the comments in the `ncclRma_t` type for a more detailed specification of each data operation.

All plugin data operations are non-blocking. The functions return an opaque `request`. NCCL
uses the returned `request` to test for completion via the plugin’s `test` function. Marking
a request complete indicates the request is locally complete. It makes no guarantees on the
state of the peer. In the case of put, this means the local buffer is available for re-use
but the data may not be visible on the peer. A `request` may only be marked complete once
all operations in the request are complete (e.g. putSignal is only complete once both the put
and signal are complete). Requests may be marked complete in any order, assuming the plugin
respects the data ordering constraints of the API.

Some implementations require help from an additional CPU Proxy thread to progress operations.
The plugin exposes a `rmaProgress` function for such cases. This function is
operation-independent and is called periodically regardless of whether
a data operation is pending.

## Example

`plugins/rma/example` contains a minimal RMA v14 plugin. It is a compile/load example,
not a functional network transport. Build it with:

```shell
make -C plugins/rma/example
```

### Loading the example plugin

Set the `LD_LIBRARY_PATH` to include the example plugin directory:

```shell
export LD_LIBRARY_PATH=/path/to/nccl/plugins/rma/example:$LD_LIBRARY_PATH
```

Set `NCCL_RMA_PLUGIN` to either the plugin name, the shared library file name, or
the absolute path to the plugin file. Any of the following can work:

```shell
export NCCL_RMA_PLUGIN=example
export NCCL_RMA_PLUGIN=libnccl-rma-example.so
export NCCL_RMA_PLUGIN=/path/to/nccl/plugins/rma/example/libnccl-rma-example.so
```

NCCL will automatically discover and load the plugin based on the exported symbol names.
