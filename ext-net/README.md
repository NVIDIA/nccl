# NCCL Net Plugin Documentation

This page describes the NCCL Net plugin API and how to implement a network plugin for NCCL.

# Overview

To allow NCCL to work on any network type, NCCL provides a way to use external plugins. Plugins
implement the NCCL network API, and decouple NCCL binary builds which are built against a
particular version of the GPU stack (i.e. CUDA) from the network code which is built against a
particular version of the networking stack. That way, we can easily integrate any CUDA version
with any network stack version.

NCCL network plugins come as a shared library called `libnccl-net.so`. That shared library
contains one or more implementations of the NCCL NET API, in the form of versioned structs,
filled with pointers to all required functions.

# Plugin architecture

## Plugin name and supporting multiple network plugins

When NCCL is initialized, it will look for a `libnccl-net.so` library and dynamically load it,
then look for symbols inside the library.

The `NCCL_NET_PLUGIN` environment variable allows multiple plugins to coexist. If set, NCCL
will look for a library with a name of `libnccl-net-${NCCL_NET_PLUGIN}.so`. It is therefore
advised to name the library following that pattern, with a symlink pointing `libnccl-net.so`
to `libnccl-net-${NCCL_NET_PLUGIN}.so`. That way, if there are multiple plugins in the path,
setting `NCCL_NET_PLUGIN` will allow users to select the right plugin.

## Struct versioning

Once a library is found, NCCL will look for a symbol named `ncclNet_vX`, with `X` increasing
over time. The versioning ensures that the plugin and the NCCL core are compatible.

Plugins are encouraged to provide multiple of those symbols, implementing multiple versions
of the NCCL NET API, so that the same plugin can be compiled and support a wide range of NCCL
versions.

Conversely, and to ease transition, NCCL can choose to support different plugin versions, looking
for the latest ncclNet struct version, but also looking for older ones so that older plugins
would still work.

## In-network collective operations, a.k.a. collNet

Additionally to the ncclNet structure, network plugins can provide a collNet structure which
implements in-network collective operations, if supported. That can be used by the NCCL collNet
algorithm to accelerate inter-node reductions in allReduce.

The collNet struct is a different, optional struct provided by the network plugin, but its
versioning is tied to the ncclNet struct and many functions are common between the two to
ease the implementation.

## Headers management

To help users build plugins effortlessly, plugins should copy the `ncclNet_vX` definitions
they support to their internal includes. An example is shown in `ext-net/example/` where we keep
all headers in the `nccl/` directory and provide thin layers to implement old versions on top
of newer ones.

The `nccl/` directory is populated with `net_vX.h` files extracting all relevant definitions
from old API versions. It also provides error codes in `err.h`.

# API (v6)

Below is the main `ncclNet_v6` struct. Each function is explained in later sections.

```
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v6_t;
```

## Error codes

All plugins functions use NCCL error codes as return value. `ncclSuccess` should be returned upon
success.

Otherwise, plugins can return one of the following:
 - `ncclSystemError` is the most common error for network plugins, when a call to the linux kernel
or a system library fails. This typically includes all network/hardware errors.
 - `ncclInternalError` is returned when the NCCL core code is using the network plugin in an
incorrect way, for example allocating more requests than it should, or passing an invalid argument
to calls.
 - `ncclInvalidUsage` should be returned when the error is most likely a user error. This can
include misconfiguration, but also sizes mismatch.
 - `ncclInvalidArgument` should usually not be used by plugins since arguments should be checked by
the NCCL core layer.
 - `ncclUnhandledCudaError` is returned when an error comes from CUDA. Since network plugins should
not need to rely on CUDA, this should not be common.

## Operation overview

NCCL will call the `init` function first, then query the number of network devices with the
`devices` function, getting each network device properties with `getProperties`.

To establish a connection between two network devices, NCCL will first call `listen` on the
receiving side, pass the returned handle to the sender side of the connection, and call `connect`
with that handle. Finally, `accept` will be called on the receiving side to finalize the connection
establishment.

Once the connection is established, communication will be done using the functions `isend`,
`irecv` and `test`. Prior to calling `isend` or `irecv`, NCCL will call the `regMr` function on
all buffers to allow RDMA NICs to prepare buffers. `deregMr` will be used to unregister buffers.

In certain conditions, `iflush` will be called after a receive calls completes to allow the network
plugin to flush data and ensure the GPU will observe the newly written data.

To close the connections NCCL will call `closeListen` to close the object returned by `listen`,
`closeSend` to close the object returned by `connect` and `closeRecv` to close the object returned
by `accept`.

## API Functions

### Initialization
`name`

The `name` field should point to a character string with the name of the network plugin. This will
be used for all logging, especially when `NCCL_DEBUG=INFO` is set.

Note: setting `NCCL_NET=<plugin name>` will ensure a specific network implementation is used, with
a matching `name`. This is not to be confused with `NCCL_NET_PLUGIN` which defines a suffix to the
`libnccl-net.so`library name to load.

`init`

As soon as NCCL finds the plugin and the correct ncclNet symbol, it will call the `init` function.
This will allow the plugin to discover network devices and make sure they are usable. If the
`init` function does not return `ncclSuccess`, then NCCL will not use the plugin and fall back on
internal ones.

To allow the plugin logs to integrate into the NCCL logs seemlessly, NCCL provides a logging
function to `init`. This function is typically used to allow for `INFO` and `WARN` macros within
the plugin code adding the following definitions:

```
#define WARN(...) logFunction(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) logFunction(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
```

`devices`

Once the plugin is initialized, NCCL will query the number of devices available. It should not
be zero, otherwise NCCL initialization will fail. If no device is present or usable, the `init`
function should not return `ncclSuccess`.

`getProperties`

Right after getting the number of devices, NCCL will query properties for each available network
device. These properties are critical when multiple adapters are present to ensure NCCL uses each
adapter in the most optimized way.

The `name` is only used for logging.

The `pciPath` is the base for all topology detection and should point to the PCI device directory
in /sys. This is typically the directory pointed by `/sys/class/net/eth0/device` or
`/sys/class/infiniband/mlx5_0/device`. If the network interface is virtual, then `pciPath` should
be `NULL`.

The `guid` field is used to determine when network adapters are connected to multiple PCI
endpoints. For normal cases, it can be set to the device number. If multiple network devices have
the same guid, then NCCL will consider the are sharing the same network port to the fabric, hence
it will not use the port multiple times.

The `ptrSupport` field indicates whether or not CUDA pointers are supported. If so, it should be
set to `NCCL_PTR_HOST|NCCL_PTR_CUDA`, otherwise it should be set to `NCCL_PTR_HOST`. If the plugin
supports `dmabuf`, it should set `ptrSupport` to `NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF` and
provide a `regMrDmaBuf` function.

The `speed` field indicates the speed of the network port in Mbps (10^6 bits per second). This is
important to ensure proper optimization of flows within the node.

The `port` field indicates the port number. This is important again for topology detection and flow
optimization within the node when a NIC with a single PCI connection is connected to the fabric
with multiple ports.

The `latency` field indicates the network latency in microseconds. This can be useful to improve
the NCCL tuning and make sure NCCL switches from tree to ring at the right size.

The `maxComms` field indicates the maximum number of connections we can create.

The `maxRecvs` field indicates the maximum number for grouped receive operations (see grouped
receive).

### Connection establishment

Connections are used in an unidirectional manner. There is therefore a sender side and a receiver
side.

`listen`

To create a connection, NCCL will start by calling `listen` on the receiver side. This function
takes a device number as input argument, and should return a local `listenComm` object, and a
`handle` to pass to the other side, so that the sender side can connect to the receiver.

The `handle` is a buffer of size `NCCL_NET_HANDLE_MAXSIZE` and is provided by NCCL.

This call should never block, but contrary to `connect` and `accept`, `listenComm` should never
be `NULL` if the call succeeds.

`connect`

NCCL will use its bootstrap infrastructure to provide the `handle` to the sender side, then call
`connect` on the sender side on a given device index `dev`, providing the `handle`. `connect`
should not block either, and instead set `sendComm` to `NULL` and return `ncclSuccess`. In that
case, NCCL will call `accept` again until it succeeds.

`accept`

To finalize the connection, the receiver side will call `accept` on the `listenComm` returned by
the `listen` call previously. If the sender did not connect yet, `accept` should not block. It
should return `ncclSuccess`, setting `recvComm` to `NULL`. NCCL will call `accept` again until it
succeeds.

`closeListen`/`closeSend`/`closeRecv`

Once a `listenComm`/`sendComm`/`recvComm` is no longer needed, NCCL will call
`closeListen`/`closeSend`/`closeRecv` to free the associated resources.

### Communication

Communication is done using asynchronous send and receive operations: `isend`, `irecv` and `test`.
To support RDMA capabilities, buffer registration and flush functions are provided.

To keep track of asynchronous send, receive and flush operations, requests are returned to NCCL,
then queried with `test`. Each `sendComm` or `recvComm` must be able to handle
`NCCL_NET_MAX_REQUESTS` requests in parallel.

Note: That value should be multiplied by the multi-receive capability of the plugin for the sender
side, so that we can effectively have `NCCL_NET_MAX_REQUESTS` multi-receive operations happening
in parallel. So, if we have a `maxRecvs`value of 8 and `NCCL_NET_MAX_REQUESTS` is 8, then each
`sendComm` must be able to handle up to 8x8=64 concurrent `isend` operations.

`regMr`

Prior to sending or receiving data, NCCL will call `regMr` with any buffers later used for
communication. It will provide a `sendComm` or `recvComm` as `comm` argument, then the buffer
pointer `data`, `size`, and `type` being either `NCCL_PTR_HOST`, or `NCCL_PTR_CUDA` if the network
supports CUDA pointers.

The network plugin can use the output argument `mhandle` to keep any reference to that memory
registration, as this `mhandle` will be passed back for all `isend`, `irecv`, `iflush` and
`deregMr` calls.

`regMrDmaBuf`

If the plugin has set the `NCCL_PTR_DMABUF` property in `ptrSupport`, NCCL will use `regMrDmaBuf`
instead of `regMr`. If the property was not set, `regMrDmaBuf` can be set to `NULL`.


`deregMr`

When buffers will no longer be used for communication, NCCL will call `deregMr` to let the plugin
free resources. This function is used to deregister handles returned by both `regMr` and
`regMrDmaBuf`.

`isend`

Data will be sent through the connection using `isend`, passing the `sendComm` previously
created by `connect`, and the buffer described by `data`, `size`, and `mhandle`. A `tag` must be
used if the network supports multi-receive operations (see `irecv`) to distinguish between
different sends matching the same multi-receive. Otherwise it can be set to 0.

The `isend` operation returns a handle in the `request` argument for further calls to `test`. If
the `isend` operation cannot be initiated, `request` can be set to `NULL` and NCCL will call
`isend` again later.

`irecv`

To receive data, NCCL will call `irecv` with the `recvComm` returned by `accept`. The argument
`n` will allow NCCL to perform a multi-receive, to allow grouping of multiple sends through a
single network connection. Each buffer will be described by the `data`, `sizes`, and `mhandles`
arrays. `tags` will specify a tag for each receive so that each of the `n` independent `isend`
operations is received into the right buffer.

If all receive operations can be initiated, `irecv` will return a handle in the `request` pointer,
otherwise it will set it to `NULL`. In the case of multi-receive, all `n` receive operations are
handled by a single request handle.

The sizes provided to `irecv` can (and will) be larger than the size of the `isend` operation.
The contrary (receive size being lower than the send size) is an error, however.

Note: for a given connection, send/receive operations should always match in the order they were
posted. Tags provided for receive operations are only used to assign a given send operation to one
of the buffers of the first (multi-)receive in the queue, not to allow for out-of-order tag
matching on any receive operation posted.

`test`

After an `isend` or `irecv` operation is initiated, NCCL will call `test` on the request handles
until they complete. When that happens, `done` will be set to 1 and `sizes` will be set to the
real size sent or received, the latter being potentially lower than the size passed to `irecv`.

In the case of a multi-receive, all receives will be considered as done as a single operation (the
goal being to allow aggregation), hence they share a single request and a single `done` status.
However, they can have different sizes, so when `done` is non-zero, the `sizes` array should
contain the `n` sizes corresponding to the buffers passed to `irecv`.

Once `test` returns 1 in `done`, the request handle can be freed, meaning that NCCL will never
call `test` again on that request (until it is reallocated by another call to `isend` or `irecv`).

`iflush`

After a receive operation completes, if the operation was targeting GPU memory and received a
non-zero number of bytes, NCCL will call `iflush` to let the network flush any buffer and ensure
the GPU can read it right after without seeing stale data. This flush operation is decoupled from
the `test` code to improve latency of `LL*` protocols, as those are capable of determining when
data is valid or not.

`iflush` returns a request which needs to be queried with `test` until it completes.
