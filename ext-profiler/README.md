# NCCL Profiler Plugin Documentation

This page describes the NCCL Profiler plugin API and how to implement a profiler plugin for NCCL.

# Overview

To allow NCCL to better integrate with DL frameworks, NCCL v2.23 introduced a profiler plugin
interface. Any NCCL user can write profiler plugins to extract performance data from NCCL and
use it for debugging and analysis.

Similarly to other plugins (e.g., network plugin), the profiler plugins come as a shared library
called `libnccl-profiler.so`. That shared library contains one or more implementations of the
NCCL PROFILER API, in the form of versioned structs, filled with pointers to all required
functions.

# Plugin architecture

## Plugin name and supporting multiple profiler plugins

When NCCL is initialized, it will look for a `libnccl-profiler.so` library and dynamically load
it, then look for symbols inside the library.

The `NCCL_PROFILER_PLUGIN` environment variable allows multiple plugins to coexist. If set, NCCL
will look for a library with a name of `libnccl-profiler-${NCCL_PROFILER_PLUGIN}.so`. It is therefore
advised to name the library following that pattern, with a symlink pointing `libnccl-profiler.so`
to `libnccl-profiler-${NCCL_PROFILER_PLUGIN}.so`. That way, if there are multiple plugins in the
path, setting `NCCL_PROFILER_PLUGIN` will allow users to select the right plugin. Alternatively,
the user can also set `NCCL_PROFILER_PLUGIN` to the pathname of the `libnccl-profiler.so` library.

## Struct versioning

Once a library is found, NCCL will look for a symbol named `ncclProfiler_vX`, with `X` increasing
over time. The versioning ensures that the plugin and the NCCL core are compatible.

Plugins are encouraged to provide multiple of those symbols, implementing multiple versions of the
NCCL PROFILER API, so that the same plugin can be compiled and support a wide range of NCCL versions.

Conversely, and to ease transition, NCCL can choose to support different plugin versions, looking
for the latest ncclProfiler struct version, but also looking for older ones so that older plugins
would still work.

## Headers management

To help users build plugins effortlessly, plugins should copy the `ncclProfiler_vX` definitions
they support to their internal includes. An example is shown in `ext-profiler/example` where we
keep all headers in the `nccl/` directory and provide thin layers to implement old version on top
of newer ones.

The `nccl/` directory is populated with `profiler_vX.h` files extracting all relevant definitions
from old API versions. It also provides error codes in `err.h`.

# API (v2)

Below is the main `ncclProfiler_v2` struct. Each function is explained in later sections.

```
typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  // Input
  //  - context        : opaque profiler context object for separating profiler behavior across comms
  // Output
  //  - eActivationMask: bitmask of active events set by the plugin
  ncclResult_t (*init)(void** context, int* eActivationMask);

  // startEvent - initialize and start a new event for the supplied event descriptor inside the eventset
  // Input
  //  - context: opaque profiler context object
  //  - eDescr : pointer to ncclProfilerEventDescr_t object
  // Output
  //  - eHandle: return event handle for supplied event descriptor object
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v2_t* eDescr);

  // stopEvent - stop/finalize an event inside and event set
  // Input
  //  - eHandle: handle to event object
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and event attribute updates
  // Input
  //  - eHandle   : handle to event object created through startEvent
  //  - eStateArgs: optional argument used to capture event attribute updates associated with the state transition
  //  - eState    : event state transition
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v2_t eState, ncclProfilerEventStateArgs_v2_t* eStateArgs);

  // finalize - finalize the profiler plugin
  // Input
  //  - context: opaque profiler context object
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v2_t;
```

## Error codes

As rule of thumb, profiler generated errors should not be propagated to NCCL and alter its normal
functioning. Nevertheless, the profiler interface returns NCCL error codes, in case any need for
them arises in the future. For now, any profiler interface call should only return `ncclSuccess`.
The only exception is `init` that can return an error so that NCCL can disable the plugin.

## Operation overview

NCCL will call the `init` function first for every new communicator that is initialized. The profiler
returns an opaque context handle that is used to isolate profiler instances across communicators.
Similarly, NCCL will call `finalize` to destroy the profiler context, thus freeing resources.

The NCCL core code is instrumented with calls to `startEvent`, `stopEvent` and `recordEventState`.
These are used to start, stop and update events in the profiler, respectively.

## API Functions

### Initialization

#### name

The `name` field should point to a character string with the name of the profiler plugin. This will
be used for all logging, especially when `NCCL_DEBUG=INFO` is set.

#### init

As soon as NCCL finds the plugin and the correct ncclProfiler symbol, it calls its `init` function.
This allows the plugin to initialize its internal context, used during profiling of NCCL events.
If the `init` function does not return `ncclSuccess`, NCCL disables the plugin.

#### finalize

When the profiler is no longer needed, a call to `finalize` destroys the profiler context and frees
up resources.

### Profiling

#### startEvent

When NCCL needs to start profiling a new event it calls `startEvent`. `startEvent` takes the profiler
context, previously created by `init`, an event descriptor of type `ncclProfilerEventDescr_t` and
returns an opaque profiler event handle that can be passed to other profiler functions, as discussed
later in the document.


The event descriptor contains all the event metadata. Every event type has its own descriptor. Below
is the `ncclProfilerEventDescr_t` struct.

```
typedef struct {
  uint8_t type;             // event type (e.g., ncclProfileGroup, ncclProfileColl, ...)
  void* parentObj;          // pointer to parent event used to expose the event hierarchy to the profiler
  int rank;                 // rank that generated the event
  union {
    struct {                // collective events metadata
      const char* name;     // string containing name of the communicator
      uint64_t commHash;    // unique hash/id for the communicator
      uint64_t seqNumber;   // sequence number of this collective operation in the communicator
      const char* func;     // string containing name of the collective
      void const* sendBuff; // address of send buffer
      void* recvBuff;       // address of recv buffer
      size_t count;         // data count
      int root;             // root rank
      const char* datatype; // string containing the name of the datatype
      size_t trafficBytes;  // number of transfer bytes
      uint8_t nMaxChannels; // max number of channels for this collective
      uint8_t nWarps;       // number of GPU warps for this collective
      const char* algo;     // string containing name of the algorithm for this collective
      const char* proto;    // string containing name of the protocol for this collective
    } coll;

    struct {                // point-to-point events metadata
      const char* name;
      uint64_t commHash;
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;             // peer rank for this point-to-point
    } p2p;

    struct {                // proxyOp events metadata
      pid_t pid;            // process id that generated the associated `ncclProxyOp` object
      uint8_t channelId;    // id of the channel used by the associated `ncclProxyOp` object
      int peer;             // peer rank
      int nSteps;           // number of network transfers/steps required by the `ncclProxyOp`
      int chunkSize;        // chunk size for this `ncclProxyOp`
      int isSend;           // set to 1 for sends and 0 for recvs
    } proxyOp;

    struct {                // proxyStep events metadata
      int step;             // individual step in `ncclProxyOp`
    } proxyStep;
  };
} ncclProfilerEventDescr_v2_t;
```

NCCL defines the following events: `ncclProfileGroup`, `ncclProfileColl`, `ncclProfileP2p`,
`ncclProfileProxyOp`, `ncclProfileProxyStep`, and `ncclProfileProxyCtrl`.

#### stopEvent

`stopEvent` takes the event handle returned by `startEvent` to stop the event. After the event
has been stopped the handle can no longer be used with other profiler calls. Using the event
handle after `eventStop` is undefined behavior.

#### recordEventState

Some events can only be started and stopped. For example, `ncclProfileGroup`, `ncclProfileColl`,
`ncclProfileP2p`, cannot be updated through calls to `recordEventState`.

`ncclProfileProxyOp`, `ncclProfileProxyStep` and `ncclProfileProxyCtrl` can be updated through
calls to `recordEventState`.

The state of proxy generated events can be updated, along with event attributes, using
`recordEventState`. These events can go through several states during their lifecycle.
The list of supported states for the proxy-defined events is reported below.

```
typedef enum {
  // ncclProfileProxyOp event states
  ncclProfilerProxyOpSendPosted,        // state marks the posting of send buffer to GPU for given network transfer/step
  ncclProfilerProxyOpSendRemFifoWait,   // state marks the waiting of CTS credits from peer rank
  ncclProfilerProxyOpSendTransmitted,   // state marks the sending of network transfer/step to peer rank
  ncclProfilerProxyOpSendDone,          // state marks the ending  of network transfer/step
  ncclProfilerProxyOpRecvPosted,        // state marks the posting of recv to network for given network transfer/step
  ncclProfilerProxyOpRecvReceived,      // state marks the recving of network transfer/step from peer rank
  ncclProfilerProxyOpRecvTransmitted,   // state marks the ending  of the network transfer/step
  ncclProfilerProxyOpRecvDone,          // state marks the consuming of data from GPU

  // ncclProfileProxyStep event states
  ncclProfilerProxyStepSendGPUWait,     // state marks the waiting of send data from GPU for given network transfer/step
  ncclProfilerProxyStepSendWait,        // state marks the waiting of send data from network for given network transfer/step
  ncclProfilerProxyStepRecvWait,        // state marks the waiting of recv data from network for given network transfer/step
  ncclProfilerProxyStepRecvFlushWait,   // state marks the waiting of recv data flush to GPU for given network transfer/step
  ncclProfilerProxyStepRecvGPUWait,     // state marks the waiting of recv data consumption from GPU for given network transfer/step

  // ncclProfileProxyCtrl event states
  ncclProfilerProxyCtrlIdle,            // state marks proxy progress thread idle
  ncclProfilerProxyCtrlActive,          // state marks proxy progress thread active
  ncclProfilerProxyCtrlSleep,           // state marks proxy progress thread sleeping
  ncclProfilerProxyCtrlWakeup,          // state marks proxy progress thread waking up
  ncclProfilerProxyCtrlAppend,          // state marks append of new network work item begin
  ncclProfilerProxyCtrlAppendEnd,       // state marks append of new network work item end
} ncclProfilerEventState_v2_t;
```

`ncclProfileProxyOp` events are generated by the proxy progress thread while it is processing
network requests for the GPU kernel. ProxyOp events are generated for every active channel and
provide a summary of the activity of the proxy progress thread for that channel.

`ncclProfileProxyStep` events are generated by the proxy progress thread while it is processing
network requests for the GPU kernel. ProxyStep events describe individual network transfer in
the channel. Thus, they provide a more fine-grained view w.r.t. ProxyOp events.

`ncclProfileProxyCtrl` events are generated by the proxy progress thread while it is not processing
network requests for the GPU kernel. This includes everything else that the proxy thread might be
doing, including appending new `ncclProxyOp` objects to the list of work elements to process.

State transitions for the events described can also come with event attribute updates. For this
reason the profiler defines the `ncclProfilerEventStateArgs_t` struct, reported below.

```
typedef union {
  struct {                // attributes to update for ncclProfileProxyOp events
    size_t transSize;     // data transferred thus far
    int steps;            // network transfer/steps processed thus far
  } proxyOp;

  struct {                // attributes to update for ncclProfileProxyCtrl
    int appendedProxyOps; // number of appended proxy ops thus far
  } proxyCtrl;
} ncclProfilerEventStateArgs_v2_t;
```

The example profiler in `ext-profiler/example` contains details on how to capture and use the events above.

### Event hierarchy

NCCL core events (reported above) are organized into a hierarchy as reported below:

```
Group event
   |
   +- Collective event
   |  |
   |  +- ProxyOp event
   |     |
   |     +- ProxyStep event
   |
   +- Point-to-point event
      |
      +- ProxyOp event
         |
         +- ProxyStep event

ProxyCtrl event
```

# Profiler instrumentation and logging

## Profiling of collective and p2p operations

The NCCL code is instrumented with profiler callbacks at different levels to capture start/stop of groups,
collective and point-to-point operations, as well as proxy progress activity. Due to the asynchronous nature
of NCCL operations, events associated to collective and point-to-point operations are not easy to delimit
precisely. For example, without both proxy and/or kernel activity it is impossible for the profiler to
figure out when a collective operation completes. Therefore, `stopEvent` for collectives simply indicates to
the profiler that the collective has been enqueued. The profiler can leverage proxy event information, if
these are enabled, to estimate when the collective ends. In this case, the profiler can look at the `stopEvent`
call of the last `ncclProfileProxyOp` event to mark the completion of the associated collective event. This
can be achieved by reference counting the collective event and letting calls to `startEvent` and `stopEvent`
increment and decrement the reference counter, respectively.

## PXN

PXN causes some proxy operations to be processed in a remote proxy thread that differs from the one that
generated the operation. When this happens, the event hierarchy reported above breaks. Because the
profiler can use the hierarchy information, provided by NCCL in the event descriptor, to dereference the
parent event during `startEvent`, the remote proxy thread must be in the same address space of the proxy
thread originating the operation. To avoid the profiler instance in the remote proxy address space to
dereference a pointer from another address space the event descriptor includes the PID of the originator.
The profiler plugin needs to check that the originator PID matches the local PID before dereferencing the
parent event.
