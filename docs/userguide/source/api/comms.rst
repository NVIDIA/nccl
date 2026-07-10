**********************************************
Communicator Creation and Management Functions
**********************************************

The following functions are public APIs exposed by NCCL to create and manage the collective communication operations.

ncclGetLastError
----------------

.. c:function:: const char* ncclGetLastError(ncclComm_t comm)

Returns a human-readable string corresponding to the last error that occurred in NCCL.
Note: The error is not cleared by calling this function.
Please note that the string returned by ncclGetLastError could be unrelated to the current call
and can be a result of previously launched asynchronous operations, if any.

ncclGetErrorString
------------------

.. c:function:: const char* ncclGetErrorString(ncclResult_t result)

Returns a human-readable string corresponding to the passed error code.

ncclGetVersion
--------------

.. c:function:: ncclResult_t ncclGetVersion(int* version)

The ncclGetVersion function returns the version number of the currently linked NCCL library.
The NCCL version number is returned in *version* and encoded as an integer which includes the
:c:macro:`NCCL_MAJOR`, :c:macro:`NCCL_MINOR` and :c:macro:`NCCL_PATCH` levels.
The version number returned will be the same as the :c:macro:`NCCL_VERSION_CODE` defined in *nccl.h*.
NCCL version numbers can be compared using the supplied macro :c:macro:`NCCL_VERSION` as ``NCCL_VERSION(MAJOR,MINOR,PATCH)``

ncclGetUniqueId
---------------

.. c:function:: ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)

Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
called once when creating a communicator and the Id should be distributed to all ranks in the
communicator before calling ncclCommInitRank. *uniqueId* should point to a ncclUniqueId object allocated by the user.

ncclCommInitRank
----------------

.. c:function:: ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)

Creates a new communicator (multi thread/process version).
*rank* must be between 0 and *nranks*-1 and unique within a communicator clique.
Each rank is associated to a CUDA device, which has to be set before calling
ncclCommInitRank.
ncclCommInitRank implicitly synchronizes with other ranks, hence it must be
called by different threads/processes or used within ncclGroupStart/ncclGroupEnd.

ncclCommInitAll
---------------

.. c:function:: ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist)

Creates a clique of communicators (single process version) in a blocking way.
This is a convenience function to create a single-process communicator clique.
Returns an array of *ndev* newly initialized communicators in *comms*.
*comms* should be pre-allocated with size at least ndev*sizeof(:c:type:`ncclComm_t`).
*devlist* defines the CUDA devices associated with each rank. If *devlist* is NULL,
the first *ndev* CUDA devices are used, in order.

.. _ncclcomminitrankconfig:

ncclCommInitRankConfig
----------------------

.. c:function:: ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config)

This function works the same way as *ncclCommInitRank* but accepts a configuration argument of extra attributes for
the communicator. If config is passed as NULL, the communicator will have the default behavior, as if ncclCommInitRank
was called.

See the :ref:`init-rank-config` section for details on configuration options.

ncclCommInitRankScalable
------------------------

.. c:function:: ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config)


This function works the same way as *ncclCommInitRankConfig* but accepts a list of ncclUniqueIds instead of a single one.
If only one ncclUniqueId is passed, the communicator will be initialized as if ncclCommInitRankConfig was called.
The provided ncclUniqueIds will all be used to initialize the single communicator given in argument.

See the :ref:`init-rank-config` section for details on how to create and distribute the list of ncclUniqueIds.

ncclCommSplit
-------------

.. c:function:: ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config)

The *ncclCommSplit* is a collective function and creates a set of new communicators from an existing one. Ranks which
pass the same *color* value will be part of the same group; color must be a non-negative value. If it is
passed as *NCCL_SPLIT_NOCOLOR*, it means that the rank will not be part of any group, therefore returning NULL
as newcomm.
The value of key will determine the rank order, and the smaller key means the smaller rank in new communicator.
If keys are equal between ranks, then the rank in the original communicator will be used to order ranks.
If the new communicator needs to have a special configuration, it can be passed as *config*, otherwise setting
config to NULL will make the new communicator inherit the original communicator's configuration.
When split, there should not be any outstanding NCCL operations on the *comm*. Otherwise, it might cause
a deadlock.

ncclCommShrink
--------------

.. c:function:: ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags)

The *ncclCommShrink* function creates a new communicator by removing specified ranks from an existing communicator.
It is a collective function that must be called by all participating ranks in the newly created communicator.
Ranks that are part of *excludeRanksList* should not call this function.
The original ranks listed in *excludeRanksList* (of size *excludeRanksCount*) will be excluded from the new communicator.
Within the new communicator, ranks will be updated to maintain a contiguous set of ids.
If the new communicator needs a special configuration, it can be passed as *config*; otherwise, setting config to NULL will make the new communicator inherit the configuration of the parent communicator.

The *shrinkFlags* parameter controls the behavior of the operation. Use *NCCL_SHRINK_DEFAULT* (or *0*) for normal operation, or *NCCL_SHRINK_ABORT* when shrinking after an error on the parent communicator.
Specifically, when using *NCCL_SHRINK_DEFAULT*, there should not be any outstanding NCCL operations on the *comm* to avoid potential deadlocks. Further, if the parent communicator has the flag config.shrinkShare set to 1, NCCL will reuse the parent communicator resources.
On the other hand, when using *NCCL_SHRINK_ABORT*, NCCL will automatically abort any outstanding operations on the parent communicator, and no resources will be shared between the parent and the newly created communicator.

ncclCommGetUniqueId
-------------------

.. c:function:: ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId)

The *ncclCommGetUniqueId* function generates a unique identifier for growing an
existing communicator exactly once. This function must be called by only one
rank (the coordinator) before each grow operation on the existing communicator.
The coordinator is responsible for distributing the *uniqueId* to all new ranks
before they join the communicator via *ncclCommGrow*. This function should only
be called when there are no outstanding NCCL operations on the communicator.

ncclCommGrow
------------

.. c:function:: ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config)

The *ncclCommGrow* function creates a new communicator by adding new ranks to an existing communicator.
It must be called by both existing ranks (from the parent communicator) and new ranks (joining the communicator).

**For existing ranks:**

- *comm* should be the parent communicator
- *rank* must be set to *-1* (existing ranks retain their original rank in the new communicator)
- *uniqueId* should be *NULL* (existing ranks receive coordination information internally)
- The function creates *newcomm* with the same rank as in the parent communicator

**For new ranks:**

- *comm* should be *NULL*
- *rank* must be set to the desired rank in the new communicator (must be >= parent communicator size)
- *uniqueId* must be the unique identifier obtained from *ncclCommGetUniqueId* called by the coordinator

The *nRanks* parameter specifies the total number of ranks in the new communicator and must be greater than the size of the parent communicator.
If the new communicator needs a special configuration, it can be passed as *config*; otherwise, setting config to NULL will make the new communicator inherit the configuration of the parent communicator (for existing ranks) or use default configuration (for new ranks).

There should not be any outstanding NCCL operations on the parent communicator when calling this function to avoid potential deadlocks.
After the grow operation completes, the parent communicator should be destroyed using *ncclCommDestroy* to free resources.

**Example workflow:**

1. Coordinator rank calls *ncclCommGetUniqueId* to generate the grow identifier
2. Coordinator distributes the *uniqueId* to all new ranks (out-of-band)
3. All existing ranks call *ncclCommGrow* with *comm*\=parent, *rank*\=-1, *uniqueId*\=NULL (except for Coordinator rank which passes the *uniqueId*)
4. All new ranks call *ncclCommGrow* with *comm*\=NULL, *rank*\=new_rank, *uniqueId*\=received_id

ncclCommRevoke
--------------

.. c:function:: ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags)

Revokes in-flight operations on a communicator without destroying resources. Successful return may be *ncclInProgress* (non-blocking) while revocation completes asynchronously; applications can query *ncclCommGetAsyncError* until it returns *ncclSuccess*.

*revokeFlags* must be set to *NCCL_REVOKE_DEFAULT* (0). Other values are reserved for future use.

After revoke completes, the communicator is quiesced and safe for destroy, split, and shrink. Launching new collectives on a revoked communicator returns *ncclInvalidUsage*. Calling *ncclCommFinalize* after revoke is not supported. Resource sharing via *splitShare*/*shrinkShare* is disabled when the parent communicator is revoked.

ncclCommFinalize
----------------

.. c:function:: ncclResult_t ncclCommFinalize(ncclComm_t comm)

Finalize a communicator object *comm*. When the communicator is marked as nonblocking, *ncclCommFinalize* is a
nonblocking function. Successful return from it will set communicator state as *ncclInProgress* and indicates
the communicator is under finalization where all uncompleted operations and the network-related resources are
being flushed and freed.
Once all NCCL operations are complete, the communicator will transition to the *ncclSuccess* state. Users
can query that state with *ncclCommGetAsyncError*.

ncclCommDestroy
---------------

.. c:function:: ncclResult_t ncclCommDestroy(ncclComm_t comm)

Destroy a communicator object *comm*. If *ncclCommFinalize* is called by users, users should guarantee that the state
of the communicator becomes *ncclSuccess* before calling *ncclCommDestroy*. In all cases, the communicator should no
longer be accessed after *ncclCommDestroy* returns. It is recommended that users call *ncclCommFinalize* and then
*ncclCommDestroy*.

*ncclCommDestroy* will call *ncclCommFinalize* internally, unless *ncclCommFinalize* was previously called on the
communicator. If *ncclCommFinalize* was previously called on the communicator object *comm*, then *ncclCommDestroy* is a
purely local operation.

This function is an intra-node collective call, which all ranks on the same node should call to avoid a hang.

ncclCommAbort
-------------

.. c:function:: ncclResult_t ncclCommAbort(ncclComm_t comm)

*ncclCommAbort* frees resources that are allocated to a communicator object *comm* and aborts any uncompleted
operations before destroying the communicator. All active ranks are required to call this function in order to
abort the NCCL communicator successfully. For more use cases, please check :ref:`ft`.

ncclCommGetAsyncError
---------------------

.. c:function:: ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)

Queries the progress and potential errors of asynchronous NCCL operations.
Operations which do not require a stream argument (e.g. ncclCommFinalize) can be considered complete as soon
as the function returns *ncclSuccess*; operations with a stream argument (e.g. ncclAllReduce) will return
*ncclSuccess* as soon as the operation is posted on the stream but may also report errors through
ncclCommGetAsyncError() until they are completed. If the return code of any NCCL function is *ncclInProgress*,
it means the operation is in the process of being enqueued in the background, and users must query the states
of the communicators until all the states become *ncclSuccess* before calling another NCCL function. Before the
states change into *ncclSuccess*, users are not allowed to issue CUDA kernel to the streams being used by NCCL.
If there has been an error on the communicator, user should destroy the communicator with :c:func:`ncclCommAbort`.
If an error occurs on the communicator, nothing can be assumed about the completion or correctness of operations
enqueued on that communicator.

ncclCommCount
-------------

.. c:function:: ncclResult_t ncclCommCount(const ncclComm_t comm, int* count)

Returns in *count* the number of ranks in the NCCL communicator *comm*.

ncclCommCuDevice
----------------

.. c:function:: ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device)

Returns in *device* the CUDA device associated with the NCCL communicator *comm*.

ncclCommUserRank
----------------

.. c:function:: ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)

Returns in *rank* the rank of the caller in the NCCL communicator *comm*.

ncclCommRegister
----------------

.. c:function:: ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle)

Registers the buffer *buff* with *size* under communicator *comm* for zero-copy communication; *handle* is
returned for future deregistration. See *buff* and *size* requirements and more instructions in :ref:`user_buffer_reg`.

ncclCommDeregister
------------------

.. c:function:: ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle)

Deregister buffer represented by *handle* under communicator *comm*.

ncclCommWindowRegister
----------------------

.. c:function:: ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags)

Collectively register local buffer *buff* with *size* under communicator *comm* into NCCL window. Since this is a collective call,
every rank in the communicator needs to participate in the registration. Size may differ across ranks; callers are
responsible for ensuring later operations only access ranges that are valid for the ranks participating in that operation. *win* is
returned for future deregistration (if called within a group, the value may not be filled in until ncclGroupEnd() has completed).
See *buff* requirement and more instructions in :ref:`user_buffer_reg`. User can also pass
different win flags to control the registration behavior. For more win flags information, please refer to :ref:`win_flags`.
Host APIs do not accept buffers that are symmetrically registered more than once. Passing such buffers to host APIs results
in undefined behavior.

ncclCommWindowDeregister
------------------------

.. c:function:: ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win)

Deregister NCCL window represented by *win* under communicator *comm*. Deregistration is local to the rank, and
caller needs to make sure the corresponding buffer within the window is not being accessed by any NCCL operation.

ncclMemAlloc
------------

.. c:function:: ncclResult_t ncclMemAlloc(void **ptr, size_t size)

Allocate a GPU buffer with *size*. Allocated buffer head address will be returned by *ptr*,
and the actual allocated size can be larger than requested because of the buffer granularity
requirements from all types of NCCL optimizations.

ncclMemFree
-----------

.. c:function:: ncclResult_t ncclMemFree(void *ptr)

Free memory allocated by *ncclMemAlloc()*.

ncclCommSuspend
---------------

.. c:function:: ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags)

Suspend communicator operations to free resources. The communicator cannot be used for any NCCL operations
while suspended. There should be no outstanding NCCL operations on *comm* when this function is called.

The *flags* parameter controls which resources are released:

- *NCCL_SUSPEND_MEM* (``0x01``) -- Release dynamic GPU memory allocations held by the communicator.

A suspended communicator can be restored to an active state by calling *ncclCommResume*.

ncclCommResume
--------------

.. c:function:: ncclResult_t ncclCommResume(ncclComm_t comm)

Resume all previously suspended resources on communicator *comm*. After this call returns successfully, the
communicator is fully operational and can be used for NCCL operations again.

ncclCommMemStats
----------------

.. c:function:: ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value)

Query communicator memory statistics. The *stat* parameter selects which statistic to retrieve, and the
result is written to *\*value*. See :c:type:`ncclCommMemStat_t` for the list of available statistics.
