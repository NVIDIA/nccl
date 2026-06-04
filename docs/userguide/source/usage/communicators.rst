.. _communicator-label:

***********************
Creating a Communicator
***********************

When creating a communicator, a unique rank between 0 and n-1 has to be assigned to each of the n CUDA devices which
are part of the communicator. Using the same CUDA device multiple times as different ranks of the same NCCL
communicator is not supported and may lead to hangs.

Given a static mapping of ranks to CUDA devices, the :c:func:`ncclCommInitRank`, :c:func:`ncclCommInitRankConfig` and
:c:func:`ncclCommInitAll` functions will create communicator objects, each communicator object being associated to a
fixed rank and CUDA device. Those objects will then be used to launch communication operations.

Before calling :c:func:`ncclCommInitRank`, you need to first create a unique object which will be used by all processes
and threads to synchronize and understand they are part of the same communicator. This is done by calling the
:c:func:`ncclGetUniqueId` function.

The :c:func:`ncclGetUniqueId` function returns an ID which has to be broadcast to all participating threads and
processes using any CPU communication system, for example, passing the ID pointer to multiple threads, or broadcasting
it to other processes using MPI or another parallel environment using, for example, sockets.

You can also call the ncclCommInitAll operation to create n communicator objects at once within a single process. As it
is limited to a single process, this function does not permit inter-node communication. ncclCommInitAll is equivalent
to calling a combination of ncclGetUniqueId and ncclCommInitRank.

The following sample code is a simplified implementation of ncclCommInitAll.

.. code:: C

 ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
   ncclUniqueId Id;
   ncclGetUniqueId(&Id);
   ncclGroupStart();
   for (int i=0; i<ndev; i++) {
     cudaSetDevice(devlist[i]);
     ncclCommInitRank(comm+i, ndev, Id, i);
   }
   ncclGroupEnd();
 }

Related links:

 * :c:func:`ncclCommInitAll`
 * :c:func:`ncclGetUniqueId`
 * :c:func:`ncclCommInitRank`

.. _init-rank-config:

Creating a communicator with options
-------------------------------------

The :c:func:`ncclCommInitRankConfig` function allows creating a NCCL communicator with specific options.

The config parameters NCCL supports are listed here :ref:`ncclconfig`.

For example, "blocking" can be set to 0 to ask NCCL to never block in any NCCL call, and at the same time
other config parameters can be set as well to more precisely define communicator behavior. A simple example
code is shown below:

.. code:: C

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  config.minCTAs = 4;
  config.maxCTAs = 16;
  config.cgaClusterSize = 2;
  config.netName = "Socket";
  CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
  do {
    CHECK(ncclCommGetAsyncError(comm, &state));
    // Handle outside events, timeouts, progress, ...
  } while(state == ncclInProgress);

Related link: :c:func:`ncclCommGetAsyncError`

Creating a communicator using multiple ncclUniqueIds
----------------------------------------------------

The :c:func:`ncclCommInitRankScalable` function enables the creation of a NCCL communicator using many ncclUniqueIds.
All NCCL ranks have to provide the same array of ncclUniqueIds (same ncclUniqueIds, and in the same order).
For the best performance, we recommend distributing the ncclUniqueIds as evenly as possible amongst the NCCL ranks.

Internally, NCCL ranks will mostly communicate with a single ncclUniqueId.
Therefore, to obtain the best results, we recommend to evenly distribute ncclUniqueIds across the ranks.

The following function can be used to decide if a NCCL rank should create a ncclUniqueIds:

.. code:: C

 bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
   const int rmr = nRanks % nIds;
   const int rpr = nRanks / nIds;
   const int rlim = rmr * (rpr+1);
   if (rank < rlim) {
     return !(rank % (rpr + 1));
   } else {
     return !((rank - rlim) % rpr);
   }
 }

For example, if 3 ncclUniqueIds are to be distributed across 7 NCCL ranks, the first ncclUniqueId will be associated to
ranks 0-2, while the others will be associated to ranks 3-4, and 5-6.
This function will therefore return true on rank 0, 3, and 5, and false otherwise.

Note: only the first ncclUniqueId will be used to create the communicator hash id, which is used to identify the
communicator in the log file and in the replay tool.

Shrinking a communicator
------------------------

The :c:func:`ncclCommShrink` function allows you to create a new communicator by removing specific ranks from an existing one.
This is useful when you need to exclude certain GPUs or nodes from a collective operation, for example in fault tolerance scenarios or when dynamically adjusting resource utilization.

The following example demonstrates how to create a new communicator by excluding rank 1:

.. code:: C

  int excludeRanks[] = {1};  // Rank to exclude
  int excludeCount = 1;      // Number of ranks to exclude
  ncclComm_t newcomm;

  // Only ranks that will be in the new communicator should call ncclCommShrink
  if (myRank != 1) {
    ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL, NCCL_SHRINK_DEFAULT);
    if (res != ncclSuccess) {
      // Handle error
    }
    // Use the new communicator for collective operations
    // ...
    // When done, destroy the new communicator
    ncclCommDestroy(newcomm);
  }

When recovering from communication errors, you may want to use the error mode:

.. code:: C

  if (myRank != 1) {
    // When shrinking after an error, use NCCL_SHRINK_ABORT to abort operations on the parent communicator
    // This mode is also useful when there might be ongoing operations on the parent communicator
    ncclResult_t res = ncclCommShrink(comm, excludeRanks, excludeCount, &newcomm, NULL, NCCL_SHRINK_ABORT);
    // ...
  }

Note that:

1. Only ranks that will be part of the new communicator should call :c:func:`ncclCommShrink`.
2. Ranks listed in the exclusion list should not call this function.
3. The new communicator will have ranks re-ordered to maintain contiguous numbering.
4. You can use the ncclGroupStart/ncclGroupEnd mechanism to synchronize the creation of new communicators.

Related link: :c:func:`ncclCommShrink`

Growing a communicator
----------------------

The :c:func:`ncclCommGrow` function allows you to create a new communicator by adding new ranks to an existing one.
This is useful when you need to dynamically scale up your computation by adding more GPUs or nodes to a running collective operation.

Growing a communicator involves coordination between existing ranks (from the parent communicator) and new ranks (joining the communicator).
The process requires a coordinator rank from the existing communicator to generate a unique identifier using :c:func:`ncclCommGetUniqueId`,
which is then distributed to all new ranks through an out-of-band mechanism (e.g., MPI, sockets, or shared memory).

The following example demonstrates how to grow a 4-rank communicator to 8 ranks:

.. code:: C

  // Step 1: Coordinator (e.g., rank 0) generates the grow identifier
  ncclUniqueId growId;
  if (myRank == 0) {
    ncclResult_t res = ncclCommGetUniqueId(comm, &growId);
    if (res != ncclSuccess) {
      // Handle error
    }
    // Distribute growId to all new ranks using out-of-band communication
    // (e.g., MPI_Send, sockets, shared memory, etc.)
  }

  // Step 2: All existing ranks call ncclCommGrow
  ncclComm_t newcomm;
  ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, NULL);
  if (res != ncclSuccess) {
    // Handle error
  }

  // Step 3: New ranks (4-7) call ncclCommGrow with the received growId
  cudaSetDevice(myDevice);
  ncclComm_t newcomm;
  ncclResult_t res = ncclCommGrow(NULL, 8, &growId, myNewRank, &newcomm, NULL);

  // Step 4: Wait for grow operation to complete (if non-blocking)
  ncclResult_t asyncErr;
  do {
    res = ncclCommGetAsyncError(newcomm, &asyncErr);
  } while (asyncErr == ncclInProgress);

  // Step 5: Use the new communicator for collective operations
  // ...

  // Step 6: Existing ranks should destroy the parent communicator
  ncclCommDestroy(comm);

  // Step 7: When done, destroy the new communicator
  ncclCommDestroy(newcomm);

For non-blocking grow operations with error handling:

.. code:: C

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;  // Non-blocking mode

  // Existing ranks
  ncclComm_t newcomm;
  ncclResult_t res = ncclCommGrow(comm, 8, NULL, -1, &newcomm, &config);

  // Poll for completion
  ncclResult_t asyncErr;
  do {
    res = ncclCommGetAsyncError(newcomm, &asyncErr);
    if (res != ncclSuccess) {
      // Handle error
      ncclCommAbort(newcomm);
      break;
    }
    // Handle timeouts or other events
  } while (asyncErr == ncclInProgress);

  if (asyncErr == ncclSuccess) {
    // Grow completed successfully
    // Destroy parent communicator
    ncclCommDestroy(comm);
  }

Important considerations:

1. **Coordinator selection**: Any rank from the existing communicator can be the coordinator. The coordinator calls :c:func:`ncclCommGetUniqueId` to generate the grow identifier.

2. **Rank assignment**: Existing ranks retain their original rank numbers in the new communicator. New ranks must be assigned ranks starting from the size of the parent communicator.

3. **Out-of-band communication**: The grow identifier must be distributed from the coordinator to all new ranks using a communication mechanism outside of NCCL (e.g., MPI, sockets, shared files).

4. **Parent communicator cleanup**: After the grow operation completes successfully, existing ranks should destroy the parent communicator using :c:func:`ncclCommDestroy` to free resources.

5. **No outstanding operations**: There should not be any outstanding NCCL operations on the parent communicator when calling :c:func:`ncclCommGrow` to avoid potential deadlocks.

6. **Configuration inheritance**: The new communicator inherits the configuration from the parent communicator for existing ranks. New ranks use the provided configuration or default settings.

Related links:

* :c:func:`ncclCommGrow`
* :c:func:`ncclCommGetUniqueId`

Creating more communicators
---------------------------

The ncclCommSplit function can be used to create communicators based on an existing one. This allows splitting an existing
communicator into multiple sub-partitions, duplicate an existing communicator, or even create a single communicator with
fewer ranks.

The ncclCommSplit function needs to be called by all ranks in the original communicator. If some ranks will not be part
of any sub-group, they still need to call ncclCommSplit with color being NCCL_SPLIT_NOCOLOR.

Newly created communicators will inherit the parent communicator configuration (e.g. non-blocking).
If the parent communicator operates in non-blocking mode, a ncclCommSplit operation may be stopped by calling ncclCommAbort
on the parent communicator, then on any new communicator returned. This is because a hang could happen during
operations on any of the two communicators.

The following code duplicates an existing communicator:

.. code:: C

 int rank;
 ncclCommUserRank(comm, &rank);
 ncclCommSplit(comm, 0, rank, &newcomm, NULL);

This splits a communicator in two halves:

.. code:: C

 int rank, nranks;
 ncclCommUserRank(comm, &rank);
 ncclCommCount(comm, &nranks);
 ncclCommSplit(comm, rank/(nranks/2), rank%(nranks/2), &newcomm, NULL);

This creates a communicator with only the first 2 ranks:

.. code:: C

 int rank;
 ncclCommUserRank(comm, &rank);
 ncclCommSplit(comm, rank<2 ? 0 : NCCL_SPLIT_NOCOLOR, rank, &newcomm, NULL);


Related links:

 * :c:func:`ncclCommSplit`

.. _multi-thread-concurrent-usage:

Using multiple NCCL communicators concurrently
----------------------------------------------

Prior to NCCL 2.26, using multiple NCCL communicators per-device required serializing the order of all communication operations (via CUDA stream dependencies or synchronization) into a consistent total global order otherwise deadlocks could ensue. As of 2.26, NCCL introduces :ref:`NCCL_LAUNCH_ORDER_IMPLICIT` which when enabled implicitly creates this order dynamically by following the order operations are issued from the host. Thus to remain deadlock free, users must ensure the order of host-side launches matches for all devices. This is most easily accomplished by using a determinstic order issued from a single host thread per-device. For example:

.. code:: C

  ncclAllReduce(..., comm1, stream1); // all ranks do this first
  ncclAllReduce(..., comm2, stream2); // and this second

When NCCL is captured in a CUDA graph the same rules apply to both capture time and launch time. At capture time this means NCCL calls in the same graph must be captured in the same order:

.. code:: C

  // both stream1 and stream2 are capturing in the same graph
  ncclAllReduce(..., comm1, stream1); // all ranks do this first
  ncclAllReduce(..., comm2, stream2); // and this second

And at graph launch time different graphs must be launched in a globally consistent order:

.. code:: C

  cudaGraphLaunch(graph1, stream1); // all ranks do this first
  cudaGraphLaunch(graph2, stream2); // and this second

When running on CUDA 12.3 or later, the implicit ordering of the operations is created using CUDA launch completion events which permits parallel execution of the two communicator's kernels.


Finalizing a communicator
-------------------------

ncclCommFinalize will transition a communicator from the *ncclSuccess* state to the *ncclInProgress* state, start
completing all operations in the background and synchronize with other ranks which may be using resources for their
communications with other ranks.
All uncompleted operations and network-related resources associated to a communicator will be flushed and freed with
ncclCommFinalize.
Once all NCCL operations are complete, the communicator will transition to the *ncclSuccess* state. Users can
query that state with ncclCommGetAsyncError.
If a communicator is marked as nonblocking, this operation is nonblocking; otherwise, it is blocking.

Related link: :c:func:`ncclCommFinalize`

Destroying a communicator
-------------------------

Once a communicator has been finalized, the next step is to free all resources, including the communicator itself.
Local resources associated to a communicator can be destroyed with ncclCommDestroy. If the state of a communicator
is *ncclSuccess* when calling ncclCommDestroy, the call is guaranteed to be nonblocking; otherwise
ncclCommDestroy might block.
In all cases, ncclCommDestroy call will free the resources of the communicator and return, and
the communicator should no longer be accessed after ncclCommDestroy returns.

Related link: :c:func:`ncclCommDestroy`

*************************************
Error handling and communicator abort
*************************************

All NCCL calls return a NCCL error code which is summarized in the table below. If a NCCL call returns an error code
different from ncclSuccess and ncclInternalError, and if NCCL_DEBUG is set to WARN, NCCL will print a human-readable
message explaining what happened.
If NCCL_DEBUG is set to INFO, NCCL will also print the call stack which led to the error.
This message is intended to help the user fix the problem.

The table below summarizes how different errors should be understood and handled. Each case is explained in details
in the following sections.

.. list-table:: NCCL Errors
   :widths: 20 50 10 10 10
   :header-rows: 1

   * - Error
     - Description
     - Resolution
     - Error handling
     - Group behavior
   * - ncclSuccess
     - No error
     - None
     - None
     - None
   * - ncclUnhandledCudaError
     - Error during a CUDA call (1)
     - CUDA configuration / usage (1)
     - Communicator abort (5)
     - Global (6)
   * - ncclSystemError
     - Error during a system call (1)
     - System configuration / usage (1)
     - Communicator abort (5)
     - Global (6)
   * - ncclInternalError
     - Error inside NCCL (2)
     - Fix in NCCL (2)
     - Communicator abort (5)
     - Global (6)
   * - ncclInvalidArgument
     - An argument to a NCCL call is invalid (3)
     - Fix in the application (3)
     - None (3)
     - Individual (3)
   * - ncclInvalidUsage
     - The usage of NCCL calls is invalid (4)
     - Fix in the application (4)
     - Communicator abort (5)
     - Global (6)
   * - ncclInProgress
     - The NCCL call is still in progress
     - Poll for completion using ncclCommGetAsyncError
     - None
     - None


(1) ncclUnhandledCudaError and ncclSystemError indicate that a call NCCL made to an external component failed,
which caused the NCCL operation to fail. The error message should explain which component the user should look
at and try to fix, potentially with the help of the administrators of the system.

(2) ncclInternalError denotes a NCCL bug. It might not report a message with NCCL_DEBUG=WARN since it requires a
fix in the NCCL source code. NCCL_DEBUG=INFO will print the back trace which led to the error.

(3) ncclInvalidArgument indicates an argument value is incorrect, like a NULL pointer or an out-of-bounds value.
When this error is returned, the NCCL call had no effect. The group state remains unchanged, the communicator is
still functioning normally. The application can call ncclCommAbort or continue as if the call did not happen.
This error will be returned immediately for a call happening within a group and applies to that specific NCCL
call. It will not be returned by ncclGroupEnd since ncclGroupEnd takes no argument.

(4) ncclInvalidUsage is returned when a dynamic condition causes a failure, which denotes an incorrect usage of
the NCCL API.

(5) These errors are fatal for the communicator. To recover, the application needs to call ncclCommAbort on the
communicator and re-create it.

(6) Dynamic errors for operations within a group are always reported by ncclGroupEnd and apply to all operations
within the group, which may or may not have completed. The application must call ncclCommAbort on all communicators
within the group.

Asynchronous errors and error handling
--------------------------------------

Some communication errors, and in particular network errors, are reported through the ncclCommGetAsyncError function.
Operations experiencing an asynchronous error will usually not progress and never complete. When an asynchronous error
happens, the operation should be aborted and the communicator destroyed using ncclCommAbort.
When waiting for NCCL operations to complete, applications should call ncclCommGetAsyncError and destroy the
communicator when an error happens.

The following code shows how to wait on NCCL operations and poll for asynchronous errors, instead of using
cudaStreamSynchronize.

.. code:: C

 int ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm) {
   cudaError_t cudaErr;
   ncclResult_t ncclErr, ncclAsyncErr;
   while (1) {
    cudaErr = cudaStreamQuery(stream);
    if (cudaErr == cudaSuccess)
      return 0;

    if (cudaErr != cudaErrorNotReady) {
      printf("CUDA Error : cudaStreamQuery returned %d\n", cudaErr);
      return 1;
    }

    ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
    if (ncclErr != ncclSuccess) {
      printf("NCCL Error : ncclCommGetAsyncError returned %d\n", ncclErr);
      return 1;
    }

    if (ncclAsyncErr != ncclSuccess) {
      // An asynchronous error happened. Stop the operation and destroy
      // the communicator
      ncclErr = ncclCommAbort(comm);
      if (ncclErr != ncclSuccess)
        printf("NCCL Error : ncclCommDestroy returned %d\n", ncclErr);
      // Caller may abort or try to create a new communicator.
      return 2;
    }

    // We might want to let other threads (including NCCL threads) use the CPU.
    sched_yield();
   }
 }

Related links:

 * :c:func:`ncclCommGetAsyncError`
 * :c:func:`ncclCommAbort`

.. _ft:

***************
Fault Tolerance
***************

NCCL provides a set of features to allow applications to recover from fatal errors such as a network failure,
a node failure, or a process failure. When such an error happens, the application should be able to call *ncclCommAbort*
on the communicator to free all resources, then create a new communicator to continue.

For more advanced recovery, the *ncclCommShrink* function with *NCCL_SHRINK_ABORT* can be used to create a new communicator
by removing failed ranks from the existing communicator while safely handling in-progress operations. This approach is
particularly useful in distributed environments where only some ranks have failed.

In order to abort NCCL communicators safely, NCCL requires applications to set communicators as nonblocking and make sure
no thread is calling any NCCL operations while calling *ncclCommAbort*. After nonblocking is set, all NCCL calls
(except *ncclCommDestroy/Abort*) become nonblocking so that *ncclCommAbort* can be called at any point, during initialization,
communication or finalizing the communicator. If NCCL communicators are set blocking, the thread can possibly get stuck inside
NCCL calls due to network errors; in this case, NCCL communicators might hang forever.

To correctly abort, when any rank in a communicator fails (e.g., due to a segmentation fault), all other ranks need to
call *ncclCommAbort* to abort their own NCCL communicator.
Users can implement methods to decide when and whether to abort the communicators and restart the NCCL operation.
Here is an example showing how to initialize and split a communicator in a non-blocking manner, allowing for an abort at any point:

.. code:: C

  bool globalFlag;
  bool abortFlag = false;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  /* set communicator as nonblocking */
  config.blocking = 0;
  CHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
  do {
    CHECK(ncclCommGetAsyncError(comm, &state));
  } while(state == ncclInProgress && checkTimeout() != true);

  if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

  /* sync abortFlag among all healthy ranks. */
  reportErrorGlobally(abortFlag, &globalFlag);

  if (globalFlag) {
    /* time is out or initialization failed: every rank needs to abort and restart. */
    ncclCommAbort(comm);
    /* restart NCCL; this is a user implemented function, it might include
     * resource cleanup and ncclCommInitRankConfig() to create new communicators. */
    restartNCCL(&comm);
  }

  /* nonblocking communicator split. */
  CHECK(ncclCommSplit(comm, color, key, &childComm, &config));
  do {
    CHECK(ncclCommGetAsyncError(comm, &state));
  } while(state == ncclInProgress && checkTimeout() != true);

  if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

  /* sync abortFlag among all healthy ranks. */
  reportErrorGlobally(abortFlag, &globalFlag);

  if (globalFlag) {
    ncclCommAbort(comm);
    /* if chilComm is not NCCL_COMM_NULL, user should abort child communicator
     * here as well for resource reclamation. */
    if (childComm != NCCL_COMM_NULL) ncclCommAbort(childComm);
    restartNCCL(&comm);
  }
  /* application workload */

The *checkTimeout* function needs to be provided by users to determine what is the longest time the application should wait for
NCCL initialization; likewise, users can apply other methods to detect errors besides a timeout function. Similar methods can be applied
to NCCL finalization as well.

******************
Quality of Service
******************

Applications which overlap communication may benefit from network Quality of
Service (QoS) features. NCCL allows an application to assign a traffic class (TC) to
each communicator to identify the communication requirements of the
communicator. All network operations on a communicator will use the assigned
TC.

The meaning of TC is specific to the network plugin in use by the communicator
(e.g. IB networks use service level, RoCE networks use type of service). TCs are defined
by the system configuration. Applications must understand the TCs available on
a system and their relative behavior in order to use them effectively.

TC is specified during communicator creation using :ref:`ncclconfig`.

.. code:: C

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.trafficClass = 1;
  CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

Infiniband networks support QoS through the use of Service Levels (SL). Each IB SL
is mapped to Virtual Lane (VL), which defines the relative priority of traffic. SL
behavior is defined within the subnet manager, such as OpenSM. Refer to subnet
manager documentation for more detail. An example configuration is shown below.

.. code:: C

  ...
  qos_max_vls 2
  qos_high_limit 255
  qos_vlarb_high 1:4
  qos_vlarb_low 0:1,1:4
  qos_sl2vl 0,1

  max_op_vls 2
  ....

The example defines one low priority and one high priority VL which
are mapped to SL 0 and 1, respectively. The high priority SL will be
given a larger share of network bandwidth at each port. In NCCL, the
communicator's traffic class corresponds to the SL on IB networks. Using
this configuration, applications can assign TC 0 to low-priority communicators
and TC 1 to high-priority ones.

On RoCE networks, the NCCL communicator trafficClass is interpreted as an IP
Type of Service (ToS). Refer to network management tools to understand how to
configure QoS for a given workload.
