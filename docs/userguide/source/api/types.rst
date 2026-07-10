*****
Types
*****

The following types are used by the NCCL library.

ncclComm_t
----------

.. c:type:: ncclComm_t

 NCCL communicator. Points to an opaque structure inside NCCL.

ncclResult_t
------------

.. c:type:: ncclResult_t

 Return values for all NCCL functions. Possible values are:

 .. c:macro:: ncclSuccess

   (``0``)
   Function succeeded.
 .. c:macro:: ncclUnhandledCudaError

   (``1``)
   A call to a CUDA function failed.

 .. c:macro:: ncclSystemError

   (``2``)
   A call to the system failed.

 .. c:macro:: ncclInternalError

   (``3``)
   An internal check failed. This is due to either a bug in NCCL or a memory corruption.

 .. c:macro:: ncclInvalidArgument

   (``4``)
   An argument has an invalid value.

 .. c:macro:: ncclInvalidUsage

   (``5``)
   The call to NCCL is incorrect. This is usually reflecting a programming error.

 .. c:macro:: ncclRemoteError

   (``6``)
   A call failed possibly due to a network error or a remote process exiting prematurely.

 .. c:macro:: ncclInProgress

   (``7``)
   A NCCL operation on the communicator is being enqueued and is being progressed in the background.

 Whenever a function returns an error (neither ncclSuccess nor ncclInProgress), NCCL should print a more detailed message when the environment variable :ref:`NCCL_DEBUG` is set to "WARN".

ncclDataType_t
--------------

.. c:type:: ncclDataType_t

 NCCL defines the following integral and floating data-types.

 .. c:macro:: ncclInt8

  Signed 8-bits integer

 .. c:macro:: ncclChar

  Signed 8-bits integer

 .. c:macro:: ncclUint8

  Unsigned 8-bits integer

 .. c:macro:: ncclInt32

  Signed 32-bits integer

 .. c:macro:: ncclInt

  Signed 32-bits integer

 .. c:macro:: ncclUint32

  Unsigned 32-bits integer

 .. c:macro:: ncclInt64

  Signed 64-bits integer

 .. c:macro:: ncclUint64

  Unsigned 64-bits integer

 .. c:macro:: ncclFloat16

  16-bits floating point number (half precision)

 .. c:macro:: ncclHalf

  16-bits floating point number (half precision)

 .. c:macro:: ncclFloat32

  32-bits floating point number (single precision)

 .. c:macro:: ncclFloat

  32-bits floating point number (single precision)

 .. c:macro:: ncclFloat64

  64-bits floating point number (double precision)

 .. c:macro:: ncclDouble

  64-bits floating point number (double precision)

 .. c:macro:: ncclBfloat16

  16-bits floating point number (truncated precision in bfloat16 format, CUDA 11 or later)

 .. c:macro:: ncclFloat8e4m3

  8-bits floating point number, 4 exponent bits, 3 mantissa bits (CUDA >= 11.8 and SM >= 90)

 .. c:macro:: ncclFloat8e5m2

  8-bits floating point number, 5 exponent bits, 2 mantissa bits (CUDA >= 11.8 and SM >= 90)


ncclRedOp_t
-----------

.. c:type:: ncclRedOp_t

 Defines the reduction operation.

 .. c:macro:: ncclSum

  Perform a sum (+) operation

 .. c:macro:: ncclProd

  Perform a product (*) operation

 .. c:macro:: ncclMin

  Perform a min operation

 .. c:macro:: ncclMax

 Perform a max operation

 .. c:macro:: ncclAvg

 Perform an average operation, i.e. a sum across all ranks, divided by the number of ranks.


ncclScalarResidence_t
---------------------

.. c:type:: ncclScalarResidence_t

 Indicates where (memory space) scalar arguments reside and when they can be
 dereferenced.

 .. c:macro:: ncclScalarHostImmediate

  The scalar resides in host memory and should be dereferenced in the most immediate
  way.

 .. c:macro:: ncclScalarDevice

  The scalar resides on device visible memory and should be dereferenced once
  needed.

.. _ncclconfig:

ncclConfig_t
------------

.. c:type:: ncclConfig_t

 A structure-based configuration users can set to initialize a communicator; a
 newly created configuration must be initialized by NCCL_CONFIG_INITIALIZER.

 .. c:macro:: NCCL_CONFIG_INITIALIZER

  A configuration macro initializer which must be assigned to a newly created configuration.

 .. c:macro:: blocking

  This attribute can be set as integer 0 or 1 to indicate nonblocking or blocking
  communicator behavior correspondingly. Blocking is the default behavior.

 .. c:macro:: cgaClusterSize

  Set Cooperative Group Array (CGA) size of kernels launched by NCCL.
  This attribute can be set between 0 and 8, and the default value is 4 since sm90 architecture
  and 0 for older architectures.

 .. c:macro:: minCTAs

  Set the minimal number of CTAs NCCL should use for each kernel.
  Set to a positive integer value, up to 32. The default value is 1.

 .. c:macro:: maxCTAs

  Set the maximal number of CTAs NCCL should use for each kernel.
  Set to a positive integer value, up to 32. The default value is 32.

 .. c:macro:: netName

  Specify the network module name NCCL should use for network communication. The value of netName must match
  exactly the name of the network module (case-insensitive). NCCL internal network module names are "IB"
  (generic IB verbs) and "Socket" (TCP/IP sockets). External network plugins define their own names.
  The default value is undefined, and NCCL will choose the network module automatically.

 .. c:macro:: splitShare

  Specify whether to share resources with child communicator during communicator split.
  Set the value of splitShare to 0 or 1. The default value is 0.
  When the parent communicator is created with `splitShare=1` during `ncclCommInitRankConfig`, the child
  communicator can share internal resources of the parent during communicator split. Split communicators
  are in the same family. When resources are shared, aborting any communicator can result in
  other communicators in the same family becoming unusable. Irrespective of whether sharing resources or not, users should
  always abort/destroy all no longer needed communicators to free up resources.
  Note: when the parent communicator has been revoked, resource sharing during split is disabled regardless of this flag.

 .. c:macro:: shrinkShare

  Specify whether to share resources with child communicator during communicator shrink.
  Set the value of shrinkShare to 0 or 1. The default value is 0.
  Note: when shrink is used with NCCL_SHRINK_ABORT, the value of shrinkShare is ignored and no resources are shared. When the parent communicator has been revoked, resource sharing is also disabled.
  The behavior of this flag is similar to `splitShare`, see above.

 .. c:macro:: trafficClass

  Set the traffic class (TC) to use for network operations on the communicator.
  The meaning of TC is specific to the network plugin in use by the
  communicator (e.g. IB networks use service level, RoCE networks use type of service).
  Assigning different TCs to each communicator can benefit workloads which
  overlap communication. TCs are defined by the system configuration and should be greater
  than or equal to 0. Note that environment variables, such as `NCCL_IB_SL` and `NCCL_IB_TC`,
  take precedence over user-specified TC values. To utilize user-defined TCs, ensure that
  these environment variables are unset.

 .. c:macro:: collnetEnable

  Set 1/0 to enable/disable IB SHARP on the communicator. The default value is 0 (disabled).

 .. c:macro:: CTAPolicy

  Set the policy for the communicator. The full list of supported policies can be found in :ref:`cta_policy_flags`.
  The default value is `NCCL_CTA_POLICY_DEFAULT`.

 .. c:macro:: nvlsCTAs

  Set the total number of CTAs NCCL should use for NVLS kernels.
  Set to a positive integer value. By default, NCCL will automatically determine the best number of CTAs based on
  the system configuration.

 .. c:macro:: commName

  Specify the user defined name for the communicator.
  The communicator name can be used by NCCL to enrich logging and profiling.

 .. c:macro:: nChannelsPerNetPeer

  Set the number of network channels to be used for pairwise communication. The value must be a positive integer and will be round up to the next power of 2. The default value is optimized for the AlltoAll communication pattern. Consider increasing the value to increase the bandwidth for send/recv communication.

 .. c:macro:: graphUsageMode

  Set the graph usage mode for the communicator. It support three possible values: 0 (no graphs), 1 (one graph) and 2 (either multiple graphs or mix of graph and non-graph). The default value is 2. If :ref:`NCCL_GRAPH_STREAM_ORDERING` or :c:macro:`graphStreamOrdering` disables capture-time stream ordering (``0``), **graph mixing must be off**—use ``graphUsageMode`` ``0`` or ``1`` only; ``graphUsageMode=2`` must not be combined with ordering ``0`` (see :ref:`NCCL_GRAPH_STREAM_ORDERING`).

 .. c:macro:: graphStreamOrdering

  (since 2.30)

  Per-communicator override of :ref:`NCCL_GRAPH_STREAM_ORDERING`. ``1`` keeps
  NCCL's default capture-time serialization of communication kernels. ``0``
  disables it for this communicator—kernels are placed on the capture stream
  and the application must guarantee correct ordering (see
  :ref:`NCCL_GRAPH_STREAM_ORDERING`).

  Defaults to ``NCCL_CONFIG_UNDEF_INT`` (inherits
  :ref:`NCCL_GRAPH_STREAM_ORDERING`). ``0`` or ``1`` overrides the env var
  for this communicator.

  ``graphStreamOrdering=0`` requires ``graphUsageMode`` ``0`` or ``1``
  (mixing **off**). Combining it with ``graphUsageMode=2`` is **not
  supported**; see :ref:`NCCL_GRAPH_STREAM_ORDERING`.

  **Mixed values on one GPU:** A communicator set to ``1`` still receives
  NCCL's internal serialization for its own kernels, but NCCL does **not**
  insert cross-communicator ordering with a peer set to ``0``—its kernels may
  overlap in situations NCCL would have serialized. Use ``0`` only when the
  application guarantees ordering of **all** NCCL communication kernels that
  may run concurrently on the GPU.

 .. c:macro:: maxP2pPeers

  Set the maximum number of peers any rank will concurrently communicate with using P2P communication. Setting this value will influence all send/recv and send/recv-based collectives (all-to-all, scatter, gather). Values less than one or greater than the number of ranks will default to the number of ranks in the communicator.

.. _ncclsiminfo:

ncclSimInfo_t
-------------

.. c:type:: ncclSimInfo_t

 This struct will be used by ncclGroupSimulateEnd() to return information about the calls.

 .. c:macro:: NCCL_SIM_INFO_INITIALIZER

 NCCL_SIM_INFO_INITIALIZER is a configuration macro initializer which must be assigned
 to a newly created ncclSimInfo_t struct.

 .. c:macro:: estimatedTime

 Estimated time for the operation(s) in the group call will be returned in this attribute.

ncclCommMemStat_t
-----------------

.. c:type:: ncclCommMemStat_t

 Memory statistic selectors for :c:func:`ncclCommMemStats`.

 .. c:macro:: ncclStatGpuMemSuspend

  Communicator allocated GPU memory that can be released via suspend (bytes).

 .. c:macro:: ncclStatGpuMemSuspended

  Whether communicator allocated GPU memory is currently suspended (``0`` = active, ``1`` = suspended).

 .. c:macro:: ncclStatGpuMemPersist

  Communicator allocated GPU memory that cannot be suspended (bytes).

 .. c:macro:: ncclStatGpuMemTotal

  Total communicator allocated GPU memory that is tracked by NCCL (bytes).

ncclWindow_t
------------

.. c:type:: ncclWindow_t

  NCCL window object for window registration and deregistration.
