.. _device_api_cft:

Device API - CFT
****************

Compute Fabric Transport
========================

Compute Fabric Transport (CFT) support is available starting with NCCL 2.31.
CFT exposes CUDA fabric logical endpoints to device kernels so they can move
data across a CFT-capable fabric using fabric instructions. NCCL provides the
team, logical endpoint, transfer, reduction, and barrier helpers needed to use
those endpoints from the Device API.

CFT is intended for systems where all participating ranks can create compatible
CUDA fabric logical endpoints. Kernels that use the CFT helpers require CUDA
Toolkit support for fabric PTX instructions and must be compiled for
architectures that support those instructions.

CFT Communicator Requirements
=============================

Request CFT resources when creating the device communicator with :c:func:`ncclDevCommCreate`.

``cftCaps``
-----------

   Bitmask of :c:macro:`NCCL_CFT` and :c:macro:`NCCL_CFT_MULTIMEM` values.
   If CFT resources are requested on a communicator where not all ranks support
   CFT, :c:func:`ncclDevCommCreate` fails.

``cftBarrierCount``
-------------------

   Number of CFT barriers to allocate for :cpp:class:`ncclCftBarrierSession`.
   This should match the number of independently addressed barrier slots used
   by the kernel, commonly one per CTA with ``blockIdx.x`` as the barrier index.

.. code-block:: C

   ncclDevComm devComm;
   ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;

   reqs.cftCaps = NCCL_CFT | NCCL_CFT_MULTIMEM;
   reqs.cftBarrierCount = nCTAs;

   NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

Use :c:macro:`NCCL_CFT` to request unicast CFT logical endpoints. Use :c:macro:`NCCL_CFT_MULTIMEM`
when the kernel also uses multicast CFT operations or multimem CFT barriers.

.. c:macro:: NCCL_CFT_NONE

   No CFT capability is requested.

.. c:macro:: NCCL_CFT

   Request unicast CFT logical endpoint support.

.. c:macro:: NCCL_CFT_MULTIMEM

   Request multicast CFT logical endpoint support.


CFT Teams
=========

CFT operations may use :c:type:`ncclTeam_t` and :cpp:type:`ncclTeam` values to
address peers.

.. c:function:: ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode)

   Returns the host-side unicast CFT team containing ranks that are part of the CFT unicast group.
   *comm* is the host-side communicator and *mode* is a selector used to filter the subset of CFT
   unicast group ranks that are part of the team (useful to build hierarchical communication patterns).

.. c:function:: ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm)

   Returns the host-side multicast CFT team containing all the ranks that are part of the CFT multicast group.

.. cpp:function:: ncclTeam ncclTeamCft(ncclDevComm const& comm, ncclCftTeamMode_t mode)

   Returns the device-side unicast CFT team containing ranks that are part of the CFT unicast group.
   *comm* is the device-side communicator and *mode* is a selector used to filter the subset of CFT
   unicast group ranks that are part of the team (useful to build hierarchical communication patterns).

.. cpp:function:: ncclTeam ncclTeamCftMultimem(ncclDevComm const& comm)

   Returns the device-side multicast CFT team containing all the ranks that are part of the CFT multicast group.

.. c:type:: ncclCftTeamMode_t

   Selects the CFT team layout returned by :c:func:`ncclTeamCft`.

   .. c:macro:: NCCL_CFT_TEAM_FLAT

      Flat CFT team includes all the ranks that are part of the CFT unicast group.

   .. c:macro:: NCCL_CFT_TEAM_HIER_MULTIMEM

      Hierarchical CFT team includes ranks that are part of the CFT unicast group and share the same index across different multicast CFT groups.

   .. c:macro:: NCCL_CFT_TEAM_HIER_LSA

      Hierarchical CFT team includes ranks that are part of the CFT unicast group and share the same index across different LSA groups.

Logical Endpoint Queries
========================

CFT data movement routines operate on a logical endpoint ID and an offset within
that endpoint. NCCL provides host-side and device-side helpers for translating
registered windows and peer ranks into those values.

.. c:function:: ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam, ncclCftLeId* leId, size_t* leOffset)

   Host-side query for the logical endpoint *leId* and *leOffset* corresponding to *window*, byte *offset*, and *peerCft* within *cftTeam*.

.. c:function:: ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId, size_t* leOffset)

   Host-side query for the logical endpoint *leId* and *leOffset* corresponding to *window*, byte *offset* and *peerWorld* within the world team.

.. c:function:: ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId, size_t* leOffset)

   Host-side query for the multicast logical endpoint *leId* and *leOffset* corresponding to *window* and byte *offset*.

.. cpp:function:: void ncclGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam cftTeam, ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset)

   Device-side query for the logical endpoint *leId* and *leOffset* corresponding to *window*, byte *offset*, *peerCft* within *cftTeam* and device *comm*.

.. cpp:function:: void ncclGetPeerLeInfo(ncclWindow_t w, size_t offset, int peerWorld, ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset)

   Device-side query for the logical endpoint *leId* and *leOffset* corresponding to *window*, byte *offset*, *peerWorld* within the world team and device *comm*.

.. cpp:function:: void ncclGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset)

   Device-side query for the multicast logical endpoint *leId* and *leOffset* corresponding to *window*, byte *offset* and device *comm*.

Host-side CFT queries require the communicator to create CFT logical endpoints
before the query is issued. Set ``hostCftMode`` in :c:type:`ncclConfig_t` when
creating the communicator if the application needs to query CFT endpoint
information from host code before creating a CFT-enabled device communicator.

CFT Operations
==============

.. cpp:struct:: ncclCftSmem

   A user-managed shared memory state object used by :cpp:class:`ncclCft` to track progress of outstanding CFT operations.
   The object can track a specific amount of in-flight data, limiting the number of outstanding CFT operations. Specifically,
   the object can track up to 16MB of non-fetching CFT operations (i.e., ``put``, ``red`` and their multicast variants) and up
   to 1MB of fetching CFT operations (i.e., ``get``, ``pullred``).
   
.. cpp:class:: template<typename Coop> ncclCft

   A class encompassing major elements of CFT support.

   .. cpp:function:: ncclCft(Coop coop, ncclCftSmem& cftSmem)

      Initializes a new ``ncclCft`` object. *cftSmem* is a user-managed shared memory state used
      by threads in *coop* to make progress on fabric operations and to check for their completion.

   .. cpp:function:: void put(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes)

      Initiates an asynchronous transfer of *bytes* bytes from the shared memory buffer referenced by *smemSource*
      to the unicast logical endpoint referenced by *leId* at offset *leOffset*.

      *coop* can include a subset of the threads in the cooperative group used to create the CFT object.
      Cooperative groups initiating fabric operations independently under the same CFT object share its state.

   .. cpp:function:: void putCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes, uint16_t cpMask)

      Variant of :cpp:func:`ncclCft::put` with an explicit copy mask. The copy mask is a 16-bit integer where bits
      correspond to the bytes of every 16-byte word being copied. A bit is set to 1 if the corresponding byte ought
      to be copied to the destination logical endpoint, or it is set to 0 otherwise.

   .. cpp:function:: void putMultimem(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes)

      Multicast variant of :cpp:func:`ncclCft::put` where *leId* references a logical endpoint for which the
      the :c:macro:`NCCL_CFT_MULTIMEM` bit was set in the :c:macro:`cftCaps` bitmask at device communicator
      creation time.

   .. cpp:function:: void putMultimemCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource, uint32_t bytes, uint16_t cpMask)

      Multicast variant of :cpp:func:`ncclCft::put` with an explicit copy mask.

   .. cpp:function:: void get(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemDestination, uint32_t bytes)

      Initiates an asynchronous transfer of *bytes* bytes from the unicast logical endpoint referenced by *leId* at offset
      *leOffset* to the shared memory buffer referenced by *smemDestination*.

      *coop* can include a subset of the threads in the cooperative group used to create the CFT object.
      Cooperative groups initiating fabric operations independently under the same CFT object share its state.

   .. cpp:function:: void red(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource, uint32_t bytes)

      Initiates an asynchronous reduction operation (see :ref:`cft_reduction_operators` for reduction tags and data types)
      for *bytes* bytes from the shared memory buffer referenced by *smemSource* to the unicast logical endpoint referenced by
      *leId* at offset *leOffset*.

      *coop* can include a subset of the threads in the cooperative group used to create the CFT object.
      Cooperative groups initiating fabric operations independently under the same CFT object share its state.

   .. cpp:function:: void redMultimem(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource, uint32_t bytes)

      Multicast reduction variant of :cpp:func:`ncclCft::red` where *leId* references a logical endpoint for which
      the :c:macro:`NCCL_CFT_MULTIMEM` bit was set in the :c:macro:`cftCaps` bitmask at device communicator creation
      time.

   .. cpp:function:: void pullRed(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemDestination, uint32_t bytes)

      Initiates an asynchronous reduction operation (see :ref:`cft_reduction_operators` for reduction tags and data types)
      for *bytes* bytes from the multicast logical endpoint referenced by *leId* at offset *leOffset* to the shared memory buffer
      referenced by *smemDestination*. *leId* references a logical endpoint for which the :c:macro:`NCCL_CFT_MULTIMEM` bit
      was set in the :c:macro:`cftCaps` bitmask at device communicator creation time.

      *coop* must be ncclCoopWarp.

   .. cpp:function:: void submit(OpCoop coop)

      Submits previously initiated CFT operations for all the threads in *coop*, allowing them to make forward
      progress. Every cooperative group (``OpCoop``) that initiated fabric operations must call submit before
      :cpp:func:`ncclCft::flushSmem` and :cpp:func:`ncclCft::flush`.

   .. cpp:function:: void flushSmem(OpCoop coop)

      Makes all the threads in *coop* wait for CFT operations that consume shared memory data. All the
      shared memory buffers that source data to outstanding fabric operations can be reused after the
      function returns. Users can call this function to wait for shared memory buffers to be consumed
      by previously initiated CFT operations before initiating new ones, and call :cpp:func:`ncclCft::flush`
      only when they need to wait for the completion of all the outstanding CFT operations, e.g., because
      they reached the number of in-flight bytes that the CFT shared memory state object can track. 

   .. cpp:function:: void flush(OpCoop coop, bool* hasReport = nullptr, uint32_t* report = nullptr)

      Flushes outstanding CFT operations for all the threads in *coop* and waits for their completion.
      When flush returns shared memory buffers are safe to reuse. *coop* must include the threads of
      all cooperative groups that initiated fabric operations under the same CFT object. No other
      cooperative group can initiate fabric operations until flush has completed. If the operation
      succeeds *hasReport* is set to false and *report* is not modified. If the operation fails
      *hasReport* is set to ``true`` and *report* is set to an error code that can be decoded via
      the ``cudaFabricOpErrorStatusGet/Count``.

``smemSource`` and ``smemDestination`` pointers must be 16-byte aligned, and ``bytes`` must be a multiple of 16.
CFT operates on shared memory; a typical kernel stages data through shared memory, issues one or more operations,
then calls :cpp:func:`ncclCft::submit` and :cpp:func:`ncclCft::flush` or :cpp:func:`ncclCft::flushSmem`
as required by the reuse and ordering rules of the kernel.

.. _cft_reduction_operators:

Reduction Operations
====================

Reduction fabric operations use operation tag types:

.. cpp:struct:: ncclCftOpSum
.. cpp:struct:: ncclCftOpAnd
.. cpp:struct:: ncclCftOpXor
.. cpp:struct:: ncclCftOpOr
.. cpp:struct:: ncclCftOpMin
.. cpp:struct:: ncclCftOpMax

Refer to CFT PTX documentation for supported reductions and data types.

CFT Barriers
============

Applications request CFT barrier resources with ``cftBarrierCount`` in
:c:type:`ncclDevCommRequirements`, then use :cpp:class:`ncclCftBarrierSession`
from device code. The barrier count should cover every barrier index used by
the kernel.

.. cpp:class:: template<typename Coop> ncclCftBarrierSession

   A CFT-backed barrier session.

   .. cpp:function:: ncclCftBarrierSession(Coop coop, ncclDevComm const& comm, uint32_t index, bool multimem = false)

      Initializes a new CFT barrier session. *coop* represents a cooperative group (see :ref:`devapi_coops`) of threads
      that synchronize throught the barrier. *comm* is the device communicator for which barrier resources were requested.
      *index* selects the barrier slot from the device communicator. *multimem* can be set to ``true`` to select multicast
      CFT barrier resources (requires CFT barriers to be created with :c:macro:`NCCL_CFT_MULTIMEM` capability).

   .. cpp:function:: void arrive(Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer)

      Signals arrival at the barrier. *order* dictates the semantics used to order memory accesses from different proxies
      around the barrier. *producer* indicates the type of proxy that stored the data into global memory. *consumer*
      indicates the type of proxy that will be loading the *producer* data from global memory.

   .. cpp:function:: void wait(Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer)

      Waits for the barrier to complete. It can be used to establish a ``happens-before`` relationship between *producer*
      and *consumer* when *order* complements the memory ordering semantics of the matching
      :cpp:func:`ncclCftBarrierSession::arrive`.

   .. cpp:function:: void sync(Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer)

      Combines :cpp:func:`ncclCftBarrierSession::arrive` and :cpp:func:`ncclCftBarrierSession::wait`.

   .. cpp:function:: ncclResult_t wait(Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer, uint64_t timeoutCycles)
   .. cpp:function:: ncclResult_t sync(Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer, uint64_t timeoutCycles)

      Timeout-returning variants of :cpp:func:`ncclCftBarrierSession::wait` and
      :cpp:func:`ncclCftBarrierSession::sync`.

.. cpp:function:: void ncclMemFence(Coop coop, cuda::memory_order order, ncclMemProxyType producer, ncclMemProxyType consumer, ncclMemFenceScope scope)

   Issues a memory fence between producer and consumer proxy domains for the participating threads. *coop* represents a cooperative
   group of threads which memory accesses are ordered around the fence. *order* dictates the semantics used to order memory accesses
   from different proxies around the fence. *producer* indicates the type of proxy that stored the data into memory. *consumer*
   indicates the type of proxy that will be loading the *producer* data from memory. *scope* indicates the point of consistency for
   the memory accesses around the fence.

.. cpp:enum-class:: ncclMemProxyType

   Describes which memory proxy produced or consumes memory around a CFT barrier or a fence.

   .. cpp:enumerator:: Generic

      Indicates that the method of memory access is the SM load/store unit.

   .. cpp:enumerator:: Fabric

      Indicates that the method of memory access is the fabric.

.. cpp:enum-class:: ncclMemFenceScope

   Describes the scope of a CFT memory fence.

   .. cpp:enumerator:: Cta

      Indicates that the point of consistency for memory accesses is the SM shared memory.

   .. cpp:enumerator:: Sys

      Indicates that the point of consistency for memory accesses is the L2 cache.
