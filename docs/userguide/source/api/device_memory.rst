.. _device_api_memory:

Device API – Memory and LSA
***************************

This page documents device-side memory and LSA (load/store accessible) functionality. For host-accessible device
pointer functions, see :ref:`Host-Accessible Device Pointer Functions <device_api_host_functions>` in the setup guide.

Memory types
============

.. cpp:class:: template<typename T> ncclSymPtr

   A symmetric pointer represented by a NCCL window and byte offset. It does
   not by itself identify a peer. APIs accepting ``ncclSymPtr`` combine it with
   a team, multimem handle, or device communicator to select the target memory.
   See :ref:`device_api_reducecopy` for usage examples.

.. c:type:: ncclMultimemHandle_t

   Handle to a multicast mapping for a team. It can be passed to multimem
   pointer and reduce/copy APIs.

.. c:type:: ncclDevResourceHandle_t

   Handle identifying an allocation in a device communicator's
   symmetric resource buffer.

LSA
===

**Device functions.** The following are callable from device (GPU) code only. LSA is used by the pointer accessors
below.

ncclLsaBarrierSession
---------------------

.. cpp:class:: template<typename Coop> ncclLsaBarrierSession

   A class representing a memory barrier session.

   .. cpp:function:: ncclLsaBarrierSession(Coop coop, ncclDevComm const& comm, ncclTeamTagLsa tag, uint32_t index, bool multimem=false)

      Initializes a new memory barrier session.  *coop* represents a cooperative group (see :ref:`devapi_coops`).
      *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
      *ncclTeamTagLsa* is here to indicate which subset of ranks the barrier will apply to.  The identifier of the underlying
      barrier to use is provided by *index* (it should be different for each *coop*; typically set to ``blockIdx.x`` to
      ensure uniqueness between CTAs).  *multimem* requests a hardware-accelerated implementation using memory multicast.

   .. cpp:function:: void arrive(Coop, cuda::memory_order order)

      Signals the arrival of the thread at the barrier session.

   .. cpp:function:: void wait(Coop, cuda::memory_order order)

      Blocks until all threads of all team members arrive at the barrier session.

   .. cpp:function:: void sync(Coop, cuda::memory_order order)

      Synchronizes all threads of all team members that participate in the barrier session (combines ``arrive`` and
      ``wait``).

.. c:type:: ncclLsaBarrierHandle_t

   Opaque handle that identifies a set of LSA barriers. It is valid until the
   device communicator is destroyed.

ncclGetPeerPointer
------------------

.. cpp:function:: void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer)

   Returns a load/store accessible pointer to the memory buffer of device *peer* within the window *w*.  *offset* is
   byte-based.  *peer* is a rank index within the world team (see :ref:`devapi_teams`).  This function will return an invalid pointer if
   *peer* is not load/store accessible from the calling rank.

.. cpp:function:: void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam team, int peer)

   Returns a load/store-accessible pointer to *offset* in *w* on rank *peer* of
   *team*. The selected peer must be load/store accessible from the calling
   rank. The function does not validate this.

ncclGetLsaPointer
-----------------

.. cpp:function:: void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer)

   Returns a load/store accessible pointer to the memory buffer of device *lsaPeer* within the window *w*.  *offset* is
   byte-based.  This is similar to ``ncclGetPeerPointer``, but here *lsaPeer* is a rank index within the :ref:`LSA <device_api_lsa>` team (see
   :ref:`devapi_teams`). For high-level reduce and copy operations over LSA memory, see :ref:`device_api_reducecopy`.

ncclGetLocalPointer
-------------------

.. cpp:function:: void* ncclGetLocalPointer(ncclWindow_t w, size_t offset)

   Returns a load-store accessible pointer to the memory buffer of the current device within the window *w*.  *offset*
   is byte-based.  This is just a shortcut version of ``ncclGetPeerPointer`` with *devComm.rank* as *peer*, or
   ``ncclGetLsaPointer`` with *devComm.lsaRank* as *lsaPeer*.

Multimem
========

ncclGetMultimemPointer
----------------------

.. cpp:function:: void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle_t multimem)

   Returns a multicast memory pointer to byte *offset* in *w* using *multimem*. The
   handle must describe a multimem mapping for a team containing the window.
   The function does not validate the handle.

ncclGetLsaMultimemPointer
-------------------------

.. cpp:function:: void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& devComm)

   Returns a multicast memory pointer associated with the window *w* and device communicator *devComm*.  *offset*
   is byte-based.  Availability of multicast memory is hardware-dependent. Currently unsupported for memory regions
   that contain host-backed segments (CU_MEM_LOCATION_TYPE_HOST_NUMA).

.. _device_api_resource_buffer:

Resource-buffer addressing
==========================

The following functions compute addresses within a device communicator's
symmetric resource buffer. They do not allocate memory, communicate, or
synchronize. Each handle must belong to the supplied device communicator.

ncclGetResourceBufferOffset
---------------------------

.. cpp:function:: size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t handle)

   Returns the byte offset represented by *handle*. This function is callable
   from host and device code.

ncclGetResourceBuffer
---------------------

.. cpp:function:: ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const& comm, ncclDevResourceHandle_t handle)

   Returns a symmetric pointer to a resource represented by *handle* in
   *comm*'s resource window.

ncclGetResourceBufferLocalPointer
---------------------------------

.. cpp:function:: void* ncclGetResourceBufferLocalPointer(ncclDevComm const& comm, ncclDevResourceHandle_t handle)

   Returns a load/store pointer to a resource represented by *handle* in the
   calling rank's resource buffer.

ncclGetResourceBufferLsaPointer
-------------------------------

.. cpp:function:: void* ncclGetResourceBufferLsaPointer(ncclDevComm const& comm, ncclDevResourceHandle_t handle, int peer)

   Returns a load/store pointer to a resource represented by *handle* on rank
   *peer* of the calling rank's LSA team. *peer* must be in ``[0, ncclTeamLsa(comm).nRanks)``.

ncclGetResourceBufferPeerPointer
--------------------------------

.. cpp:function:: void* ncclGetResourceBufferPeerPointer(ncclDevComm const& comm, ncclDevResourceHandle_t handle, ncclTeam team, int peer)

   Returns a load/store pointer to a resource represented by *handle* on rank
   *peer* of *team*. The selected peer must be load/store accessible from the
   calling rank.

ncclGetResourceBufferMultimemPointer
------------------------------------

.. cpp:function:: void* ncclGetResourceBufferMultimemPointer(ncclDevComm const& comm, ncclDevResourceHandle_t handle, ncclMultimemHandle_t multimem)

   Returns a multicast pointer to a resource represented by *handle* using
   *multimem*. *multimem* must describe a multimem mapping that includes the
   resource window.

ncclGetResourceBufferLsaMultimemPointer
---------------------------------------

.. cpp:function:: void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const& comm, ncclDevResourceHandle_t handle)

   Returns a multicast pointer to a resource represented by *handle* for the
   LSA team.
