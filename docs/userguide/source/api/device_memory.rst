.. _device_api_memory:

Device API – Memory and LSA
***************************

This page documents device-side memory and LSA (load/store accessible) functionality. For host-accessible device
pointer functions, see :ref:`Host-Accessible Device Pointer Functions <device_api_host_functions>` in the setup guide.

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

ncclGetPeerPointer
------------------

.. cpp:function:: void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer)

   Returns a load/store accessible pointer to the memory buffer of device *peer* within the window *w*.  *offset* is
   byte-based.  *peer* is a rank index within the world team (see :ref:`devapi_teams`).  This function will return NULL if
   the *peer* is not within the :ref:`LSA <device_api_lsa>` team.

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

ncclGetLsaMultimemPointer
-------------------------

.. cpp:function:: void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& devComm)

   Returns a multicast memory pointer associated with the window *w* and device communicator *devComm*.  *offset*
   is byte-based.  Availability of multicast memory is hardware-dependent. Currently unsupported for memory regions
   that contain host-backed segments (CU_MEM_LOCATION_TYPE_HOST_NUMA).
