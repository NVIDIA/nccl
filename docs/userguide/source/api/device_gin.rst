.. _device_api_gin:

Device API – GIN
****************

GIN
===

**Device functions.** The following are callable from device (GPU) code only. GIN is supported since NCCL 2.28.7.

Types and limits
----------------

.. cpp:type:: ncclGinRequest_t

   Opaque request storage initialized by a GIN operation (e.g., :cpp:func:`ncclGin::flushAsync`) and
   consumed by :cpp:func:`ncclGin::wait`.

Shared-memory descriptors
--------------------------

.. cpp:struct:: ncclGinDescriptorSmem

   Optional shared-memory storage for a GIN operation descriptor. Use of caller-supplied storage depends
   on the backend and operation; backends that do not use it ignore it. For operations that use this storage,
   declare an object in CUDA shared memory and keep it available until the operation has completed.
   Concurrent operations must not reuse the same descriptor storage.

.. cpp:struct:: ncclGin_DescriptorSmem

   .. cpp:member:: ncclGinDescriptorSmem* descriptor

      Pointer to the shared-memory descriptor storage supplied by the caller.

Pass ``ncclGin_DescriptorSmem{&storage}`` as the *descriptor* argument of a GIN operation to use that storage.
The default ``ncclGin_None{}`` requests no caller-supplied descriptor.

ncclGin
-------

.. cpp:class:: ncclGin

   A class encompassing major elements of the GIN support.

   .. cpp:function:: ncclGin(ncclDevComm const&, int contextIndex, ncclGinResourceSharingMode resourceSharingMode = NCCL_GIN_RESOURCE_SHARING_GPU)

      Initializes a new ``ncclGin`` object. The first argument is the device communicator created using :c:func:`ncclDevCommCreate`.
      *contextIndex* is the index of the GIN context -- a network communication channel.  Using multiple GIN contexts allows
      the implementation to spread traffic onto multiple connections, avoiding locking and bottlenecks.  Therefore,
      performance-oriented kernels should cycle among the available contexts to improve resource utilization (the number of
      available contexts is available via :c:member:`ncclDevComm.ginContextCount`).

      *resourceSharingMode* selects the scope of resource sharing for this object: GPU-wide
      (``NCCL_GIN_RESOURCE_SHARING_GPU``, the default), within a CTA (``NCCL_GIN_RESOURCE_SHARING_CTA``),
      or per thread (``NCCL_GIN_RESOURCE_SHARING_THREAD``). It is selected per object, not per context.

      ``ncclGin`` always represents one specific context. To run a barrier whose fence covers every GIN context on the comm
      (useful when operations have been sharded across multiple contexts -- e.g. multi-NIC), pass
      ``ncclGinAllContexts(comm)`` to :cpp:func:`ncclGinBarrier` in place of an ``ncclGin``.

   .. cpp:function:: void put(ncclTeam team, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, \
      size_t srcOffset, size_t bytes, \
      RemoteAction remoteAction = ncclGin_None{}, LocalAction localAction = ncclGin_None{}, \
      Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, \
      cuda::thread_scope givenRelease = cuda::thread_scope_thread, \
      cuda::thread_scope requiredRelease = cuda::thread_scope_device, \
      uint32_t optFlags = ncclGinOptFlagsDefault, SegmentType bufType = ncclGin_SegmentDevice{}) const

      Schedules a device-initiated, one-sided data transfer operation from a local buffer to a remote buffer on a peer.

      *peer* is a rank within *team* (see :ref:`devapi_teams`); it may refer to the local rank (a loopback).  The destination
      and source buffers are each specified using the window (*dstWnd*, *srcWnd*) and a byte-based offset (*dstOffset*,
      *srcOffset*).  *bytes* specifies the data transfer count in bytes. If GIN is initialized with connection
      type :c:enumerator:`NCCL_GIN_CONNECTION_RAIL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_RAIL>`, *peer* must be within the same rail team as the local rank.

      Arguments beyond the first seven are optional. *remoteAction* and *localAction* default to
      ``ncclGin_None{}`` (no action). Other action types leave signals and counters unchanged.

      *remoteAction* specifies a signal update on the destination peer after the payload has settled. Supported
      signal actions include:

      * Indexed signals: :cpp:struct:`ncclGin_StrongSignalInc`, :cpp:struct:`ncclGin_StrongSignalAdd`,
        :cpp:struct:`ncclGin_WeakSignalInc`, and :cpp:struct:`ncclGin_WeakSignalAdd`.
      * VA signals: :cpp:struct:`ncclGin_StrongVASignalInc`, :cpp:struct:`ncclGin_StrongVASignalAdd`,
        :cpp:struct:`ncclGin_WeakVASignalInc`, and :cpp:struct:`ncclGin_WeakVASignalAdd`.

      The :cpp:struct:`ncclGin_SignalInc`, :cpp:struct:`ncclGin_SignalAdd`,
      :cpp:struct:`ncclGin_VASignalInc`, and :cpp:struct:`ncclGin_VASignalAdd` actions have been deprecated since NCCL 2.30.5
      and remain supported.

      *localAction* specifies a local counter update after the input has been consumed. The supported counter
      actions are :cpp:struct:`ncclGin_WeakCounterInc` and :cpp:struct:`ncclGin_CounterInc`
      (deprecated since NCCL 2.30.5).
      See :ref:`Signals and Counters <devapi_signals>` below for the completion guarantees of signals and counters.

      *coop* indicates the set of threads participating in this operation (see
      :ref:`devapi_coops`); it defaults to ``ncclCoopThread`` (a single device thread), which is the recommended model.
      *bufType* specifies the physical memory composition of the source and destination buffers (see
      :ref:`devapi_segment_types`); it defaults to ``ncclGin_SegmentDevice``. *optFlags* is a bitwise OR of backend optimization flags
      (see :ref:`gin_opt_flags`) and defaults to ``ncclGinOptFlagsDefault``.

      The visibility of the signal on the destination peer implies the visibility of the put data it is
      attached to. Depending on the signal type, the visibility of the signal may also imply the visibility
      of all the preceding puts to the same peer on the same context.

      The API also defines an alternative, "convenience" variant of this method that uses ``ncclSymPtr`` types to specify the
      buffers and expects size to be conveyed in terms of the number of elements instead of the byte count.  There are also
      two ``putValue`` variants that take a single element at a time (no greater than eight bytes), passed by value.

   .. cpp:function:: template<typename T> void put(ncclTeam team, int peer, ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts, RemoteAction remoteAction = ncclGin_None{}, LocalAction localAction = ncclGin_None{}, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, cuda::thread_scope givenRelease = cuda::thread_scope_thread, cuda::thread_scope requiredRelease = cuda::thread_scope_device, uint32_t optFlags = ncclGinOptFlagsDefault, SegmentType bufType = ncclGin_SegmentDevice{}) const

      Transfers *nElts* elements of type *T* between symbolic pointers. The remaining arguments have the same
      meaning as in the window-and-offset overload; *peer* is a rank within *team*.

   .. cpp:function:: template<typename T> void putValue(ncclTeam team, int peer, ncclWindow_t dstWnd, size_t dstOffset, T value, RemoteAction remoteAction = ncclGin_None{}, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, cuda::thread_scope givenRelease = cuda::thread_scope_thread, cuda::thread_scope requiredRelease = cuda::thread_scope_device, uint32_t optFlags = ncclGinOptFlagsDefault) const
   .. cpp:function:: template<typename T> void putValue(ncclTeam team, int peer, ncclSymPtr<T> dst, T value, RemoteAction remoteAction = ncclGin_None{}, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, cuda::thread_scope givenRelease = cuda::thread_scope_thread, cuda::thread_scope requiredRelease = cuda::thread_scope_device, uint32_t optFlags = ncclGinOptFlagsDefault) const

      Transfers the single *value* to rank *peer* of *team*. ``sizeof(T)`` must be at most eight bytes.
      *dstOffset* is measured in bytes. The optional remote signal and ordering arguments have the same
      meaning as in :cpp:func:`put`. These overloads have no local-counter action argument.

   .. cpp:function:: void get(ncclTeam team, int peer, ncclWindow_t remoteWnd, size_t remoteOffset, ncclWindow_t localWnd, \
      size_t localOffset, size_t bytes, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, \
      uint32_t optFlags = ncclGinOptFlagsDefault, SegmentType bufType = ncclGin_SegmentDevice{}) const

      Schedules a device-initiated, one-sided data transfer operation from a remote buffer to a local buffer
      (available since NCCL 2.30.3).

      *peer* is a rank within *team* (see :ref:`devapi_teams`); it may refer to the local rank (a loopback).  The remote
      and local buffers are each specified using the window (*remoteWnd*, *localWnd*) and a byte-based offset (*remoteOffset*,
      *localOffset*).  *bytes* specifies the data transfer count in bytes. If GIN is initialized with connection
      type :c:enumerator:`NCCL_GIN_CONNECTION_RAIL <ncclGinConnectionType_t.NCCL_GIN_CONNECTION_RAIL>`, *peer* must be within the same rail team as the local rank.
      *bufType* specifies the physical memory composition of the source and destination buffers (see
      :ref:`devapi_segment_types`); it defaults to ``ncclGin_SegmentDevice``.
      *optFlags* is a bitwise OR of optional flags to the backend (see :ref:`gin_opt_flags`); it defaults to
      ``ncclGinOptFlagsDefault``.

   .. cpp:function:: void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire, DescriptorSmem descriptor = ncclGin_None{}) const

      Ensures that all the pending transfer operations scheduled by any threads of *coop* are locally consumed. For put
      operations, this means that the source buffers are safe to reuse; this makes no claims regarding the completion
      status on the remote peer(s). For get operations, this means that the data is visible to the local rank.

   .. cpp:function:: void flushAsync(ncclTeam team, uint32_t peer, ncclGinRequest_t* request, Coop coop = ncclCoopThread{}, \
        uint32_t optFlags = ncclGinOptFlagsDefault, DescriptorSmem descriptor = ncclGin_None{}) const

      Initiates a non-blocking flush operation for one peer (see :cpp:func:`ncclGin::flush`). *peer* is a rank within *team*
      (see :ref:`devapi_teams`). *request* is supplied by the caller and initialized by ``flushAsync``.
      The caller may use *request* to determine when the flush is complete (see :cpp:func:`ncclGin::wait`).
      Available since NCCL 2.30.3.

   .. cpp:function:: void wait(ncclGinRequest_t& request, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, \
        cuda::memory_order ord = cuda::memory_order_acquire) const

      Blocks until *request* is complete. Available since NCCL 2.30.3.

   .. cpp:function:: ncclResult_t wait(ncclGinRequest_t& request, Coop coop, DescriptorSmem descriptor, cuda::memory_order ord, uint64_t timeoutCycles) const

      Waits for *request*, returning :c:enumerator:`ncclTimeout <ncclResult_t.ncclTimeout>` if the wait exceeds
      *timeoutCycles* device clock cycles. Check the returned status before reusing the associated buffers.

   .. cpp:function:: ncclResult_t flush(Coop coop, cuda::memory_order ord, DescriptorSmem descriptor, uint64_t timeoutCycles) const

      Waits for local completion with the same memory-ordering semantics as the void-returning overload.
      Returns :c:enumerator:`ncclTimeout <ncclResult_t.ncclTimeout>` if the wait exceeds *timeoutCycles*
      device clock cycles. A timeout does not establish completion or make outstanding buffers safe to reuse.

.. _devapi_signals:

Signals and Counters
--------------------

.. cpp:type:: ncclGinSignal_t

   Signals are used to trigger actions on remote peers, most commonly on the completion of a :cpp:func:`ncclGin::put` operation.  They each
   have a 64-bit integer value associated with them that can be manipulated atomically.

   Since NCCL 2.30.5, there are two types of signals: *strong* and *weak*. Strong signals imply the visibility of all the
   preceding puts to the same peer on the same context. Weak signals imply only the visibility of the put data the signal is
   attached to.

.. cpp:struct:: ncclGin_StrongSignalInc

   .. cpp:member:: ncclGinSignal_t signal

.. cpp:struct:: ncclGin_StrongSignalAdd

   .. cpp:member:: ncclGinSignal_t signal
   .. cpp:member:: uint64_t value

.. cpp:struct:: ncclGin_WeakSignalInc

   .. cpp:member:: ncclGinSignal_t signal

.. cpp:struct:: ncclGin_WeakSignalAdd

   .. cpp:member:: ncclGinSignal_t signal
   .. cpp:member:: uint64_t value

These objects can be passed as the *remoteAction* arguments of methods such as :cpp:func:`ncclGin::put` and :cpp:func:`ncclGin::signal` to describe the
actions to perform on the peer on receipt -- in this case, increase the value of a *signal* specified by
index. ``SignalInc{signalIdx}`` is functionally equivalent to ``SignalAdd{signalIdx, 1}``; however, it
may not be mixed with other signal-modifying operations without an intervening signal reset (see below).  Signal values
use "rolling" comparison logic to ensure that an unsigned overflow maintains the property of ``x < x + 1``.

.. cpp:struct:: ncclGin_StrongVASignalInc

   .. cpp:member:: ncclWindow_t signalWindow
   .. cpp:member:: size_t signalOffset

.. cpp:struct:: ncclGin_StrongVASignalAdd

   .. cpp:member:: ncclWindow_t signalWindow
   .. cpp:member:: size_t signalOffset
   .. cpp:member:: uint64_t value

.. cpp:struct:: ncclGin_WeakVASignalInc

   .. cpp:member:: ncclWindow_t signalWindow
   .. cpp:member:: size_t signalOffset

.. cpp:struct:: ncclGin_WeakVASignalAdd

   .. cpp:member:: ncclWindow_t signalWindow
   .. cpp:member:: size_t signalOffset
   .. cpp:member:: uint64_t value

These objects represent "VA signals": signals that are located at an arbitrary VA (window and offset pair) instead
of a pre-allocated signal index. Like the ``ncclGin_StrongSignalInc`` and ``ncclGin_StrongSignalAdd`` objects,
these objects can be passed as the *remoteAction* arguments of methods such as :cpp:func:`ncclGin::put`
and :cpp:func:`ncclGin::signal` to increment a signal on the peer. To use a VA signal, the window must be
registered with flags :c:macro:`NCCL_WIN_STRICT_ORDERING`. When an address is used as a signal, all reads
and writes to the address must be issued via GIN (i.e., a ``RemoteAction`` or GIN signal method).

.. cpp:struct:: ncclGin_VASignalInc

   .. deprecated:: 2.30.5

      Prefer :cpp:struct:`ncclGin_StrongVASignalInc` or :cpp:struct:`ncclGin_WeakVASignalInc` instead.

   .. cpp:member:: ncclWindow_t signalWindow
   .. cpp:member:: size_t signalOffset

.. cpp:struct:: ncclGin_VASignalAdd

   .. deprecated:: 2.30.5

      Prefer :cpp:struct:`ncclGin_StrongVASignalAdd` or :cpp:struct:`ncclGin_WeakVASignalAdd` instead.

   .. cpp:member:: ncclWindow_t signalWindow
   .. cpp:member:: size_t signalOffset
   .. cpp:member:: uint64_t value

.. cpp:struct:: ncclGin_SignalInc

   .. deprecated:: 2.30.5

      Prefer :cpp:struct:`ncclGin_StrongSignalInc` or :cpp:struct:`ncclGin_WeakSignalInc` instead.

   .. cpp:member:: ncclGinSignal_t signal

.. cpp:struct:: ncclGin_SignalAdd

   .. deprecated:: 2.30.5

      Prefer :cpp:struct:`ncclGin_StrongSignalAdd` or :cpp:struct:`ncclGin_WeakSignalAdd` instead.

   .. cpp:member:: ncclGinSignal_t signal
   .. cpp:member:: uint64_t value

Since NCCL 2.30.5, these signal types are deprecated in favor of explicitly strong and weak signal objects.
The strength of these signals is determined by the value of :c:member:`ginStrongSignalsRequired <ncclDevCommRequirements.ginStrongSignalsRequired>`
when creating the device communicator.

**Signal methods of** :cpp:class:`ncclGin`:

.. cpp:function:: void ncclGin::signal(ncclTeam team, int peer, RemoteAction remoteAction, Coop coop = ncclCoopThread{}, \
        DescriptorSmem descriptor = ncclGin_None{}, cuda::thread_scope givenRelease = cuda::thread_scope_thread, \
        cuda::thread_scope requiredRelease = cuda::thread_scope_device, uint32_t optFlags = ncclGinOptFlagsDefault) const

.. cpp:function:: uint64_t ncclGin::readSignal(ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const
.. cpp:function:: void ncclGin::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const
.. cpp:function:: void ncclGin::resetSignal(ncclGinSignal_t signal) const

These are signal-specific methods of :cpp:class:`ncclGin`.  :cpp:func:`ncclGin::signal` implements an explicit signal notification without
an accompanying data transfer operation; it takes a subset of arguments of :cpp:func:`ncclGin::put`.  :cpp:func:`ncclGin::readSignal` returns the
bottom *bits* of the value of the *signal*.  :cpp:func:`ncclGin::waitSignal` waits for the bottom *bits* of the *signal* value to meet
or exceed *least*.  Finally, :cpp:func:`ncclGin::resetSignal` resets the *signal* value to ``0`` (this method may not race with
concurrent modifications to the signal).

.. cpp:function:: uint64_t ncclGin::readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const
.. cpp:function:: void ncclGin::waitSignal(Coop coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const
.. cpp:function:: void ncclGin::resetSignal(ncclWindow_t signalWindow, size_t signalOffset) const

These are VA signal-specific methods of :cpp:class:`ncclGin`.

.. cpp:function:: ncclResult_t ncclGin::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits, cuda::memory_order ord, uint64_t timeoutCycles) const
.. cpp:function:: ncclResult_t ncclGin::waitSignal(Coop coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits, cuda::memory_order ord, uint64_t timeoutCycles) const

   Waits for an indexed or VA signal using the same rolling comparison and memory ordering as the
   void-returning overload. Returns :c:enumerator:`ncclTimeout <ncclResult_t.ncclTimeout>` if the wait exceeds
   *timeoutCycles* device clock cycles. Check the status before consuming the data protected by the signal.

Signal shadows
^^^^^^^^^^^^^^

Each indexed signal has a caller-accessible shadow value that can track the next expected signal value.
The shadow is local bookkeeping; changing it does not send a signal to a peer.

.. cpp:function:: uint64_t* ncclGin::getSignalShadowPtr(ncclGinSignal_t signal) const

   Returns a pointer to the shadow of *signal*.

.. cpp:function:: void ncclGin::increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const

   Adds *delta* to the shadow of *signal*.

.. cpp:function:: void ncclGin::waitSignalMeetShadow(Coop coop, ncclGinSignal_t signal, int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire) const

   Waits until *signal* meets or exceeds its shadow value, using the same rolling comparison as
   :cpp:func:`ncclGin::waitSignal`. The shadow value is unchanged.

.. cpp:function:: template<typename Uint> void ncclGin::waitSignalFollowShadow(Coop coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire) const

   Waits for *signal* to advance by at least *leastDelta* beyond its shadow, then updates the shadow to the
   observed value. Writes the previous shadow to *before* and the observed increment to *delta*, masked to
   *bits*. Both output pointers must be valid for every participating thread. Coordinate shadow access
   between callers to avoid concurrent updates of the same shadow.

The indexed-signal overload of :cpp:func:`ncclGin::resetSignal` resets both the signal and its shadow to zero.

Counters
^^^^^^^^

.. cpp:type:: ncclGinCounter_t

   Counters are used to trigger actions on the local rank; as such, they are complementary to signals, which are meant for
   remote actions.  Like signals, they use "rolling" comparison logic, but they are limited to storing values of at most 56
   bits.

.. cpp:struct:: ncclGin_WeakCounterInc

   .. cpp:member:: ncclGinCounter_t counter

This object can be passed as the *localAction* argument of methods such as :cpp:func:`ncclGin::put`.
The counter increment guarantees that the bundled put is locally complete; it does not guarantee completion
of preceding puts.

.. cpp:struct:: ncclGin_CounterInc

   .. deprecated:: 2.30.5

      Use :cpp:struct:`ncclGin_WeakCounterInc` instead.

   .. cpp:member:: ncclGinCounter_t counter

**Counter methods of** :cpp:class:`ncclGin`:

.. cpp:function:: uint64_t ncclGin::readCounter(ncclGinCounter_t counter, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire) const
.. cpp:function:: void ncclGin::waitCounter(Coop coop, ncclGinCounter_t counter, uint64_t least, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire) const
.. cpp:function:: void ncclGin::resetCounter(ncclGinCounter_t counter) const

These are counter-specific methods of :cpp:class:`ncclGin` and they are functionally equivalent to their signal
counterparts discussed above.

.. cpp:function:: ncclResult_t ncclGin::waitCounter(Coop coop, ncclGinCounter_t counter, uint64_t least, int bits, cuda::memory_order ord, uint64_t timeoutCycles) const

   Waits for the counter using the same comparison and memory-ordering rules as the void-returning overload.
   Returns :c:enumerator:`ncclTimeout <ncclResult_t.ncclTimeout>` if the wait exceeds *timeoutCycles*
   device clock cycles. A timeout does not establish local completion of the outstanding transfers.

ncclGinBarrier
--------------

.. cpp:struct:: ncclGinAllContexts

   .. cpp:function:: ncclGinAllContexts(ncclDevComm const& comm_)

      Selects every GIN context of *comm_* for the barrier fence.

   A :c:type:`ncclDevComm` can also be passed directly to :cpp:func:`ncclGinBarrier` or a
   :cpp:class:`ncclGinBarrierSession` constructor in place of ``ncclGinAllContexts``. It is implicitly
   converted to ``ncclGinAllContexts`` and selects every GIN context of that device communicator.
   The device communicator must remain valid for the lifetime of the barrier session.

.. cpp:function:: template<typename Coop> void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeamTagWorld tag, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

.. cpp:function:: template<typename Coop> void ncclGinBarrier(Coop coop, ncclGinAllContexts gin, ncclTeamTagWorld tag, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

   Constructs, synchronizes, and destroys a world-team barrier session. The ``ncclGin`` variant fences
   one context; the ``ncclGinAllContexts`` variant fences every context of the communicator.
   The *index* selects a barrier slot reserved through :c:member:`ncclDevCommRequirements.worldGinBarrierCount`.
   The *fence* has the same meaning as in :cpp:func:`ncclGinBarrierSession::sync`.
   Overloads also accept ``ncclTeamTagRail``, or an explicit ``ncclTeam`` and ``ncclGinBarrierHandle``.

.. cpp:function:: template<typename Coop> void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)
.. cpp:function:: template<typename Coop> void ncclGinBarrier(Coop coop, ncclGinAllContexts gin, ncclTeamTagRail tag, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

   Rail-team variants. Reserve the barrier slots through ``railGinBarrierCount``.

.. cpp:function:: template<typename Coop> void ncclGinBarrier(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)
.. cpp:function:: template<typename Coop> void ncclGinBarrier(Coop coop, ncclGinAllContexts gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

   Explicit-team variants. *handle* must identify barriers allocated for *team*; *index* must be within
   the allocated barrier count.

ncclGinBarrierSession
---------------------

.. cpp:class:: template<typename Coop> ncclGinBarrierSession

   A class representing a network barrier session.

   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index)

      Initializes a new network barrier session.  *coop* represents a cooperative group (see :ref:`devapi_coops`).  *gin* is
      a previously initialized :cpp:class:`ncclGin` object.  ``ncclTeamTagRail`` indicates that the barrier will apply to all
      peers on the same rail as the local rank (see :ref:`devapi_teams`).  *index* identifies the underlying barrier to use
      (it should be different for each *coop*; typically set to ``blockIdx.x`` to ensure uniqueness between CTAs).

   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeamTagWorld tag, uint32_t index)

      Initializes a world-team session. Reserve *index* through ``worldGinBarrierCount`` when creating
      the device communicator.

   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGinAllContexts gin, ncclTeamTagWorld tag, uint32_t index)
   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGinAllContexts gin, ncclTeamTagRail tag, uint32_t index)
   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGinAllContexts gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index)

      Initializes a session whose fence covers every GIN context of the communicator.
      The team, barrier handle, and index requirements match the corresponding single-context constructor.

   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index)

      Initializes a new network barrier session.  This is the general-purpose variant to be used, e.g., when communicating
      with ranks from the world team (see :ref:`devapi_teams`), whereas the previous variant was specific to the rail team.
      This variant expects *team* to be passed as an argument, and also takes an extra *handle* argument indicating the
      location of the underlying barriers. Use ``worldGinBarrier`` for the world team or ``railGinBarrier``
      for the rail team; the handle must match *team*.

   .. cpp:function:: void sync(Coop coop, cuda::memory_order order, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

      Synchronizes all threads of all team members that participate in the barrier session. The *fence* argument is a
      bit-flag enum selecting which prior network operations must be complete after the barrier returns; if omitted it
      defaults to ``ncclGinFenceLevel::Put | ncclGinFenceLevel::Get`` so callers who do not opt in explicitly get the
      strongest guarantee:

      * ``ncclGinFenceLevel::None`` — pure synchronization, no drain.
      * ``ncclGinFenceLevel::Put`` — after the barrier returns, puts issued by other team members targeting the
        calling rank prior to the barrier are visible in the calling rank's memory.
      * ``ncclGinFenceLevel::Get`` — after the barrier returns, gets issued by the calling rank prior to the barrier have
        landed in the calling rank's local memory.

      The fence values are bit flags and compose via bitwise OR. To request both ``Put`` and ``Get`` semantics, pass
      ``ncclGinFenceLevel::Put | ncclGinFenceLevel::Get``. ``ncclGinFenceLevel::Relaxed`` has been deprecated since
      NCCL 2.30.5 and is preserved as an alias for ``None`` for source-level backward compatibility;
      new code should use ``None``.

   .. cpp:function:: ncclResult_t sync(Coop coop, cuda::memory_order order, ncclGinFenceLevel fence, uint64_t timeoutCycles)

      Synchronizes with the selected fence semantics and a timeout measured in device clock cycles.
      A :c:enumerator:`ncclTimeout <ncclResult_t.ncclTimeout>` result does not establish barrier completion.
      All participating threads must call the overload together and check the result before using protected data.

.. c:type:: ncclGinBarrierHandle_t

   Opaque handle that identifies a set of GIN barriers. It is valid until the device communicator is destroyed.

ncclBarrierSession
------------------

.. cpp:class:: template<typename Coop> ncclBarrierSession

   A class representing a hybrid barrier session.  It combines a memory barrier over the :ref:`LSA <device_api_lsa>`
   team (:cpp:class:`ncclLsaBarrierSession`) with a network barrier (:cpp:class:`ncclGinBarrierSession`).

   .. cpp:function:: ncclBarrierSession(Coop coop, ncclTeamTagWorld tag, ncclGin gin, uint32_t index, bool multimem = false)

      Initializes a new hybrid barrier session covering the world team (see :ref:`devapi_teams`).  *coop* represents a
      cooperative group (see :ref:`devapi_coops`).  *gin* is a previously initialized :cpp:class:`ncclGin` object,
      used for the network part of the barrier.  *index* identifies the underlying barrier to use (it should be
      unique for each *coop*; typically set to ``blockIdx.x`` to ensure uniqueness between CTAs).  *multimem*
      requests memory multicast for the LSA part of the barrier.

   .. cpp:function:: ncclBarrierSession(Coop coop, ncclTeamTagRail tag, ncclGin gin, uint32_t index)

      Initializes a new hybrid barrier session covering the rail team, i.e., all peers on the same rail as the local
      rank (see :ref:`devapi_teams`).

   .. cpp:function:: ncclBarrierSession(Coop coop, ncclTeamTagLsa tag, ncclDevComm const& comm, uint32_t index, bool multimem = false)

      Initializes a new hybrid barrier session covering the LSA team.  Since no network traffic is involved, this
      variant takes the device communicator *comm* (created using :c:func:`ncclDevCommCreate`) instead of an
      :cpp:class:`ncclGin` object, and only the LSA barrier takes part in the session.

   .. cpp:function:: ncclBarrierSession(Coop coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin gin, \
        ncclLsaBarrierHandle innerBarHandle, ncclGinBarrierHandle outerBarHandle, uint32_t index, \
        bool multimem = false, ncclMultimemHandle innerMmHandle = {})

      Initializes a new hybrid barrier session.  This is the general-purpose variant, to be used when the teams and
      the barrier locations are not the ones baked into the constructors above.  *innerTeam* and *innerBarHandle*
      describe the memory barrier, while *outerTeam* and *outerBarHandle* describe the network barrier. *innerMmHandle* provides
      the multicast handle backing the *multimem* implementation of the memory barrier.

   .. cpp:function:: void sync(Coop coop, cuda::memory_order order, \
        ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

      Synchronizes all threads of all team members that participate in the barrier session.  *fence* selects which
      prior network operations must be complete after the barrier returns; it has the same meaning and the same
      default as in :cpp:func:`ncclGinBarrierSession::sync`.

      The selected barrier implementation is decided at run-time from the session's team, the GIN connectivity of the device communicator
      (see :c:member:`ginConnectionType <ncclDevCommRequirements.ginConnectionType>`), and *fence*.

   .. cpp:function:: ncclLsaBarrierSession<Coop>& lsaBarrier()
   .. cpp:function:: ncclGinBarrierSession<Coop>& ginBarrier()

      Returns the reference to the memory barrier or the rail network barrier underlying this hybrid session, so that it can be
      driven individually.

   .. cpp:function:: ncclResult_t sync(Coop coop, cuda::memory_order order, ncclGinFenceLevel fence, uint64_t timeoutCycles)

      Timeout-returning hybrid synchronization. *timeoutCycles* bounds the wait in device clock cycles.
      Check the returned status; :c:enumerator:`ncclTimeout <ncclResult_t.ncclTimeout>` does not establish
      synchronization or completion of the operations covered by *fence*.

ncclBarrier
------------

.. cpp:function:: template<typename Coop> void ncclBarrier(Coop coop, ncclTeamTagWorld tag, ncclGin gin, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get, bool multimem = false)
.. cpp:function:: template<typename Coop> void ncclBarrier(Coop coop, ncclTeamTagRail tag, ncclGin gin, uint32_t index, cuda::memory_order order = cuda::memory_order_acq_rel, ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)

   Performs a single hybrid barrier using :cpp:class:`ncclBarrierSession`. The arguments and resource requirements
   match the corresponding session constructor and :cpp:func:`ncclBarrierSession::sync`. All threads in *coop*
   and all ranks of the selected team must participate.
