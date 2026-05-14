.. _device_api_gin:

Device API – GIN
****************

GIN
===

**Device functions.** The following are callable from device (GPU) code only. GIN is supported since NCCL 2.28.7.

ncclGin
-------

.. cpp:class:: ncclGin

   A class encompassing major elements of the GIN support.

   .. cpp:function:: ncclGin(ncclDevComm const& comm, int contextIndex)

      Initializes a new ``ncclGin`` object. *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
      *contextIndex* is the index of the GIN context -- a network communication channel.  Using multiple GIN contexts allows
      the implementation to spread traffic onto multiple connections, avoiding locking and bottlenecks.  Therefore,
      performance-oriented kernels should cycle among the available contexts to improve resource utilization (the number of
      available contexts is available via :c:macro:`ginContextCount`).

      ``ncclGin`` always represents one specific context. To run a barrier whose fence covers every GIN context on the comm
      (useful when operations have been sharded across multiple contexts -- e.g. multi-NIC), pass
      ``ncclGinAllContexts(comm)`` to :cpp:func:`ncclGinBarrier` in place of an ``ncclGin``.

   .. cpp:function:: void put(ncclTeam team, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, \
      size_t srcOffset, size_t bytes, \
      RemoteAction remoteAction, LocalAction localAction, Coop coop, DescriptorSmem descriptor, \
                      cuda::thread_scope alreadyReleased, cuda::thread_scope expected_scope, SegmentType bufType)

      Schedules a device-initiated, one-sided data transfer operation from a local buffer to a remote buffer on a peer.

      *peer* is a rank within *team* (see :ref:`devapi_teams`); it may refer to the local rank (a loopback).  The destination
      and source buffers are each specified using the window (*dstWnd*, *srcWnd*) and a byte-based offset (*dstOffset*,
      *srcOffset*).  *bytes* specifies the data transfer count in bytes. If GIN is initialized with connection
      type :c:macro:`NCCL_GIN_CONNECTION_RAIL`, *peer* must be within the same rail team as the local rank.

      Arguments beyond the first seven are optional.  *remoteAction* and *localAction* specify actions
      to undertake on the destination peer and on the local rank when the payload has been settled and the input has been
      consumed (respectively).  They default to ``ncclGin_None`` (no action); other options include
      ``ncclGin_Signal{Inc|Add}`` (for *remoteAction*) and ``ncclGin_CounterInc`` (for *localAction*); see
      :ref:`devapi_signals` below for more details.  *coop* indicates the set of threads participating in this operation (see
      :ref:`devapi_coops`); it defaults to ``ncclCoopThread`` (a single device thread), which is the recommended model.
      *bufType* specifies the physical memory composition of the source and destination buffers (see
      :ref:`devapi_segment_types`); it defaults to ``ncclGin_SegmentDevice``.

      The visibility of the signal on the destination peer implies the visibility of the put data it is
      attached to. Depending on the signal type, the visibility of the signal may also imply the visibility
      of all the preceding puts to the same peer on the same context.

      The API also defines an alternative, "convenience" variant of this method that uses ``ncclSymPtr`` types to specify the
      buffers and expects size to be conveyed in terms of the number of elements instead of the byte count.  There are also
      two ``putValue`` variants that take a single element at a time (no greater than eight bytes), passed by value.

   .. cpp:function:: void get(ncclTeam team, int peer, ncclWindow_t remoteWnd, size_t remoteOffset, ncclWindow_t localWnd, \
      size_t localOffset, size_t bytes, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, \
      uint32_t optFlags = ncclGinOptFlagsDefault, SegmentType bufType = ncclGin_SegmentDevice{})

      Schedules a device-initiated, one-sided data transfer operation from a remote buffer to a local buffer
      (available since NCCL 2.30.3).

      *peer* is a rank within *team* (see :ref:`devapi_teams`); it may refer to the local rank (a loopback).  The remote
      and local buffers are each specified using the window (*remoteWnd*, *localWnd*) and a byte-based offset (*remoteOffset*,
      *localOffset*).  *bytes* specifies the data transfer count in bytes. If GIN is initialized with connection
      type :c:macro:`NCCL_GIN_CONNECTION_RAIL`, *peer* must be within the same rail team as the local rank.
      *bufType* specifies the physical memory composition of the source and destination buffers (see
      :ref:`devapi_segment_types`); it defaults to ``ncclGin_SegmentDevice``.

   .. cpp:function:: void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire)

      Ensures that all the pending transfer operations scheduled by any threads of *coop* are locally consumed. For put
      operations, this means that the source buffers are safe to reuse; this makes no claims regarding the completion
      status on the remote peer(s). For get operations, this means that the data is visible to the local rank.

   .. cpp:function:: void flushAsync(ncclTeam team, uint32_t peer, ncclGinRequest_t* request, Coop coop = ncclCoopThread{}, \
        uint32_t optFlags = ncclGinOptFlagsDefault, DescriptorSmem descriptor = ncclGin_None{})

      Initiates a non-blocking flush operation for one peer (see :cpp:func:`ncclGin::flush`). *peer* is a rank within *team*
      (see :ref:`devapi_teams`). *request* is supplied by the caller and initialized by ``flushAsync``.
      The caller may use *request* to determine when the flush is complete (see :cpp:func:`ncclGin::wait`).
      Available since NCCL 2.30.3.

   .. cpp:function:: void wait(ncclGinRequest_t& request, Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{}, \
        cuda::memory_order ord = cuda::memory_order_acquire)

      Blocks until *request* is complete. Available since NCCL 2.30.3.

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
registered with flags :c:macro:`NCCL_WIN_COLL_STRICT_ORDERING`. When an address is used as a signal, all reads
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
The strength of these signals is determined by the value of :c:member:`ginStrongSignalsRequired`
when creating the device communicator.

**Signal methods of ncclGin:**

.. cpp:function:: void ncclGin::signal(ncclTeam team, int peer, RemoteAction remoteAction, Coop coop, \
        DescriptorSmem descriptor, cuda::thread_scope alreadyReleased, \
        cuda::thread_scope expected_scope)

.. cpp:function:: uint64_t ncclGin::readSignal(ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::resetSignal(ncclGinSignal_t signal)

These are signal-specific methods of :cpp:class:`ncclGin`.  :cpp:func:`ncclGin::signal` implements an explicit signal notification without
an accompanying data transfer operation; it takes a subset of arguments of :cpp:func:`ncclGin::put`.  :cpp:func:`ncclGin::readSignal` returns the
bottom *bits* of the value of the *signal*.  :cpp:func:`ncclGin::waitSignal` waits for the bottom *bits* of the *signal* value to meet
or exceed *least*.  Finally, :cpp:func:`ncclGin::resetSignal` resets the *signal* value to ``0`` (this method may not race with
concurrent modifications to the signal).

.. cpp:function:: uint64_t ncclGin::readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::waitSignal(Coop coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::resetSignal(ncclWindow_t signalWindow, size_t signalOffset)

These are VA signal-specific methods of :cpp:class:`ncclGin`.

.. cpp:type:: ncclGinCounter_t

   Counters are used to trigger actions on the local rank; as such, they are complementary to signals, which are meant for
   remote actions.  Like signals, they use "rolling" comparison logic, but they are limited to storing values of at most 56
   bits.

.. cpp:struct:: ncclGin_CounterInc

   .. cpp:member:: ncclGinCounter_t counter

This object can be passed as the *localAction* argument of methods such as :cpp:func:`ncclGin::put`.  It is the only action
defined for counters.

**Counter methods of ncclGin:**

.. cpp:function:: uint64_t ncclGin::readCounter(ncclGinCounter_t counter, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::waitCounter(Coop coop, ncclGinCounter_t counter, uint64_t least, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::resetCounter(ncclGinCounter_t counter)

These are counter-specific methods of :cpp:class:`ncclGin` and they are functionally equivalent to their signal
counterparts discussed above.

ncclGinBarrierSession
---------------------

.. cpp:class:: template<typename Coop> ncclGinBarrierSession

   A class representing a network barrier session.

   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index)

      Initializes a new network barrier session.  *coop* represents a cooperative group (see :ref:`devapi_coops`).  *gin* is
      a previously initialized :cpp:class:`ncclGin` object.  *ncclTeamTagRail* indicates that the barrier will apply to all
      peers on the same rail as the local rank (see :ref:`devapi_teams`).  *index* identifies the underlying barrier to use
      (it should be different for each *coop*; typically set to ``blockIdx.x`` to ensure uniqueness between CTAs).

   .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index)

      Initializes a new network barrier session.  This is the general-purpose variant to be used, e.g., when communicating
      with ranks from the world team (see :ref:`devapi_teams`), whereas the previous variant was specific to the rail team.
      This variant expects *team* to be passed as an argument, and also takes an extra *handle* argument indicating the
      location of the underlying barriers (typically set to the ``railGinBarrier`` field of the device communicator).

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
      ``ncclGinFenceLevel::Put | ncclGinFenceLevel::Get``. ``ncclGinFenceLevel::Relaxed`` is preserved as a deprecated alias
      for ``None`` for source-level backward compatibility; new code should use ``None``.
