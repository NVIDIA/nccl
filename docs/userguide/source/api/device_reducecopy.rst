.. _device_api_reducecopy:

Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels
*****************************************************************************************

**Device functions.** All functions on this page are callable from device (GPU) code only. They are **building blocks
for computation-fused kernels**: they implement reduce, copy (broadcast), and fused reduce-then-copy operations, keeping
communication and computation in a single kernel.

Key points:

* **Communication patterns:** Sources and destinations can be on remote ranks (using :c:type:`ncclWindow_t` as
  input and output of the API), enabling direct implementation of patterns such as
  :ref:`AllReduce <allreduce>`, :ref:`AllGather <allgather>`, and :ref:`ReduceScatter <reducescatter>`.
* **Building blocks:** Each function implements one peak-bandwidth **communication building block**, not a full
  algorithm. You can combine these blocks (and your own computation) in tandem to implement custom communication
  patterns.
  The three building blocks are:

  * :ref:`ReduceSum <device_api_reducecopy_reducesum>` — *reduce*; e.g. reduce phase of AllReduce or ReduceScatter
  * :ref:`Copy <device_api_reducecopy_copy>` — *broadcast/copy*; e.g. broadcast phase of AllReduce or copy in
    AllGather
  * :ref:`ReduceSumCopy <device_api_reducecopy_reducesumcopy>` — *fused reduce-then-copy*; e.g. one-step AllReduce
    or reduce-to-chunks for ReduceScatter

  For non-sum reductions, see :ref:`Custom Reduction Operators <device_api_reducecopy_custom_redop>`.
* **API forms:** All functions are device-only (callable from ``__device__`` code) and come in two forms:
  **high-level convenience overloads** (the direct summation overloads described in the sections below; they work
  with NCCL windows, teams, and
  device communicators) and **lambda-based overloads**, which offer more flexibility for custom layouts (see
  :ref:`Lambda-Based (Custom Layouts) <device_api_reducecopy_lambda>`).
* **GIN:** This API does not support :ref:`GIN <device_api_gin>` (GPU-Initiated Networking) implicitly; use this
  API within the :ref:`LSA <device_api_lsa>` domain and implement an explicit hierarchical design with NCCL GIN to
  exchange data between LSA domains.
* **Invocation model (not rank-collective):** These functions are not rank-collective (unlike host API such as
  :c:func:`ncclAllReduce`). For a given memory region (e.g. a :c:type:`ncclWindow_t`, offset, and count), only a
  single rank must issue the API call that uses that region. The per-operation sections specify whether each role is
  multi-rank (each rank issues for its own region) or single-rank (one rank issues for that region).
* **Memory:** Source and destination regions must not overlap (except when exactly in-place: same buffer and same
  offset); otherwise behavior is undefined. The caller must ensure all arguments and buffer layouts meet the documented
  requirements; the API does not perform runtime checks.
* **Alignment:** For best performance, use 16-byte aligned source and destination pointers.

.. _device_api_reducecopy_compile:

Compile-Time Requirements
-------------------------

``NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE`` must be defined to ``1`` before including the NCCL device headers so that all
block sizes and type combinations are supported for **multimem** operations (see
:ref:`multimem reduce <ncclMultimemReduceSum-symptr>`, :ref:`multimem copy <ncclMultimemCopy-symptr>`, and related
multimem APIs below). Without it, certain combinations of low-precision types, *count*, and pointer alignment
in multimem operations may hit runtime asserts. Because only one rank might trigger an assert, this can also lead to
hangs. Defining it means the user acknowledges that they are willing to use cutting-edge APIs that might change between
releases.

**Lambda-based overloads** The API uses device-side C++ lambda functions for overloads that take callables
(e.g. lambdas) to describe source or destination layouts. The API also offers user-facing lambda-based overloads; see
:ref:`Lambda-Based (Custom Layouts) <device_api_reducecopy_lambda>`. Code that includes the NCCL device headers for
this API must always be compiled with CUDA extended lambdas enabled (e.g. ``--extended-lambda`` with nvcc); otherwise
you may get a compile-time static assert. See the
`CUDA documentation for extended lambdas <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambdas>`_.

API Overview
------------

* :ref:`ReduceSum <device_api_reducecopy_reducesum>` — Reduce building block.
* :ref:`Copy <device_api_reducecopy_copy>` — Broadcast/copy building block.
* :ref:`ReduceSumCopy <device_api_reducecopy_reducesumcopy>` — Fused reduce-then-copy building block.

.. _device_api_reducecopy_common_params:

**Common template parameters**

  **T**
    Element type. Supported types are: ``float``, ``double``, ``half``, ``int8``, ``int16``, ``int32``, ``int64``;
    and, when available, the following low-precision types: ``__nv_bfloat16``, ``__nv_fp8_e4m3``, and ``__nv_fp8_e5m2``.
    For low-precision types, sum reduction is accumulated in a wider type:

    .. list-table::
       :header-rows: 1
       :widths: 20 20

       * - **T**
         - **Accumulation type**
       * - ``half``
         - ``float``
       * - ``__nv_bfloat16``
         - ``float``
       * - ``__nv_fp8_e4m3``
         - ``half``
       * - ``__nv_fp8_e5m2``
         - ``half``

    For multimem reduce, this wider accumulation is performed on the NVLink Switch.

  **Coop**
    Cooperation level (see :ref:`devapi_coops`), e.g. ``ncclCoopCta`` or ``ncclCoopThread``. All threads in the
    cooperative group defined by *Coop* must participate in the call.

  **IntCount**
    Type for the element count. The user can choose a 32-bit integer type (e.g. ``unsigned int``) or a 64-bit
    integer type (e.g. ``size_t``) depending on the size of the block region the API operates on.

  **UNROLL**
    Optional; default ``4*16/sizeof(T)``. UNROLL represents the tradeoff between register usage and achievable peak
    bandwidth; the optimal value depends on the register usage of the surrounding kernel. Higher *UNROLL* allows
    vectorized load/store and more loop unrolling, which helps achieve peak
    bandwidth. High register usage can lower occupancy and may lead to register spilling; see the
    `CUDA Programming Guide section on kernel launch and occupancy
    <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy>`_.
    The default is chosen to make good performance possible on most systems.

  Example (ReduceSumCopy with *T* = ``float``, *Coop* = ``ncclCoopCta``, *IntCount* = ``size_t``, and *UNROLL* set to
  the default for float, ``4*16/sizeof(float)`` = 16):

    .. code-block:: cpp

       size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
       size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
       ncclLsaReduceSumCopy<float, ncclCoopCta, size_t, 16>(ctaCoop, sendwin, srcOffset, recvwin, dstOffset, count, team);

.. _device_api_reducecopy_reducesum:

ReduceSum — N Sources to One Destination
-----------------------------------------

All ReduceSum variants reduce from N sources to one destination using sum. See :ref:`common template parameters
<device_api_reducecopy_common_params>` (*T*, *Coop*, *IntCount*, *UNROLL*).

.. _ncclLsaReduceSum-window-devComm:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSum(Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count, ncclDevComm_t devComm)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Reduces from the symmetric buffer at *window* + *offset* on all :ref:`LSA <device_api_lsa>` peers into local
   *dstPtr*. The reduction is over all LSA ranks in the communicator; pass *devComm* (the device communicator).

   *coop* is the cooperative group (see :ref:`devapi_coops`). *window* is the window handle from a prior host-side
   ``ncclCommWindowRegister`` and must be the same window (and communicator) as *devComm*; the buffer region must
   remain registered for the duration
   of the call. *offset* is the byte offset into *window* where the source buffer starts on each peer;
   *offset* + *count* × ``sizeof(T)`` must not exceed the size of the registered window. *dstPtr* is the local
   device pointer to the destination buffer; it must point to at least *count* elements of type *T* and must be
   accessible by all participating threads according to *coop*. *count* is the number of elements to reduce; it
   must be the same on all :ref:`LSA <device_api_lsa>` ranks, non-negative, and consistent with *IntCount*.
   *devComm* is the device communicator.

   **Barrier usage:** When using remote memory, synchronize before and after the call (see example below).

   Example:

   .. code-block:: cpp

      ncclCoopCta ctaCoop;
      ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
      bar.sync(ctaCoop, cuda::memory_order_acquire);

      size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
      size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
      T* dstPtr = (T*)ncclGetLocalPointer(recvwin, dstOffset);

      ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, dstPtr, count, devComm);

      bar.sync(ctaCoop, cuda::memory_order_release);

.. _ncclLsaReduceSum-window-team:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSum(Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count, ncclTeam team)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSum-window-devComm>`, except the user passes *team* explicitly (e.g.
   ``ncclTeamLsa(devComm)``) instead of *devComm*. All other parameters, invocation, and barrier usage are as
   documented for the overload above.

   *team* is the team of :ref:`LSA <device_api_lsa>` ranks (see :ref:`devapi_teams`).

   Example:

   .. code-block:: cpp

      ncclTeam team = ncclTeamLsa(devComm);
      ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, dstPtr, count, team);

.. _ncclLsaReduceSum-symptr-devComm:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSum(Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclDevComm_t devComm)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSum-window-devComm>`, but the source is given by symmetric pointer
   *src* instead of (window, offset). *dstPtr*, *count*, and *devComm* are as for
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>`. With ``ncclSymPtr`` you can construct with 0 offset
   and use ``src + elementOffset`` (offset in elements, no ``sizeof(T)``).

   Example:

   .. code-block:: cpp

      ncclSymPtr<float> src{sendwin, 0};
      ncclLsaReduceSum<float, ncclCoopCta, size_t>(ctaCoop, src + elementOffset, dstPtr, count, devComm);

.. _ncclLsaReduceSum-symptr-team:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSum(Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclTeam team)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSum-symptr-devComm>`, except the user passes *team* explicitly instead of
   *devComm*. The team is derived from the device communicator (e.g. ``ncclTeamLsa(devComm)``).

   Example:

   .. code-block:: cpp

      ncclSymPtr<float> src{sendwin, 0};
      ncclTeam team = ncclTeamLsa(devComm);
      ncclLsaReduceSum<float, ncclCoopCta, size_t>(ctaCoop, src + elementOffset, dstPtr, count, team);

.. _ncclLocalReduceSum-strided:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLocalReduceSum(Coop coop, int nSrc, T* basePtr, size_t displ, T* dstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`, but over **local** sources only (no other ranks,
   no remote memory). Sources are strided: the *i*-th source is at ``basePtr + i*displ`` for *i* = 0…*nSrc* − 1.
   *basePtr* is the base pointer, *displ* is the stride in bytes; *dstPtr* and *count* are as for
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`.

**Multimem reduce (ncclMultimemReduceSum)** — Multimem reduce uses NVLink SHARP (NVLS) multicast; the NVLink
Switch performs the reduction from multimem sources. To query NVLS/multimem capability from the host, call
:c:func:`ncclCommQueryProperties` and check the :c:member:`multimemSupport` field of :c:type:`ncclCommProperties_t`.
**Multimem restriction:** The local rank (self) must always be part of the multimem reduction or store; the multimem
source or destination logically includes the calling rank. For multimem reduce, supported element types are
``float``, ``double``, ``half``; the low-precision types when available: ``__nv_bfloat16``, ``__nv_fp8_e4m3``,
``__nv_fp8_e5m2``; and ``int32``, ``uint32``, ``int64``, ``uint64``. The
define described above (e.g. ``NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1``) may need to be set for all type
and block-size combinations.

.. _ncclMultimemReduceSum-symptr:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSum(Coop coop, ncclSymPtr<T> src, T* dstPtr, IntCount count, ncclMultimemHandle multimemHandle)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Reduces from the multimem source *src* (one logical buffer maps to all participating ranks) into local *dstPtr*.
   Invocation as for :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>`.
   *src* is the symmetric pointer to the multimem source; *dstPtr* is the local destination; *count* is the number of
   elements; *multimemHandle* identifies the multimem context. To obtain it, set :c:member:`lsaMultimem` to true in
   :c:type:`ncclDevCommRequirements` when calling :c:func:`ncclDevCommCreate`; the handle is then available from the
   device communicator in device code.

.. _ncclMultimemReduceSum-raw:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSum(Coop coop, T* mcSrcPtr, T* dstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemReduceSum-symptr>`, but the source is given by raw multimem pointer *mcSrcPtr*
   instead of ``ncclSymPtr`` + handle (e.g. from host-side :c:func:`ncclGetLsaMultimemDevicePointer`). *dstPtr* and
   *count* are as for :ref:`ncclMultimemReduceSum <ncclMultimemReduceSum-symptr>`.

.. _ncclMultimemReduceSum-window:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSum(Coop coop, ncclWindow_t window, size_t offset, T* dstPtr, IntCount count, ncclMultimemHandle multimemHandle)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemReduceSum-symptr>`, but the source is given by *window* + *offset* (byte offset)
   and *multimemHandle*, analogous to the window-based LSA overload. *dstPtr* and *count* are as for
   :ref:`ncclMultimemReduceSum <ncclMultimemReduceSum-symptr>`.

.. _device_api_reducecopy_copy:

Copy (Broadcast) — One Source to N Destinations
------------------------------------------------

All Copy variants copy from one source to N destinations. See :ref:`common template parameters
<device_api_reducecopy_common_params>`. Invocation as for
:ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>`; for Copy, the source is local to the invoking rank.

.. _ncclLsaCopy-window-devComm:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaCopy(Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count, ncclDevComm_t devComm)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Copies from local *srcPtr* into the symmetric buffer at *window* + *offset* on all :ref:`LSA <device_api_lsa>`
   peers. Pass *devComm* (the device communicator). *srcPtr* is the local source; *window* and *offset* define the
   destination region on each
   peer; *count* is the number of elements. Barrier usage: synchronize before and after when using remote memory (see
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>`).

   Example (e.g. broadcast phase of AllGather):

   .. code-block:: cpp

      ncclCoopCta ctaCoop;
      ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
      bar.sync(ctaCoop, cuda::memory_order_acquire);

      size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
      size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
      T* srcPtr = (T*)ncclGetLocalPointer(sendwin, srcOffset);
      ncclLsaCopy<T, ncclCoopCta, size_t>(ctaCoop, srcPtr, recvwin, dstOffset, count, devComm);

      bar.sync(ctaCoop, cuda::memory_order_release);

.. _ncclLsaCopy-window-team:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaCopy(Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count, ncclTeam team)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaCopy-window-devComm>`, except the user passes *team* explicitly (e.g.
   ``ncclTeamLsa(devComm)``) instead of *devComm*.

.. _ncclLsaCopy-symptr-devComm:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaCopy(Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclDevComm_t devComm)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaCopy-window-devComm>`, but the destination is given by symmetric pointer *dst* instead
   of (window, offset). *srcPtr*, *count*, and *devComm* are as for
   :ref:`ncclLsaCopy <ncclLsaCopy-window-devComm>`. You can construct *dst* with 0 offset and use
   ``dst + elementOffset`` for element-based indexing.

.. _ncclLsaCopy-symptr-team:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaCopy(Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclTeam team)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaCopy-symptr-devComm>`, except the user passes *team* explicitly instead of *devComm*.

.. _ncclLocalCopy-strided:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLocalCopy(Coop coop, T* srcPtr, int nDst, T* basePtr, size_t displ, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaCopy <ncclLsaCopy-window-team>` for thread-cooperation behavior, but **local** only (no other
   ranks, no remote memory): copies from single source *srcPtr* to *nDst* strided destinations at
   ``basePtr + i*displ`` for
   *i* = 0…*nDst* − 1. *displ* is the stride in bytes; *count* is the number of elements copied to each destination.

**Multimem copy (ncclMultimemCopy)** — Copies from one local source to one multimem destination (one logical buffer
over all ranks). Uses NVLink SHARP (NVLS) multicast. Query :c:member:`multimemSupport` for capability; *multimemHandle*
as for :ref:`multimem reduce <ncclMultimemReduceSum-symptr>`. All element types are supported; for types less than
32 bits wide, the define described above (e.g. ``NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1``) must be set for some
count and pointer combinations.

.. _ncclMultimemCopy-symptr:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemCopy(Coop coop, T* srcPtr, ncclSymPtr<T> dst, IntCount count, ncclMultimemHandle multimemHandle)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Copies from local *srcPtr* into the multimem destination *dst* (one logical buffer maps to all participating ranks).
   Invocation as for :ref:`ncclLsaCopy <ncclLsaCopy-window-devComm>`. *multimemHandle* as for
   :ref:`ncclMultimemReduceSum <ncclMultimemReduceSum-symptr>`.

.. _ncclMultimemCopy-raw:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemCopy(Coop coop, T* srcPtr, T* mcDstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemCopy-symptr>`, but the destination is given by raw multimem pointer *mcDstPtr*
   (e.g. from :c:func:`ncclGetLsaMultimemDevicePointer`). *srcPtr* and *count* are as for
   :ref:`ncclMultimemCopy <ncclMultimemCopy-symptr>`.

.. _ncclMultimemCopy-window:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemCopy(Coop coop, T* srcPtr, ncclWindow_t window, size_t offset, IntCount count, ncclMultimemHandle multimemHandle)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemCopy-symptr>`, but the destination is given by *window* + *offset* (byte offset)
   and *multimemHandle*. *srcPtr* and *count* are as for :ref:`ncclMultimemCopy <ncclMultimemCopy-symptr>`.

.. _device_api_reducecopy_reducesumcopy:

ReduceSumCopy
-------------

ReduceSumCopy combines reduction and copy into a single call. See :ref:`common template parameters
<device_api_reducecopy_common_params>`. Invocation as documented for the
:ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>` and
:ref:`ncclLsaCopy <ncclLsaCopy-window-devComm>` overloads above.

LSA ReduceSumCopy (ncclLsaReduceSumCopy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _ncclLsaReduceSumCopy-window-devComm:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumCopy(Coop coop, ncclWindow_t srcWindow, size_t srcOffset, ncclWindow_t dstWindow, size_t dstOffset, IntCount count, ncclDevComm_t devComm)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Reduces from the :ref:`LSA <device_api_lsa>` source at *srcWindow* + *srcOffset* (all LSA peers) and copies the
   result to the LSA destination at *dstWindow* + *dstOffset* (all LSA peers) in one call. Pass *devComm* (the
   device communicator).
   *srcOffset* and *dstOffset* are byte offsets; *count* is the number of elements. When using remote memory, barrier
   usage is the same as for :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>` and
   :ref:`ncclLsaCopy <ncclLsaCopy-window-devComm>`: synchronize before and after the call (see the examples there).

   Example (e.g. LSA AllReduce; see ``test/perf/all_reduce.cu`` for block-parallel chunking):

   .. code-block:: cpp

      ncclCoopCta ctaCoop;
      ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
      bar.sync(ctaCoop, cuda::memory_order_acquire);

      size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
      size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
      ncclLsaReduceSumCopy<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, recvwin, dstOffset, count, devComm);

      bar.sync(ctaCoop, cuda::memory_order_release);

.. _ncclLsaReduceSumCopy-window-team:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumCopy(Coop coop, ncclWindow_t srcWindow, size_t srcOffset, ncclWindow_t dstWindow, size_t dstOffset, IntCount count, ncclTeam team)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSumCopy-window-devComm>`, except the user passes *team* explicitly (e.g.
   ``ncclTeamLsa(devComm)``) instead of *devComm*.

.. _ncclLsaReduceSumCopy-symptr-devComm:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclSymPtr<T> dst, IntCount count, ncclDevComm_t devComm)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSumCopy-window-devComm>`, but the source is given by symmetric pointer *src* and
   the destination by symmetric pointer *dst* instead of (window, offset). *count* and *devComm* are as for
   :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-devComm>`.

.. _ncclLsaReduceSumCopy-symptr-team:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclSymPtr<T> dst, IntCount count, ncclTeam team)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSumCopy-symptr-devComm>`, except the user passes *team* explicitly instead of
   *devComm*.

.. _ncclLsaReduceSumCopy-different-teams:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-team>`, but source and destination use different
   teams (*srcTeam* and *dstTeam*). Ranks in one team must still be load-store accessible
   (LSA) from ranks in the other (same LSA communicator; involved ranks must be able to access each other's
   registered memory). *src* is the source symmetric pointer over *srcTeam*; *dst* is the destination symmetric
   pointer over *dstTeam*.

.. _ncclLocalReduceSumCopy-strided:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLocalReduceSumCopy(Coop coop, int nSrc, T* srcBasePtr, size_t srcDispl, int nDst, T* dstBasePtr, size_t dstDispl, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-team>` for thread-cooperation behavior, but
   **local** only (no other ranks, no remote memory). Reduces from *nSrc* strided sources at
   ``srcBasePtr + i*srcDispl`` (i = 0…*nSrc*
   − 1) and copies the result to *nDst* strided destinations at ``dstBasePtr + j*dstDispl`` (j = 0…*nDst* − 1).
   *srcDispl* and *dstDispl* are strides in bytes; *count* is the number of elements per source/destination.

Multimem ReduceSumCopy (ncclMultimemReduceSumCopy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reduces from one multimem source and copies to one multimem destination (each one logical buffer maps to all
participating ranks) in one call. To query multimem capability from the host, call :c:func:`ncclCommQueryProperties`
and check the :c:member:`multimemSupport` field of :c:type:`ncclCommProperties_t`. The multimem handle is obtained by
setting :c:member:`lsaMultimem` to true in :c:type:`ncclDevCommRequirements` when calling :c:func:`ncclDevCommCreate`;
it is then available from the device communicator in device code.

.. _ncclMultimemReduceSumCopy-window:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSumCopy(Coop coop, ncclWindow_t srcWindow, size_t srcOffset, ncclMultimemHandle srcHandle, ncclWindow_t dstWindow, size_t dstOffset, ncclMultimemHandle dstHandle, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Reduces from the multimem source at *srcWindow* + *srcOffset* and copies to the multimem destination at
   *dstWindow* + *dstOffset* in one call. *srcHandle* and *dstHandle* identify the multimem contexts (may be the
   same or different). Invocation as documented for the :ref:`overload above <ncclLsaReduceSum-window-team>`.

.. _ncclMultimemReduceSumCopy-symptr:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSumCopy(Coop coop, ncclSymPtr<T> src, ncclMultimemHandle srcHandle, ncclSymPtr<T> dst, ncclMultimemHandle dstHandle, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemReduceSumCopy-window>`, but the source is given by symmetric pointer *src* and
   the destination by symmetric pointer *dst* instead of (window, offset). *srcHandle* and *dstHandle* are as for
   :ref:`ncclMultimemReduceSumCopy <ncclMultimemReduceSumCopy-window>`.

.. _ncclMultimemReduceSumCopy-raw:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSumCopy(Coop coop, T* mcSrcPtr, T* mcDstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemReduceSumCopy-symptr>`, but source and destination are given by raw multimem
   pointers *mcSrcPtr* and *mcDstPtr* (e.g. from :c:func:`ncclGetLsaMultimemDevicePointer`).

Mixed LSA and Multimem ReduceSumCopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reduce from :ref:`LSA <device_api_lsa>` and write to multimem, or reduce from multimem and write to LSA, in one call.

.. _ncclLsaReduceSumMultimemCopy-symptr:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumMultimemCopy(Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, ncclSymPtr<T> dst, ncclMultimemHandle dstHandle, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Reduces from :ref:`LSA <device_api_lsa>` source *src* over *srcTeam* and copies to multimem destination *dst* (one
   logical buffer maps to all participating ranks). *dstHandle* as for
   :ref:`ncclMultimemCopy <ncclMultimemCopy-symptr>`. Invocation as documented for
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>` and :ref:`ncclMultimemCopy <ncclMultimemCopy-symptr>`.

.. _ncclLsaReduceSumMultimemCopy-raw:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclLsaReduceSumMultimemCopy(Coop coop, ncclSymPtr<T> src, ncclTeam srcTeam, T* mcDstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclLsaReduceSumMultimemCopy-symptr>`, but the multimem destination is given by raw pointer
   *mcDstPtr* instead of ``ncclSymPtr`` + handle.

.. _ncclMultimemReduceSumLsaCopy-symptr:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSumLsaCopy(Coop coop, ncclSymPtr<T> src, ncclMultimemHandle srcHandle, ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Reduces from multimem source *src* and copies to :ref:`LSA <device_api_lsa>` destination *dst* over *dstTeam*.
   *srcHandle* as for :ref:`ncclMultimemReduceSum <ncclMultimemReduceSum-symptr>`. Invocation as documented for
   :ref:`ncclMultimemReduceSum <ncclMultimemReduceSum-symptr>` and :ref:`ncclLsaCopy <ncclLsaCopy-window-devComm>`.

.. _ncclMultimemReduceSumLsaCopy-raw:

.. cpp:function:: template<typename T, typename Coop, typename IntCount, int UNROLL> void ncclMultimemReduceSumLsaCopy(Coop coop, T* mcSrcPtr, ncclSymPtr<T> dst, ncclTeam dstTeam, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`above <ncclMultimemReduceSumLsaCopy-symptr>`, but the multimem source is given by raw pointer
   *mcSrcPtr* instead of ``ncclSymPtr`` + handle.

.. _device_api_reducecopy_lambda:

Lambda-Based (Custom Layouts)
-----------------------------

Lambda-based overloads give more flexibility and allow custom memory layouts for reduce and/or copy. They can mix
local and :ref:`LSA <device_api_lsa>`-remote sources or destinations (e.g. one source from local memory, others from
LSA windows), and can express non-contiguous or index-dependent addressing that the fixed window/symptr overloads do
not support.

**Conditions for the lambda.** The callable (e.g. lambda) must return ``T*`` and be invocable from device code with
a single index argument. When the callable is a lambda, it must be qualified with ``__device__`` and the build must
have CUDA extended lambdas enabled (see :ref:`Compile-Time Requirements <device_api_reducecopy_compile>`). Let *n* be
the associated count (*nSrc* or *nDst* depending on the API).

#. *n* > 0. Otherwise behavior is undefined.
#. For every index *i* in [0, *n*), the call *lambda*(*i*) must return a pointer to the start of a valid region of
   at least *count* contiguous elements of type *T*. The same restrictions apply as for the corresponding non-lambda
   API: e.g. for :ref:`LSA <device_api_lsa>` sources/destinations the region must be in registered LSA memory and
   remain valid for the duration of the call; for local pointers they must be accessible to all threads in *coop*.
#. When the API designates source or destination as **multimem**, every pointer returned by the lambda for that side
   must be a **multimem pointer** (e.g. from :c:func:`ncclGetLsaMultimemDevicePointer`). Using LSA or local pointers
   for the multimem side is undefined behavior. Conversely, multimem pointers cannot be used where LSA or local
   pointers are accepted.
#. The relationship between *n* and the logical set of sources or destinations is as documented for each API
   (e.g. *nSrc* sources meaning *nSrc* distinct source regions, or *nDst* destinations meaning *nDst* distinct
   destination regions).

Violating any of these conditions is undefined behavior; the API does not perform runtime checks.

Example: reduce from a window over a team, but **exclude the local rank** (e.g. reduce only from remote peers).
Use a source lambda that maps index *i* to the *i*-th *other* rank and pass *nSrc* = *team*\ ``.nRanks`` − 1:

.. code-block:: cpp

   size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
   ncclTeam team = ncclTeamLsa(devComm);
   int myRank = devComm.rank;
   int nSrc = team.nRanks - 1;   // all ranks except this one

   auto srcLambda = [=] __device__ (int i) -> T* {
     int peer = (i < myRank) ? i : i + 1;   // skip myRank
     return (T*)ncclGetLsaPointer(sendwin, srcOffset, peer);
   };

   ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, srcLambda, nSrc, dstPtr, count);

.. _ncclLsaReduceSum-lambda:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL> void ncclLsaReduceSum(Coop coop, SrcLambda srcLambda, int nSrc, T* dstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`, but the source layout is given by *srcLambda*(index)
   returning ``T*`` for each of *nSrc* sources; result to local *dstPtr*. *coop* and *count* as for
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`. *srcLambda* is called with indices 0 to *nSrc* − 1.

.. _ncclLocalReduceSum-lambda:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL> void ncclLocalReduceSum(Coop coop, SrcLambda srcLambda, int nSrc, T* dstPtr, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`, but over **local** sources only (no other ranks,
   no remote memory). *srcLambda*(index) returns ``T*`` for each of *nSrc* sources; result to *dstPtr*. *coop* and
   *count* as for :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`. *srcLambda* is called with indices 0 to
   *nSrc* − 1.

.. _ncclLsaCopy-lambda:

.. cpp:function:: template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL> void ncclLsaCopy(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaCopy <ncclLsaCopy-window-team>`, but the destination layout is given by *dstLambda*(index)
   returning ``T*`` for each of *nDst* destinations. *srcPtr* is the local source; *coop* and *count* as for
   :ref:`ncclLsaCopy <ncclLsaCopy-window-team>`. *dstLambda* is called with indices 0 to *nDst* − 1.

.. _ncclLocalCopy-lambda:

.. cpp:function:: template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL> void ncclLocalCopy(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaCopy <ncclLsaCopy-window-team>`, but **local** only (no other ranks, no remote memory).
   *dstLambda*(index) returns ``T*`` for each of *nDst* destinations; *srcPtr*, *coop*, and *count* as for
   :ref:`ncclLsaCopy <ncclLsaCopy-window-team>`. *dstLambda* is called with indices 0 to *nDst* − 1.

.. _ncclLsaReduceSumLsaCopy-lambda:

The following four overloads are the lambda-based forms of ReduceSumCopy. They differ by whether the **source** and
**destination** are **:ref:`LSA <device_api_lsa>`** or **multimem**. For any **multimem** side, the common case is a
single multimem pointer (one already represents multiple remote spaces). Multiple multimem pointers are also
supported; the API then
initiates multicast to or from all of them.

.. warning::
   Multicast always includes the self rank. With more than one multimem source or destination, this creates
   overlapping ranks. The user must ensure correctness.

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL> void ncclLsaReduceSumLsaCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   **Source: :ref:`LSA <device_api_lsa>`. Destination: LSA.** Same as
   :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-team>`, but the source layout is given by
   *srcLambda*(*i*) returning ``T*`` for each of *nSrc* sources (*i* = 0 to *nSrc* − 1)
   and the destination layout by *dstLambda*(*j*) for each of *nDst* destinations (*j* = 0 to *nDst* − 1). *coop* and
   *count* as for :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-team>`. When using remote memory, barrier
   usage is as for :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-devComm>`.

.. _ncclLsaReduceSumMultimemCopy-lambda:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL> void ncclLsaReduceSumMultimemCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   **Source: :ref:`LSA <device_api_lsa>`. Destination: multimem.** Same as
   :ref:`ncclLsaReduceSumMultimemCopy <ncclLsaReduceSumMultimemCopy-symptr>`, but the source layout is given by
   *srcLambda*(*i*) for *i* = 0 to *nSrc* − 1 and the destination layout by *dstLambda*(*j*) for *j* = 0 to
   *nDst* − 1. *srcLambda* returns :ref:`LSA <device_api_lsa>` pointers; *dstLambda* must return multimem pointers
   (e.g. from :c:func:`ncclGetLsaMultimemDevicePointer`). The common case is *nDst* = 1. *coop* and *count* as for
   :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-team>`. Invocation as documented for
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>` and :ref:`ncclMultimemCopy <ncclMultimemCopy-symptr>`.

.. _ncclMultimemReduceSumLsaCopy-lambda:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL> void ncclMultimemReduceSumLsaCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   **Source: multimem. Destination: :ref:`LSA <device_api_lsa>`.** Same as
   :ref:`ncclMultimemReduceSumLsaCopy <ncclMultimemReduceSumLsaCopy-symptr>`, but the source layout is given by
   *srcLambda*(*i*) for *i* = 0 to *nSrc* − 1 and the destination layout by *dstLambda*(*j*) for *j* = 0 to
   *nDst* − 1. *srcLambda* must return multimem pointers (e.g. from
   :c:func:`ncclGetLsaMultimemDevicePointer`); *dstLambda* returns :ref:`LSA <device_api_lsa>` pointers. The common
   case is *nSrc* = 1. *coop* and *count* as for
   :ref:`ncclMultimemReduceSumLsaCopy <ncclMultimemReduceSumLsaCopy-symptr>`.
   Invocation as documented for :ref:`ncclMultimemReduceSum <ncclMultimemReduceSum-symptr>` and
   :ref:`ncclLsaCopy <ncclLsaCopy-window-devComm>`.

.. _ncclMultimemReduceSumMultimemCopy-lambda:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL> void ncclMultimemReduceSumMultimemCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   **Source: multimem. Destination: multimem.** Same as
   :ref:`ncclMultimemReduceSumCopy <ncclMultimemReduceSumCopy-window>`, but the source layout is given by
   *srcLambda*(*i*) for *i* = 0 to *nSrc* − 1 and the destination layout by
   *dstLambda*(*j*) for *j* = 0 to *nDst* − 1. *srcLambda* and *dstLambda* must each return multimem pointers (e.g.
   from :c:func:`ncclGetLsaMultimemDevicePointer`). The common case is a single multimem source and a single multimem
   destination. *coop* and *count* as for :ref:`ncclMultimemReduceSumCopy <ncclMultimemReduceSumCopy-window>`.
   Invocation as documented for :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-team>`.

.. _device_api_reducecopy_custom_redop:

Custom Reduction Operators
--------------------------

The APIs below take an explicit reduction operator (*redOp*) instead of a fixed sum, enabling custom reductions
(e.g. min, max, product). **Restrictions for *redOp*:**

* *redOp* is a callable (e.g. functor or lambda) that takes two arguments of type *T* (or ``const T&``) and returns
  *T* (the combined value).
* No order is guaranteed in which elements are combined; the reduction may be applied in any order across the sources.
* The callable must be **const**: it must not modify internal state. If *redOp* is a functor, its ``operator()`` must
  be ``const``; stateless lambdas satisfy this by default. Violating this is undefined behavior.

.. _ncclLsaReduceLsaCopy:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount, int UNROLL> void ncclLsaReduceLsaCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, RedOp const& redOp, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSumLsaCopy <ncclLsaReduceSumLsaCopy-lambda>`, but the reduction is performed with the
   explicit *redOp* callable instead of sum. *redOp* must satisfy the restrictions above. Sources and destinations are
   :ref:`LSA <device_api_lsa>`. *srcLambda* and *dstLambda* are as for the
   :ref:`lambda-based ReduceSumCopy APIs <ncclLsaReduceSumLsaCopy-lambda>`; *coop*, *count*, and barrier usage as for
   :ref:`ncclLsaReduceSumCopy <ncclLsaReduceSumCopy-window-team>`.

.. _ncclLsaReduceMultimemCopy:

.. cpp:function:: template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount, int UNROLL> void ncclLsaReduceMultimemCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst, RedOp const& redOp, IntCount count)

   For shared requirements (invocation model, memory, alignment), see the :ref:`introduction <device_api_reducecopy>`.

   Same as :ref:`ncclLsaReduceSumMultimemCopy <ncclLsaReduceSumMultimemCopy-lambda>`, but the reduction is performed
   with the explicit *redOp* callable instead of sum. *redOp* must satisfy the restrictions above. Sources are
   :ref:`LSA <device_api_lsa>`; destinations are multimem (one logical buffer maps to all participating ranks).
   *dstLambda* must return multimem pointers (e.g. from :c:func:`ncclGetLsaMultimemDevicePointer`). For custom
   reduction operators, use this API with :ref:`LSA <device_api_lsa>` sources and multimem destinations; the multimem
   hardware path supports only sum. *coop* and *count* as for
   :ref:`ncclLsaReduceSumMultimemCopy <ncclLsaReduceSumMultimemCopy-lambda>`. Invocation as documented for
   :ref:`ncclLsaReduceSum <ncclLsaReduceSum-window-devComm>` and :ref:`ncclMultimemCopy <ncclMultimemCopy-symptr>`.
