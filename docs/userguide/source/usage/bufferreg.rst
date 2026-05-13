.. _user_buffer_reg:

************************
User Buffer Registration
************************

User Buffer Registration is a feature that allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy (zero-copy).
It can accelerate collectives and greatly reduce the resource usage (e.g. #channel usage). NCCL provides two ways to register user buffers; one is *CUDA Graph*
registration, and the other is *Local* registration. NCCL requires that for all NCCL communication function calls (e.g., allreduce, sendrecv, and so on), if any
rank in a communicator passes registered buffers to a NCCL communication function, all other ranks in the same communicator must pass their registered buffers;
otherwise, mixing registered and non-registered buffers can result in undefined behavior; in addition, source and destination buffers must be registered in order
to enable user buffer registration for NCCL operations.

NVLink Sharp Buffer Registration
--------------------------------

Since 2.19.x, NCCL supports user buffer registration for NVLink Sharp (NVLS); any NCCL collectives (e.g., allreduce) that support NVLS algorithm can utilize this feature.

To enable the *CUDA Graph* based buffer registration for NVLS, users have to comply with several requirements:

 * The buffer is allocated through :c:func:`ncclMemAlloc` or a qualified allocator (see :ref:`mem_allocator`).
 * The NCCL operation is launched on a stream captured by a CUDA graph for each rank.
 * Offset to the head address of the buffer is the same in collectives for each rank.

Registered buffers will be deregistered when the CUDA graph is destroyed. Here is a CUDA graph based buffer registration example:

.. code:: C

  void* sendbuff;
  void* recvbuff;
  size_t count = 1 << 25;
  CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
  CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

  cudaGraph_t graph;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
  // Same offset to the sendbuff and recvbuff head address for each rank
  CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t instance;
  CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  CHECK(cudaGraphLaunch(instance, stream));
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaGraphExecDestroy(instance));
  CHECK(cudaGraphDestroy(graph));

  CHECK(ncclMemFree(sendbuff));
  CHECK(ncclMemFree(recvbuff));

On the other hand, to enable the *Local* based buffer registration for NVLS, users have to comply with the following requirements:

 * The buffer is allocated through :c:func:`ncclMemAlloc` or a qualified allocator (see :ref:`mem_allocator`).
 * Register buffer with :c:func:`ncclCommRegister` before calling collectives for each rank.
 * Call NCCL collectives as usual but similarly keep the offset to the head address of the buffer the same for each rank.

Registered buffers will be deregistered when users explicitly call :c:func:`ncclCommDeregister`. Here is a local based buffer registration example:

.. code:: C

  void* sendbuff;
  void* recvbuff;
  size_t count = 1 << 25;
  void* sendRegHandle;
  void* recvRegHandle;
  CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
  CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

  CHECK(ncclCommRegister(comm, sendbuff, count * sizeof(float), &sendRegHandle));
  CHECK(ncclCommRegister(comm, recvbuff, count * sizeof(float), &recvRegHandle));

  CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(ncclCommDeregister(comm, sendRegHandle));
  CHECK(ncclCommDeregister(comm, recvRegHandle));

  CHECK(ncclMemFree(sendbuff));
  CHECK(ncclMemFree(recvbuff));

For local based registration, users can register the buffer once at the beginning of the program and reuse the buffer multiple times to utilize
registration benefits.

To save the memory, it is also valid to allocate a large chunk of buffer and register it once. `sendbuff` and `recvbuff` can be further
allocated through the big chunk for zero-copy NCCL operations as long as `sendbuff` and `recvbuff` satisfy the offset requirements. The following
example shows a use case:

.. code:: C

  void* buffer;
  void* handle;
  void* sendbuff;
  void* recvbuff;
  size_t size = 1 << 29;

  CHECK(ncclMemAlloc(&buffer, size));
  CHECK(ncclCommRegister(comm, buffer, size, &handle));

  // assign buffer chunk to sendbuff and recvbuff
  sendbuff = buffer;
  recvbuff = (void*)((uint8_t*)buffer + (1 << 20));

  CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(ncclAllGather(sendbuff, recvbuff, 1024, ncclInt8, comm, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(ncclCommDeregister(comm, handle));

  CHECK(ncclMemFree(sendbuff));

IB Sharp Buffer Registration
----------------------------

NCCL 2.21.x supports IB Sharp buffer registration, any NCCL collectives that support IB Sharp algorithm can benefit from the feature such as allreduce,
reducescatter, and allgather. Currently, NCCL only supports IB Sharp buffer registration for the communicators which contain 1 rank per node, and the
registration can reduce the number of NCCL SM usage down to 1.

To enable IB Sharp buffer registration by CUDA graph:

 * Allocate send and recv buffer with any CUDA allocator (e.g., cudaMalloc/ncclMemAlloc)
 * Launch NCCL collectives with CUDA graph

To enable IB Sharp buffer registration by local registration:

 * Allocate send and recv buffer with any CUDA allocator (e.g., cudaMalloc/ncclMemAlloc)
 * Register send and recv buffer for each rank in the communicator with `ncclCommRegister`
 * Launch NCCL collectives

General Buffer Registration
---------------------------

Since 2.23.x, NCCL supports intra-node buffer registration, which targets all peer-to-peer intra-node communications (e.g., Allgather Ring) and brings less memory pressure, better communication and computation overlap performance. Either registering buffers by `ncclCommRegister` in the beginning or applying CUDA graph can enable intra-node buffer registration for NCCL collectives and sendrecv.

The user buffers can be allocated through VMM API (i.e., `cuMem*`), any VMM-based allocators (:ref:`mem_allocator`) or `ncclMemAlloc` will work.
The buffers allocated through legacy cuda API (e.g., `cudaMalloc`) can also be used for registration. However, it is not safe due to the potential hang during execution and segmentation fault during failure and abort, so using legacy buffers for registration is not recommended; currently, legacy buffer registration is disabled by default, users can set `NCCL_LEGACY_CUDA_REGISTER=1` to enable it.

Buffer Registration, GPU Direct RDMA, and MPS with MLOPart
----------------------------------------------------------

To ensure optimal performance for scale-out communications, we recommend the usage of `ncclMemAlloc`, or alternatively `cuMemCreate` with the attribute `gpuDirectRDMACapable`.
Failing to do so might force NCCL to use an internal staging buffer and therefore offset the gain provided by the user-buffer registration.

Further, mixing buffers allocated with different allocators maybe result in undefined behavior.

Buffer Registration and PXN
---------------------------

Buffer registration for network communication (e.g., InfiniBand) and PXN are inherently incompatible. PXN is enabled by default in NCCL as long as the platform supports it, and it can be used for sendrecv-based operations and collectives. When PXN is enabled, the network buffer registration will not be enabled even if users have called `ncclCommRegister` to register the buffers. To enable network buffer registration, users can set `NCCL_PXN_DISABLE=1` to disable PXN.

.. _mem_allocator:

Memory Allocator
----------------

For convenience, NCCL provides `ncclMemAlloc` function to help users to allocate buffers through VMM API, which can be used for NCCL registration later. It is only designed for NCCL so that it is not recommended to use `ncclMemAlloc` allocated buffers everywhere in the applications.

For advanced users, if you want to create your own memory allocator for NVLS UB, the allocated buffer of the allocator needs to satisfy the following requirements:

 * Allocate buffer with shared flag `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` and also `CU_MEM_HANDLE_TYPE_FABRIC` on GPUs where it's supported.
 * Buffer physical memory size is multiple of CUMEM recommended granularity (i.e. cuMemGetAllocationGranularity(..., `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED``))
 * Buffer virtual head address is at least aligned to CUMEM recommended granularity and size is multiple of CUMEM recommended granularity.

For general buffer registration with VMM API, the allocator needs to satisfy the same requirements as NVLS UB allocators.

.. _window_reg:

Window Registration
-------------------

Since 2.27, NCCL supports window registration, which allows users to register local buffers into NCCL window and enables extremely low latency and high bandwidth communication in NCCL. Currently, window registration supports input buffers only from VMM-based allocators (:ref:`mem_allocator`) and `ncclMemAlloc`; any other type of cuda buffers will fail to be registered.

NCCL window registration is enabled by default. However, if users do not use window registration and need to turn it off, set `NCCL_WIN_ENABLE=0` to disable it. In addition, users can also control the behavior of window registration through flags in :ref:`win_flags`.

.. _device_api_lsa:

For the device API, symmetrically registered windows (e.g. with :c:macro:`NCCL_WIN_COLL_SYMMETRIC`) provide **LSA** (load/store accessible) memory: device code can access peer buffers via load/store operations. See :ref:`device_api_memory` for device-side pointer accessors and reduce/copy operations.

The following example shows how to register buffers into NCCL window and use it for communication:

.. code:: C

  void* src;
  void* dst;
  ncclWindow_t src_win;
  ncclWindow_t dst_win;

  CHECK(ncclMemAlloc(&src, src_size));
  CHECK(ncclMemAlloc(&dst, dst_size));
  // Passing NCCL_WIN_COLL_SYMMETRIC requires users to provide the symmetric buffers among all ranks in collectives.
  // Every rank needs to call ncclCommWindowRegister to register its buffers.
  CHECK(ncclCommWindowRegister(comm, src, src_size, &src_win, NCCL_WIN_COLL_SYMMETRIC));
  CHECK(ncclCommWindowRegister(comm, dst, dst_size, &dst_win, NCCL_WIN_COLL_SYMMETRIC));
  // Use the registered buffers for communication to enable symmetric communication benefits.
  // In this example, every rank has 0x1000 offset and 0x2000 offset from the head address of
  // src and dst respectively, which satisfies the symmetric buffer requirement.
  CHECK(ncclAllGather((uint8_t*)src + 0x1000, (uint8_t*)dst + 0x2000, 1, ncclInt8, comm, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(ncclCommWindowDeregister(comm, src_win));
  CHECK(ncclCommWindowDeregister(comm, dst_win));

  CHECK(ncclMemFree(src));
  CHECK(ncclMemFree(dst));

See the description of :c:func:`ncclCommWindowRegister` and :c:func:`ncclCommWindowDeregister` for additional details.

Zero-CTA Optimization
------------------------

Since NCCL version 2.28, NCCL supports zero-CTA optimization. Zero-CTA optimization aims to avoid the use of CTA for communication and to overlap communication and computation.

Current zero-CTA optimization supports using the Copy Engine (CE) to perform the communication. The following are the requirements to enable zero-CTA optimization with CE:

 * CUDA driver version >= 12.5
 * Collectives run within a single NVL or MNNVL domain (does not support network, e.g., IB/ROCE)
 * The buffer is symmetrically registered with the NCCL window
 * The communicator is configured with the ``NCCL_CTA_POLICY_ZERO`` flag (please see :ref:`cta_policy_flags`)
 * Supported collectives are AlltoAll, AllGather, Scatter, and Gather

The following example shows how to enable zero-CTA optimization:

.. code:: C

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // NCCL_CTA_POLICY_ZERO to enable zero-CTA optimization whenever possible
  config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

  void* src;
  void* dst;
  ncclWindow_t src_win;
  ncclWindow_t dst_win;

  CHECK(ncclMemAlloc(&src, src_size));
  CHECK(ncclMemAlloc(&dst, dst_size));

  // Register the buffers into NCCL symmetric window
  CHECK(ncclCommWindowRegister(comm, src, src_size, &src_win, NCCL_WIN_COLL_SYMMETRIC));
  CHECK(ncclCommWindowRegister(comm, dst, dst_size, &dst_win, NCCL_WIN_COLL_SYMMETRIC));

  CHECK(ncclAllGather(src, dst, 1, ncclInt8, comm, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(ncclCommWindowDeregister(comm, src_win));
  CHECK(ncclCommWindowDeregister(comm, dst_win));

  CHECK(ncclMemFree(src));
  CHECK(ncclMemFree(dst));
