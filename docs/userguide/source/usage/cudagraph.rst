.. _using-nccl-with-cuda-graphs:

***************************
Using NCCL with CUDA Graphs
***************************

Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs.

CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. More details about CUDA Graphs can be found in the `CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>`_.

NCCL's collective, P2P and group operations all support CUDA Graph captures. This support requires a minimum CUDA version of 11.3.

Requirements and Limitations
----------------------------

Using multiple GPUs per process with CUDA graph capture may result in deadlocks. A deadlock is likely to occur with single-threaded applications: in some cases ``cudaGraphLaunch`` may block, preventing the launch across all GPUs. We recommend running with a single GPU per process for the most reliable experience.

Whether an operation launch is graph-captured is considered a collective property of that operation and therefore must be uniform over all ranks participating in the launch (for collectives this is all ranks in the communicator, for peer-to-peer this is both the sender and receiver). The launch of a graph (via cudaGraphLaunch, etc.) containing a captured NCCL operation is considered collective for the same set of ranks that were present in the capture, and each of those ranks must be using the graph derived from that collective capture.

The following sample code shows how to capture computational kernels and NCCL operations in a CUDA Graph: ::

  cudaGraph_t graph;
  cudaStreamBeginCapture(stream);
  kernel_A<<< ..., stream >>>(...);
  kernel_B<<< ..., stream >>>(...);
  ncclAllreduce(..., stream);
  kernel_C<<< ..., stream >>>(...);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

Starting with NCCL 2.11, when NCCL communication is captured and the CollNet algorithm is used, NCCL allows for further performance improvement via user buffer registration. For details, please see the environment variable :ref:`NCCL_GRAPH_REGISTER`.

Mixing graph-captured and non-graph-captured NCCL operations is supported by NCCL. However, when graphs involving multiple communicators are ``cudaGraphLaunch``'d from the same thread, the internal mechanism NCCL uses to support this mixing can contribute to the deadlocks described above. To disable this mechanism, see the environment variable :ref:`NCCL_GRAPH_MIXING_SUPPORT`.

Disabling NCCL's capture-time serialization of communication kernels (see
:ref:`NCCL_GRAPH_STREAM_ORDERING` and :c:macro:`graphStreamOrdering` in
:ref:`ncclconfig`) together with graph mixing (communicator ``graphUsageMode=2``)
is **not supported**. If ordering is disabled for a communicator, **graph mixing
must be off** (``graphUsageMode`` ``0`` or ``1``); workloads that need mixing must
keep the default ordering (``1``).
