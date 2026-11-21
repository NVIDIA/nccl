{#rel_2_0_2}
# NCCL Release 2.0.2

## Key Features and Enhancements

This NCCL release includes the following key features and enhancements.

-   NCCL 2.0.2 provides support for intra-node and inter-node communication.
-   NCCL optimizes intra-node communication using NVLink, PCI express, and shared memory.
-   Between nodes, NCCL implements fast transfers over sockets or InfiniBand verbs.
-   GPU-to-GPU and GPU-to-Network direct transfers, using the GPU Direct technology, is extensively used when the hardware topology permits it.

## Using NCCL 2.0.2

Ensure you are familiar with the following notes when using this release.

-   The NCCL 2.0 API is different from NCCL 1.x. Some porting may be needed for NCCL 1.x applications to work correctly. Refer to the migration documentation in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Known Issues

-   NCCL 2.0.2 is known to not work with CUDA driver 384.40 and later.
-   If NCCL returns any error code, set the environment variable `NCCL_DEBUG` to `WARN` to receive an explicit error message.
-   NCCL 2.0.2 does not support RoCE, that is, InfiniBand cards using Ethernet as link layer. The presence of an RoCE card on a node will make NCCL fail even when run within the node.
-   Using multiple processes in conjunction with multiple threads to manage the different GPUs may in some cases cause `ncclCommInitRank` to fail while establishing IPCs \(`cudaIpcOpenMemHandle`\). This problem does not appear when using only processes or only threads.


