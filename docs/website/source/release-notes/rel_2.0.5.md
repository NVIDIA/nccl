{#rel_2_0_5}
# NCCL Release 2.0.5

## Key Features and Enhancements

This NCCL release includes the following key features and enhancements.

-   NCCL 2.0.5 provides support for intra-node and inter-node communication.
-   NCCL optimizes intra-node communication using NVLink, PCI express, and shared memory.
-   Between nodes, NCCL implements fast transfers over sockets or InfiniBand verbs.
-   GPU-to-GPU and GPU-to-Network direct transfers, using the GPU Direct technology, are extensively used when the hardware topology permits it.

## Using NCCL 2.0.5

Ensure you are familiar with the following notes when using this release.

-   The NCCL 2.0 API is different from NCCL 1.x. Some porting may be needed for NCCL 1.x applications to work correctly. Refer to the migration documentation in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).
-   NCCL 2.0.5 has the new configuration file support. The NCCL environment variables can now be set in `~/.nccl.conf` and `/etc/nccl.conf`.
-   Values defined in `~/.nccl.conf` take precedence over ones in `/etc/nccl.conf`.
-   The syntax for each line of the NCCL configuration file is `<NCCL_VAR_NAME>=<VALUE>`.

## Known Issues

-   If NCCL returns any error code, set the environment variable `NCCL_DEBUG` to `WARN` to receive an explicit error message.
-   RoCE support is experimental.
-   Using multiple processes in conjunction with multiple threads to manage the different GPUs may in some cases cause `ncclCommInitRank` to fail while establishing IPCs \(`cudaIpcOpenMemHandle`\). This problem does not appear when using only processes or only threads.


