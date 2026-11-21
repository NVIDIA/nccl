{#rel_2_1_4}
# NCCL Release 2.1.4

## Key Features and Enhancements

This NCCL release includes the following key features and enhancements.

-   Added support for InfiniBand GID selection, enabling the use of RoCE v2.
-   Added support for InfiniBand Service Level \(SL\) selection.

## Using NCCL 2.1.4

Ensure you are familiar with the following notes when using this release.

-   The NCCL 2.x API is different from NCCL 1.x. Some porting may be needed for NCCL 1.x applications to work correctly. Refer to the migration documentation in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Known Issues

-   If NCCL returns an error code, set the environment variable `NCCL_DEBUG to WARN` to receive an explicit error message.
-   Using multiple processes in conjunction with multiple threads to manage the different GPUs may in some cases cause `ncclCommInitRank` to fail while establishing IPCs \(`cudaIpcOpenMemHandle`\). This problem does not appear when using only processes or only threads.
-   NCCL uses CUDA® 9 cooperative group launch by default, which may induce increased latencies in multi-threaded programs. See the `NCCL_LAUNCH_MODE` knob in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html) to restore the original behavior.
-   NCCL 2.1.4-1 embeds `libstdc++` and exports its symbols. This can break C++ applications.

## Fixed Issues

-   Fixed bug causing CUDA IPC to fail in some situations.
-   Fixed bug causing a crash when p2p mappings are exhausted instead of returning an error.


