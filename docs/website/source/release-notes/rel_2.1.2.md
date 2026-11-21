{#rel_2_1_2}
# NCCL Release 2.1.2

## Key Features and Enhancements

This NCCL release includes the following key features and enhancements.

-   New algorithms for improved latency communication
-   RoCE support

## Using NCCL 2.1.2

Ensure you are familiar with the following notes when using this release.

-   The NCCL 2.x API is different from NCCL 1.x. Some porting may be needed for NCCL 1.x applications to work correctly. Refer to the migration documentation in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Known Issues

-   If NCCL returns an error code, set the environment variable `NCCL_DEBUG to WARN` to receive an explicit error message.
-   Using multiple processes in conjunction with multiple threads to manage the different GPUs may in some cases cause `ncclCommInitRank` to fail while establishing IPCs \(`cudaIpcOpenMemHandle`\). This problem does not appear when using only processes or only threads.
-   NCCL uses CUDA 9 cooperative group launch by default, which may induce increased latencies in multi-threaded programs. See the `NCCL_LAUNCH_MODE` knob in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html) to restore the original behavior.

## Fixed Issues

-   NCCL now uses CUDA 9 cooperative groups to launch the CUDA kernels, fixing a long-standing issue with CUDA and NCCL operations being interleaved and potentially causing hangs.


