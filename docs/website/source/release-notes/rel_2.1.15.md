{#rel_2_1_15}
# NCCL Release 2.1.15

## Key Features and Enhancements

This NCCL release includes the following key features and enhancements.

-   Added a variable to control InfiniBand Traffic Class and Retry Count.

## Using NCCL 2.1.15

Ensure you are familiar with the following notes when using this release.

-   The NCCL 2.x API is different from NCCL 1.x. Some porting may be needed for NCCL 1.x applications to work correctly. Refer to the migration documentation in the [NCCL Developer Guide](http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).
-   Starting in 2.2, NCCL supports collective aggregation. You can put multiple collectives in between `ncclGroupStart()` and `ncclGroupEnd()` to enable this feature.

## Known Issues

-   If NCCL returns an error code, set the environment variable `NCCL_DEBUG` to `WARN` to receive an explicit error message.
-   Using multiple processes in conjunction with multiple threads to manage the different GPUs may in some cases cause `ncclCommInitRank` to fail while establishing IPCs \(`cudaIpcOpenMemHandle`\). This problem does not appear when using only processes or only threads.
-   NCCL uses CUDA 9 cooperative group launch by default, which may induce increased latencies in multi-threaded programs. See the `NCCL_LAUNCH_MODE` knob to restore the original behavior.

## Fixed Issues

-   Fixed CPU usage and scheduling of NCCL network threads.
-   Fixed CUDA launch crash when mixing different types of GPUs in a node.
-   Fixed a performance problem on Skylake CPUs.
-   Fixed hanging issues with cudaFree and inter-node communication.
-   Restored library installation path to `/usr/lib/x86_64-linux-gnu` in debian packages.
-   Fixed RoCEv2 failure when using a non-zero GID.
-   No longer link to `stdc++` library statically as this can cause issues with C++ applications.
-   Fixed PyTorch hanging issues when using multiple rings and many back-to-back broadcast operations.


