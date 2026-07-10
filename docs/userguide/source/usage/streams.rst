*********************
CUDA Stream Semantics
*********************


NCCL calls are associated to a stream which is passed as the last argument of the collective communication function. The NCCL call returns when the operation has been effectively enqueued to the given stream, or returns an error. The collective operation is then executed asynchronously on the CUDA device. The operation status can be queried using standard CUDA semantics, for example, calling cudaStreamSynchronize or using CUDA events.


Mixing Multiple Streams within the same ncclGroupStart/End() group
------------------------------------------------------------------

NCCL allows for using multiple streams within a group call. This will enforce
a stream dependency of all streams before the NCCL kernel starts and block all
streams until the NCCL kernel completes.

It will behave as if the NCCL group operation was posted on every stream, but
given it is a single operation, it will cause a global synchronization point
between the streams.
