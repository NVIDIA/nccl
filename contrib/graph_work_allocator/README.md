# External CUDA Graph work allocator

Prototype for the local work-descriptor slice of [#2416](https://github.com/NVIDIA/nccl/issues/2416).
Set `ncclConfig_t.workAllocator` before communicator initialization. The descriptor
is copied by value; its `context` remains caller-owned. Both callbacks are required.
A split communicator without an overriding config inherits the descriptor.

Only captured kernel plans whose work descriptors do not fit in kernel arguments
use the allocator. Ordinary enqueue, transport buffers, user buffers, registrations,
NVLS windows and the memory manager's suspend/offload allocations retain their current
allocators. This is not an implementation of the entire external allocator RFE.

`alloc(context, &ptr, bytes, device, stream)` must return at least 16-byte-aligned
storage on `device`, usable in order on `stream`. NCCL copies the descriptors on
that stream. A callback error propagates without internal fallback; success with a
null pointer returns `ncclInvalidUsage`. The callback must retain any rounded size
or pool metadata it needs: NCCL supplies the original request size to `free`.

`free(context, ptr, bytes, device)` receives exactly the allocated pointer and request
size after graph users release the plan. External pointers never enter `cudaFree`
or `ncclCuMemFree` inside NCCL. The context and callback code must remain alive until
all graphs and graph executables are destroyed and communicator destruction completes.
Callbacks can run on internal threads; they must be thread-safe and must not call NCCL.
A callback returning an allocation error must not retain an allocation for NCCL.

An allocator callback does not shorten the plan's lifetime or return its storage
between replays. The capacity benefit is limited to sharing the pool that backs this
specific local allocation class; there is no claim of reclaiming transport memory.

## GPU regression

`test.cu` supplies an independent CUDA memory pool and checks actual graph AllReduce
results, pointer/size pairing, current device, allocation errors, legacy config size,
and balanced alloc/free counts after graph and communicator destruction.

```sh
nvcc -std=c++17 -arch=sm_89 -DWORK_ALLOCATOR -DOMPI_SKIP_MPICXX=1 \
  $(mpicxx --showme:compile) test.cu -I/path/to/nccl/include \
  -L/path/to/nccl/lib -lnccl $(mpicxx --showme:link) -o test
export LD_LIBRARY_PATH=/path/to/nccl/lib
export CUDA_VISIBLE_DEVICES=0,1 NCCL_ALGO=Ring NCCL_WORK_ARGS_BYTES=256
for mode in default external legacy split invalid fail; do
  timeout 120 mpirun -np 2 ./test "$mode"
done
```

Use the actual GPU architecture in the compile command. The small work-argument
budget forces the persistent-buffer path; it is a test setting, not a tuning recommendation.
Run the matrix again with `NCCL_ENQUEUE_REARCH_ENABLE=1` to check the alternate enqueue path.
To build against an unmodified NCCL, omit `-DWORK_ALLOCATOR` and run `default` only.
The printed capture/replay times are diagnostics; compare independent A/P runs
with matched conditions before making a performance claim.
