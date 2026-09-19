# AlltoAll per-call CTA cap regression

Regression for [#2421](https://github.com/NVIDIA/nccl/issues/2421). Each MPI rank owns
one CUDA device. The test checks exact output and observes kernel grids independently:
CUPTI driver launch callbacks for eager execution, CUDA driver kernel-node inspection
plus three validated replays for captured graphs. Warmup is excluded from observations.

Cases cover unset/1/3/7 caps, a grouped unset/3/1 sequence, capped AlltoAll mixed with
Send/Recv and capped AllReduce, ordinary Send/Recv after a cap, and restoration of the
default AlltoAll mapping. A separate mixed case checks an AllReduce with a wider
cap; its grids are reported separately without applying the AlltoAll cap to that
AllReduce. Grouped AlltoAll calls use the tightest cap for their shared
P2P plan. Because channel mapping uses powers of two, maxCTAs=3 may select two CTAs;
the cap is an upper bound, not a request for an exact grid size.

```sh
nvcc -std=c++17 -arch=sm_89 -DOMPI_SKIP_MPICXX=1 test.cu \
  -I/path/to/nccl/include -I/usr/local/cuda/extras/CUPTI/include \
  $(mpicxx --showme:compile) $(mpicxx --showme:link) \
  -L/path/to/nccl/lib -lnccl -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -lcuda -o test
export LD_LIBRARY_PATH=/path/to/nccl/lib:/usr/local/cuda/extras/CUPTI/lib64
export CUDA_VISIBLE_DEVICES=0,1
for mode in eager graph; do
  for count in 0 1 257 2097152; do
    timeout 120 mpirun -np 2 ./test "$mode" "$count"
  done
done
```

Use the GPU architecture and toolkit paths for the host. Repeat with
`NCCL_ENQUEUE_REARCH_ENABLE=1`, and with four visible GPUs / `-np 4` when available.
Exit 0 means all checked outputs and bounds passed; exit 2 means a CTA bound was
violated. CUDA/NCCL/MPI failures or output mismatches abort with exit 1. Empty calls
must not require a kernel. This diagnostic is not a performance benchmark.

As with collective resource settings, all ranks must use matching per-call caps.
A cap applies to P2P work fused in the same group, so uncapped Send/Recv in that group
may use fewer channels. The communicator's default mapping is retained for later groups.
