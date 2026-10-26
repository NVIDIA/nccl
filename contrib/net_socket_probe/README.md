# NET confidential-computing copy path

Reproducers and correctness checks for NCCL issue #2503. Test configurations,
benchmark results and remaining validation gaps are described in
[NCCL PR #2511](https://github.com/NVIDIA/nccl/pull/2511).

## Configuration

The current implementation automatically selects staging for eligible non-GDR
host connections when NCCL's NVML wrapper reports confidential computing enabled.
It does not require `NCCL_PROTO=Simple`, `NCCL_NET_SHARED_BUFFERS=0`,
`NCCL_P2P_LL_THRESHOLD=0`, or `TORCH_NCCL_BLOCKING_WAIT`.

`NCCL_NET_CC_STAGING` is an optional override: unset / `-1` selects automatically,
`0` retains the original path, and `1` enables staging on eligible host connections
irrespective of CC detection. It does not redirect GDR or remote-proxy connections
through this copy path. Selection is per connection, so mixed connections use the
same existing NET progress routines without a global dispatch override.

`with_cc_staging.sh` selects the custom library only. It preserves the caller's
transport, protocol, buffer and framework wait settings:

```sh
NCCL_LIBRARY=/absolute/path/libnccl.so.2 ./with_cc_staging.sh COMMAND [ARGS...]
```

## Implementation

On the tested H20 TDX VMs, mapped CUDA host allocations have CUDA memory type
`cudaMemoryTypeManaged`. Kernel access and CPU/GPU polling of this memory can be
very slow or stall. CUDA-only controls reproduced the expensive handoff without
NCCL or PyTorch. No driver-internal fault mechanism is claimed without UVM traces.

The GPU instead uses VMM device buffers, allocated with NCCL's existing cuMem
allocator. Host staging buffers, observation storage, streams and events belong
to a separate reference-counted CUDA context per device. This prevents application
module loading from excluding proxy CUDA calls while waiting for an NCCL kernel.
Moving only the stream was insufficient: host allocations must also belong to
the copy context. Polls use finite copies and events, never an unsatisfied stream
wait on a future GPU counter.

The implementation extends the existing NET send/receive progression:

- Preserve network request grouping, registration handles, profiling, slot
  accounting and the existing collective algorithms.
- Retain reference-counted shared pools. Publish a slot's offset before its credit
  or receive tail on the same copy stream.
- Copy only GPU-owned control fields back. Invalidate outstanding snapshots when
  a send slot is released so an old size cannot reappear after its reset.
- Support SIMPLE, LL and LL128. Staged LL sends publish a completion counter after
  payload and cleanup flags are stored. Staged LL/LL128 receives wait for the
  copy-completion counter before examining in-band flags. Their wire formats do
  not change. Unstaged connections keep their existing synchronization branches.
- Require network receive completion before copying a staged payload. Publish the
  GPU tail after the copy; GPU acknowledgement then protects host-buffer reuse.

Keeping LL on the old mapped-host path did not suffice: on these CC machines both
that intermediate implementation and staging-disabled NCCL stalled during the
mixed-operation test's 1,024-element LL AllReduce. The device-buffer implementation
passed this case and the explicit LL matrix. Restricting selection to SIMPLE
would have hidden this failure.

The independent driver-symbol fix in `src/misc/cudawrap.cc` selects versioned
runtime lookup from CUDA 12.5 onward. CUDA 12.8's unversioned lookup otherwise
returned `cuCtxCreate_v4` through a v3 function-pointer declaration.

## Build and correctness checks

Use a full build for collective and SGLang tests:

```sh
make -j8 src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
nvcc -O2 -std=c++11 -arch=sm_90 -Ibuild/include \
  contrib/net_socket_probe/protocols.cu -Lbuild/lib -lnccl -o protocols
# Run one rank on each host, using the same library on both:
LD_LIBRARY_PATH=build/lib timeout --kill-after=5 120 \
  ./protocols RANK RANK0_IPV4 PORT 20
```

`protocols.cu` covers AllReduce, Broadcast, Reduce, AllGather, ReduceScatter,
bidirectional grouped P2P, and repeated queued P2P. Counts are 0, 1, 17, 1,024,
65,536 and 1,048,576 integers. Patterns vary with rank, iteration and element
position; every output element is checked. Run with default protocol/buffer
selection, each explicit protocol, and shared buffers enabled and disabled.

Other controls:

- `late_module.py` / `late_module.cu`: delayed peer send plus module loading after
  asynchronous `Work.wait()`, with fresh payload and kernel-output checks.
- `late_module_native.cu`: corresponding control without PyTorch.
- `stress.cu`: zero, unaligned and large byte counts, fresh payload/ACK per iteration.
- `abort.cu`: an unmatched 64 MiB send followed by abort; use an external timeout.
- `host_access.cu`, `host_handshake.cu`, `device_handshake.cu`: CUDA-only access
  and synchronization controls.

## Current validation scope

The broad implementation has passed the following on two H20 hosts, one GPU per
host, TDX and GPU CC ON:

| Check | Outcome |
| --- | --- |
| Default protocol and shared-buffer selection | Both ranks pass data checks |
| Explicit SIMPLE, LL, LL128 | Both ranks pass each protocol |
| Shared buffers disabled / enabled | Both ranks pass both settings |
| Default asynchronous late module loading, 5 repetitions | Pass with fresh data |
| Abort, both rank placements | Both ranks exit normally |
| SGLang main, Qwen3.5-35B-A3B, TP1/PP2, NEXTN, decode CUDA graphs | 80 requests complete |
| CUDA 12.8 with native driver, asynchronous control in both directions | Both ranks pass |

The SGLang tests omit protocol, shared-buffer, P2P LL-threshold, staging and
blocking-wait overrides. Transport/interface selectors and custom-library paths
are recorded separately with the commands. Source hashes, library hashes,
commands, statuses and logs are exported outside the source tree.

The final 2026-10-23 E2E run produced exactly the same output tokens for all 80
requests as the earlier SGLang main run. Each rank recorded 1,795 decode graph
replays with no fallback; measured blocks had no new compilation. Aggregate
throughput across three measured blocks was 143.30 output tokens/s at concurrency
1 and 316.43 at concurrency 4. This is a functional regression run, not a new
matched Gloo comparison.

Final library SHA256:

- CUDA 13: `f319155575fae9263001081507ef9caf08e32af577956082db15a0aee811b65a`
- Native CUDA 12.8: `e5f5a27326459b7eb0ff261677d6d28e907a66391fe8f0ba0537a3a4203737f0`

Both builds completed without compiler warnings. Native controls reported
`runtime=12080 driver_api=12080`. Both test containers were stopped afterwards.

This validates these configurations, not all hardware topologies. Multi-GPU
Protected PCIe, GDR/PXN, external NET plugins and performance on non-CC GPUs still
need their own coverage. In particular, the added LL/LL128 device state needs
non-CC performance review before claiming no performance regression. The original
issue's Protected PCIe deployment has not been retested here.

The remaining measurements below describe earlier restricted prototypes. Their
configuration requirements and LL rejection test do not describe current behavior.

## Historical SIMPLE-only measurements (2026-10-22)

CUDA 13.0 runtime / compatibility driver 580.95.05 / host driver 570.148.08,
same-source library, rank-0 RTT in milliseconds (20 timed samples per size):

| Bytes | Staging off median | Staging on medians, two runs | Largest timed on sample |
| ---: | ---: | ---: | ---: |
| 4096 | 7.502907 | 0.446236 / 0.460130 | 1.758125 |
| 65536 | 13.125334 | 0.639853 / 0.652093 | 0.840164 |
| 1048576 | 37.367422 | 1.777374 / 1.763296 | 2.761601 |

The matched baseline had a 7028.870265 ms maximum timed sample. Both modified
runs completed with data validation and exit code zero on both ranks.

The stress test completed in both directions: 10 sizes × 100 timed iterations,
plus five warmups per size, with every payload and ACK checked. Both ranks exited
zero. Two unmatched-send abort tests also completed, with abort calls taking
approximately 511–514 ms. An existing GPU process on the second host remained
running throughout; it was not stopped or modified.

The raw commands, exit codes, library hashes and logs are retained in the test
host's isolated experiment directory and exported separately, rather than added
to the source tree. The final source also built with CUDA 12.8 (C++14) and CUDA 13.0 (C++17).
Native CUDA 12.8 / driver 570.148.08 was explicitly selected and verified with
`runtime=12080 driver_api=12080`; the compatibility library was excluded.

| Native-driver test | 4 KiB median | 64 KiB median | 1 MiB median | Outcome |
| --- | ---: | ---: | ---: | --- |
| Staging off, 20 samples | 6.837368 ms | 11.881693 ms | 35.581107 ms | Both ranks exited 0 |
| Staging off, 500 samples | 6.942794 ms | Incomplete | Not reached | Both ranks timed out at 85 s |
| Staging on, 500 samples | 0.452869 ms | 0.640906 ms | 1.712391 ms | Both ranks exited 0; validated |

The final modified native-driver run's largest timed sample was 2.750991 ms.
The final CUDA 13 build repeated both 100-iteration-per-size stress directions
successfully. An explicit `NCCL_PROTO=LL` negative test with staging enabled
failed promptly on both ranks (exit 3), rather than timing out. The modified library itself logged `hostMemoryType=3`, confirming
the NET allocation was managed memory before its device mirror was introduced.

The tested final `src/transport/net.cc` SHA256 is
`92da20ba9e235b04fae66c22898d39a7eee4b2e964cff33a9e7b3c63ae6b70e4`.

NVIDIA's [R580 trusted-computing release notes](https://docs.nvidia.com/580trd1-trusted-computing-solutions-release-notes.pdf)
list CUDA forward compatibility as unsupported in CC modes. The native-driver
control above is therefore important: both the poor baseline and the improvement
were reproduced without that compatibility layer.
