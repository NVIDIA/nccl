# GPU min/max special values

`minmax.cc` checks the documented GPU reduction semantics for FP32, FP16 and
BF16 with `ncclMin` and `ncclMax`. Each output is checked against an integer
oracle over the original input encodings:

- NaNs are ignored when any numeric input exists. All-NaN inputs accept any NaN
  result, without fixing its payload or sign.
- FP16 and BF16 order `-0` below `+0`. FP32 accepts either sign for a zero result.
- Finite values, infinities and nonzero results must match the expected bits.

The input table includes distinct quiet NaNs, both zeros, infinities, normal
range boundaries and ordinary finite values. The test covers in-place and
out-of-place calls, checks every result and allocation guard, and checks that
out-of-place input buffers are unchanged. ReduceScatter uses distinct data
phases for each destination slice, including when the receive count is 17.

## Build

Build against an existing NCCL installation and a CUDA toolkit with BF16
support. The test contains no device kernels and can be compiled with a host
C++ compiler. The full matrix requires two or six visible GPUs of SM80 or
newer; NCCL itself must contain code for the devices being tested.

```sh
CUDA_ROOT=/usr/local/cuda
NCCL_BUILD=/path/to/nccl/build
g++ -std=c++14 -O2 -Wall -Wextra -Werror \
  -I"$CUDA_ROOT/include" -I"$NCCL_BUILD/include" \
  public_tests/determinism/minmax.cc \
  -L"$CUDA_ROOT/lib64" -L"$NCCL_BUILD/lib" \
  -Wl,-rpath,"$CUDA_ROOT/lib64" -Wl,-rpath,"$NCCL_BUILD/lib" \
  -lcudart -lnccl -ldl -o minmax
```

## Run

Device ordinals are `0..N-1` within `CUDA_VISIBLE_DEVICES`. The default is two
GPUs, AllReduce, Ring and Simple. Select one collective and one protocol per
process:

```sh
./minmax --self-test
timeout 120s ./minmax
timeout 120s ./minmax --gpus 6 --collective reducescatter --proto LL
timeout 120s ./minmax --gpus 2 --collective reduce --root 1
timeout 120s ./minmax --gpus 2 --collective allreduce --algo Tree
```

`--collective` accepts `allreduce`, `reduce` and `reducescatter`; `--proto`
accepts `Simple` and `LL`. Tree is supported only for AllReduce. For Reduce,
the default root rotates with the case index; `--root N` runs every case with
the specified root. Run once per root to cover the full matrix at each root.

Each full two-GPU run executes 636 grouped collectives: 53 input shapes,
three types, two operations and two placements. Shapes use counts 1, 3, 4
and 17, rotating the 17 semantic pairs through each element position. Six-GPU
runs execute 708 collectives, adding sparse numeric inputs among NaNs and
rotating their source ranks. The final line reports planned and completed
case counts. An unavailable GPU configuration exits with status 77; API,
guard or value failures exit with status 1.

The test sets and prints its GPU arithmetic profile before initializing NCCL:
the requested Ring/Tree algorithm and Simple/LL protocol, empty
`NCCL_SYM_KERNEL`, disabled NVLS and CollNet, `NCCL_TUNER_PLUGIN=none`,
`NCCL_CONF_FILE=/dev/null`, and `NCCL_ENQUEUE_REARCH_ENABLE=1`. It also reports
the loaded NCCL library, CUDA versions and GPU compute capabilities. Record
the library hash with validation results. To inspect actual selection on a
short run, use:

```sh
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=TUNING \
  timeout 120s ./minmax --case-limit 2
```

The current default policy selects legacy enqueue for communicators whose
ranks are driven by one host thread, as in this test; setting
`NCCL_ENQUEUE_REARCH_ENABLE=1` retains that automatic choice. Its general-kernel
selector reports `Best tuning, algorithm, RING, protocol, SIMPLE` (or the
requested Tree/LL combination) at INFO level. Check the selected algorithm and
protocol for every rank. Builds with detailed tracing may also report a
structured `Best tuning` line; there should be no selected symmetric kernel.
`--case-limit N` runs only the first N cases and reports
`full=0`; this is useful for diagnostics, not full-matrix validation.

## Checker controls

The host-only `--self-test` checks the oracle against explicit expected
answers. It exercises NaN/finite substitutions, incorrect 16-bit zero signs,
permitted FP32 zero signs, poison values and swapped ReduceScatter slices.
It runs before every GPU test as well.

`--inject-error` overwrites the last valid output element of the last case
with `+3`, a value absent from all expected results. The normal output checker
must report the injected case/rank/element mismatch and exit with status 1;
an unrelated failure or a successful exit does not pass this negative control:

```sh
timeout 120s ./minmax --collective reducescatter --inject-error
```

Run the normal command separately to verify GPU correctness. The test is a
correctness check; its reported duration is the cost of the test body.
