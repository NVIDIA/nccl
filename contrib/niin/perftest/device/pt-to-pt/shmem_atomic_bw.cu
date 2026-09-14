/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// GPUNetIO remote-atomic bandwidth.  This follows NVSHMEM's
// shmem_atomic_bw operation/size matrix, but deliberately issues from one
// CUDA thread.  The current NIIN direct provider owns one QP and one response
// slot per destination PE, and serializes the slot until CQ completion.  A
// wider grid would therefore measure lock-spin contention rather than useful
// independent AMO throughput.

#include "perftest.h"

#include "niin/gpunetio/host.h"

#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstring>

namespace {

constexpr double kBytesPerGb = 1000.0 * 1000.0 * 1000.0;
constexpr double kOperationsPerM = 1000.0 * 1000.0;
constexpr double kMillisecondsPerSecond = 1000.0;

enum class AtomicOp : int {
  Inc,
  FetchInc,
  Set,
  Add,
  FetchAdd,
  And,
  FetchAnd,
  Or,
  FetchOr,
  Xor,
  FetchXor,
  Swap,
  CompareSwap,
};

const char* atomicOpName(AtomicOp op) {
  switch (op) {
    case AtomicOp::Inc: return "inc";
    case AtomicOp::FetchInc: return "fetch_inc";
    case AtomicOp::Set: return "set";
    case AtomicOp::Add: return "add";
    case AtomicOp::FetchAdd: return "fetch_add";
    case AtomicOp::And: return "and";
    case AtomicOp::FetchAnd: return "fetch_and";
    case AtomicOp::Or: return "or";
    case AtomicOp::FetchOr: return "fetch_or";
    case AtomicOp::Xor: return "xor";
    case AtomicOp::FetchXor: return "fetch_xor";
    case AtomicOp::Swap: return "swap";
    case AtomicOp::CompareSwap: return "compare_swap";
  }
  return "unknown";
}

bool parseAtomicOp(const char* name, AtomicOp* op) {
  static constexpr AtomicOp kOps[] = {
      AtomicOp::Inc,          AtomicOp::FetchInc, AtomicOp::Set,
      AtomicOp::Add,          AtomicOp::FetchAdd, AtomicOp::And,
      AtomicOp::FetchAnd,     AtomicOp::Or,       AtomicOp::FetchOr,
      AtomicOp::Xor,          AtomicOp::FetchXor, AtomicOp::Swap,
      AtomicOp::CompareSwap,
  };
  for (AtomicOp candidate : kOps) {
    if (std::strcmp(name, atomicOpName(candidate)) == 0) {
      *op = candidate;
      return true;
    }
  }
  return false;
}

[[noreturn]] void providerCheck(ncclResult_t result, const char* what) {
  std::fprintf(stderr, "shmem_atomic_bw: %s: %s\n", what, ncclGetErrorString(result));
#ifdef NIIN_HAS_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(1);
}

template <AtomicOp Op>
__global__ void atomicBwKernel(uint64_t* data, size_t elements, int peer, int iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  // Preserve returned values in a register and materialize it once.  This
  // keeps fetch AMOs observable without adding a device-memory operation per
  // AMO to the timed path.
  uint64_t consumed = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const uint64_t setValue = static_cast<uint64_t>(iteration) + 1;
    for (size_t element = 0; element < elements; ++element) {
      uint64_t* destination = data + element;
      if constexpr (Op == AtomicOp::Inc) {
        nvshmem_uint64_atomic_inc(destination, peer);
      } else if constexpr (Op == AtomicOp::FetchInc) {
        consumed ^= nvshmem_uint64_atomic_fetch_inc(destination, peer);
      } else if constexpr (Op == AtomicOp::Set) {
        nvshmem_uint64_atomic_set(destination, setValue, peer);
      } else if constexpr (Op == AtomicOp::Add) {
        nvshmem_uint64_atomic_add(destination, UINT64_C(1), peer);
      } else if constexpr (Op == AtomicOp::FetchAdd) {
        consumed ^= nvshmem_uint64_atomic_fetch_add(destination, UINT64_C(1), peer);
      } else if constexpr (Op == AtomicOp::And) {
        const unsigned int shift = static_cast<unsigned int>(iteration < 63 ? iteration + 1 : 63);
        nvshmem_uint64_atomic_and(destination, data[element] << shift, peer);
      } else if constexpr (Op == AtomicOp::FetchAnd) {
        const unsigned int shift = static_cast<unsigned int>(iteration < 63 ? iteration + 1 : 63);
        consumed ^= nvshmem_uint64_atomic_fetch_and(destination, data[element] << shift, peer);
      } else if constexpr (Op == AtomicOp::Or) {
        const unsigned int shift = static_cast<unsigned int>(iteration < 63 ? iteration : 63);
        nvshmem_uint64_atomic_or(destination, data[element] << shift, peer);
      } else if constexpr (Op == AtomicOp::FetchOr) {
        const unsigned int shift = static_cast<unsigned int>(iteration < 63 ? iteration : 63);
        consumed ^= nvshmem_uint64_atomic_fetch_or(destination, data[element] << shift, peer);
      } else if constexpr (Op == AtomicOp::Xor) {
        nvshmem_uint64_atomic_xor(destination, UINT64_C(1), peer);
      } else if constexpr (Op == AtomicOp::FetchXor) {
        consumed ^= nvshmem_uint64_atomic_fetch_xor(destination, UINT64_C(1), peer);
      } else if constexpr (Op == AtomicOp::Swap) {
        consumed ^= nvshmem_uint64_atomic_swap(destination, setValue, peer);
      } else if constexpr (Op == AtomicOp::CompareSwap) {
        consumed ^= nvshmem_uint64_atomic_compare_swap(destination, static_cast<uint64_t>(iteration),
                                                        setValue, peer);
      }
    }
  }

  // Every direct AMO has already waited for CQ completion on return, so no
  // nvshmem_quiet() is needed here.  This local store makes `consumed` live
  // until after the final AMO without touching the remote target.
  data[0] = consumed;
}

template <AtomicOp Op>
void launchAtomicKernel(uint64_t* data, size_t elements, int peer, int iterations) {
  atomicBwKernel<Op><<<1, 1>>>(data, elements, peer, iterations);
}

void launchAtomicKernel(AtomicOp op, uint64_t* data, size_t elements, int peer, int iterations) {
  switch (op) {
    case AtomicOp::Inc: return launchAtomicKernel<AtomicOp::Inc>(data, elements, peer, iterations);
    case AtomicOp::FetchInc: return launchAtomicKernel<AtomicOp::FetchInc>(data, elements, peer, iterations);
    case AtomicOp::Set: return launchAtomicKernel<AtomicOp::Set>(data, elements, peer, iterations);
    case AtomicOp::Add: return launchAtomicKernel<AtomicOp::Add>(data, elements, peer, iterations);
    case AtomicOp::FetchAdd: return launchAtomicKernel<AtomicOp::FetchAdd>(data, elements, peer, iterations);
    case AtomicOp::And: return launchAtomicKernel<AtomicOp::And>(data, elements, peer, iterations);
    case AtomicOp::FetchAnd: return launchAtomicKernel<AtomicOp::FetchAnd>(data, elements, peer, iterations);
    case AtomicOp::Or: return launchAtomicKernel<AtomicOp::Or>(data, elements, peer, iterations);
    case AtomicOp::FetchOr: return launchAtomicKernel<AtomicOp::FetchOr>(data, elements, peer, iterations);
    case AtomicOp::Xor: return launchAtomicKernel<AtomicOp::Xor>(data, elements, peer, iterations);
    case AtomicOp::FetchXor: return launchAtomicKernel<AtomicOp::FetchXor>(data, elements, peer, iterations);
    case AtomicOp::Swap: return launchAtomicKernel<AtomicOp::Swap>(data, elements, peer, iterations);
    case AtomicOp::CompareSwap:
      return launchAtomicKernel<AtomicOp::CompareSwap>(data, elements, peer, iterations);
  }
}

void initializeData(uint64_t* data, size_t bytes, AtomicOp op) {
  const int byteValue = (op == AtomicOp::And || op == AtomicOp::FetchAnd || op == AtomicOp::Or ||
                         op == AtomicOp::FetchOr)
                            ? 0xff
                            : 0;
  CUDA_CHECK(cudaMemset(data, byteValue, bytes));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void printAtomicBwTable(const char* operation, uint64_t* sizes, double* bandwidth,
                        double* amoRate, int entries) {
  std::printf("\nshmem_atomic_%s (GPUNetIO direct, serialized endpoint)\n", operation);
  std::printf("%-16s %12s %12s\n", "size (Bytes)", "BW (GB/s)", "MAMO/s");
  std::printf("%-16s %12s %12s\n", "----------------", "------------", "------------");
  for (int i = 0; i < entries; ++i) {
    std::printf("%-16lu %12.3f %12.3f\n", static_cast<unsigned long>(sizes[i]), bandwidth[i],
                amoRate[i]);
  }
  std::printf("\n");
  std::fflush(stdout);
}

bool helpRequested(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) return true;
  }
  return false;
}

void printAtomicUsage() {
  std::printf(
      "Usage: shmem_atomic_bw [options]\n"
      "  -b, --min_size <bytes>    Minimum payload size (default: 8)\n"
      "  -e, --max_size <bytes>    Maximum payload size (default: 65536)\n"
      "  -f, --step <factor>       Size step factor (default: 2)\n"
      "  -i, --iters <n>           Timed iterations (default: 10)\n"
      "  -w, --warmup <n>          Warmup iterations (default: 10)\n"
      "  -a, --atomic_op <op>      inc, fetch_inc, set, add, fetch_add, and, fetch_and,\n"
      "                              or, fetch_or, xor, fetch_xor, swap, compare_swap (default: inc)\n"
      "  -t, --threads <n>         Must be 1 (one serialized direct endpoint issuer)\n"
      "  -n, --blocks <n>          Must be 1 (one serialized direct endpoint issuer)\n");
}

}  // namespace

int main(int argc, char* argv[]) {
  // A serialized request/completion endpoint cannot use the generic RMA
  // defaults (4 MiB x 200 iterations) responsibly.  Match the upstream AMO
  // test's compact sweep; explicit command-line arguments still override
  // every one of these values in read_args().
  min_size = sizeof(uint64_t);
  max_size = 64 * 1024;
  iters = 10;
  warmup_iters = 10;
  num_blocks = 1;
  threads_per_block = 1;
  if (helpRequested(argc, argv)) {
    printAtomicUsage();
    return 0;
  }
  read_args(argc, argv);

  AtomicOp op;
  if (!parseAtomicOp(atomic_op, &op)) {
    std::fprintf(stderr,
                 "Unsupported --atomic_op '%s'. Supported operations: inc, fetch_inc, set, add, "
                 "fetch_add, and, fetch_and, or, fetch_or, xor, fetch_xor, swap, compare_swap\n",
                 atomic_op);
    return 1;
  }
  if (min_size < sizeof(uint64_t) || min_size > max_size) {
    std::fprintf(stderr, "shmem_atomic_bw requires 8 <= min_size <= max_size\n");
    return 1;
  }
  if (iters == 0 || iters > static_cast<size_t>(INT_MAX) ||
      warmup_iters > static_cast<size_t>(INT_MAX)) {
    std::fprintf(stderr, "shmem_atomic_bw requires 1 <= iters and 32-bit iteration counts\n");
    return 1;
  }
  if (num_blocks != 1 || threads_per_block != 1) {
    std::fprintf(stderr,
                 "shmem_atomic_bw requires --blocks 1 and --threads 1: the current direct provider "
                 "serializes each destination endpoint through one response slot\n");
    return 1;
  }

  init_wrapper(&argc, &argv);

  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  if (npes != 2) {
    if (mype == 0) std::fprintf(stderr, "This test requires exactly 2 PEs\n");
    finalize_wrapper();
    return 1;
  }

  // Public atomic APIs deliberately choose CUDA atomics for LSA peers.  Make
  // this a two-node test so the measured operation is the bound GPUNetIO QP.
  auto& state = niin::detail::state();
  if (state.lsaSize != 1) {
    if (mype == 0) {
      std::fprintf(stderr,
                   "shmem_atomic_bw requires the two PEs on distinct LSA domains; "
                   "same-node peers use CUDA atomics instead of GPUNetIO\n");
    }
    finalize_wrapper();
    return 1;
  }

  // nvshmem_init() brings the provider up and binds it; this test only needs
  // to confirm it is actually there before reporting network AMO numbers.
  if (state.gpunetioAtomics == nullptr) {
    if (mype == 0) {
      std::fprintf(stderr,
                   "shmem_atomic_bw: no network atomic provider was bound. Link the NIIN "
                   "GPUNetIO provider and leave NIIN_GPUNETIO_ATOMICS unset (or =1 to see why)\n");
    }
    finalize_wrapper();
    return 1;
  }

  uint64_t* data = static_cast<uint64_t*>(nvshmem_malloc(max_size));
  if (data == nullptr) {
    if (mype == 0) std::fprintf(stderr, "shmem_atomic_bw: nvshmem_malloc(%zu) failed\n", max_size);
    finalize_wrapper();
    return 1;
  }

  void** tables = nullptr;
  alloc_tables(&tables, 3, static_cast<int>(max_size_log + 1));
  uint64_t* sizes = static_cast<uint64_t*>(tables[0]);
  double* bandwidth = static_cast<double*>(tables[1]);
  double* amoRate = static_cast<double*>(tables[2]);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int peer = 1;
  int entries = 0;
  for (size_t requestedBytes = min_size; requestedBytes <= max_size;) {
    const size_t elements = requestedBytes / sizeof(uint64_t);
    const size_t effectiveBytes = elements * sizeof(uint64_t);
    if (elements != 0) {
      sizes[entries] = effectiveBytes;

      // Every PE initializes its local target before the source PE issues a
      // new phase.  The host barriers prevent a target reset from racing a
      // completed remote AMO phase.
      initializeData(data, effectiveBytes, op);
      nvshmem_barrier_all();
      if (mype == 0) {
        launchAtomicKernel(op, data, elements, peer, static_cast<int>(warmup_iters));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      nvshmem_barrier_all();

      initializeData(data, effectiveBytes, op);
      nvshmem_barrier_all();
      if (mype == 0) {
        CUDA_CHECK(cudaEventRecord(start));
        launchAtomicKernel(op, data, elements, peer, static_cast<int>(iters));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        const double seconds = milliseconds / kMillisecondsPerSecond;
        const double operations = static_cast<double>(elements) * static_cast<double>(iters);
        bandwidth[entries] = static_cast<double>(effectiveBytes) * static_cast<double>(iters) /
                             seconds / kBytesPerGb;
        amoRate[entries] = operations / seconds / kOperationsPerM;
      }
      nvshmem_barrier_all();
      ++entries;
    }

    if (requestedBytes > max_size / step_factor) break;
    requestedBytes *= step_factor;
  }

  if (mype == 0) printAtomicBwTable(atomicOpName(op), sizes, bandwidth, amoRate, entries);

  CUDA_CHECK(cudaDeviceSynchronize());
  nvshmem_barrier_all();
  nvshmem_free(data);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  free_tables(tables, 3);
  finalize_wrapper();
  return 0;
}
