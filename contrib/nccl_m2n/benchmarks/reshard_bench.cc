/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard Benchmark (User Window API)
 *
 * Uses ncclReshardWithWindow with caller-registered ncclWindow_t.
 *
 * Usage:
 *   mpirun -np <N> reshard_bench [options]
 *
 * Example:
 *   mpirun -np 6 reshard_bench \
 *       --src-mesh-dims 1,4 --dst-mesh-dims 1,2 \
 *       --tensor-dims 256,128,64 \
 *       --src-shard-dim 0 --dst-shard-dim 0 \
 *       --validate --verbose
 *
 ************************************************************************/

#include "bench_common.h"
#include "bench_common_kernels.h"

#include "nccl_m2n.h"

static void printUsage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("\nRequired:\n");
  printf("  --src-mesh-dims <rep>,<shard>    Source mesh dimensions\n");
  printf("  --dst-mesh-dims <rep>,<shard>    Dest mesh dimensions\n");
  printf("  --tensor-dims <d0>,<d1>[,<d2>]   Global tensor dims (2D or 3D)\n");
  printf("  --src-shard-dim <0|1|2>          Source sharding dimension\n");
  printf("  --dst-shard-dim <0|1|2>          Dest sharding dimension (can "
         "differ!)\n");
  printf("\nOptional:\n");
  printf("  --iterations <N>                 Timed iterations (default: 10)\n");
  printf("  --warmup <N>                     Warmup iterations (default: 2)\n");
  printf("  --validate                       Validate data correctness\n");
  printf("  --algorithm <algo>               Algorithm: 'ring' (default) or "
         "'direct'\n");
  printf("  --lb-mode <uniform|node>         Load balancing: 'uniform' "
         "(default) or 'node'\n");
  printf("  --verbose                        Enable debug output\n");
  printf("  --print-all-ranks                Print per-rank timing\n");
  printf("  --use-default-stream             Pass nullptr to "
         "ncclReshardWithWindow so the\n");
  printf("                                   library substitutes a stream from "
         "its internal\n");
  printf("                                   pool — exercises the "
         "default-stream code path.\n");
  printf("\nExamples:\n");
  printf("  # Same-dim sharding (partial overlap)\n");
  printf("  mpirun -np 6 %s --src-mesh-dims 1,4 --dst-mesh-dims 1,2 \\\n", prog);
  printf("         --tensor-dims 256,128,64 --src-shard-dim 0 --dst-shard-dim 0 "
         "--validate\n");
  printf("\n  # Cross-dim sharding (all-to-all)\n");
  printf("  mpirun -np 6 %s --src-mesh-dims 1,4 --dst-mesh-dims 1,2 \\\n", prog);
  printf("         --tensor-dims 256,128,64 --src-shard-dim 0 --dst-shard-dim 1 "
         "--validate\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));

  int mpiRank, mpiSize;
  MPICHECK(MPI_Comm_rank(benchMpiWorld(), &mpiRank));
  MPICHECK(MPI_Comm_size(benchMpiWorld(), &mpiSize));

  // Default parameters
  int srcMeshDims[2] = {0, 0};
  int dstMeshDims[2] = {0, 0};
  size_t globalTensorDims[3] = {0, 0, 0};
  int ndims = 0;
  int srcShardDim = -1;
  int dstShardDim = -1;
  int iterations = 10;
  int warmup = 2;
  bool validate = false;
  bool verbose = false;
  bool printAllRanks = false;
  bool useDefaultStream = false;
  const char* algorithm = "RING";
  const char* lbMode = "UNIFORM";

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--src-mesh-dims") == 0) {
      benchParseMeshDims(argv[++i], srcMeshDims);
    } else if (strcmp(argv[i], "--dst-mesh-dims") == 0) {
      benchParseMeshDims(argv[++i], dstMeshDims);
    } else if (strcmp(argv[i], "--tensor-dims") == 0) {
      ndims = benchParseTensorDims(argv[++i], globalTensorDims);
    } else if (strcmp(argv[i], "--src-shard-dim") == 0) {
      srcShardDim = benchParseInt(argv[++i]);
    } else if (strcmp(argv[i], "--dst-shard-dim") == 0) {
      dstShardDim = benchParseInt(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0) {
      iterations = benchParseInt(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0) {
      warmup = benchParseInt(argv[++i]);
    } else if (strcmp(argv[i], "--validate") == 0) {
      validate = true;
    } else if (strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "--print-all-ranks") == 0) {
      printAllRanks = true;
    } else if (strcmp(argv[i], "--use-default-stream") == 0) {
      useDefaultStream = true;
    } else if (strcmp(argv[i], "--algorithm") == 0) {
      ++i;
      if (strcmp(argv[i], "direct") == 0) {
        algorithm = "DIRECT";
      } else if (strcmp(argv[i], "ring") == 0) {
        algorithm = "RING";
      } else {
        if (mpiRank == 0) {
          printf("ERROR: Unknown algorithm '%s'. Use 'ring' or "
                 "'direct'\n",
                 argv[i]);
        }
        MPI_Finalize();
        return 1;
      }
    } else if (strcmp(argv[i], "--lb-mode") == 0) {
      ++i;
      if (strcmp(argv[i], "node") == 0) {
        lbMode = "NODE_AWARE";
      } else if (strcmp(argv[i], "uniform") == 0) {
        lbMode = "UNIFORM";
      } else {
        if (mpiRank == 0) {
          printf("ERROR: Unknown lb-mode '%s'. Use 'uniform' or "
                 "'node'\n",
                 argv[i]);
        }
        MPI_Finalize();
        return 1;
      }
    } else if (strcmp(argv[i], "--help") == 0) {
      if (mpiRank == 0) printUsage(argv[0]);
      MPI_Finalize();
      return 0;
    }
  }

  // Configure reshard library via env vars (applied in ncclM2nInit).
  if (verbose) benchSetEnv("NCCL_RESHARD_LOG_LEVEL", "DEBUG");
  benchSetEnv("NCCL_RESHARD_ALGORITHM", algorithm);
  benchSetEnv("NCCL_RESHARD_LB_MODE", lbMode);
  NCCLCHECK(ncclM2nInit(NULL));

  // Validate required parameters
  if (srcMeshDims[0] <= 0 || srcMeshDims[1] <= 0 || dstMeshDims[0] <= 0 || dstMeshDims[1] <= 0 || ndims < 2 ||
      ndims > 3 || srcShardDim < 0 || srcShardDim >= ndims || dstShardDim < 0 || dstShardDim >= ndims) {
    if (mpiRank == 0) {
      printf("ERROR: Missing or invalid required parameters\n");
      printUsage(argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  // Calculate total ranks needed
  int srcTotal = srcMeshDims[0] * srcMeshDims[1];
  int dstTotal = dstMeshDims[0] * dstMeshDims[1];
  int totalExpected = srcTotal + dstTotal;

  if (mpiSize != totalExpected) {
    if (mpiRank == 0)
      printf("ERROR: Expected %d processes (src=%d + dst=%d), got %d\n", totalExpected, srcTotal, dstTotal, mpiSize);
    MPI_Finalize();
    return 1;
  }

  // Determine role
  bool isSource = (mpiRank < srcTotal);
  bool isDest = (mpiRank >= srcTotal);

  // Compute shard counts
  int srcShardCount = srcMeshDims[1]; // Sharding is on mesh dim 1
  int dstShardCount = dstMeshDims[1];

  // Compute local tensor dimensions
  size_t srcLocalDims[3], dstLocalDims[3];
  for (int d = 0; d < ndims; d++) {
    if (d == srcShardDim) srcLocalDims[d] = globalTensorDims[d] / srcShardCount;
    else srcLocalDims[d] = globalTensorDims[d];
    if (d == dstShardDim) dstLocalDims[d] = globalTensorDims[d] / dstShardCount;
    else dstLocalDims[d] = globalTensorDims[d];
  }

  // Print configuration
  if (mpiRank == 0) {
    printf("=== Tensor Reshard Benchmark ===\n");
    printf("Using: ncclReshardWithWindow (user window API)\n");
    printf("Global tensor: [%zu", globalTensorDims[0]);
    for (int d = 1; d < ndims; d++) printf(", %zu", globalTensorDims[d]);
    printf("] (%dD)\n", ndims);
    printf("Source shard dim: %d, Dest shard dim: %d%s\n", srcShardDim, dstShardDim,
           srcShardDim == dstShardDim ? " (same-dim)" : " (CROSS-DIM!)");
    printf("Source: %d ranks = %d reps x %d shards, local=[%zu", srcTotal, srcMeshDims[0], srcMeshDims[1],
           srcLocalDims[0]);
    for (int d = 1; d < ndims; d++) printf(", %zu", srcLocalDims[d]);
    printf("]\n");
    printf("Dest: %d ranks = %d reps x %d shards, local=[%zu", dstTotal, dstMeshDims[0], dstMeshDims[1],
           dstLocalDims[0]);
    for (int d = 1; d < ndims; d++) printf(", %zu", dstLocalDims[d]);
    printf("]\n");
    printf("Algorithm: %s\n", algorithm);
    if (strcmp(algorithm, "RING") == 0) printf("Load Balance Mode: %s\n", lbMode);
    printf("Iterations: %d (warmup: %d), Validate: %s\n", iterations, warmup, validate ? "yes" : "no");
    fflush(stdout);
  }

  // Setup CUDA device
  int numDevices;
  CUDACHECK(cudaGetDeviceCount(&numDevices));
  CUDACHECK(cudaSetDevice(mpiRank % numDevices));

  // Create NCCL communicator
  ncclUniqueId worldId;
  if (mpiRank == 0) NCCLCHECK(ncclGetUniqueId(&worldId));
  MPICHECK(MPI_Bcast(&worldId, sizeof(worldId), benchMpiByte(), 0, benchMpiWorld()));

  ncclComm_t worldComm;
  NCCLCHECK(ncclCommInitRank(&worldComm, mpiSize, worldId, mpiRank));

  // Allocate buffer (symmetric memory required for one-sided ops)
  size_t srcBufferSize = 1, dstBufferSize = 1;
  for (int d = 0; d < ndims; d++) {
    srcBufferSize *= srcLocalDims[d];
    dstBufferSize *= dstLocalDims[d];
  }
  size_t allocSize = std::max(srcBufferSize, dstBufferSize);
  const size_t NCCL_MIN_ALLOC = 4096;
  if (allocSize < NCCL_MIN_ALLOC) allocSize = NCCL_MIN_ALLOC;

  void* buffer;
  NCCLCHECK(ncclMemAlloc(&buffer, allocSize));
  CUDACHECK(cudaMemset(buffer, 0xDE, allocSize)); // Initialize with pattern

  // Register window for user-window API
  ncclWindow_t window = nullptr;
  NCCLCHECK(ncclCommWindowRegister(worldComm, buffer, allocSize, &window, NCCL_WIN_COLL_SYMMETRIC));

  // Setup mesh structures
  ncclMesh_t srcMesh = {.dims = {srcMeshDims[0], srcMeshDims[1]},
                                   .startRank = 0,
                                   .placement = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_SHARD(srcShardDim)}};

  ncclMesh_t dstMesh = {.dims = {dstMeshDims[0], dstMeshDims[1]},
                                   .startRank = srcTotal,
                                   .placement = {NCCL_RESHARD_REPLICATE, NCCL_RESHARD_SHARD(dstShardDim)}};

  // Create CUDA stream
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Stream we actually pass to ncclReshardWithWindow.  When
  // --use-default-stream is set we pass nullptr, exercising the
  // library's internal stream pool; otherwise we hand the explicit
  // stream through.  Init / validation kernels still use the
  // explicit stream regardless.
  cudaStream_t reshardStream = useDefaultStream ? (cudaStream_t)0 : stream;

  // Initialize source data for validation
  if (isSource && validate) {
    // Compute shard index: position within shard dimension of mesh
    int localRank = mpiRank - srcMesh.startRank;
    int shardIdx = localRank % srcMeshDims[1]; // Shard dim is mesh dim 1

    benchInitSourceData((char*)buffer, srcLocalDims, ndims, srcShardDim, shardIdx, srcShardCount, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  MPICHECK(MPI_Barrier(benchMpiWorld()));

  MPICHECK(MPI_Barrier(benchMpiWorld()));

  // Build src/dst descriptors once. dataPtr=NULL is the role signal;
  // localShape entries are zeroed on the side this rank doesn't own.
  ncclDistTensor_t srcTensor = {};
  srcTensor.dataPtr = isSource ? buffer : nullptr;
  srcTensor.ndims = ndims;
  srcTensor.dtype = ncclInt8; // bench validates byte patterns
  srcTensor.mesh = &srcMesh;
  if (isSource)
    for (int d = 0; d < ndims; d++) srcTensor.localShape[d] = srcLocalDims[d];

  ncclDistTensor_t dstTensor = {};
  dstTensor.dataPtr = isDest ? buffer : nullptr;
  dstTensor.ndims = ndims;
  dstTensor.dtype = ncclInt8;
  dstTensor.mesh = &dstMesh;
  if (isDest)
    for (int d = 0; d < ndims; d++) dstTensor.localShape[d] = dstLocalDims[d];

  // Lambda for running one iteration. NCCLCHECK aborts on any non-success
  // return so a contract violation (null window, mismatched offsets, etc.)
  // fails the bench instead of being silently dropped.
  auto runOneIteration = [&]() {
    NCCLCHECK(ncclReshardWithWindow(worldComm, window, &srcTensor, &dstTensor, reshardStream));
  };

  // Warmup
  if (mpiRank == 0) printf("\nRunning %d warmup iterations...\n", warmup);

  for (int i = 0; i < warmup; i++) {
    runOneIteration();
    CUDACHECK(cudaStreamSynchronize(reshardStream));
    MPICHECK(MPI_Barrier(benchMpiWorld()));
  }

  if (mpiRank == 0) printf("Warmup complete.\n");

  // Validation (after warmup). Result is propagated to the process exit
  // code so a corrupted reshard fails the bench instead of silently
  // printing FAILED while returning success.
  int validationRc = 0;
  if (validate) {
    bool localValid = true;

    if (isDest) {
      int localRank = mpiRank - dstMesh.startRank;
      int shardIdx = localRank % dstMeshDims[1];

      localValid = benchValidateDestData((const char*)buffer, dstLocalDims, ndims, dstShardDim, shardIdx, dstShardCount,
                                         mpiRank, stream);
      if (localValid) printf("[Rank %d] VALIDATION PASSED: %zu bytes correct\n", mpiRank, dstBufferSize);
    }

    int localResult = localValid ? 1 : 0;
    int globalResult = 0;
    MPICHECK(MPI_Allreduce(&localResult, &globalResult, 1, benchMpiInt(), benchMpiMin(), benchMpiWorld()));

    if (globalResult == 0) {
      if (mpiRank == 0) printf("\n*** VALIDATION FAILED ***\n\n");
      validationRc = 1;
    } else {
      if (mpiRank == 0) printf("\n*** VALIDATION PASSED ***\n\n");
    }

    // Reset dest buffer for timing runs
    if (isDest) CUDACHECK(cudaMemset(buffer, 0xDE, dstBufferSize));
    MPICHECK(MPI_Barrier(benchMpiWorld()));
  }

  // Timed iterations
  if (mpiRank == 0) printf("\nRunning %d timed iterations...\n", iterations);

  MPICHECK(MPI_Barrier(benchMpiWorld()));
  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iterations; iter++) {
    runOneIteration();
    CUDACHECK(cudaStreamSynchronize(reshardStream));
    MPICHECK(MPI_Barrier(benchMpiWorld()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
  double avgTimeMs = elapsedMs / iterations;

  // Compute bandwidth statistics
  size_t totalData = 1;
  for (int d = 0; d < ndims; d++) totalData *= globalTensorDims[d];
  double bandwidthGbps = ((double)totalData / (avgTimeMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

  size_t myData = isSource ? srcBufferSize : dstBufferSize;
  double myBwGbps = ((double)myData / (avgTimeMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

  // Gather statistics
  double bwMin, bwMax, bwSum;
  MPICHECK(MPI_Reduce(&myBwGbps, &bwMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&myBwGbps, &bwMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&myBwGbps, &bwSum, 1, benchMpiDouble(), benchMpiSum(), 0, benchMpiWorld()));

  double timeMin, timeMax;
  MPICHECK(MPI_Reduce(&avgTimeMs, &timeMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&avgTimeMs, &timeMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));

  // Source-only stats
  double trainerBwForMin = isSource ? myBwGbps : 1e20;
  double trainerBwForMax = isSource ? myBwGbps : -1e20;
  double trainerBwForSum = isSource ? myBwGbps : 0.0;
  double trainerTimeForMin = isSource ? avgTimeMs : 1e20;
  double trainerTimeForMax = isSource ? avgTimeMs : -1e20;

  double trainerBwMin, trainerBwMax, trainerBwSum;
  double trainerTimeMin, trainerTimeMax;
  MPICHECK(MPI_Reduce(&trainerBwForMin, &trainerBwMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerBwForMax, &trainerBwMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerBwForSum, &trainerBwSum, 1, benchMpiDouble(), benchMpiSum(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerTimeForMin, &trainerTimeMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&trainerTimeForMax, &trainerTimeMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));

  // Dest-only stats
  double genBwForMin = isDest ? myBwGbps : 1e20;
  double genBwForMax = isDest ? myBwGbps : -1e20;
  double genBwForSum = isDest ? myBwGbps : 0.0;
  double genTimeForMin = isDest ? avgTimeMs : 1e20;
  double genTimeForMax = isDest ? avgTimeMs : -1e20;

  double genBwMin, genBwMax, genBwSum;
  double genTimeMin, genTimeMax;
  MPICHECK(MPI_Reduce(&genBwForMin, &genBwMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genBwForMax, &genBwMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genBwForSum, &genBwSum, 1, benchMpiDouble(), benchMpiSum(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genTimeForMin, &genTimeMin, 1, benchMpiDouble(), benchMpiMin(), 0, benchMpiWorld()));
  MPICHECK(MPI_Reduce(&genTimeForMax, &genTimeMax, 1, benchMpiDouble(), benchMpiMax(), 0, benchMpiWorld()));

  // Print per-rank stats if requested
  if (printAllRanks) {
    for (int r = 0; r < mpiSize; r++) {
      if (mpiRank == r) {
        printf("[Rank %3d] %s: time=%.3f ms, bw=%.2f GB/s\n", mpiRank, isSource ? "Source" : "Dest  ", avgTimeMs,
               myBwGbps);
        fflush(stdout);
      }
      MPICHECK(MPI_Barrier(benchMpiWorld()));
    }
  }

  // Print summary
  if (mpiRank == 0) {
    printf("\n=================================\n");
    printf("       BENCHMARK RESULTS\n");
    printf("=================================\n");
    printf("Iterations: %d (warmup: %d)\n", iterations, warmup);
    printf("Total data: %zu bytes (%.2f MB)\n", totalData, (double)totalData / (1024.0 * 1024.0));
    printf("Sources: %d ranks, Destinations: %d ranks\n", srcTotal, dstTotal);
    printf("Sharding: src_dim=%d, dst_dim=%d (%s)\n", srcShardDim, dstShardDim,
           srcShardDim == dstShardDim ? "same-dim" : "cross-dim");

    printf("\n--- Overall (all ranks) ---\n");
    printf("Time per iteration (ms):  Min=%.3f  Max=%.3f\n", timeMin, timeMax);
    printf("Bandwidth (GB/s):         Min=%.2f  Max=%.2f  Avg=%.2f\n", bwMin, bwMax, bwSum / mpiSize);

    printf("\n--- Sources only (%d ranks) ---\n", srcTotal);
    printf("Time per iteration (ms):  Min=%.3f  Max=%.3f\n", trainerTimeMin, trainerTimeMax);
    printf("Bandwidth (GB/s):         Min=%.2f  Max=%.2f  Avg=%.2f\n", trainerBwMin, trainerBwMax,
           trainerBwSum / srcTotal);

    printf("\n--- Destinations only (%d ranks) ---\n", dstTotal);
    printf("Time per iteration (ms):  Min=%.3f  Max=%.3f\n", genTimeMin, genTimeMax);
    printf("Bandwidth (GB/s):         Min=%.2f  Max=%.2f  Avg=%.2f\n", genBwMin, genBwMax, genBwSum / dstTotal);

    printf("\n--- Effective bandwidth ---\n");
    printf("Total data throughput: %.2f GB/s\n", bandwidthGbps);
    printf("=================================\n");
    fflush(stdout);
  }

  // Cleanup order matters: deregister window, finalize library, then free
  // the buffer.
  ncclCommWindowDeregister(worldComm, window);
  ncclM2nFinalize();
  NCCLCHECK(ncclMemFree(buffer));
  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(worldComm);

  MPICHECK(MPI_Finalize());

  if (mpiRank == 0) {
    if (validationRc == 0) printf("\nBenchmark completed successfully!\n");
    else printf("\nBenchmark completed with VALIDATION FAILURES.\n");
  }

  return validationRc;
}
