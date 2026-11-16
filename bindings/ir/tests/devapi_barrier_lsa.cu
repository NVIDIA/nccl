/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

// Host driver for devapi_barrier_lsa_ir.cu.

#include <cuda.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "cumod_common.h"

int runTest(int rank, int nRanks, int dev, ncclNvlsHostMode_t nvlsHostMode,
            bool lsaMultimem, bool* multimemSupport = nullptr) {
  ncclUniqueId id;
  ncclComm_t comm;
  if (rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(dev));

  CUstream stream;
  CU_CHECK(cuStreamCreate(&stream, 0));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  config.nvlsHostMode = nvlsHostMode;
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  if (multimemSupport != nullptr) {
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(comm, &props));
    *multimemSupport = props.multimemSupport;
  }

  ncclDevComm dcomm;
  { ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = 16;
    reqs.lsaMultimem = lsaMultimem;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));
  }

  // run kernel
  printf("[MPI Rank %d] Starting kernel\n", rank);

  CUmodule mymodule = NULL;
#ifdef USE_LTO_IR
  initCumoduleFromLtoir(&mymodule, "devapi_barrier_lsa.ltoir");
#else
  initCumodule(&mymodule, "devapi_barrier_lsa_ir.cubin");
#endif
  CUfunction kernel;
  initTestCaseKernel(mymodule, &kernel, "runDevice");

  // Pass the explicit handle separately for the epoch/reuse checks in the IR kernel.
  void* args[] = {&dcomm, &dcomm.lsaBarrier};
  CU_CHECK(cuLaunchKernel(kernel, 16, 1, 1, 512, 1, 1, 0, stream, args, NULL));
  CU_CHECK(cuStreamSynchronize(stream));

  printf("[MPI Rank %d] Completed kernel\n", rank);

  // cleanup
  finiCumodule(&mymodule);
  CU_CHECK(cuStreamDestroy(stream));

  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  return 0;
}

int main(int argc, char** argv) {
  int rank, nRanks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r = 0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  // Default config: UC-only barrier.
  bool multimemSupport = false;
  int ret = runTest(rank, nRanks, dev, ncclNvlsHostModeDefault,
                    /*lsaMultimem=*/false, &multimemSupport);

  // All ranks must take the same path through the second communicator creation.
  int allRanksSupportMultimem = multimemSupport ? 1 : 0;
  MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &allRanksSupportMultimem, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD));

  // nvlsHostModeDisable must not affect the device API: the LSA barrier must still
  // work, and lsaMultimem must still produce a valid handle when nvlsSupport=1.
  if (ret == 0 && allRanksSupportMultimem)
    ret = runTest(rank, nRanks, dev, ncclNvlsHostModeDisable, /*lsaMultimem=*/true);

  MPICHECK(MPI_Finalize());
  return ret;
}
