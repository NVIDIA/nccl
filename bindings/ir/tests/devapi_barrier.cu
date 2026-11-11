/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

// Host driver for devapi_barrier_ir.cu.

#include <cuda.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "cumod_common.h"

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
  for (int r=0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  ncclUniqueId id;
  ncclComm_t comm;
  if (rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(dev));

  CUstream stream;
  CU_CHECK(cuStreamCreate(&stream, 0));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  ncclDevComm dcomm;
  { ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.barrierCount = 16;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));
  }
  // run kernel
  printf("[MPI Rank %d] Starting kernel\n", rank);

  CUmodule mymodule = NULL;
#ifdef USE_LTO_IR
  initCumoduleFromLtoir(&mymodule, "devapi_barrier.ltoir");
#else
  initCumodule(&mymodule, "devapi_barrier_ir.cubin");
#endif
  CUfunction kernel;
  initTestCaseKernel(mymodule, &kernel, "runDevice");

  // Pass explicit handles separately for the epoch/reuse checks in the IR kernel.
  void* args[] = {&dcomm, &dcomm.hybridLsaBarrier, &dcomm.hybridRailGinBarrier};
  CU_CHECK(cuLaunchKernel(kernel, 16, 1, 1, 512, 1, 1, 0, stream, args, NULL));
  CU_CHECK(cuStreamSynchronize(stream));

  printf("[MPI Rank %d] Completed kernel\n", rank);

  // cleanup
  finiCumodule(&mymodule);
  CU_CHECK(cuStreamDestroy(stream));

  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());
  return 0;
}
