/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

// Host driver for devapi_data_ring2_ir.cu.

#include <cuda.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "common.h"
#include "cumod_common.h"

constexpr int BlockPerRank = 16;
constexpr int BufElts = 1<<20;

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
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = 2*BlockPerRank;
  reqs.ginCounterCount = BlockPerRank;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginStrongSignalsRequired = true;

  ncclDevResourceHandle hBuf;
  ncclDevResourceRequirements bufReq = {};
  bufReq.bufferSize = 2*BlockPerRank*BufElts*sizeof(int);
  bufReq.outBufferHandle = &hBuf;
  bufReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &bufReq;

  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));

  // run kernel
  printf("[MPI Rank %d] Starting kernels\n", rank);

  CUmodule mymodule = NULL;
#ifdef USE_LTO_IR
  initCumoduleFromLtoir(&mymodule, "devapi_data_ring2.ltoir");
#else
  initCumodule(&mymodule, "devapi_data_ring2_ir.cubin");
#endif
  CUfunction kernel;
  initTestCaseKernel(mymodule, &kernel, "runDevice");

  for (int kernelNum=0; kernelNum < 100; kernelNum++) {
    void* args[] = {&dcomm, &hBuf, &kernelNum};
    CU_CHECK(cuLaunchKernel(kernel, BlockPerRank, 1, 1, 512, 1, 1, 0, stream, args, NULL));
  }
  CU_CHECK(cuStreamSynchronize(stream));

  printf("[MPI Rank %d] Completed kernels\n", rank);

  // cleanup
  finiCumodule(&mymodule);
  CU_CHECK(cuStreamDestroy(stream));

  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());
  return 0;
}
