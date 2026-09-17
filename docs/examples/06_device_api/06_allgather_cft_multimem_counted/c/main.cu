/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"
#include "utils.h"
#include <cuda/barrier>
#include <stdio.h>
#include <stdlib.h>

#define NCCL_DEVICE_CTA_COUNT 8
#define NCCL_DEVICE_THREADS_PER_CTA 256
#define ELEMENTS_PER_RANK 1024
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#define CFT_SMEM_OFFSET 0
#define TMA_BARRIER_OFFSET ALIGN_UP(CFT_SMEM_OFFSET + sizeof(ncclCftSmem), alignof(cuda::barrier<cuda::thread_scope_block>))
#define TMA_BUFFER_OFFSET ALIGN_UP(TMA_BARRIER_OFFSET + sizeof(cuda::barrier<cuda::thread_scope_block>), 16)
#define DYNAMIC_SMEM_BYTES (TMA_BUFFER_OFFSET + NCCL_DEVICE_THREADS_PER_CTA * sizeof(float))

__global__ void allGatherCftMultimemCountedKernel(ncclWindow_t sendWin, ncclWindow_t recvWin, size_t count,
                                                  ncclDevComm devComm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(CUDART_VERSION) && CUDART_VERSION >= 13030
  ncclCoopCta coop;
  ncclTeam team = ncclTeamCft(devComm);

  int rank = team.rank;
  int nRanks = team.nRanks;
  size_t bytes = count * sizeof(float);

  const int globalNthreads = blockDim.x * gridDim.x;

  float* send = (float*)ncclGetLocalPointer(sendWin, 0);
  float* recv = (float*)ncclGetLocalPointer(recvWin, 0);
  uintptr_t base = reinterpret_cast<uintptr_t>(recv);
  uintptr_t end = base + sizeof(float) * count * nRanks;
  uintptr_t counterAddr = ALIGN_UP(end, uintptr_t(256u));
  uint64_t* counterPtr = reinterpret_cast<uint64_t*>(counterAddr);

  ncclCftBarrierSession<ncclCoopCta> bar { coop, devComm, blockIdx.x };
  bar.sync(coop, cuda::memory_order_relaxed, ncclMemProxyType::Generic, ncclMemProxyType::Generic);

  extern __shared__ __align__(16) unsigned char smemBytes[];
  ncclCftSmem* cftSmem = reinterpret_cast<ncclCftSmem*>(smemBytes + CFT_SMEM_OFFSET);
  cuda::barrier<cuda::thread_scope_block>* mbar =
    reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(smemBytes + TMA_BARRIER_OFFSET);
  float* smem = reinterpret_cast<float*>(smemBytes + TMA_BUFFER_OFFSET);
  ncclCft<ncclCoopCta> cft { coop, *cftSmem };

  if (coop.thread_rank() == 0) init(mbar, 1);

  for (size_t base = blockIdx.x * blockDim.x; base < count; base += globalNthreads) {
    uint32_t chunkElems = (uint32_t)((count - base) < (size_t)blockDim.x ? (count - base) : (size_t)blockDim.x);
    uint32_t chunkBytes = chunkElems * sizeof(float);
    if (coop.thread_rank() == 0) {
      cuda::device::memcpy_async_tx(smem, send + base, cuda::aligned_size_t<16>(chunkBytes), *mbar);
      cuda::barrier<cuda::thread_scope_block>::arrival_token token =
        cuda::device::barrier_arrive_tx(*mbar, 1, chunkBytes);
      mbar->wait(std::move(token));
    }
    coop.sync();
    ncclCftLeId leId;
    size_t leOffset;
    ncclGetMultimemLeInfo(recvWin, rank * count * sizeof(float), devComm, &leId, &leOffset);
    size_t dataOffset = leOffset + base * sizeof(float);
    size_t counterOffset = ALIGN_UP(leOffset + (nRanks - rank) * count * sizeof(float), 256u);
    cft.putMultimemCounted(coop, leId, dataOffset, counterOffset, smem, chunkBytes);
    cft.submit(coop);
    cft.flush(coop);
  }
  cft.waitCounted(coop, cuda::memory_order_acquire, ncclMemProxyType::Generic, counterPtr, bytes * nRanks,
                  devComm.abortFlag);
#else
  (void)sendWin;
  (void)recvWin;
  (void)count;
  (void)devComm;
#endif
}

void* allGatherCftMultimemCounted(int myRank, int totalRanks, int localDevice, int devicesPerRank) {
  (void)devicesPerRank;
  ncclComm_t comm;
  ncclUniqueId id;

  if (myRank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  util_broadcast(0, myRank, &id);

  CUDACHECK(cudaSetDevice(localDevice));
#if !defined(CUDART_VERSION) || CUDART_VERSION < 13030
  printf("rank %d allgather_cft_multimem_counted: SKIPPED, requires CUDA Toolkit 13.3 or newer\n", myRank);
  return NULL;
#endif
  cudaDeviceProp prop;
  CUDACHECK(cudaGetDeviceProperties(&prop, localDevice));
  if (prop.major * 10 + prop.minor < 100) {
    printf("rank %d allgather_cft_multimem_counted: SKIPPED, requires compute capability 10.0 or newer\n", myRank);
    return NULL;
  }
  NCCLCHECK(ncclCommInitRank(&comm, totalRanks, id, myRank));

  size_t count = ELEMENTS_PER_RANK;
  size_t counterBytes = /*counter*/8u + /*alignment*/255u;
  size_t sendBytes = count * sizeof(float);
  size_t recvBytes = count * totalRanks * sizeof(float);

  float* hSend = (float*)malloc(sendBytes);
  float* hRecv = (float*)malloc(recvBytes);
  for (size_t i = 0; i < count; i++) hSend[i] = (float)(myRank * 1000 + i);

  void* dSend;
  void* dRecv;
  ncclWindow_t sendWin;
  ncclWindow_t recvWin;
  NCCLCHECK(ncclMemAlloc(&dSend, sendBytes));
  NCCLCHECK(ncclMemAlloc(&dRecv, recvBytes + counterBytes));
  CUDACHECK(cudaMemcpy(dSend, hSend, sendBytes, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(dRecv, 0, recvBytes));
  CUDACHECK(cudaMemset(reinterpret_cast<char*>(dRecv) + recvBytes, 0, counterBytes));

  NCCLCHECK(ncclCommWindowRegister(comm, dSend, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, dRecv, recvBytes + counterBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC|NCCL_WIN_CFT_COUNTED));

  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.cftCaps = NCCL_CFT | NCCL_CFT_MULTIMEM;
  reqs.cftBarrierCount = NCCL_DEVICE_CTA_COUNT;
  reqs.lsaMultimem = true;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  allGatherCftMultimemCountedKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA,
                                      DYNAMIC_SMEM_BYTES, stream>>>(sendWin, recvWin, count, devComm);
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaMemcpy(hRecv, dRecv, recvBytes, cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int r = 0; r < totalRanks; r++) {
    for (size_t i = 0; i < count; i++) {
      float expected = (float)(r * 1000 + i);
      float got = hRecv[r * count + i];
      if (got != expected) {
        printf("rank %d mismatch at source rank %d element %zu: got %f expected %f\n", myRank, r, i, got, expected);
        ok = false;
        goto done_check;
      }
    }
  }
done_check:
  printf("rank %d allgather_cft_multimem_counted: %s\n", myRank, ok ? "PASSED" : "FAILED");

  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommWindowDeregister(comm, recvWin));
  NCCLCHECK(ncclCommWindowDeregister(comm, sendWin));
  NCCLCHECK(ncclMemFree(dRecv));
  NCCLCHECK(ncclMemFree(dSend));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));
  free(hRecv);
  free(hSend);
  return NULL;
}

int main(int argc, char* argv[]) {
  return run_example(argc, argv, allGatherCftMultimemCounted);
}
