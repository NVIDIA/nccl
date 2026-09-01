/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cuda_runtime.h"
#include "kernels.cuh"
#include "nccl.h"
#include "nccl_device.h"
#include "utils.h"
#include <atomic>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define WARMUP_ITERS 10
#define TIMED_ITERS 100
#define BUFFER_SIZE_BYTES ((size_t)CTA_COUNT * PUTS_PER_CTA * ELEMS_PER_PUT * sizeof(int))
#define ELEMENT_VALUE_PERIOD 4096
#define MAX_VALIDATION_RANKS \
  (INT_MAX / (CTA_COUNT * PUTS_PER_CTA * ELEMENT_VALUE_PERIOD) + 1)

static std::atomic<bool> validationSuccess{true};

static void fillSendBuffer(int* sendBuff, int myRank) {
  for (int ctaIndex = 0; ctaIndex < CTA_COUNT; ++ctaIndex) {
    for (int putIndex = 0; putIndex < PUTS_PER_CTA; ++putIndex) {
      for (size_t elem = 0; elem < ELEMS_PER_PUT; ++elem) {
        const size_t idx = ((size_t)ctaIndex * PUTS_PER_CTA + putIndex) * ELEMS_PER_PUT + elem;
        sendBuff[idx] = (((myRank * CTA_COUNT + ctaIndex) * PUTS_PER_CTA + putIndex) *
                         ELEMENT_VALUE_PERIOD) + (int)(elem % ELEMENT_VALUE_PERIOD);
      }
    }
  }
}

static bool verifyRecvBuffer(int myRank, int totalRanks, const int* recvBuff, const char* label) {
  const int srcRank = (myRank - 1 + totalRanks) % totalRanks;
  for (int ctaIndex = 0; ctaIndex < CTA_COUNT; ++ctaIndex) {
    for (int putIndex = 0; putIndex < PUTS_PER_CTA; ++putIndex) {
      for (size_t elem = 0; elem < ELEMS_PER_PUT; ++elem) {
        const size_t idx = ((size_t)ctaIndex * PUTS_PER_CTA + putIndex) * ELEMS_PER_PUT + elem;
        const int expected = (((srcRank * CTA_COUNT + ctaIndex) * PUTS_PER_CTA + putIndex) *
                              ELEMENT_VALUE_PERIOD) + (int)(elem % ELEMENT_VALUE_PERIOD);
        if (recvBuff[idx] != expected) {
          printf("  Rank %d %s mismatch at CTA %d put %d elem %zu: got %d, expected %d\n",
                 myRank, label, ctaIndex, putIndex, elem, recvBuff[idx], expected);
          return false;
        }
      }
    }
  }
  return true;
}

static float runTimedKernel(int myRank, int caseIndex, ncclComm_t comm,
                            ncclWindow_t sendWin, ncclWindow_t recvWin, void* dRecvBuff,
                            int* hRecvBuff, ncclDevComm devComm, cudaStream_t stream, bool* success) {
  CUDACHECK(cudaMemset(dRecvBuff, 0, BUFFER_SIZE_BYTES));
  switch (caseIndex) {
    case 0:
      ctaCooperativePutKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, WARMUP_ITERS, devComm);
      break;
    case 1:
      ctaCooperativePutAggregateRequestsKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, WARMUP_ITERS, devComm);
      break;
    case 2:
      threadPerPutKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, WARMUP_ITERS, devComm);
      break;
    case 3:
      threadPerPutAggregateRequestsKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, WARMUP_ITERS, devComm);
      break;
  }
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));

  CUDACHECK(cudaMemset(dRecvBuff, 0, BUFFER_SIZE_BYTES));

  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  CUDACHECK(cudaEventRecord(start, stream));
  switch (caseIndex) {
    case 0:
      ctaCooperativePutKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, TIMED_ITERS, devComm);
      break;
    case 1:
      ctaCooperativePutAggregateRequestsKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, TIMED_ITERS, devComm);
      break;
    case 2:
      threadPerPutKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, TIMED_ITERS, devComm);
      break;
    case 3:
      threadPerPutAggregateRequestsKernel<<<CTA_COUNT, PUTS_PER_CTA, 0, stream>>>(
          sendWin, recvWin, TIMED_ITERS, devComm);
      break;
  }
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  float milliseconds = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  CUDACHECK(cudaMemcpy(hRecvBuff, dRecvBuff, BUFFER_SIZE_BYTES, cudaMemcpyDeviceToHost));
  const char* verifyLabels[4] = {
    "cta_coop weak per-put signals",
    "cta_coop weak AggregateRequests",
    "thread_per_put weak per-put signals",
    "thread_per_put weak AggregateRequests"
  };
  *success = verifyRecvBuffer(myRank, devComm.nRanks, hRecvBuff, verifyLabels[caseIndex]);

  int passedRanks = *success ? 1 : 0;
  CUDACHECK(cudaMemcpy(dRecvBuff, &passedRanks, sizeof(passedRanks), cudaMemcpyHostToDevice));
  NCCLCHECK(ncclAllReduce(dRecvBuff, dRecvBuff, 1, ncclInt, ncclSum, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaMemcpy(&passedRanks, dRecvBuff, sizeof(passedRanks), cudaMemcpyDeviceToHost));
  *success = passedRanks == devComm.nRanks;

  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));
  return milliseconds;
}

void* ringExchangeExample(int myRank, int totalRanks, int localDevice, int devicesPerRank) {
  (void)devicesPerRank;

  if (totalRanks < 2) {
    if (myRank == 0) {
      printf("GIN Ring Exchange example requires at least 2 ranks.\n");
    }
    return NULL;
  }
  if (totalRanks > MAX_VALIDATION_RANKS) {
    if (myRank == 0) {
      printf("WARNING: GIN Ring Exchange validation uses int payload tags and supports at most %d ranks; "
             "got %d. Skipping the run to avoid validation tag overflow.\n",
             MAX_VALIDATION_RANKS, totalRanks);
    }
    return NULL;
  }

  ncclComm_t comm;
  ncclUniqueId commId;

  if (myRank == 0) {
    printf("Starting GIN Ring Exchange initialization\n");
  }

  if (myRank == 0) {
    NCCLCHECK(ncclGetUniqueId(&commId));
  }
  util_broadcast(0, myRank, &commId);

  CUDACHECK(cudaSetDevice(localDevice));
  printf("  Rank %d using GPU device %d\n", myRank, localDevice);

  NCCLCHECK(ncclCommInitRank(&comm, totalRanks, commId, myRank));

  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));
  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", myRank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }
  if (props.ginType == NCCL_GIN_TYPE_NONE) {
    printf("ERROR: rank %d communicator does not support GIN!\n", myRank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }

  if (myRank == 0) {
    printf("GIN Ring Exchange: ctas=%d, puts_per_cta=%d, elems_per_put=%d\n",
           CTA_COUNT, PUTS_PER_CTA, ELEMS_PER_PUT);
  }

  int* hSendBuff = (int*)malloc(BUFFER_SIZE_BYTES);
  int* hRecvBuff = (int*)malloc(BUFFER_SIZE_BYTES);
  if (hSendBuff == NULL || hRecvBuff == NULL) {
    printf("ERROR: rank %d failed to allocate host buffers\n", myRank);
    free(hSendBuff);
    free(hRecvBuff);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return NULL;
  }

  void* dSendBuff = NULL;
  void* dRecvBuff = NULL;
  ncclWindow_t sendWin;
  ncclWindow_t recvWin;
  NCCLCHECK(ncclMemAlloc(&dSendBuff, BUFFER_SIZE_BYTES));
  NCCLCHECK(ncclMemAlloc(&dRecvBuff, BUFFER_SIZE_BYTES));

  NCCLCHECK(ncclCommWindowRegister(comm, dSendBuff, BUFFER_SIZE_BYTES, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, dRecvBuff, BUFFER_SIZE_BYTES, &recvWin, NCCL_WIN_COLL_SYMMETRIC));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.worldGinBarrierCount = CTA_COUNT;
  reqs.ginSignalCount = CTA_COUNT;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginStrongSignalsRequired = false;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  if (myRank == 0) {
    printf("\n=== Comparing GIN ring-exchange implementations ===\n");
    printf("Timing columns: us/batch is whole-batch time; us/put is derived from us/batch / (ctas * puts_per_cta).\n");
    printf("Speedups are relative to configuration case 1.\n");
    printf("\nConfiguration cases:\n");
    printf("  %-4s %-14s %-22s\n", "Case", "Producer", "Completion");
    printf("  %-4s %-14s %-22s\n", "----", "--------------", "----------------------");
    printf("  1    %-14s %-22s (baseline)\n", "cta_coop", "weak per-put signals");
    printf("  2    %-14s %-22s\n", "cta_coop", "weak AggregateRequests");
    printf("  3    %-14s %-22s\n", "thread_per_put", "weak per-put signals");
    printf("  4    %-14s %-22s\n", "thread_per_put", "weak AggregateRequests");
  }

  fillSendBuffer(hSendBuff, myRank);
  CUDACHECK(cudaMemcpy(dSendBuff, hSendBuff, BUFFER_SIZE_BYTES, cudaMemcpyHostToDevice));

  float elapsedMs[4];
  bool passed[4];
  bool success = false;

  // Case 1: CTA-cooperative producer without aggregate-request hints.
  elapsedMs[0] = runTimedKernel(myRank, 0, comm, sendWin, recvWin,
                                dRecvBuff, hRecvBuff, devComm, stream, &success);
  passed[0] = success;

  // Case 2: CTA-cooperative producer with aggregate-request hints.
  success = false;
  elapsedMs[1] = runTimedKernel(myRank, 1, comm, sendWin, recvWin,
                                dRecvBuff, hRecvBuff, devComm, stream, &success);
  passed[1] = success;

  // Case 3: One producer thread per put without aggregate-request hints.
  success = false;
  elapsedMs[2] = runTimedKernel(myRank, 2, comm, sendWin, recvWin,
                                dRecvBuff, hRecvBuff, devComm, stream, &success);
  passed[2] = success;

  // Case 4: One producer thread per put with aggregate-request hints.
  success = false;
  elapsedMs[3] = runTimedKernel(myRank, 3, comm, sendWin, recvWin,
                                dRecvBuff, hRecvBuff, devComm, stream, &success);
  passed[3] = success;

  const bool allCasesPassed = passed[0] && passed[1] && passed[2] && passed[3];
  if (!allCasesPassed) {
    validationSuccess.store(false, std::memory_order_relaxed);
  }
  if (myRank == 0) {
    printf("\n  %-4s %12s %12s %13s %8s\n",
           "Case", "us/batch", "us/put", "speedup", "result");
    printf("  %-4s %12s %12s %13s %8s\n",
           "----", "------------", "------------", "-------------", "--------");
    for (int caseIndex = 0; caseIndex < 4; ++caseIndex) {
      const double usPerBatch = elapsedMs[caseIndex] * 1000.0 / TIMED_ITERS;
      const double usPerPut = usPerBatch / (CTA_COUNT * PUTS_PER_CTA);
      const double speedup = (elapsedMs[0] > 0.0 && elapsedMs[caseIndex] > 0.0) ?
                             elapsedMs[0] / elapsedMs[caseIndex] : 0.0;

      printf("  %-4d %12.2f %12.2f %12.2fx %8s\n",
             caseIndex + 1, usPerBatch, usPerPut, speedup,
             passed[caseIndex] ? "PASSED" : "FAILED");
    }
    printf("\nGIN Ring Exchange result: %s\n", allCasesPassed ? "PASSED" : "FAILED");
  }

  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommWindowDeregister(comm, sendWin));
  NCCLCHECK(ncclCommWindowDeregister(comm, recvWin));
  NCCLCHECK(ncclMemFree(dSendBuff));
  NCCLCHECK(ncclMemFree(dRecvBuff));

  CUDACHECK(cudaStreamDestroy(stream));
  free(hSendBuff);
  free(hRecvBuff);

  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  return NULL;
}

int main(int argc, char* argv[]) {
  const int result = run_example(argc, argv, ringExchangeExample);
  if (result != 0) return result;
  return validationSuccess.load(std::memory_order_relaxed) ? 0 : 1;
}
