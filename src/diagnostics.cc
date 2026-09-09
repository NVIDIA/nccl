/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "diagnostics.h"

#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"

#include <stdlib.h>
#include "diagnostics_log.h"
#include "diagnostics/p2p.h"
#include "graph.h"
#include "graph/topo.h"
#include "param.h"

#include <algorithm>
#include <chrono>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>

// NCCL_RUN_DIAGNOSTICS=1 runs NCCL comm init diagnostics;
NCCL_PARAM(Diagnostics, "RUN_DIAGNOSTICS", 0);

#define CHILD_KILL_GRACE_SEC 2 // `timeout -k`: SIGKILL this long after the initial SIGTERM

int ncclDiagChildRun(const char* command, int timeoutSec, char* output, int outputSize) {
  if (output != nullptr && outputSize > 0) output[0] = '\0';
  if (command == nullptr || timeoutSec < 1) return -1;

  char wrapped[2048];
  int n = snprintf(wrapped, sizeof(wrapped), "timeout -k %d %d stdbuf -oL %s </dev/null 2>&1", CHILD_KILL_GRACE_SEC,
                   timeoutSec, command);
  if (n < 0 || n >= (int)sizeof(wrapped)) return -1;

  FILE* stream = popen(wrapped, "r");
  if (stream == nullptr) return -1;

  int used = 0;
  char buffer[4096];
  size_t got;
  while ((got = fread(buffer, 1, sizeof(buffer), stream)) > 0) {
    if (output != nullptr && used + 1 < outputSize) {
      int copy = std::min(static_cast<int>(got), outputSize - used - 1);
      memcpy(output + used, buffer, copy);
      used += copy;
      output[used] = '\0';
    }
  }
  int status = pclose(stream);
  return (status >= 0 && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
}

static ncclResult_t ncclDiagDetectTransportMask(struct ncclComm* comm, unsigned int* outUnionMask) {
  ncclResult_t ret = ncclSuccess;
  unsigned int* allPathMasks = nullptr;
  unsigned int localPathMask = 0;
  if (outUnionMask != nullptr) *outUnionMask = 0;

  int myGpuIdx = -1;
  if (comm->topo != nullptr) {
    ncclTopoRankToIndex(comm->topo, comm->rank, &myGpuIdx, false);
  }

  for (int peer = 0; peer < comm->nRanks; peer++) {
    int t = PATH_DIS;
    if (peer == comm->rank) {
      t = PATH_LOC;
    } else {
      int peerGpuIdx = -1;
      if (comm->topo != nullptr) {
        ncclTopoRankToIndex(comm->topo, peer, &peerGpuIdx, false);
      }
      if (myGpuIdx < 0 || peerGpuIdx < 0) {
        t = PATH_NET;
      } else {
        t = comm->topo->nodes[GPU].nodes[myGpuIdx].paths[GPU][peerGpuIdx].type;
      }
    }
    if (t < 0 || t > PATH_DIS) t = PATH_DIS;
    localPathMask |= 1u << t;
  }

  NCCLCHECK(ncclCalloc(&allPathMasks, comm->nRanks));
  allPathMasks[comm->rank] = localPathMask;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allPathMasks, sizeof(unsigned int)), ret, fail);

  if (outUnionMask != nullptr) {
    unsigned int unionMask = 0;
    for (int r = 0; r < comm->nRanks; r++) unionMask |= allPathMasks[r];
    *outUnionMask = unionMask;
  }

fail:
  free(allPathMasks);
  return ret;
}

ncclResult_t ncclRunDiagnosticsActive(struct ncclComm* comm) {
  auto t0 = std::chrono::steady_clock::now();

  if (comm->rank == 0) DIAG_PRINT("NCCL DIAG === NCCL Diagnostics ===");

  ncclResult_t r;
  unsigned int transportMask = 0;
  r = ncclDiagDetectTransportMask(comm, &transportMask);
  if (comm->rank == 0 && r != ncclSuccess) DIAG_PRINT("NCCL DIAG [INFO] transport detect returned %d", r);

  r = ncclDiagP2pRun(comm);
  if (comm->rank == 0 && r != ncclSuccess) DIAG_PRINT("NCCL DIAG [INFO] p2p: active check returned %d", r);

  auto t1 = std::chrono::steady_clock::now();
  double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (comm->rank == 0) {
    DIAG_PRINT("NCCL DIAG NCCL diagnostics completed in %.1f ms across %d ranks", elapsedMs, comm->nRanks);
  }

  return ncclSuccess;
}