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
#include "diagnostics/ib_write_bw.h"
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

// Runs `command` under `timeout` + `stdbuf -oL` and streams its combined output: the child stays
// line-buffered, so `onLine` fires for each line as the child emits it (or for partial lines
// longer than the read buffer), letting callers react to child milestones (e.g. a benchmark
// server announcing its listen socket) while the child is still running.
int ncclDiagChildRunStream(const char* command, int timeoutSec, char* output, int outputSize,
                           ncclDiagChildLineFn onLine, void* onLineCtx, bool* outputTruncated) {
  if (output != nullptr && outputSize > 0) output[0] = '\0';
  if (outputTruncated != nullptr) *outputTruncated = false;
  if (command == nullptr || timeoutSec < 1) return -1;

  char wrapped[2048];
  int n = snprintf(wrapped, sizeof(wrapped), "timeout -k %d %d stdbuf -oL %s </dev/null 2>&1", CHILD_KILL_GRACE_SEC,
                   timeoutSec, command);
  if (n < 0 || n >= (int)sizeof(wrapped)) return -1;

  FILE* stream = popen(wrapped, "r");
  if (stream == nullptr) return -1;

  int used = 0;
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
    if (onLine != nullptr) onLine(buffer, onLineCtx);
    int got = static_cast<int>(strlen(buffer));
    int copy = 0;
    if (output != nullptr && used + 1 < outputSize) {
      copy = std::min(got, outputSize - used - 1);
      memcpy(output + used, buffer, copy);
      used += copy;
      output[used] = '\0';
    }
    if (outputTruncated != nullptr && copy < got) *outputTruncated = true;
  }
  int status = pclose(stream);
  return (status >= 0 && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
}

int ncclDiagChildRun(const char* command, int timeoutSec, char* output, int outputSize, bool* outputTruncated) {
  return ncclDiagChildRunStream(command, timeoutSec, output, outputSize, nullptr, nullptr, outputTruncated);
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
  // We can only classify a peer's transport if we have a trustworthy local topology anchored on our
  // own GPU. Without it (topo absent, or our own GPU missing) "peer not found" tells us nothing --
  // it must NOT be read as "reached over the network", or it would spuriously assert an IB path.
  bool haveTopo = comm->topo != nullptr && myGpuIdx >= 0;

  for (int peer = 0; peer < comm->nRanks; peer++) {
    int t;
    if (peer == comm->rank) {
      t = PATH_LOC;
    } else if (!haveTopo) {
      t = PATH_DIS; // undetermined: no local topology to classify against
    } else {
      int peerGpuIdx = -1;
      ncclTopoRankToIndex(comm->topo, peer, &peerGpuIdx, false);
      // A peer absent from our valid local topology is off-node, i.e. reached over the network.
      t = peerGpuIdx < 0 ? PATH_NET : comm->topo->nodes[GPU].nodes[myGpuIdx].paths[GPU][peerGpuIdx].type;
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

ncclResult_t ncclRunDiagnostics(struct ncclComm* comm) {
  auto t0 = std::chrono::steady_clock::now();

  if (comm->rank == 0) DIAG_PRINT("NCCL DIAG === NCCL Diagnostics ===");

  ncclResult_t r;
  unsigned int transportMask = 0;
  r = ncclDiagDetectTransportMask(comm, &transportMask);
  if (comm->rank == 0 && r != ncclSuccess) DIAG_PRINT("NCCL DIAG [INFO] transport detect returned %d", r);

  // Only exercise ib_write_bw when the communicator uses the network transport across at least two
  // hosts (nNodes counts hosts not connected via MNNVL). The union mask is allgathered and nNodes is
  // identical everywhere, so every rank makes the same decision and stays in lockstep for the
  // check's collectives.
  if ((transportMask & (1u << PATH_NET)) && comm->nNodes >= 2) {
    ncclDiagRunIbWriteBw(comm);
  } else if (comm->rank == 0) {
    DIAG_PRINT("NCCL DIAG [OK] net bw: skipped (single host or no network transport)");
  }

  r = ncclDiagP2pRun(comm);
  if (comm->rank == 0 && r != ncclSuccess) DIAG_PRINT("NCCL DIAG [INFO] p2p: check returned %d", r);

  auto t1 = std::chrono::steady_clock::now();
  double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (comm->rank == 0) {
    DIAG_PRINT("NCCL DIAG NCCL diagnostics completed in %.1f ms across %d ranks", elapsedMs, comm->nRanks);
  }

  return ncclSuccess;
}
