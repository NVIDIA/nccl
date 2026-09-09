/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_DIAGNOSTICS_H_
#define NCCL_RAS_DIAGNOSTICS_H_

#include <stddef.h>
#include <stdint.h>

#include "nccl.h"
#include "ras_internal.h"

struct ncclComm;

typedef enum {
  RAS_DIAG_CHECK_GPU_MODEL = 0,
  RAS_DIAG_CHECK_CUDA_DRIVER_VERSION = 1,
  RAS_DIAG_CHECK_ECC = 2,
  RAS_DIAG_CHECK_NVLINK = 3,
  RAS_DIAG_CHECK_NCCL_ENV = 4,
  // Must remain last. Add new check IDs above this sentinel and add the corresponding dispatch table entry.
  RAS_DIAG_CHECK_COUNT
} rasDiagnosticsCheckId;

// Dispatcher metadata. This describes the scope of the diagnostics request.
struct rasDiagnosticsContext {
  bool hasCommFilter;
  struct rasCommId commFilter;
  int commNRanks; // Rank count of the filtered communicator; 0 when unscoped.
};

struct rasDiagnosticsLocalData {
  char* records;
  int recordsBytes;
  int recordStride;
  int nRecords;
};

struct rasDiagnosticsReporter {
  ncclResult_t (*emit)(void* target, const char* line);
  ncclResult_t (*finish)(void* target, ncclResult_t result);
  void* target;
};

typedef ncclResult_t (*rasDiagnosticsCollectLocalFn)(const struct rasDiagnosticsContext* ctx,
                                                     struct rasDiagnosticsLocalData* data);
typedef ncclResult_t (*rasDiagnosticsSummarizeFn)(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData);

struct rasDiagnosticsCheck {
  rasDiagnosticsCheckId id;
  rasDiagnosticsCollectLocalFn collectLocal;
  rasDiagnosticsSummarizeFn summarize;
};

// First field of every per-check record. A check payload is a flat array of fixed-size records.
struct rasDiagnosticsRankHeader {
  struct rasCommId commId;
  int commRank;
  int commNRanks;
};

// Local-only view of a communicator. The collector builds this while holding ncclCommsMutex,
// then check-specific code consumes it after the lock is released.
struct rasDiagnosticsCommSnapshot {
  struct rasDiagnosticsRankHeader rank;
  int cudaDev;
  int nvmlDev;
  int64_t busId;
  int localRank;
  int localRanks;
};

// One RAS diagnostics gather contribution from one peer:
//   [rasDiagnosticsPeerPayloadHeader]
//   [rasDiagnosticsCheckPayloadHeader][records for that check]
//   ...
struct rasDiagnosticsPeerPayloadHeader {
  int nChecks;
  int payloadBytes; // Includes this header.
};

struct rasDiagnosticsCheckPayloadHeader {
  rasDiagnosticsCheckId checkId;
  int recordStride;
  int nRecords;
  int payloadBytes; // Only the records following this header.
};

ncclResult_t rasDiagnosticsContextInit(struct rasDiagnosticsContext* ctx, const struct ncclComm* comm);
ncclResult_t rasDiagnosticsFormatLine(char* out, size_t outSize, const char* line);
ncclResult_t rasLocalHandleRunDiag(const struct rasDiagnosticsContext* ctx);
ncclResult_t rasDiagnosticsClientInit(struct rasClient* client, const struct rasDiagnosticsContext* ctx,
                                      const struct rasDiagnosticsReporter* reporter);
ncclResult_t rasDiagnosticsStart(struct rasClient* client);
bool rasDiagnosticsInProgress();
void rasDiagnosticsCancelTarget(void* target);
void rasDiagnosticsClientCleanup(struct rasClient* client);

// RAS_COLL_DIAG collective hooks (variable-size gather to the requester). Implemented in diagnostics.cc and dispatched
// from collectives.cc, following the RAS_COLL_COMMS *Init/*Merge pattern.
ncclResult_t rasCollDiagInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData);
ncclResult_t rasCollDiagMerge(struct rasCollective* coll, struct rasMsg* msg);
// Locally-initiated completion: summarizes the gathered peer payloads owned by the client.
ncclResult_t rasDiagnosticsResume(struct rasClient* client);

#endif // NCCL_RAS_DIAGNOSTICS_H_
