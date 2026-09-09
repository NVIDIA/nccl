/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "diagnostics_log.h"
#include "diagnostics.h"
#include "diagnostics_checks.h"
#include "transport.h"

ncclResult_t rasDiagnosticsFormatLine(char* out, size_t outSize, const char* line) {
  int len;

  if (out == nullptr || outSize == 0 || line == nullptr) {
    WARN("RAS diagnostics line formatting received invalid arguments");
    return ncclInternalError;
  }
  diagLogInit();
  len = snprintf(out, outSize, "%s:%d NCCL DIAG %s", diagLogHost, diagLogPid, line);
  if (len < 0 || len >= (int)outSize) {
    WARN("RAS diagnostics line formatting failed");
    return ncclInternalError;
  }
  return ncclSuccess;
}

// TODO: Replace this stdout fallback with the Notifier once diagnostics supports it.
static ncclResult_t rasDiagnosticsDefaultEmit(void* target, const char* line) {
  (void)target;
  DIAG_PRINT("NCCL DIAG %s", line);
  if (fflush(stdout) != 0) return ncclSystemError;
  return ncclSuccess;
}

// Used when a client-backed reporter target disappears before the async diagnostics collective completes.
// The diagnostics client state stays alive until completion, but it must not write to the closed client socket.
static ncclResult_t rasDiagnosticsNoopEmit(void* target, const char* line) {
  (void)target; // unused
  (void)line; // unused
  return ncclSuccess;
}

static const struct rasDiagnosticsReporter rasDiagnosticsDefaultReporter = {
  rasDiagnosticsDefaultEmit,
  nullptr,
  nullptr,
};
static const struct rasDiagnosticsReporter rasDiagnosticsNoopReporter = {rasDiagnosticsNoopEmit, nullptr, nullptr};

// Dispatcher table for diagnostics checks. Each entry binds the stable request id
// to the local collection and summary callbacks for that check.
static const struct rasDiagnosticsCheck rasDiagnosticsChecks[RAS_DIAG_CHECK_COUNT] = {
  {RAS_DIAG_CHECK_GPU_MODEL, rasDiagnosticsGpuModelCollectLocal, rasDiagnosticsGpuModelSummarize},
  {RAS_DIAG_CHECK_CUDA_DRIVER_VERSION, rasDiagnosticsCudaDriverVersionCollectLocal,
   rasDiagnosticsCudaDriverVersionSummarize},
  {RAS_DIAG_CHECK_ECC, rasDiagnosticsEccCollectLocal, rasDiagnosticsEccSummarize},
  {RAS_DIAG_CHECK_NVLINK, rasDiagnosticsNvLinkCollectLocal, rasDiagnosticsNvLinkSummarize},
  {RAS_DIAG_CHECK_NCCL_ENV, rasDiagnosticsNcclEnvCollectLocal, rasDiagnosticsNcclEnvSummarize},
};

static ncclResult_t rasDiagnosticsSummarizePeerPayloads(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData);

ncclResult_t rasDiagnosticsContextInit(struct rasDiagnosticsContext* ctx, const struct ncclComm* comm) {
  if (ctx == nullptr) {
    WARN("RAS diagnostics context init received null context");
    return ncclInternalError;
  }

  memset(ctx, 0, sizeof(*ctx));

  if (comm == nullptr) return ncclSuccess;
  if (!COMPILER_ATOMIC_LOAD(&comm->peerInfoValid, std::memory_order_acquire)) {
    WARN("RAS diagnostics requested before communicator peerInfo is available");
    return ncclInternalError;
  }

  ctx->hasCommFilter = true;
  ctx->commFilter.commHash = comm->commHash;
  ctx->commFilter.hostHash = comm->peerInfo[0].hostHash;
  ctx->commFilter.pidHash = comm->peerInfo[0].pidHash;
  ctx->commNRanks = comm->nRanks;
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsGetCheck(rasDiagnosticsCheckId checkId, const struct rasDiagnosticsCheck** check) {
  int id = (int)checkId;

  if (check == nullptr) {
    WARN("RAS diagnostics check lookup received null output pointer");
    return ncclInternalError;
  }
  if (id < 0 || id >= RAS_DIAG_CHECK_COUNT) {
    WARN("RAS diagnostics check id %d is out of range", id);
    return ncclInternalError;
  }

  *check = &rasDiagnosticsChecks[id];
  if ((*check)->id != checkId) {
    WARN("RAS diagnostics check id %d does not match dispatch table entry %d", id, (int)(*check)->id);
    return ncclInternalError;
  }

  if ((*check)->collectLocal == nullptr || (*check)->summarize == nullptr) {
    WARN("RAS diagnostics check id %d has incomplete dispatch table entry", id);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsSummarizeCheck(rasDiagnosticsCheckId checkId, const struct rasDiagnosticsContext* ctx,
                                                 const struct rasDiagnosticsReporter* reporter, const char* data,
                                                 int nData) {
  const struct rasDiagnosticsCheck* check = nullptr;

  NCCLCHECK(rasDiagnosticsGetCheck(checkId, &check));
  NCCLCHECK(check->summarize(ctx, reporter, data, nData));
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsAppendData(char** data, int* nData, const void* extra, int nExtra) {
  if (data == nullptr || nData == nullptr || *nData < 0 || nExtra < 0 || *nData > INT_MAX - nExtra) {
    WARN("RAS diagnostics payload size is invalid");
    return ncclInternalError;
  }
  if (nExtra == 0) return ncclSuccess;
  if (extra == nullptr) {
    WARN("RAS diagnostics payload append received null data");
    return ncclInternalError;
  }

  NCCLCHECK(ncclRealloc(data, *nData, *nData + nExtra));
  memcpy(*data + *nData, extra, nExtra);
  *nData += nExtra;
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsValidateLocalData(rasDiagnosticsCheckId checkId,
                                                    const struct rasDiagnosticsLocalData* data) {
  int id = (int)checkId;

  if (data == nullptr) {
    WARN("RAS diagnostics check id %d produced null local payload metadata", id);
    return ncclInternalError;
  }
  if (data->recordsBytes < 0 || data->recordStride < 0 || data->nRecords < 0) {
    WARN("RAS diagnostics check id %d produced invalid local payload metadata", id);
    return ncclInternalError;
  }
  if (data->nRecords == 0) {
    if (data->records != nullptr || data->recordsBytes != 0) {
      WARN("RAS diagnostics check id %d produced records for an empty local payload", id);
      return ncclInternalError;
    }
    return ncclSuccess;
  }
  if (data->records == nullptr || data->recordStride == 0 || data->nRecords > INT_MAX / data->recordStride ||
      data->recordsBytes != data->nRecords * data->recordStride) {
    WARN("RAS diagnostics check id %d produced malformed local payload", id);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsAppendCheckPayload(const struct rasDiagnosticsContext* baseCtx,
                                                     const struct rasDiagnosticsCheck& check, char** payload,
                                                     int* payloadBytes) {
  struct rasDiagnosticsLocalData localData = {};
  struct rasDiagnosticsCheckPayloadHeader checkHeader = {};

  NCCLCHECK(check.collectLocal(baseCtx, &localData));
  ncclUniquePtr<char> recordsOwner(localData.records);
  NCCLCHECK(rasDiagnosticsValidateLocalData(check.id, &localData));
  if (localData.nRecords == 0) return ncclSuccess;

  checkHeader.checkId = check.id;
  checkHeader.recordStride = localData.recordStride;
  checkHeader.nRecords = localData.nRecords;
  checkHeader.payloadBytes = localData.recordsBytes;
  NCCLCHECK(rasDiagnosticsAppendData(payload, payloadBytes, &checkHeader, sizeof(checkHeader)));
  NCCLCHECK(rasDiagnosticsAppendData(payload, payloadBytes, localData.records, localData.recordsBytes));

  ((struct rasDiagnosticsPeerPayloadHeader*)*payload)->nChecks++;
  ((struct rasDiagnosticsPeerPayloadHeader*)*payload)->payloadBytes = *payloadBytes;
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsCollectLocalPeerPayload(const struct rasDiagnosticsContext* baseCtx, char** data,
                                                          int* nData) {
  ncclResult_t ret = ncclSuccess;
  char* payload = nullptr;
  int payloadBytes = 0;
  struct rasDiagnosticsPeerPayloadHeader peerHeader = {};

  if (data == nullptr || nData == nullptr) {
    WARN("RAS diagnostics local collection received null output pointer");
    return ncclInternalError;
  }
  *data = nullptr;
  *nData = 0;
  if (baseCtx == nullptr) {
    WARN("RAS diagnostics local collection requested with null context");
    return ncclInternalError;
  }

  peerHeader.payloadBytes = sizeof(peerHeader);
  NCCLCHECKGOTO(rasDiagnosticsAppendData(&payload, &payloadBytes, &peerHeader, sizeof(peerHeader)), ret, fail);

  for (const struct rasDiagnosticsCheck& tableCheck : rasDiagnosticsChecks) {
    NCCLCHECKGOTO(rasDiagnosticsAppendCheckPayload(baseCtx, tableCheck, &payload, &payloadBytes), ret, fail);
  }

  *data = payload;
  *nData = payloadBytes;
  return ncclSuccess;

fail:
  free(payload);
  return ret;
}

struct rasDiagnosticsClientState {
  struct rasDiagnosticsContext ctx;
  struct rasDiagnosticsReporter reporter;
};

// Stores diagnostics context and reporter state on a client before the collective starts.
ncclResult_t rasDiagnosticsClientInit(struct rasClient* client, const struct rasDiagnosticsContext* ctx,
                                      const struct rasDiagnosticsReporter* reporter) {
  if (client == nullptr || ctx == nullptr) {
    WARN("RAS diagnostics client init received invalid arguments");
    return ncclInternalError;
  }
  if (reporter != nullptr && reporter->emit == nullptr) {
    WARN("RAS diagnostics client init received invalid reporter");
    return ncclInternalError;
  }
  if (client->diagnostics != nullptr) {
    WARN("RAS diagnostics client init requested with existing diagnostics state");
    return ncclInternalError;
  }

  NCCLCHECK(ncclCalloc(&client->diagnostics, 1));
  client->diagnostics->ctx = *ctx;
  client->diagnostics->reporter = (reporter != nullptr ? *reporter : rasDiagnosticsDefaultReporter);
  return ncclSuccess;
}

void rasDiagnosticsClientCleanup(struct rasClient* client) {
  if (client == nullptr) return;
  free(client->diagnostics);
  client->diagnostics = nullptr;
}

static bool rasDiagnosticsClientHasActiveRequest(const struct rasClient* client) {
  return client != nullptr && client->diagnostics != nullptr;
}

bool rasDiagnosticsInProgress() {
  for (struct rasClient* client = rasClientsHead; client; client = client->next) {
    if (rasDiagnosticsClientHasActiveRequest(client)) return true;
  }
  return false;
}

void rasDiagnosticsCancelTarget(void* target) {
  if (target == nullptr) return;
  for (struct rasClient* client = rasClientsHead; client; client = client->next) {
    if (client->diagnostics != nullptr && client->diagnostics->reporter.target == target) {
      client->diagnostics->reporter = rasDiagnosticsNoopReporter;
    }
  }
}

// RAS_COLL_DIAG init hook: runs on every peer handling the request. Builds this peer's diagnostics payload,
// scoped by the filter carried in the request.
ncclResult_t rasCollDiagInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData) {
  struct rasDiagnosticsContext ctx = {};

  ctx.hasCommFilter = (*pReq)->diag.hasCommFilter;
  if (ctx.hasCommFilter) ctx.commFilter = (*pReq)->diag.commFilter;

  *pReqLen = rasCollDataLength(RAS_COLL_DIAG);
  NCCLCHECK(rasDiagnosticsCollectLocalPeerPayload(&ctx, pData, pNData));
  return ncclSuccess;
}

// RAS_COLL_DIAG merge hook: variable-size payload concatenation. Peer payloads are self-delimiting
// (rasDiagnosticsPeerPayloadHeader.payloadBytes), so we simply append the incoming bytes in peer order.
ncclResult_t rasCollDiagMerge(struct rasCollective* coll, struct rasMsg* msg) {
  const int dataAlign = alignof(int64_t);
  int dataOffset;
  int nData;

  if (coll == nullptr || msg == nullptr || coll->nData < 0 || msg->collResp.nData < 0 || msg->collResp.nPeers < 0) {
    WARN("RAS diagnostics merge received malformed collective payload size");
    return ncclInternalError;
  }
  if (msg->collResp.nData == 0) return ncclSuccess;
  if (msg->collResp.nPeers >
      (INT_MAX - (int)rasMsgLength(RAS_MSG_COLLRESP) - (dataAlign - 1)) / (int)sizeof(*msg->collResp.peers)) {
    WARN("RAS diagnostics merge received too many peer entries");
    return ncclInternalError;
  }
  if (coll->nData > INT_MAX - msg->collResp.nData) {
    WARN("RAS diagnostics merged payload is too large");
    return ncclInternalError;
  }

  dataOffset = rasMsgLength(RAS_MSG_COLLRESP) + msg->collResp.nPeers * sizeof(*msg->collResp.peers);
  ALIGN_SIZE(dataOffset, dataAlign);
  nData = coll->nData + msg->collResp.nData;
  NCCLCHECK(ncclRealloc(&coll->data, coll->nData, nData));
  memcpy(coll->data + coll->nData, ((char*)msg) + dataOffset, msg->collResp.nData);
  coll->nData = nData;
  return ncclSuccess;
}

// Locally-initiated RAS_COLL_DIAG completion: summarize the gathered peer payloads owned by the client.
ncclResult_t rasDiagnosticsResume(struct rasClient* client) {
  ncclResult_t ret = ncclSuccess;
  struct rasDiagnosticsClientState* diagnostics;
  struct rasCollective* coll;
  char line[64];
  int nRanks;

  if (client == nullptr || client->diagnostics == nullptr || client->coll == nullptr) {
    WARN("RAS diagnostics resume requested with invalid client state");
    return ncclInternalError;
  }
  diagnostics = client->diagnostics;
  coll = client->coll;

  // A comm-scoped request reports the communicator's rank count; an unscoped one falls back to responding peers.
  nRanks = diagnostics->ctx.hasCommFilter ? diagnostics->ctx.commNRanks : coll->nPeers;

  if (!client->internal) (void)diagnostics->reporter.emit(diagnostics->reporter.target, "=== RAS Diagnostics ===");
  ret = rasDiagnosticsSummarizePeerPayloads(&diagnostics->ctx, &diagnostics->reporter, coll->data, coll->nData);
  if (ret != ncclSuccess) WARN("RAS diagnostics summary returned %d", ret);
  snprintf(line, sizeof(line), "RAS diagnostics completed in %.1f ms across %d %s",
           (double)(clockNano() - coll->startTime) / (CLOCK_UNITS_PER_SEC / 1000), nRanks,
           diagnostics->ctx.hasCommFilter ? "ranks" : "RAS peers");
  (void)diagnostics->reporter.emit(diagnostics->reporter.target, line);
  if (diagnostics->reporter.finish != nullptr) {
    ncclResult_t finishRet = diagnostics->reporter.finish(diagnostics->reporter.target, ret);
    if (finishRet != ncclSuccess) WARN("RAS diagnostics reporter finish returned %d", finishRet);
  }

  rasCollFree(coll);
  client->coll = nullptr;
  rasDiagnosticsClientCleanup(client);
  return ret;
}

// Accumulates per-check record totals while enforcing stride consistency.
static ncclResult_t rasDiagnosticsAccountCheckRecords(struct rasDiagnosticsLocalData* combined,
                                                      const struct rasDiagnosticsCheckPayloadHeader* checkHeader) {
  if (checkHeader->nRecords == 0) return ncclSuccess;
  if (combined->nRecords == 0) {
    combined->recordStride = checkHeader->recordStride;
  } else if (combined->recordStride != checkHeader->recordStride) {
    WARN("RAS diagnostics check id %d changed record stride from %d to %d", (int)checkHeader->checkId,
         combined->recordStride, checkHeader->recordStride);
    return ncclInternalError;
  }
  if (combined->nRecords > INT_MAX - checkHeader->nRecords) {
    WARN("RAS diagnostics check id %d gathered too many records", (int)checkHeader->checkId);
    return ncclInternalError;
  }
  if (combined->recordsBytes > INT_MAX - checkHeader->payloadBytes) {
    WARN("RAS diagnostics check id %d gathered too many records", (int)checkHeader->checkId);
    return ncclInternalError;
  }
  combined->nRecords += checkHeader->nRecords;
  combined->recordsBytes += checkHeader->payloadBytes;
  return ncclSuccess;
}

static ncclResult_t rasDiagnosticsSummarizePeerPayloads(const struct rasDiagnosticsContext* baseCtx,
                                                        const struct rasDiagnosticsReporter* reporter, const char* data,
                                                        int nData) {
  ncclResult_t ret = ncclSuccess;
  struct rasDiagnosticsLocalData combined[RAS_DIAG_CHECK_COUNT] = {};
  int writeOffset[RAS_DIAG_CHECK_COUNT] = {};
  int offset;

  if (baseCtx == nullptr) {
    WARN("RAS diagnostics summary requested with null context");
    return ncclInternalError;
  }
  if (nData == 0) return ncclSuccess;
  if (data == nullptr || nData < 0) {
    WARN("RAS diagnostics summary received malformed gathered payload size %d", nData);
    return ncclInternalError;
  }

  // Validate the gathered buffer and compute per-check storage requirements.
  for (offset = 0; offset < nData;) {
    const struct rasDiagnosticsPeerPayloadHeader* peerHeader;
    int peerOffset, peerEnd;

    if (nData - offset < (int)sizeof(*peerHeader)) {
      WARN("RAS diagnostics gathered payload has truncated peer header");
      ret = ncclInternalError;
      goto exit;
    }
    peerHeader = (const struct rasDiagnosticsPeerPayloadHeader*)(data + offset);
    if (peerHeader->nChecks < 0 || peerHeader->payloadBytes < (int)sizeof(*peerHeader) ||
        peerHeader->payloadBytes > nData - offset) {
      WARN("RAS diagnostics gathered payload has malformed peer payload header");
      ret = ncclInternalError;
      goto exit;
    }

    peerOffset = offset + sizeof(*peerHeader);
    peerEnd = offset + peerHeader->payloadBytes;
    for (int checkIdx = 0; checkIdx < peerHeader->nChecks; checkIdx++) {
      const struct rasDiagnosticsCheckPayloadHeader* checkHeader;
      const struct rasDiagnosticsCheck* check = nullptr;
      int id;

      if (peerEnd - peerOffset < (int)sizeof(*checkHeader)) {
        WARN("RAS diagnostics gathered payload has truncated check header");
        ret = ncclInternalError;
        goto exit;
      }

      checkHeader = (const struct rasDiagnosticsCheckPayloadHeader*)(data + peerOffset);
      NCCLCHECKGOTO(rasDiagnosticsGetCheck(checkHeader->checkId, &check), ret, exit);
      id = (int)check->id;
      if (checkHeader->recordStride <= 0 || checkHeader->nRecords < 0 || checkHeader->payloadBytes < 0 ||
          checkHeader->payloadBytes > peerEnd - peerOffset - (int)sizeof(*checkHeader) ||
          checkHeader->nRecords > INT_MAX / checkHeader->recordStride ||
          checkHeader->payloadBytes != checkHeader->nRecords * checkHeader->recordStride) {
        WARN("RAS diagnostics check id %d has malformed gathered payload metadata", id);
        ret = ncclInternalError;
        goto exit;
      }

      NCCLCHECKGOTO(rasDiagnosticsAccountCheckRecords(&combined[id], checkHeader), ret, exit);
      peerOffset += sizeof(*checkHeader) + checkHeader->payloadBytes;
    }

    if (peerOffset != peerEnd) {
      WARN("RAS diagnostics gathered payload has trailing bytes in peer payload");
      ret = ncclInternalError;
      goto exit;
    }
    offset = peerEnd;
  }

  for (int id = 0; id < RAS_DIAG_CHECK_COUNT; id++) {
    if (combined[id].recordsBytes > 0) {
      NCCLCHECKGOTO(ncclCalloc(&combined[id].records, combined[id].recordsBytes), ret, exit);
    }
  }

  // Copy each peer's records into the pre-sized buffers.
  for (offset = 0; offset < nData;) {
    const struct rasDiagnosticsPeerPayloadHeader* peerHeader =
      (const struct rasDiagnosticsPeerPayloadHeader*)(data + offset);
    int peerOffset = offset + sizeof(*peerHeader);

    for (int checkIdx = 0; checkIdx < peerHeader->nChecks; checkIdx++) {
      const struct rasDiagnosticsCheckPayloadHeader* checkHeader =
        (const struct rasDiagnosticsCheckPayloadHeader*)(data + peerOffset);
      int id = (int)checkHeader->checkId;

      peerOffset += sizeof(*checkHeader);
      if (checkHeader->payloadBytes > 0) {
        memcpy(combined[id].records + writeOffset[id], data + peerOffset, checkHeader->payloadBytes);
        writeOffset[id] += checkHeader->payloadBytes;
      }
      peerOffset += checkHeader->payloadBytes;
    }
    offset += peerHeader->payloadBytes;
  }

  for (const struct rasDiagnosticsCheck& tableCheck : rasDiagnosticsChecks) {
    rasDiagnosticsCheckId checkId = tableCheck.id;
    int id = (int)checkId;
    const char* records = combined[id].records;
    int recordsBytes = combined[id].recordsBytes;

    NCCLCHECKGOTO(rasDiagnosticsSummarizeCheck(checkId, baseCtx, reporter, records, recordsBytes), ret, exit);
  }

exit:
  for (int i = 0; i < RAS_DIAG_CHECK_COUNT; i++) free(combined[i].records);
  return ret;
}

// Posts the diagnostics collective and advances the client to the completion state.
ncclResult_t rasDiagnosticsStart(struct rasClient* client) {
  const struct rasDiagnosticsContext* ctx;
  struct rasCollRequest req;
  bool allDone = false;
  ncclResult_t ret = ncclSuccess;

  if (client == nullptr) {
    WARN("RAS diagnostics requested with null client");
    return ncclInternalError;
  }
  if (client->diagnostics == nullptr) {
    WARN("RAS diagnostics requested without client state");
    return ncclInternalError;
  }
  ctx = &client->diagnostics->ctx;

  // Post the RAS_COLL_DIAG variable-size gather and return; completion resumes the owning RAS client.
  // Every peer that handles the request builds its own payload in rasCollDiagInit (scoped by req.diag).
  memset(&req, '\0', sizeof(req));
  rasCollReqInit(&req);
  req.timeout = client->timeout;
  req.type = RAS_COLL_DIAG;
  req.diag.hasCommFilter = ctx->hasCommFilter;
  if (ctx->hasCommFilter) req.diag.commFilter = ctx->commFilter;

  NCCLCHECKGOTO(rasNetSendCollReq(&req, &allDone, &client->coll), ret, fail);

  client->status = RAS_CLIENT_DIAG_FINI;
  if (!allDone) ret = ncclInProgress;

  return ret;

fail:
  if (client->coll != nullptr) {
    rasCollFree(client->coll);
    client->coll = nullptr;
  }
  rasDiagnosticsClientCleanup(client);
  return ret;
}
