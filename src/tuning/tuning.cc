/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "tuning.h"
#include "tuning_int.h"
#include "cost_model.h"
#include "tuner.h"
#include "transport.h"
#include "comm.h"
#include "device.h"
#include "alloc.h"
#include <cfloat>

NCCL_PARAM(SymNoWinEnable, "SYM_NOWIN_ENABLE", 0);

extern int64_t ncclParamSingleProcMemRegEnable();

void ncclTuningResultListFree(struct ncclTuningResultList_t* list) {
  struct ncclTuningResultListNode* node = list->head;
  while (node != nullptr) {
    struct ncclTuningResultListNode* next = node->next;
    free(node);
    node = next;
  }
  list->head = nullptr;
}

ncclResult_t ncclTuningResultListPushFront(struct ncclTuningResultList_t* list, struct ncclTuningResult_t result) {
  struct ncclTuningResultListNode* node = nullptr;
  NCCLCHECK(ncclCalloc(&node, 1));
  node->result = result;
  node->next = list->head;
  list->head = node;
  return ncclSuccess;
}

/*
  Initialize the tuning subsystem.
*/
ncclResult_t ncclTuningInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), ret, fail);
  if (comm->tuner) {
    NCCLCHECKGOTO(comm->tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks, comm->nNodes, ncclDebugLog,
                                    &comm->nvlDomainInfo, &comm->tuningContext.tuningConstants),
                  ret, fail);
  }

  NCCLCHECKGOTO(ncclTuningCostModelInit(comm), ret, fail);
  ncclTuningSetThreadThresholds(comm);

  if (comm->rank == 0) {
    constexpr int lineLen = 1024;
    char line[lineLen];
    int offset = 0;
    for (int block = 0; block < DIVUP(NCCL_NUM_ALGORITHMS, 3); block++) {
      offset = snprintf(line, lineLen, "  Algorithm   |");
      for (int ba = 0; ba < 3; ba++) {
        int a = block * 3 + ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        offset +=
          snprintf(line + offset, std::max(0, lineLen - offset), " %14s   %14s   %14s |", "", ncclAlgoStr[a], "");
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, "  Protocol    |");
      for (int ba = 0; ba < 3; ba++) {
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
          offset += snprintf(line + offset, std::max(0, lineLen - offset), " %14s |", ncclProtoStr[p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      offset = snprintf(line, lineLen, " Max NThreads |");
      for (int ba = 0; ba < 3; ba++) {
        int a = block * 3 + ba;
        if (a >= NCCL_NUM_ALGORITHMS) continue;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
          offset +=
            snprintf(line + offset, std::max(0, lineLen - offset), " %14d |", comm->tuningContext.maxThreads[a][p]);
        }
      }
      for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
        offset = snprintf(line, lineLen, "%13s |", ncclFuncStr[c]);
        for (int ba = 0; ba < 3; ba++) {
          int a = block * 3 + ba;
          if (a >= NCCL_NUM_ALGORITHMS) continue;
          for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
            float lat = comm->tuningContext.generalLatencies[c][a][p];
            float bw = comm->tuningContext.generalBandwidths[c][a][p];
            offset += snprintf(line + offset, std::max(0, lineLen - offset), "%8.1f/%6.1f |", lat, bw);
          }
        }
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }
exit:
  return ret;
fail:
  goto exit;
}

/*
  Finalize the tuning subsystem.
*/
ncclResult_t ncclTuningFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclTuningCostModelFinalize(comm), ret, fail);
  if (comm->tuner != NULL) {
    NCCLCHECKGOTO(comm->tuner->finalize(comm->tunerContext), ret, fail);
    NCCLCHECKGOTO(ncclTunerPluginUnload(comm), ret, fail);
  }
exit:
  return ret;
fail:
  goto exit;
}

/*
  Get all valid tunings for the given input.
*/
ncclResult_t ncclTuningComputeAllTunings(struct ncclTuningInput_t* const input,
                                         struct ncclTuningResultList_t* const tunings) {
  ncclResult_t ret = ncclSuccess;

  for (int i = 0; i < NCCL_TUNING_COUNT; i++) {
    struct ncclTuningResult_t tuning = NCCL_TUNING_RESULT_INIT;
    tuning.id = i;
    tuning.valid = 1;

    if (!(input->tuningMask & (1lu << i))) {
      tuning.valid = 0;
      continue;
    }
    NCCLCHECK(ncclTuningExpandId(i, &tuning.algo, &tuning.proto, &tuning.symKernelId));
    NCCLCHECKGOTO(ncclTuningComputeTuning(i, input, &tuning), ret, fail);
    if (tuning.valid) NCCLCHECKGOTO(ncclTuningResultListPushFront(tunings, tuning), ret, fail);
  }
exit:
  return ret;
fail:
  goto exit;
}

/*
  Select the best tuning from the given list of tunings.
  The tuning with the lowest simulated time/cost is determined to be best.
*/
static ncclResult_t ncclTuningSelectBestTuning(struct ncclTuningResultList_t* tunings,
                                               struct ncclTuningResult_t* const bestTuning) {
  bestTuning->timeUs = FLT_MAX;
  struct ncclTuningResultListNode* node = tunings->head;
  while (node != nullptr) {
    const struct ncclTuningResult_t& tuning = node->result;
    TRACE(NCCL_TUNING, "A/P/S %s/%s/%s, time: %f", ncclAlgoToString(tuning.algo), ncclProtoToString(tuning.proto),
          ncclSymkKernelIdToString(tuning.symKernelId), tuning.timeUs);
    if (tuning.timeUs < bestTuning->timeUs) {
      *bestTuning = tuning;
    }
    node = node->next;
  }
  return ncclSuccess;
}

/*
  Get the best tuning for the given input.
  Simulates the all tunings for the given input and engages the tuner plugin if defined.
  Then selects the best tuning from all the tuning results and returns that to the caller.
*/
ncclResult_t ncclTuningCompute(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result) {
  ncclResult_t ret = ncclSuccess;
  INFO(NCCL_TUNING,
       "Input: { .comm = %p, .tuningMask = %lb, .func = %s, .redOp = %d, .devRedop = %d, .dataType = %d, .nBytes = "
       "%lu, .numPipesOps = %d, .count = %lu, .countMax = %lu, .nWorks = %d, .winRegType = %d, .regBuff = %d }",
       input->comm, input->tuningMask, ncclFuncToString(input->func), input->redOp, input->devRedOp, input->datatype,
       input->nBytes, input->numPipeOps, input->count, input->countMax, input->nWorks, input->winRegType,
       input->regBuff);
  struct ncclTuningResultList_t tunings;
  tunings.head = nullptr;
  struct ncclTuningResult_t bestTuning = NCCL_TUNING_RESULT_INIT;
  // Set tuning to Ring/Simple for single rank case
  if (input->comm->nRanks <= 1) {
    bestTuning.algo = NCCL_ALGO_RING;
    bestTuning.proto = NCCL_PROTO_SIMPLE;
    bestTuning.symKernelId = ncclSymkKernelId_Count;
    bestTuning.nChannels = 0;
    bestTuning.maxChannels = 0;
    bestTuning.nWarps = 0;
    bestTuning.forced = 0;
  } else {
    NCCLCHECKGOTO(ncclTuningComputeAllTunings(input, &tunings), ret, exit);
    if (input->comm->tuner != NULL) {
      float generalTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
      for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
        for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
          generalTable[i][j] = NCCL_TUNING_IGNORE;
        }
      }
      struct ncclTuningResultListNode* node = tunings.head;
      while (node != nullptr) {
        const struct ncclTuningResult_t& tuning = node->result;
        node = node->next;
        if (tuning.algo == NCCL_ALGO_UNDEF || tuning.proto == NCCL_PROTO_UNDEF) continue;
        generalTable[tuning.algo][tuning.proto] = tuning.timeUs;
      }
      node = tunings.head;
      int nMaxChannels = 0;
      NCCLCHECKGOTO(input->comm->tuner->getCollInfo(input->comm->tunerContext, input->func, input->nBytes,
                                                    input->numPipeOps, (float**)generalTable, NCCL_NUM_ALGORITHMS,
                                                    NCCL_NUM_PROTOCOLS, input->regBuff, &nMaxChannels),
                    ret, exit);
      while (node != nullptr) {
        struct ncclTuningResult_t& tuning = node->result;
        node = node->next;
        if (tuning.algo == NCCL_ALGO_UNDEF || tuning.proto == NCCL_PROTO_UNDEF) continue;
        tuning.maxChannels = nMaxChannels;
        tuning.timeUs = generalTable[tuning.algo][tuning.proto];
      }
      NCCLCHECKGOTO(ncclTuningSelectBestTuning(&tunings, &bestTuning), ret, exit);
    } else {
      int collNetSupport = input->collNetSupport;
      int nvlsSupport = input->nvlsSupport;
      NCCLCHECKGOTO(ncclTuningSelectBestTuning(&tunings, &bestTuning), ret, exit);
      // NCCL_CTA_POLICY_EFFICIENCY requires user (non-symmetric) buffer registration (currently unsupported with MNNVL)
      if ((input->comm->config.CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY) && ncclGetEnv("NCCL_ALGO") == NULL &&
          ncclGetEnv("NCCL_PROTO") == NULL && !input->comm->MNNVL) {
        // make algorithm selection based on buffer registration
        // there can be other specialized policies for algorithms and protocols pickup in the future
        if (input->regBuff && (input->func == ncclFuncAllGather || input->func == ncclFuncReduceScatter)) {
          if ((input->comm->nNodes > 1 && collNetSupport && nvlsSupport) || (input->comm->nNodes == 1 && nvlsSupport)) {
            int recChannels;
            NCCLCHECKGOTO(ncclNvlsRegResourcesQuery(input->comm, input->func, &recChannels), ret, exit);
            if (recChannels <= bestTuning.nChannels) {
              bestTuning.algo = NCCL_ALGO_NVLS;
              bestTuning.proto = NCCL_PROTO_SIMPLE;
              bestTuning.nChannels = recChannels;
              bestTuning.nWarps = input->comm->tuningContext.maxThreads[bestTuning.algo][bestTuning.proto] / WARP_SIZE;
            }
          }
        }
      }
    }
  }
  if (bestTuning.algo != NCCL_ALGO_UNDEF && bestTuning.proto != NCCL_PROTO_UNDEF) {
    NCCLCHECKGOTO(ncclTuningGetChannels(input, &bestTuning), ret, exit);
  }
  if ((bestTuning.symKernelId != ncclSymkKernelId_Count ||
       (input->tuningMask & NCCL_TUNING_MASK_SYM_KERNELS && bestTuning.symKernelId == ncclSymkKernelId_Count)) &&
      bestTuning.algo == NCCL_ALGO_UNDEF && bestTuning.proto == NCCL_PROTO_UNDEF) {
    bool isLLKernel = (1 << bestTuning.symKernelId) & ncclSymkLLKernelMask();
    bool isOneThreadMultiGpus = input->comm->intraRanks > 1 && !ncclParamSingleProcMemRegEnable();
    bool needFallback = bestTuning.symKernelId != ncclSymkKernelId_Count ? false : true;

    // Fallback logic for symmetric LL kernels:
    // - If both src and dst are registered, we don't fall back if a symmetric kernel is available.
    // - Otherwise, we have to fall back to generl kernel if running the selected symmetric LL kernel is
    //   not possible (if the buffers are not registered and we manage multiple GPUs).
    // - If the user forced a symmetric kernel via NCCL_SYM_KERNEL or requested preference for using
    //   symmetric kernels even without symmetric buffers via NCCL_SYM_NOWIN_ENABLE, we respect that.
    // - Otherwise, we query the general cost model and if it selects a non-LL proto, we pick that.
    if (bestTuning.symKernelId != ncclSymkKernelId_Count) {
      if (input->winRegType == ncclSymSendRegRecvReg) {
        needFallback = false;
      } else if (isLLKernel) {
        needFallback = isOneThreadMultiGpus && input->winRegType == ncclSymSendNonregRecvNonreg;
        if (!needFallback && !result->forced) {
          needFallback = !ncclParamSymNoWinEnable() && input->winRegType == ncclSymSendNonregRecvNonreg;
        }
      }
    }
    if (needFallback) {
      struct ncclTuningResult_t generalTuning = NCCL_TUNING_RESULT_INIT;
      struct ncclTuningInput_t generalInput = *input;
      generalInput.tuningMask = NCCL_TUNING_MASK_GENERAL_KERNELS;
      NOWARN(ncclTuningCompute(&generalInput, &generalTuning), NCCL_TUNING);
      if (generalTuning.proto != NCCL_PROTO_LL) {
        bestTuning = generalTuning;
      }
    }
  }

  INFO(NCCL_TUNING,
       "Best tuning { .id = %d, .valid = %d, timeUs = %f,  .algo = %s, .proto = %s, .symKernelId = %s, nChannels = %d, "
       "maxChannels = %d, nWarps = %d, forced = %d }",
       bestTuning.id, bestTuning.valid, bestTuning.timeUs, ncclAlgoToString(bestTuning.algo),
       ncclProtoToString(bestTuning.proto), ncclSymkKernelIdToString(bestTuning.symKernelId), bestTuning.nChannels,
       bestTuning.maxChannels, bestTuning.nWarps, bestTuning.forced);

  if ((bestTuning.algo == NCCL_ALGO_UNDEF || bestTuning.proto == NCCL_PROTO_UNDEF) &&
      bestTuning.symKernelId == ncclSymkKernelId_Count) {
    char ncclAlgoEnvStr[1024] = "";
    char ncclProtoEnvStr[1024] = "";
    char ncclSymKernelIdEnvStr[1024] = "";
    const char* symKernelIdEnv = ncclGetEnv("NCCL_SYM_KERNEL_ID");
    if (symKernelIdEnv) {
      snprintf(ncclSymKernelIdEnvStr, 1023, " NCCL_SYM_KERNEL_ID was set to %s.", symKernelIdEnv);
    }
    const char* algoEnv = ncclGetEnv("NCCL_ALGO");
    if (algoEnv) {
      snprintf(ncclAlgoEnvStr, 1023, " NCCL_ALGO was set to %s.", algoEnv);
    }
    const char* protoEnv = ncclGetEnv("NCCL_PROTO");
    if (protoEnv) {
      snprintf(ncclProtoEnvStr, 1023, " NCCL_PROTO was set to %s.", protoEnv);
    }
    WARN("No algorithm/protocol nor symKernelId available for function %s with datatype %s.%s%s%s",
         ncclFuncToString(input->func), ncclDatatypeToString(input->datatype), ncclAlgoEnvStr, ncclProtoEnvStr,
         ncclSymKernelIdEnvStr);
    ret = (algoEnv || protoEnv || symKernelIdEnv) ? ncclInvalidUsage : ncclInternalError;
  }
  *result = bestTuning;
exit:
  ncclTuningResultListFree(&tunings);
  return ret;
}

/*
  Collect the tuning results.
*/
ncclResult_t ncclTuningComputeTuning(int id, struct ncclTuningInput_t* const input,
                                     struct ncclTuningResult_t* const result) {
  NCCLCHECK(ncclTuningCostModelSimModel(id, input, result));
  return ncclSuccess;
}
