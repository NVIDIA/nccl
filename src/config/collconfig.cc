/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "collectives.h" // ncclFuncToString
#include "config/algorithm_parser.h"
#include "config/algorithm_registry.h"
#include "config/collconfig.h"
#include "core.h" // WARN
#include "device.h" // MAXCHANNELS
#include "nccl.h"

bool ncclCollConfigHasAlgSelection(const ncclCollConfig_t* config) {
  if (config == NULL) return false;
  return config->algSelection != NCCL_CONFIG_UNDEF_PTR && config->algSelection[0] != '\0';
}

// Helper function to set aggregation isolation
bool ncclCollConfigNeedAggIsolate(const ncclCollConfig_t* config) {
  if (config->size == 0) {
    // zero size config indicate no user config is passed, no aggregation isolation
    return false;
  }
  // check for other options to determine if aggregation isolation is needed
  if (config->minCTAs != NCCL_CONFIG_UNDEF_INT || config->maxCTAs != NCCL_CONFIG_UNDEF_INT) return true;
  if (config->nvlsCTAs != NCCL_CONFIG_UNDEF_INT) return true;
  // cgaClusterSize is a per-launch attribute applied to the whole plan, not a per-collective resource
  // cap, so it does not force aggregation isolation; it is applied to the group as-is.
  // algorithm selection need aggregation isolation so that selection does not spill to other collectives in the group.
  if (ncclCollConfigHasAlgSelection(config)) return true;
  // CTAPolicy is resolved in place (env > per-call > comm) before this runs, so its isolation is
  // decided by the caller comparing the resolved value against the comm default.
  return false;
}

// Validate a per-call config header and copy it into internal_config for safe access during
// scheduling. A NULL user config is accepted (no config); internal_config->size is set to 0 when no
// user config was passed (used to isolate configured collectives from aggregation), non-zero
// otherwise. Later commits add per-call resource knobs, resolved (env > per-call > comm) by their
// consumers via resolveCollRes.
ncclResult_t ncclParseCollConfig(const ncclCollConfig_t* config, ncclCollConfig_t* internal_config) {
  *internal_config = NCCL_COLLCONFIG_INITIALIZER;
  if (config != nullptr) {
    // ncclCollConfig_t is append-only (size only grows, fields never move). The smallest valid
    // struct is the v23100 one, used as the minimal size check.
    if (config->size < sizeof(struct ncclCollConfig_v23100)) {
      WARN("ncclCollConfig_t size %zu too small (expected >= %zu)", config->size, sizeof(ncclCollConfig_t));
      return ncclInvalidArgument;
    }
    if (config->magic != NCCL_API_MAGIC) {
      WARN("ncclCollConfig_t magic mismatch (got 0x%x, expected 0x%x)", config->magic, NCCL_API_MAGIC);
      return ncclInvalidArgument;
    }
    // copy user config to internal_config for safe access later
    if (config->version <= NCCL_VERSION_CODE) {
      // A same/older user config struct is smaller than (or equal to) current library's struct.
      memcpy(internal_config, config, config->size);
    } else {
      // A newer user config struct is larger than current library's struct
      memcpy(internal_config, config, internal_config->size);
    }

    // forceAlgSelection is boolean and only accepts 0 or 1
    if (internal_config->forceAlgSelection != 1 && internal_config->forceAlgSelection != 0) {
      WARN("config->forceAlgSelection=%d is invalid, accepted values are 0 or 1", internal_config->forceAlgSelection);
      return ncclInvalidArgument;
    }
    if (internal_config->CTAPolicy != NCCL_CONFIG_UNDEF_INT) {
      if (internal_config->CTAPolicy < 0 ||
          internal_config->CTAPolicy > (NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO)) {
        WARN("config->CTAPolicy=%d is invalid, accepted values are [1, %d]", internal_config->CTAPolicy,
             (NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO));
        return ncclInvalidArgument;
      }
    }
  } else {
    // When no user config, set collConfig.size == 0 to mark as "no user config passed"
    internal_config->size = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclCollConfigGetAlgMask(const ncclCollConfig_t* config, ncclFunc_t func, uint64_t* outMask) {
  // Each set bit keeps that algorithm in the filter; 0 == no selection == automatic (the
  // consumers skip narrowing when the mask is 0).
  *outMask = 0;
  if (!ncclCollConfigHasAlgSelection(config)) return ncclSuccess; // no selection (incl. NULL) -> automatic
  const char* sel = config->algSelection;
  // A non-empty selection that cannot be honored -- bad syntax, or nothing valid/matching for this
  // collective -- is governed by forceAlgSelection: true -> hard error; false -> fall back to automatic.
  uint64_t mask = 0;
  ncclResult_t r = ncclAlgParse(sel, &mask);
  if (r == ncclSuccess) mask &= ncclAlgValidForFuncMask(func);
  if (r != ncclSuccess || mask == 0) {
    if (config->forceAlgSelection) {
      WARN("config->algSelection \"%s\" cannot be honored for %s (set forceAlgSelection=0 to if need to allow fall "
           "back to automatic selection)",
           sel, ncclFuncToString(func));
      return ncclInvalidArgument;
    }
    INFO(NCCL_TUNING, "config->algSelection \"%s\" cannot be honored for %s; falling back to automatic selection", sel,
         ncclFuncToString(func));
    return ncclSuccess; // *outMask stays 0 -> no filtering
  }
  *outMask = mask;
  return ncclSuccess;
}

int ncclCollConfigResolveCTAPolicy(int configVal, int commVal, bool envOverridden) {
  // CTAPolicy is resolved once globally at comm init (an NCCL_CTA_POLICY env override is folded into
  // commVal there), so an env override makes commVal authoritative and the per-call config value is
  // ignored. Otherwise the per-call config value wins, with ZERO taking precedence
  // over EFFICIENCY (same as in src/init.cc)
  if (envOverridden || configVal == NCCL_CONFIG_UNDEF_INT) return commVal;
  if ((configVal & NCCL_CTA_POLICY_ZERO) && (configVal & NCCL_CTA_POLICY_EFFICIENCY)) {
    configVal &= ~NCCL_CTA_POLICY_EFFICIENCY;
  }
  return configVal;
}
