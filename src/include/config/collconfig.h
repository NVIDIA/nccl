/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef NCCL_CONFIG_COLLCONFIG_H_
#define NCCL_CONFIG_COLLCONFIG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdint.h>

// Config option resolve and setting helper macro
// It resolves one config option across the three sources, highest priority first:
//   env var  >  per-call config  >  comm config
// Note the comm's value could be user setting or default (see src/init.cc). So we assume comm's value is always
// a valid one.
// If the env var or config value is out of bound, we will ignore it and proceed to the next level of priority.
#define NCCL_CONFIG_SET(target_ptr, option, envVal, configVal, commVal, optionMin, optionMax) \
  do { \
    if ((target_ptr) != nullptr) { \
      if ((envVal) != NCCL_CONFIG_UNDEF_INT) { \
        if ((envVal) <= (optionMax) && (envVal) >= (optionMin)) { \
          (target_ptr)->option = (envVal); \
          break; \
        } else { \
          INFO(NCCL_ENV, "Env var for " #option "=%" PRId64 " is ignored because value out of bound min/max=[%d, %d]", \
               (envVal), (optionMin), (optionMax)); \
        } \
      } \
      if ((configVal) != NCCL_CONFIG_UNDEF_INT) { \
        if ((configVal) <= (optionMax) && (configVal) >= (optionMin)) { \
          (target_ptr)->option = (configVal); \
          break; \
        } else { \
          INFO(NCCL_COLL, "Config " #option "=%d is ignored because value out of bound min/max=[%d, %d]", (configVal), \
               (optionMin), (optionMax)); \
        } \
      } \
      (target_ptr)->option = (commVal); \
    } \
  } while (0)

ncclResult_t ncclParseCollConfig(const ncclCollConfig_t* config, ncclCollConfig_t* internal_config);

// True if the config carries an actual algorithm selection. NULL, UNDEF, and "" all mean
// "automatic" (no selection); see the ncclCollConfig_t::algSelection contract in nccl.h.
bool ncclCollConfigHasAlgSelection(const ncclCollConfig_t* config);

bool ncclCollConfigNeedAggIsolate(const ncclCollConfig_t* config);

ncclResult_t ncclCollConfigGetAlgMask(const ncclCollConfig_t* config, ncclFunc_t func, uint64_t* outMask);

// Resolve the effective per-call CTAPolicy from a per-call value, the comm-level policy, and whether
// an NCCL_CTA_POLICY env override is present (it is folded into commPolicy at comm init). When the
// override is present the per-call value is ignored and commPolicy wins; otherwise a valid per-call
// value wins (ZERO over EFFICIENCY); an unset or out-of-range per-call value inherits commPolicy.
int ncclCollConfigResolveCTAPolicy(int perCall, int commPolicy, bool envOverridden);

#endif
