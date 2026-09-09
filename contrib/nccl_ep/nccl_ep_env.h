/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#ifndef NCCL_EP_ENV_H_
#define NCCL_EP_ENV_H_

// ============================================================================
// Per-group NCCL EP environment configuration.
//
// All std::getenv() parsing for the NCCL EP runtime lives in nccl_ep_env.cc.
// Each knob is a ncclEpEnvVar that carries its own environment-variable name,
// so callers never hardcode the name: they read <field>.set / <field>.value and
// (for diagnostics) <field>.name. A configuration struct is populated once per
// EP group by nccl_ep_env_init() at group creation; handles then read their
// group's copy (handle->group->env) directly.
//
// nccl_ep_env_print() dumps every variable (name + provided value, or "unset")
// when NCCL_EP_ENV_VERBOSE is truthy, so the user can inspect exactly what was
// provided. The struct only reads and types the environment — range checking,
// defaulting, and clamping are the caller's responsibility.
// ============================================================================

// The kind of value an environment variable carries; selects which union
// member of ncclEpEnvVar::value is active and how it is dumped.
enum class ncclEpEnvType {
    flag,   // boolean toggle stored in value.flag
    ulong,  // unsigned long integer stored in value.ul
};

// A single environment variable bound to its name and type (both set at
// construction). `is_set` is true only when the variable is present with a
// valid value (an invalid value is warned about and left unset). `value` then
// holds the parsed result in the union member selected by `type`; callers must
// check `is_set` before reading it. The union lets more representations
// (double, string, ...) be added later without churn.
struct ncclEpEnvVar {
    const char* name;
    ncclEpEnvType type;
    bool is_set = false;
    union {
        unsigned long ul = 0;
        bool flag;
    } value;
};

struct ncclEpEnvConfig {
    int rank = 0;
    ncclEpEnvVar verbose{"NCCL_EP_ENV_VERBOSE", ncclEpEnvType::flag};
    ncclEpEnvVar debug{"NCCL_EP_DEBUG", ncclEpEnvType::flag};
    ncclEpEnvVar ht_em_local_dup{"NCCL_EP_HT_EM_LOCAL_DUP", ncclEpEnvType::flag};
    ncclEpEnvVar ht_em_nvlink_dup{"NCCL_EP_HT_EM_NVLINK_DUP", ncclEpEnvType::flag};
    ncclEpEnvVar disable_guard{"NCCL_EP_DISABLE_GUARD", ncclEpEnvType::flag};
    ncclEpEnvVar timeout_ms{"NCCL_EP_TIMEOUT_MS", ncclEpEnvType::ulong};
    ncclEpEnvVar comm_num_sms{"NCCL_EP_COMM_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar prolog_epilog_sms{"NCCL_EP_PROLOG_EPILOG_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar preprocess_num_sms{"NCCL_EP_PREPROCESS_NUM_SMS", ncclEpEnvType::ulong};
    ncclEpEnvVar tokens_per_chunk{"NCCL_EP_TOKENS_PER_CHUNK", ncclEpEnvType::ulong};
};

// True iff a flag variable was explicitly set to an "on" value (1/on/true).
// Unset, invalid, or explicitly-off all yield false.
inline bool nccl_ep_env_flag_on(const ncclEpEnvVar& var) {
    return var.is_set && var.value.flag;
}

// Verbose diagnostics was requested by the user
inline bool nccl_ep_env_verbose(const ncclEpEnvConfig& cfg) {
    return nccl_ep_env_flag_on(cfg.verbose) && cfg.rank == 0;
}

// Stamp this process's rank into *cfg.
inline void nccl_ep_env_set_rank(ncclEpEnvConfig* cfg, int rank) {
    if (cfg != nullptr) cfg->rank = rank;
}

// Read every NCCL EP environment variable into *cfg (resetting it first).
// Called once per group at creation time. No printing happens here.
void nccl_ep_env_init(ncclEpEnvConfig* cfg);

// Dump every environment variable (name + provided value, or "unset") to
// stderr. Always prints — the caller decides when (e.g. gated on verbose at
// group init, or unconditionally when reporting a misconfiguration).
void nccl_ep_env_print(const ncclEpEnvConfig& cfg);

#endif  // NCCL_EP_ENV_H_
