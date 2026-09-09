/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "nccl_ep_env.h"

#include <cstdio>
#include <cstdlib>
#include <strings.h>  // strcasecmp

namespace {

// Parse a boolean flag environment variable into var (is_set + value.flag).
// Accepts only "1"/"on"/"true" (=> true) or "0"/"off"/"false" (=> false),
// case-insensitive. Unset/empty leaves it not-set; any other value is warned
// about and also left not-set.
void parse_flag(ncclEpEnvVar& var) {
    const char* v = std::getenv(var.name);
    if (v == nullptr || v[0] == '\0') return;  // unset -> is_set stays false
    if (strcasecmp(v, "1") == 0 || strcasecmp(v, "on") == 0 || strcasecmp(v, "true") == 0) {
        var.is_set = true;
        var.value.flag = true;
    } else if (strcasecmp(v, "0") == 0 || strcasecmp(v, "off") == 0 || strcasecmp(v, "false") == 0) {
        var.is_set = true;
        var.value.flag = false;
    } else {
        std::fprintf(stderr, "[nccl_ep] %s=%s ignored (expected 1/on/true or 0/off/false)\n", var.name, v);
    }
}

// Parse an unsigned-long environment variable into var (is_set + value.ul).
// Unset/empty leaves it not-set; otherwise the raw strtoul() value is stored
// without interpretation — validity/range is the consumer's responsibility.
void parse_ulong(ncclEpEnvVar& var) {
    const char* v = std::getenv(var.name);
    if (v == nullptr || v[0] == '\0') return;  // unset/empty -> not set
    var.is_set = true;
    var.value.ul = std::strtoul(v, nullptr, 10);
}

}  // namespace

void nccl_ep_env_init(ncclEpEnvConfig* cfg) {
    if (cfg == nullptr) return;
    *cfg = ncclEpEnvConfig{};  // reset to defaults (re-binds the names) before reading

    // Boolean flags: is_set means present-and-valid, value.flag holds the bool.
    parse_flag(cfg->verbose);
    parse_flag(cfg->debug);
    parse_flag(cfg->ht_em_local_dup);
    parse_flag(cfg->ht_em_nvlink_dup);
    parse_flag(cfg->disable_guard);

    // Numeric (ulong) vars: is_set means present, value.ul holds the raw integer
    // (no range checks here — consumers in nccl_ep.cc validate per their needs).
    parse_ulong(cfg->timeout_ms);
    parse_ulong(cfg->comm_num_sms);
    parse_ulong(cfg->prolog_epilog_sms);
    parse_ulong(cfg->preprocess_num_sms);
    parse_ulong(cfg->tokens_per_chunk);
}

void nccl_ep_env_print(const ncclEpEnvConfig& cfg) {
    // Every variable, so the user can see exactly what was provided. Adding a
    // field to ncclEpEnvConfig only requires adding it to this list.
    const ncclEpEnvVar* vars[] = {
        &cfg.verbose,
        &cfg.debug,
        &cfg.ht_em_local_dup,
        &cfg.ht_em_nvlink_dup,
        &cfg.disable_guard,
        &cfg.timeout_ms,
        &cfg.comm_num_sms,
        &cfg.prolog_epilog_sms,
        &cfg.preprocess_num_sms,
        &cfg.tokens_per_chunk,
    };

    std::fprintf(stderr, "[nccl_ep][env] NCCL EP environment configuration:\n");
    for (const ncclEpEnvVar* v : vars) {
        if (!v->is_set) {
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = unset\n", v->name);
            continue;
        }
        switch (v->type) {
        case ncclEpEnvType::flag:
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = %s\n", v->name, v->value.flag ? "enabled" : "disabled");
            break;
        case ncclEpEnvType::ulong:
            std::fprintf(stderr, "[nccl_ep][env]   %-28s = %lu\n", v->name, v->value.ul);
            break;
        }
    }
}
