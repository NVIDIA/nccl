// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "runtime/config.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tide {

namespace {

const char* env(const char* name) { return std::getenv(name); }
int envInt(const char* v) { return std::atoi(v); }

}  // namespace

Config Config::fromEnv() {
  Config c;
  if (const char* v = env("TIDE_RMA_CTX"); v && envInt(v) > 0) c.rmaCtx = envInt(v);
  if (const char* v = env("TIDE_NIC_PACE")) c.paceMode = !std::strcmp(v, "auto") ? -1 : envInt(v);
  if (const char* v = env("TIDE_NIC_PACE_LEAD_MIB")) c.paceLead = static_cast<size_t>(std::max(0, envInt(v))) << 20;
  if (const char* v = env("TIDE_NIC_GROUPS")) c.nicGroups = std::max(1, envInt(v));
  if (const char* v = env("TIDE_COMM_CTAS")) c.commCtas = envInt(v);
  if (const char* v = env("TIDE_RS_CTAS")) { c.rsCtas = std::max(1, envInt(v)); c.envSet |= Config::kEnvRsCtas; }
  if (const char* v = env("TIDE_RS_MC_CTAS")) { c.rsMcCtas = std::max(0, envInt(v)); c.envSet |= Config::kEnvRsMcCtas; }
  if (const char* v = env("TIDE_RS_KWAY")) { c.rsKway = std::max(1, envInt(v)); c.envSet |= Config::kEnvRsKway; }
  if (const char* v = env("TIDE_RS_CHUNK_MIB")) {
    c.rsChunk = static_cast<size_t>(std::max(1, envInt(v))) << 20;
    c.envSet |= Config::kEnvRsChunk;
  }
  if (const char* v = env("TIDE_STRIDED_ROWS")) {
    const long rows = std::atol(v);
    c.stridedRows = rows <= 0 ? SIZE_MAX : static_cast<size_t>(rows);
  }
  if (const char* v = env("TIDE_RS_RAIL_PARTS")) c.rsRailParts = std::max(1, envInt(v));
  if (const char* v = env("TIDE_RS_PARTS")) {
    c.rsParts = std::clamp(envInt(v), 1, Config::kRsMaxParts);
    c.envSet |= Config::kEnvRsParts;
  }
  if (const char* v = env("TIDE_RS_OUTBOX")) {
    c.rsOutbox = std::max(0, envInt(v));
    c.envSet |= Config::kEnvRsOutbox;
  }
  if (const char* v = env("TIDE_RS_STAGE_SLOTS")) c.rsStageSlots = std::clamp(envInt(v), 2, Config::kRsMaxStageSlots);
  if (const char* v = env("TIDE_RS_KERNEL_PRIORITY")) c.rsKernelPriority = std::max(0, envInt(v));
  if (const char* v = env("TIDE_RS_PIPELINE")) {
    c.rsPipeline = std::clamp(envInt(v), 1, 2);
    c.envSet |= Config::kEnvRsPipeline;
  }
  if (const char* v = env("TIDE_RS_NET")) {
    if (!std::strcmp(v, "device")) c.rsDevctl = 1;
    else if (!std::strcmp(v, "host")) c.rsDevctl = 0;
    else throw std::invalid_argument(std::string("TIDE_RS_NET must be device or host, got '") + v + "'");
  }
  if (const char* v = env("TIDE_RS_NET_TIMEOUT_S")) c.rsDevctlTimeoutS = std::max(1.0, atof(v));
  if (const char* v = env("TIDE_RS_NET_DEBUG")) c.rsDevctlDebug = envInt(v) != 0;
  if (const char* v = env("TIDE_RS_NET_TRACE")) c.rsDevctlTrace = std::max(0, envInt(v));
  if (const char* v = env("TIDE_RS_WAVE_NODES")) c.rsWaveNodes = std::max(1, envInt(v));
  if (const char* v = env("TIDE_RS_DEBUG")) c.rsDebug = envInt(v);
  if (const char* v = env("TIDE_TRACE_AG")) c.traceAg = envInt(v) != 0;
  return c;
}

const Config& envConfig() {
  static const Config c = Config::fromEnv();
  return c;
}

}  // namespace tide
