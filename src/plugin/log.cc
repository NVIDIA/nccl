/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include <string.h>

#include "checks.h"
#include "debug.h"
#include "env.h"
#include "nccl_log.h"
#include "os.h"
#include "plugin.h"

static void* logPluginLib = nullptr;
static ncclLog_t* ncclLogPlugin = nullptr;

// Closes the plugin library and forgets it. Callers must have detached the sink first.
static void logPluginUnload(void) {
  ncclLogPlugin = nullptr;
  if (logPluginLib) {
    (void)ncclClosePluginLib(logPluginLib, ncclPluginTypeLog);
    logPluginLib = nullptr;
  }
}

// Loads a log plugin and installs it as the destination for NCCL's own log records.
//
// Ordering caveat, shared with the env plugin: opening the library logs, and those records are emitted
// before the sink is installed, so they go to NCCL_DEBUG_FILE or stdout. A plugin therefore sees
// everything logged after it is loaded, not the handful of records produced while loading it.
//
// Loading runs from ncclInitEnv() rather than from the logging path, so a plugin that logs during its own
// init cannot re-enter the logger that is installing it. The converse constraint applies to the plugin: it
// runs inside the std::call_once that guards ncclInitEnv(), so its init() must not call a public NCCL
// entry point -- ncclCommInitRank, ncclGetUniqueId, ncclMemAlloc and friends all route back through
// ncclInitEnv() and would re-enter the same once-flag on the same thread.
//
// A failure to load is never fatal: NCCL keeps its default output and the job continues.
ncclResult_t ncclLogPluginInit(void) {
  if (ncclLogPlugin != nullptr) return ncclSuccess;

  // Read through the env plugin, which is already initialized at this point, rather than std::getenv:
  // a site that supplies configuration from an env plugin must be able to select the log plugin too.
  // ncclGetEnv() would be wrong here -- it calls ncclInitEnv(), which is the call_once we are inside.
  const char* logName = ncclEnvPluginGetEnv("NCCL_LOG_PLUGIN");
  if (logName == nullptr) {
    // No log plugin is the norm. Without the variable set, do not probe the filesystem on every job.
    return ncclSuccess;
  }
  INFO(NCCL_ENV, "NCCL_LOG_PLUGIN set by environment to %s", logName);
  if (strcasecmp(logName, "none") == 0) return ncclSuccess;

  logPluginLib = ncclOpenLogPluginLib(logName);
  if (logPluginLib == nullptr) {
    logPluginUnload();
    return ncclSuccess;
  }
  if (ncclPluginLibPaths[ncclPluginTypeLog]) logName = ncclPluginLibPaths[ncclPluginTypeLog];

  ncclLog_t* plugin = (ncclLog_t*)ncclOsDlsym(logPluginLib, NCCL_LOG_PLUGIN_SYMBOL_NAME);
  if (plugin == nullptr) {
    ATTN("External log plugin %s does not export %s", logName, NCCL_LOG_PLUGIN_SYMBOL_NAME);
    logPluginUnload();
    return ncclSuccess;
  }

  // Handing the plugin's own struct to the public registration keeps one code path for both ways of
  // supplying a sink, so a plugin is not special-cased inside the logger.
  ncclResult_t res = ncclSetDebugLogSink(plugin);
  if (res == ncclInvalidUsage) {
    // The application registered its own sink before the first NCCL call. First claimant wins: the
    // plugin is not loaded, rather than displacing a sink the application is already using.
    ATTN("Not loading log plugin %s: a log sink is already registered", logName);
    logPluginUnload();
    return ncclSuccess;
  }
  if (res != ncclSuccess) {
    ATTN("External log plugin %s failed to initialize: %s", logName, ncclGetErrorString(res));
    logPluginUnload();
    return ncclSuccess;
  }

  ncclLogPlugin = plugin;
  INFO(NCCL_INIT, "Successfully loaded external log plugin %s (%s)", logName, plugin->name ? plugin->name : "unnamed");
  return ncclSuccess;
}
