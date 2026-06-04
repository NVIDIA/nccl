/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "nccl/env.h"

static ncclDebugLogger_t logFunction = NULL;

/**
 * Initialize the environment plugin
 *
 * This function is called by NCCL during initialization to set up the plugin.
 * It receives NCCL version information and a logging function for debug output.
 */
static ncclResult_t ncclEnvInit(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix, ncclDebugLogger_t logFn) {
  logFunction = logFn;
  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_ENV, __FILE__, __LINE__,
                "ENV/Plugin: ncclEnvExample initialized for NCCL %d.%d.%d%s",
                ncclMajor, ncclMinor, ncclPatch, suffix ? suffix : "");
  }
  return ncclSuccess;
}

/**
 * Finalize the environment plugin
 *
 * This function is called by NCCL during finalization to clean up plugin resources.
 */
static ncclResult_t ncclEnvFinalize(void) {
  logFunction = NULL;
  return ncclSuccess;
}

/**
 * Get environment variable value
 *
 * This function is called by NCCL whenever it needs to retrieve an environment variable.
 * It delegates to the system getenv function and provides optional logging.
 *
 * @param name The name of the environment variable to retrieve
 * @return Pointer to the environment variable value string, or NULL if not found
 */
static const char* ncclEnvGetEnv(const char* name) {
  return getenv(name);
}

/**
 * Export the plugin structure
 *
 * This structure must be exported with the correct symbol name for NCCL to find it.
 * The symbol name should match NCCL_ENV_PLUGIN_SYMBOL defined in nccl_env.h.
 */
const ncclEnv_v2_t ncclEnvPlugin_v2 = {
  .name = "ncclEnvExample",
  .init = ncclEnvInit,
  .finalize = ncclEnvFinalize,
  .getEnv = ncclEnvGetEnv,
};
