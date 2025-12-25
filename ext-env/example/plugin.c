/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "nccl/env.h"

/**
 * Initialize the environment plugin
 *
 * This function is called by NCCL during initialization to set up the plugin.
 * It receives NCCL version information and a logging function for debug output.
 */
static ncclResult_t ncclEnvInit(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix) {
  return ncclSuccess;
}

/**
 * Finalize the environment plugin
 *
 * This function is called by NCCL during finalization to clean up plugin resources.
 */
static ncclResult_t ncclEnvFinalize(void) {
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
const ncclEnv_v1_t ncclEnvPlugin_v1 = {
  .name = "ncclEnvExample",
  .init = ncclEnvInit,
  .finalize = ncclEnvFinalize,
  .getEnv = ncclEnvGetEnv,
};
