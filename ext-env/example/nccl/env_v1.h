/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef ENV_V1_H_
#define ENV_V1_H_

#include "err.h"

typedef struct {
  const char* name;
  // Initialize the environment plugin
  // Input
  //  - ncclMajor: NCCL major version number
  //  - ncclMinor: NCCL minor version number
  //  - ncclPatch: NCCL patch version number
  //  - suffix: NCCL version suffix string
  ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix);
  // Finalize the environment plugin
  ncclResult_t (*finalize)(void);
  // Get environment variable value
  // Input
  //  - name: environment variable name
  // Output
  //  - returns: pointer to environment variable value string, or NULL if not found. The plugin is responsible for keeping the
  //             returned value (address) valid until it is no longer needed by NCCL. This happens when NCCL calls ``finalize``
  //             or ``getEnv`` again on the same variable name. In any other case, modifying the variable (e.g., through
  //             ``setenv``) is considered undefined behavior since NCCL might access the returned address after the plugin has
  //             reset the variable.
  const char* (*getEnv)(const char* name);
} ncclEnv_v1_t;

#endif
