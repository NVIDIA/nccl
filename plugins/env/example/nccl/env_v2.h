/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef ENV_V2_H_
#define ENV_V2_H_

#include "err.h"
#include "common.h"

typedef struct {
  const char* name;
  // Initialize the environment plugin
  // Input
  //  - ncclMajor: NCCL major version number
  //  - ncclMinor: NCCL minor version number
  //  - ncclPatch: NCCL patch version number
  //  - suffix: NCCL version suffix string
  //  - logFunction: NCCL debug logging function for plugin diagnostics
  ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix, ncclDebugLogger_t logFunction);
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
} ncclEnv_v2_t;

#endif
