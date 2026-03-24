/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Common types shared between param.h (C++) and param_c.h (C API).
//
// This header is intended to be includable from both C and C++.

#ifndef PARAM_COMMON_H_INCLUDED
#define PARAM_COMMON_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

// Status codes for NcclParam C API functions
typedef enum {
  NCCL_PARAM_OK             = 0,   // Success
  NCCL_PARAM_NOT_FOUND      = 1,   // Parameter key not found in registry
  NCCL_PARAM_TYPE_MISMATCH  = 2,   // Requested type does not match parameter type
  NCCL_PARAM_VALUE_CACHED   = 3,   // Cached parameter, rejects sets after initialized
  NCCL_PARAM_INVALID_VALUE  = 4,   // Value rejected by parser (out of range, invalid format)
  NCCL_PARAM_BAD_ARGUMENT   = 5,   // Invalid argument (null pointer, etc.)
} ncclParamStatus_t;

// Source provenance: how a parameter's value was set
typedef enum {
  NCCL_PARAM_SOURCE_DEFAULT     = 0,
  NCCL_PARAM_SOURCE_ENV_VAR     = 1,  // Stub - for EnvVar that cannot be set by EnvPlugins
  NCCL_PARAM_SOURCE_ENV_PLUGIN  = 2,
  NCCL_PARAM_SOURCE_CONFIG_FILE = 3,  // Stub - not yet implemented
} ncclParamSource_t;

#ifdef __cplusplus
}
#endif

#endif /* PARAM_COMMON_H_INCLUDED */
