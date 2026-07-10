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

#include <stdint.h>
#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NCCL_PARAM_FLAG_NONE = 0,
  NCCL_PARAM_FLAG_PUBLISHED = 1ULL << 0, // public parameters in NCCL doc
  NCCL_PARAM_FLAG_DEPRECATED = 1ULL << 1,
  NCCL_PARAM_FLAG_CACHED = 1ULL << 2, // value cached, subsequent change has no effect
  NCCL_PARAM_FLAG_UNUSED = 1ULL << 3, // parameter has no effect
  NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT = 1ULL << 4 // special params that do not attempt to init
                                                // the EnvPlugin if it has not been initialized.
                                                // It will fallback to std::get_env().
} ncclParamFlag_t;

// Type IDs for param info. non-integers, non-boolean and non-const-char* is mapped to RAW type.
typedef enum {
  NCCL_PARAM_TYPE_I8 = 1,
  NCCL_PARAM_TYPE_I16,
  NCCL_PARAM_TYPE_I32,
  NCCL_PARAM_TYPE_I64,
  NCCL_PARAM_TYPE_U8,
  NCCL_PARAM_TYPE_U16,
  NCCL_PARAM_TYPE_U32,
  NCCL_PARAM_TYPE_U64,
  NCCL_PARAM_TYPE_BOOL,
  NCCL_PARAM_TYPE_CSTR,
  NCCL_PARAM_TYPE_RAW
} ncclParamTypeId_t;

// Parameter metadata. All const char* fields must point to string literals.
typedef struct {
  ncclParamTypeId_t typeId;
  uint64_t flags;
  const char* typeStr;
  const char* key;
  const char* desc;
} ncclParamInfo_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>

struct ncclParamInterface {
  virtual ~ncclParamInterface() = default;
  virtual ncclResult_t getRawData(void* out, int maxLen, int* len) = 0;
  virtual std::string toString() = 0;
  virtual std::string dump() = 0;
};
#endif

#endif /* PARAM_COMMON_H_INCLUDED */
