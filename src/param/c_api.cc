/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// C API implementation for NcclParam access.
//
// This file implements the C-facing API declared in param/c_api.h.

#include "param/param.h"
#include "param/c_api.h"
#include "core.h"
#include "debug.h"

#include <cstdio>
#include <string>
#include <mutex>
#include <unordered_set>

USE_NCCL_PARAM(ncclParamDumpAllFlag, bool);

// Helper to unwrap opaque handle to NcclParamBase*
static inline NcclParamBase* unwrap(ncclParamHandle_t* h) {
  return reinterpret_cast<NcclParamBase*>(h);
}

static inline void ncclParamCheckFlag(uint64_t flags, const char* key) {
  static std::unordered_set<std::string> warningSet;

  // skip if this key was checked before, avoid repetitive warning messages
  if (warningSet.count(key)) {
    return;
  }

  warningSet.emplace(key);
  if (flags & NCCL_PARAM_FLAG_UNUSED) {
    INFO(NCCL_ENV, "Warning: ENV %s is unused. Setting it has no effect.", key);
  } else if (flags & NCCL_PARAM_FLAG_DEPRECATED) {
    INFO(NCCL_ENV, "Warning: ENV %s is deprecated. "
         "It may be removed or become unused in future versions.", key);
  }

  if (!(flags & NCCL_PARAM_FLAG_PUBLISHED)) {
    INFO(NCCL_ENV, "ENV: %s is private. Its function may change without notice across versions.", key);
  }
}

// ============================================================================
// Handle-based Access Interface
// ============================================================================

NCCL_API(ncclParamStatus_t, ncclParamBind, ncclParamHandle_t** out, const char* key);
ncclParamStatus_t ncclParamBind(ncclParamHandle_t** out, const char* key) {
  if (!out || !key) return NCCL_PARAM_BAD_ARGUMENT;
  NcclParamBase* param = NcclParamRegistry::find(key);
  if (!param) return NCCL_PARAM_NOT_FOUND;
  ncclParamCheckFlag(param->getFlags(), key);
  *out = reinterpret_cast<ncclParamHandle_t*>(param);
  return NCCL_PARAM_OK;
}

// ============================================================================
// Signed Integer Getters
// ============================================================================

NCCL_API(ncclParamStatus_t, ncclParamGetI8, ncclParamHandle_t* h, int8_t* out);
ncclParamStatus_t ncclParamGetI8(ncclParamHandle_t* h, int8_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getI8(out));
}

NCCL_API(ncclParamStatus_t, ncclParamGetI16, ncclParamHandle_t* h, int16_t* out);
ncclParamStatus_t ncclParamGetI16(ncclParamHandle_t* h, int16_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getI16(out));
}

NCCL_API(ncclParamStatus_t, ncclParamGetI32, ncclParamHandle_t* h, int32_t* out);
ncclParamStatus_t ncclParamGetI32(ncclParamHandle_t* h, int32_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getI32(out));
}

NCCL_API(ncclParamStatus_t, ncclParamGetI64, ncclParamHandle_t* h, int64_t* out);
ncclParamStatus_t ncclParamGetI64(ncclParamHandle_t* h, int64_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getI64(out));
}

// ============================================================================
// Unsigned Integer Getters
// ============================================================================

NCCL_API(ncclParamStatus_t, ncclParamGetU8, ncclParamHandle_t* h, uint8_t* out);
ncclParamStatus_t ncclParamGetU8(ncclParamHandle_t* h, uint8_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getU8(out));
}

NCCL_API(ncclParamStatus_t, ncclParamGetU16, ncclParamHandle_t* h, uint16_t* out);
ncclParamStatus_t ncclParamGetU16(ncclParamHandle_t* h, uint16_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getU16(out));
}

NCCL_API(ncclParamStatus_t, ncclParamGetU32, ncclParamHandle_t* h, uint32_t* out);
ncclParamStatus_t ncclParamGetU32(ncclParamHandle_t* h, uint32_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getU32(out));
}

NCCL_API(ncclParamStatus_t, ncclParamGetU64, ncclParamHandle_t* h, uint64_t* out);
ncclParamStatus_t ncclParamGetU64(ncclParamHandle_t* h, uint64_t* out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getU64(out));
}

// ============================================================================
// String Accessors
// ============================================================================

NCCL_API(ncclParamStatus_t, ncclParamGetStr, ncclParamHandle_t* h, const char** out);
ncclParamStatus_t ncclParamGetStr(ncclParamHandle_t* h, const char** out) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  // Thread-local buffer to keep the returned pointer valid for the caller
  static thread_local std::string buffer;
  buffer = unwrap(h)->toString();
  *out = buffer.c_str();
  return NCCL_PARAM_OK;
}

NCCL_API(ncclParamStatus_t, ncclParamGet, ncclParamHandle_t* h, void* out, int maxLen, int* len);
ncclParamStatus_t ncclParamGet(ncclParamHandle_t* h, void* out, int maxLen, int* len) {
  if (!h || !out) return NCCL_PARAM_BAD_ARGUMENT;
  return static_cast<ncclParamStatus_t>(unwrap(h)->getRawData(out, maxLen, len));
}

// ============================================================================
// Typeless Access Interface
// ============================================================================
NCCL_API(ncclParamStatus_t, ncclParamGetAllParameterKeys, const char*** table, int* tableLen);
ncclParamStatus_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen) {
  if (!table || !tableLen) return NCCL_PARAM_BAD_ARGUMENT;

  // Thread-local storage: keyTable owns string copies so ptrTable pointers remain
  // valid even if the registry map changes between calls.
  static thread_local std::vector<std::string> keyTable;
  static thread_local std::vector<const char*> ptrTable;

  std::lock_guard<std::mutex> regLock(NcclParamRegistry::mutex());
  auto& map = NcclParamRegistry::instance();

  keyTable.clear();
  keyTable.reserve(map.size());
  for (const auto& entry : map) {
    keyTable.emplace_back(entry.first.data(), entry.first.size());
  }

  ptrTable.clear();
  ptrTable.reserve(keyTable.size());
  for (const auto& s : keyTable) {
    ptrTable.push_back(s.c_str());
  }

  *table = ptrTable.data();
  *tableLen = static_cast<int>(ptrTable.size());
  return NCCL_PARAM_OK;
}

NCCL_API(ncclParamStatus_t, ncclParamGetParameter, const char* key, const char** value, int* valueLen);
ncclParamStatus_t ncclParamGetParameter(const char* key, const char** value, int* valueLen) {
  if (!key || !value || !valueLen) return NCCL_PARAM_BAD_ARGUMENT;

  // Thread-local buffer to keep the returned pointer valid for the caller
  static thread_local std::string resultHolder;

  NcclParamBase* param = NcclParamRegistry::find(key);

  if (param != nullptr) {
    ncclParamCheckFlag(param->getFlags(), key);
    resultHolder = param->toString();
    *value = resultHolder.c_str();
    *valueLen = static_cast<int>(resultHolder.length());
    return NCCL_PARAM_OK;
  } else {
    *value = nullptr;
    *valueLen = 0;
    return NCCL_PARAM_NOT_FOUND;
  }
}

NCCL_API(void, ncclParamDumpAll);
void ncclParamDumpAll(){
  bool showAll = ncclParamDumpAllFlag();
  printf("=== NcclParam Registry Dump ===\n");
  std::lock_guard<std::mutex> regLock(NcclParamRegistry::mutex());
  for (auto& entry : NcclParamRegistry::instance()) {
    std::string d = entry.second->dump(showAll);
    if (!d.empty()) {
      printf("%s\n", d.c_str());
    }
  }
}
