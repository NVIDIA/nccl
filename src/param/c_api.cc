/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// C API implementation for ncclParam access.
//
// This file implements the C-facing API declared in param/c_api.h.

#include "param/param.h"
#include "core.h"
#include "debug.h"

#include <cstdio>
#include <string>
#include <mutex>
#include <unordered_set>

USE_NCCL_PARAM(ncclParamDumpAllFlag, bool);

// Helper to unwrap opaque handle to ncclParamRegistry::mapEntry*
static inline ncclParamRegistry::mapEntry* unwrap(ncclParamHandle_t h) {
  return reinterpret_cast<ncclParamRegistry::mapEntry*>(h);
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
    INFO(NCCL_ENV,
         "Warning: ENV %s is deprecated. "
         "It may be removed or become unused in future versions.",
         key);
  }

  if (!(flags & NCCL_PARAM_FLAG_PUBLISHED)) {
    INFO(NCCL_ENV, "ENV: %s is private. Its function may change without notice across versions.", key);
  }
}

// ============================================================================
// Handle-based Access Interface
// ============================================================================

NCCL_API(ncclResult_t, ncclParamBind, ncclParamHandle_t* out, const char* key);
ncclResult_t ncclParamBind(ncclParamHandle_t* out, const char* key) {
  if (!out || !key) return ncclInvalidArgument;
  auto* entry = ncclParamRegistry::find(key);
  if (!entry) {
    INFO(NCCL_ENV, "PARAM: key \"%s\" not found in registry", key);
    return ncclInvalidArgument;
  }
  ncclParamCheckFlag(entry->info.flags, key);
  *out = reinterpret_cast<ncclParamHandle_t>(entry);
  return ncclSuccess;
}

// ============================================================================
// Typed Getters — dispatch via typeId + getRawData()
// ============================================================================

#define NCCL_PARAM_DEFINE_TYPED_GETTER(suffix, ctype, tid) \
  NCCL_API(ncclResult_t, ncclParamGet##suffix, ncclParamHandle_t h, ctype* out); \
  ncclResult_t ncclParamGet##suffix(ncclParamHandle_t h, ctype* out) { \
    if (!h || !out) return ncclInvalidArgument; \
    auto* e = unwrap(h); \
    if (e->info.typeId != (tid)) { \
      INFO(NCCL_ENV, "PARAM: type mismatch for key \"%s\"", e->info.key); \
      return ncclInvalidArgument; \
    } \
    int len = 0; \
    return e->param->getRawData(out, sizeof(*out), &len); \
  }

NCCL_PARAM_DEFINE_TYPED_GETTER(I8, int8_t, NCCL_PARAM_TYPE_I8)
NCCL_PARAM_DEFINE_TYPED_GETTER(I16, int16_t, NCCL_PARAM_TYPE_I16)
NCCL_PARAM_DEFINE_TYPED_GETTER(I32, int32_t, NCCL_PARAM_TYPE_I32)
NCCL_PARAM_DEFINE_TYPED_GETTER(I64, int64_t, NCCL_PARAM_TYPE_I64)
NCCL_PARAM_DEFINE_TYPED_GETTER(U8, uint8_t, NCCL_PARAM_TYPE_U8)
NCCL_PARAM_DEFINE_TYPED_GETTER(U16, uint16_t, NCCL_PARAM_TYPE_U16)
NCCL_PARAM_DEFINE_TYPED_GETTER(U32, uint32_t, NCCL_PARAM_TYPE_U32)
NCCL_PARAM_DEFINE_TYPED_GETTER(U64, uint64_t, NCCL_PARAM_TYPE_U64)

#undef NCCL_PARAM_DEFINE_TYPED_GETTER

// ============================================================================
// String Accessors
// ============================================================================

NCCL_API(ncclResult_t, ncclParamGetStr, ncclParamHandle_t h, const char** out);
ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char** out) {
  if (!h || !out) return ncclInvalidArgument;
  auto* e = unwrap(h);
  if (e->info.typeId != NCCL_PARAM_TYPE_CSTR) {
    INFO(NCCL_ENV, "PARAM: type mismatch for key \"%s\"", e->info.key);
    return ncclInvalidArgument;
  }
  // Thread-local buffer to keep the returned pointer valid for the caller
  static thread_local std::string buffer;
  buffer = unwrap(h)->param->toString();
  *out = buffer.c_str();
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclParamGet, ncclParamHandle_t h, void* out, int maxLen, int* len);
ncclResult_t ncclParamGet(ncclParamHandle_t h, void* out, int maxLen, int* len) {
  if (!h || !out) return ncclInvalidArgument;
  return unwrap(h)->param->getRawData(out, maxLen, len);
}

// ============================================================================
// Typeless Access Interface
// ============================================================================
NCCL_API(ncclResult_t, ncclParamGetAllParameterKeys, const char*** table, int* tableLen);
ncclResult_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen) {
  if (!table || !tableLen) return ncclInvalidArgument;

  // Thread-local storage: keyTable owns string copies so ptrTable pointers remain
  // valid even if the registry map changes between calls.
  static thread_local std::vector<std::string> keyTable;
  static thread_local std::vector<const char*> ptrTable;

  bool showAll = ncclParamDumpAllFlag();

  std::lock_guard<std::mutex> regLock(ncclParamRegistry::mutex());
  auto& map = ncclParamRegistry::instance();

  keyTable.clear();
  keyTable.reserve(map.size());
  for (const auto& kv : map) {
    if (!showAll && !(kv.second.info.flags & NCCL_PARAM_FLAG_PUBLISHED)) continue;
    keyTable.emplace_back(kv.first.data(), kv.first.size());
  }

  ptrTable.clear();
  ptrTable.reserve(keyTable.size());
  for (const auto& s : keyTable) {
    ptrTable.push_back(s.c_str());
  }

  *table = ptrTable.data();
  *tableLen = static_cast<int>(ptrTable.size());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclParamGetParameter, const char* key, const char** value, int* valueLen);
ncclResult_t ncclParamGetParameter(const char* key, const char** value, int* valueLen) {
  if (!key || !value || !valueLen) return ncclInvalidArgument;

  // Thread-local buffer to keep the returned pointer valid for the caller
  static thread_local std::string resultHolder;

  auto* entry = ncclParamRegistry::find(key);

  if (entry != nullptr) {
    ncclParamCheckFlag(entry->info.flags, key);
    resultHolder = entry->param->toString();
    *value = resultHolder.c_str();
    *valueLen = static_cast<int>(resultHolder.length());
    return ncclSuccess;
  } else {
    *value = nullptr;
    *valueLen = 0;
    INFO(NCCL_ENV, "PARAM: key \"%s\" not found in registry", key);
    return ncclInvalidArgument;
  }
}

NCCL_API(void, ncclParamDumpAll);
void ncclParamDumpAll() {
  bool showAll = ncclParamDumpAllFlag();
  printf("=== ncclParam Registry Dump ===\n");
  std::lock_guard<std::mutex> regLock(ncclParamRegistry::mutex());
  for (auto& kv : ncclParamRegistry::instance()) {
    if (!showAll && !(kv.second.info.flags & NCCL_PARAM_FLAG_PUBLISHED)) continue;
    printf("%s\n", kv.second.param->dump().c_str());
  }
}
