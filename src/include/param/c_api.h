/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// C-facing API for NcclParam access.
//
// This header is intended to be includable from both C and C++.
// It provides API and C11 _Generic convenience macros.
//

#ifndef PARAM_C_H_INCLUDED
#define PARAM_C_H_INCLUDED

#include <stdint.h>
#include "param/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Handle-based API (allow typed access)
// -----------------------------------------------------------------------------
typedef struct ncclParamHandle ncclParamHandle_t;

ncclParamStatus_t ncclParamBind(ncclParamHandle_t** out, const char* key);

// helper macro: binds a handle and defines handle_name##St status variable.
// Works in both C and C++.
#define BIND_NCCL_PARAM(handleName, key) \
  ncclParamHandle_t* handleName = (ncclParamHandle_t*)0; \
  ncclParamStatus_t handleName##St = ncclParamBind(&handleName, (key))

// Signed integers
ncclParamStatus_t ncclParamGetI8(ncclParamHandle_t* h, int8_t* out);
ncclParamStatus_t ncclParamGetI16(ncclParamHandle_t* h, int16_t* out);
ncclParamStatus_t ncclParamGetI32(ncclParamHandle_t* h, int32_t* out);
ncclParamStatus_t ncclParamGetI64(ncclParamHandle_t* h, int64_t* out);

// Unsigned integers
ncclParamStatus_t ncclParamGetU8(ncclParamHandle_t* h, uint8_t* out);
ncclParamStatus_t ncclParamGetU16(ncclParamHandle_t* h, uint16_t* out);
ncclParamStatus_t ncclParamGetU32(ncclParamHandle_t* h, uint32_t* out);
ncclParamStatus_t ncclParamGetU64(ncclParamHandle_t* h, uint64_t* out);

// String representation (always available). Returned pointer is owned by the
// parameter system and is valid until the next ncclParamGetStr() call on the
// same thread.
ncclParamStatus_t ncclParamGetStr(ncclParamHandle_t* h, const char** out);

// Get raw data (copied) of the parameter. User need to allocate the buffer.
ncclParamStatus_t ncclParamGet(ncclParamHandle_t* h, void* out, int maxLen, int* len);

// -----------------------------------------------------------------------------
// Key-based API (no handle required, typeless access, return value as string)
// -----------------------------------------------------------------------------

// Get parameter value as string by key. Returned pointer is owned by the
// parameter system and is valid until the next ncclParamGetParameter() call
// on the same thread.
ncclParamStatus_t ncclParamGetParameter(const char* key, const char** value, int* valueLen);

// Get all registered parameter keys. Returned pointer table is owned by the
// parameter system and is valid until the next ncclParamGetAllParameterKeys()
// call on the same thread.
ncclParamStatus_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen);

// Dump all parameters to log output.
void ncclParamDumpAll(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* PARAM_C_H_INCLUDED */

