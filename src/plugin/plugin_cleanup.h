/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PLUGIN_CLEANUP_H_
#define NCCL_PLUGIN_CLEANUP_H_

typedef ncclResult_t (*ncclPluginFinalizeFn_t)(void* ctx);

static inline ncclResult_t ncclPluginFinalizeContext(ncclPluginFinalizeFn_t finalize, void** ctx) {
  if (*ctx == nullptr) return ncclSuccess;
  ncclResult_t ret = finalize(*ctx);
  *ctx = nullptr;
  return ret;
}

#endif
