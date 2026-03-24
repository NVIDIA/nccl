/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Windows-only stub for nccl_device/gin.h. Provides minimal types so that
 * barrier/gin_barrier headers compile without pulling in real GIN device headers
 * (gin/gin_device_common.h etc.). Used only when building on Windows.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SESSION_H_
#define _NCCL_DEVICE_GIN_SESSION_H_

#if !defined(NCCL_OS_WINDOWS)
#error "gin_win_stub.h is for Windows builds only"
#endif

#include "core.h"
#include "net_device.h"

/* No GIN backends on Windows */
#define NCCL_GIN_BACKEND_MASK_ALL 0u

#if NCCL_CHECK_CUDACC
template<unsigned backendMask>
struct ncclGin_BackendMask {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  NCCL_DEVICE_INLINE ncclGin_BackendMask(ncclDevComm const& c, int contextIndex)
    : comm(c), nConnections(0), connectionId(0), _ginBackend(0), contextId(0) {}
};

template<ncclNetDeviceType backend>
using ncclGin_BackendOne = ncclGin_BackendMask<(1u<<(int)backend)>;

using ncclGin = ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>;
#endif

/* GIN scratch types (mirrors gin_scratch.h / gin_scratch__types.h) */
struct ncclGinOutboxHandle {
  ncclDevResourceHandle bufHandle;
  ncclGinCounter_t counter0;
  uint32_t size_log2;
};
struct ncclGinInboxA2AHandle {
  ncclDevResourceHandle bufHandle;
  ncclGinSignal_t signals;
  uint32_t size_log2;
  uint32_t nPeers_rcp32;
};
struct ncclGinSyncHandle {
  ncclGinSignal_t railSignals;
};

NCCL_EXTERN_C __host__ ncclResult_t ncclGinOutboxCreateRequirement(
  int nBlocks, int size_log2,
  ncclGinOutboxHandle* outHandle, ncclDevResourceRequirements* outReq
);
NCCL_EXTERN_C __host__ ncclResult_t ncclGinInboxA2ACreateRequirement(
  ncclTeam peers, int nBlocks, int size_log2,
  ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq
);

#endif /* _NCCL_DEVICE_GIN_SESSION_H_ */
