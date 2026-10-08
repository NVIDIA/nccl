/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_IMPL_BARRIER_H_
#define _NCCL_DEVICE_WRAPPER_IMPL_BARRIER_H_

// Internal implementation fragment; include only from nccl_device_wrapper.cu
// after its prerequisite headers.

static_assert(sizeof(ncclIrLsaBarrierSession) == sizeof(ncclLsaBarrierSession<ncclCoopAny>),
              "ncclIrLsaBarrierSession size must match ncclLsaBarrierSession<ncclCoopAny>");
static_assert(alignof(ncclIrLsaBarrierSession) == alignof(ncclLsaBarrierSession<ncclCoopAny>),
              "ncclIrLsaBarrierSession alignment must match ncclLsaBarrierSession<ncclCoopAny>");
static_assert(sizeof(ncclIrGinBarrierSession) == sizeof(ncclGinBarrierSession<ncclCoopAny>),
              "ncclIrGinBarrierSession size must match ncclGinBarrierSession<ncclCoopAny>");
static_assert(alignof(ncclIrGinBarrierSession) == alignof(ncclGinBarrierSession<ncclCoopAny>),
              "ncclIrGinBarrierSession alignment must match ncclGinBarrierSession<ncclCoopAny>");
static_assert(sizeof(ncclIrBarrierSession) == sizeof(ncclBarrierSession<ncclCoopAny>),
              "ncclIrBarrierSession size must match ncclBarrierSession<ncclCoopAny>");
static_assert(alignof(ncclIrBarrierSession) == alignof(ncclBarrierSession<ncclCoopAny>),
              "ncclIrBarrierSession alignment must match ncclBarrierSession<ncclCoopAny>");

NCCL_DEVICE_INLINE size_t ncclIrLsaBarrierSessionSize() {
  return sizeof(ncclIrLsaBarrierSession);
}
NCCL_DEVICE_INLINE size_t ncclIrGinBarrierSessionSize() {
  return sizeof(ncclIrGinBarrierSession);
}
NCCL_DEVICE_INLINE size_t ncclIrBarrierSessionSize() {
  return sizeof(ncclIrBarrierSession);
}

NCCL_DEVICE_INLINE void ncclIrLsaBarrierSessionInit(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                    ncclDevComm const* comm, ncclTeam team, ncclLsaBarrierHandle handle,
                                                    uint32_t index, bool multimem, ncclMultimemHandle mmHandle) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  ::new (session) Session(*reinterpret_cast<ncclCoopAny const*>(coop), *comm, team, handle, index, multimem, mmHandle);
}
NCCL_DEVICE_INLINE void ncclIrLsaBarrierSessionArrive(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                      cuda::memory_order order) {
  reinterpret_cast<ncclLsaBarrierSession<ncclCoopAny>*>(session)->arrive(*reinterpret_cast<ncclCoopAny const*>(coop),
                                                                         order);
}
NCCL_DEVICE_INLINE void ncclIrLsaBarrierSessionWait(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                    cuda::memory_order order) {
  reinterpret_cast<ncclLsaBarrierSession<ncclCoopAny>*>(session)->wait(*reinterpret_cast<ncclCoopAny const*>(coop),
                                                                       order);
}
NCCL_DEVICE_INLINE void ncclIrLsaBarrierSessionSync(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                    cuda::memory_order order) {
  reinterpret_cast<ncclLsaBarrierSession<ncclCoopAny>*>(session)->sync(*reinterpret_cast<ncclCoopAny const*>(coop),
                                                                       order);
}
NCCL_DEVICE_INLINE void ncclIrLsaBarrierSessionDestroy(ncclIrLsaBarrierSession* session) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  reinterpret_cast<Session*>(session)->~Session();
}

NCCL_DEVICE_INLINE void ncclIrGinBarrierSessionInit(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                    ncclGin_C const* net, ncclTeam team, ncclGinBarrierHandle handle,
                                                    uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  ::new (session)
    Session(*reinterpret_cast<ncclCoopAny const*>(coop), reinterpret_cast<ncclGin const&>(*net), team, handle, index);
}

NCCL_DEVICE_INLINE void ncclIrGinBarrierSessionInitAllContexts(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                               ncclDevComm const* comm, ncclTeam team,
                                                               ncclGinBarrierHandle handle, uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  ::new (session) Session(*reinterpret_cast<ncclCoopAny const*>(coop), ncclGinAllContexts(*comm), team, handle, index);
}

NCCL_DEVICE_INLINE void ncclIrGinBarrierSessionSync(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                    cuda::memory_order order, ncclGinFenceLevel fence) {
  reinterpret_cast<ncclGinBarrierSession<ncclCoopAny>*>(session)->sync(*reinterpret_cast<ncclCoopAny const*>(coop),
                                                                       order, fence);
}
NCCL_DEVICE_INLINE void ncclIrGinBarrierSessionDestroy(ncclIrGinBarrierSession* session) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  reinterpret_cast<Session*>(session)->~Session();
}

NCCL_DEVICE_INLINE void ncclIrBarrierSessionInit(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                 ncclTeam innerTeam, ncclTeam outerTeam, ncclGin_C const* net,
                                                 ncclLsaBarrierHandle const innerBarHandle,
                                                 ncclGinBarrierHandle const outerBarHandle, uint32_t index,
                                                 bool multimem, ncclMultimemHandle const innerMmHandle) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  ::new (session)
    Session(*reinterpret_cast<ncclCoopAny const*>(coop), innerTeam, outerTeam, reinterpret_cast<ncclGin const&>(*net),
            innerBarHandle, outerBarHandle, index, multimem, innerMmHandle);
}

NCCL_DEVICE_INLINE void ncclIrBarrierSessionSync(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                 cuda::memory_order order, ncclGinFenceLevel fence) {
  reinterpret_cast<ncclBarrierSession<ncclCoopAny>*>(session)->sync(*reinterpret_cast<ncclCoopAny const*>(coop), order,
                                                                    fence);
}
NCCL_DEVICE_INLINE void ncclIrBarrierSessionDestroy(ncclIrBarrierSession* session) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  reinterpret_cast<Session*>(session)->~Session();
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_BARRIER_H_
