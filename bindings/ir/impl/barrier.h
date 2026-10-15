/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
// Internal implementation fragment; include only from nccl_device_ir.cu
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

NCCL_IR_DEVICE_INLINE void ncclIrLsaBarrierSessionInit(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                       ncclIrDevComm const* comm, ncclTeam team,
                                                       ncclLsaBarrierHandle handle, uint32_t index, bool multimem,
                                                       ncclMultimemHandle mmHandle) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ::new (session) Session(coopImpl, commImpl, team, handle, index, multimem, mmHandle);
}
NCCL_IR_DEVICE_INLINE void ncclIrLsaBarrierSessionInitLsa(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                          ncclIrDevComm const* comm, uint32_t index, bool multimem) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ::new (session) Session(coopImpl, commImpl, ncclTeamTagLsa{}, index, multimem);
}
NCCL_IR_DEVICE_INLINE void ncclIrLsaBarrierSessionArrive(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                         cuda::memory_order order) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  auto& sessionImpl = *reinterpret_cast<Session*>(session);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  sessionImpl.arrive(coopImpl, order);
}
NCCL_IR_DEVICE_INLINE void ncclIrLsaBarrierSessionWait(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                       cuda::memory_order order) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  auto& sessionImpl = *reinterpret_cast<Session*>(session);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  sessionImpl.wait(coopImpl, order);
}
NCCL_IR_DEVICE_INLINE void ncclIrLsaBarrierSessionSync(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                                       cuda::memory_order order) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  auto& sessionImpl = *reinterpret_cast<Session*>(session);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  sessionImpl.sync(coopImpl, order);
}
NCCL_IR_DEVICE_INLINE void ncclIrLsaBarrierSessionDestroy(ncclIrLsaBarrierSession* session) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  reinterpret_cast<Session*>(session)->~Session();
}

NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionInit(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                       ncclIrGin const* net, ncclTeam team, ncclGinBarrierHandle handle,
                                                       uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  ::new (session) Session(coopImpl, netImpl, team, handle, index);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionInitRail(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                           ncclIrGin const* net, uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  ::new (session) Session(coopImpl, netImpl, ncclTeamTagRail{}, index);
}
NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionInitWorld(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                            ncclIrGin const* net, uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  ::new (session) Session(coopImpl, netImpl, ncclTeamTagWorld{}, index);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionInitAllContexts(ncclIrGinBarrierSession* session,
                                                                  ncclIrCoop const* coop, ncclIrDevComm const* comm,
                                                                  ncclTeam team, ncclGinBarrierHandle handle,
                                                                  uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ::new (session) Session(coopImpl, ncclGinAllContexts(commImpl), team, handle, index);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionInitAllContextsRail(
  ncclIrGinBarrierSession* session, ncclIrCoop const* coop, ncclIrDevComm const* comm, uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ::new (session) Session(coopImpl, ncclGinAllContexts(commImpl), ncclTeamTagRail{}, index);
}
NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionInitAllContextsWorld(
  ncclIrGinBarrierSession* session, ncclIrCoop const* coop, ncclIrDevComm const* comm, uint32_t index) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ::new (session) Session(coopImpl, ncclGinAllContexts(commImpl), ncclTeamTagWorld{}, index);
}
NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionSync(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                       cuda::memory_order order, ncclGinFenceLevel fence) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  auto& sessionImpl = *reinterpret_cast<Session*>(session);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  sessionImpl.sync(coopImpl, order, fence);
}
NCCL_IR_DEVICE_INLINE void ncclIrGinBarrierSessionDestroy(ncclIrGinBarrierSession* session) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  reinterpret_cast<Session*>(session)->~Session();
}

NCCL_IR_DEVICE_INLINE void ncclIrBarrierSessionInit(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                    ncclTeam innerTeam, ncclTeam outerTeam, ncclIrGin const* net,
                                                    ncclLsaBarrierHandle const innerBarHandle,
                                                    ncclGinBarrierHandle const outerBarHandle, uint32_t index,
                                                    bool multimem, ncclMultimemHandle const innerMmHandle) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  ::new (session)
    Session(coopImpl, innerTeam, outerTeam, netImpl, innerBarHandle, outerBarHandle, index, multimem, innerMmHandle);
}

NCCL_IR_DEVICE_INLINE void ncclIrBarrierSessionInitWorld(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                         ncclIrGin const* net, uint32_t index, bool multimem) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  ::new (session) Session(coopImpl, ncclTeamTagWorld{}, netImpl, index, multimem);
}
NCCL_IR_DEVICE_INLINE void ncclIrBarrierSessionInitLsa(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                       ncclIrDevComm const* comm, uint32_t index, bool multimem) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& commImpl = *reinterpret_cast<ncclDevComm const*>(comm);
  ::new (session) Session(coopImpl, ncclTeamTagLsa{}, commImpl, index, multimem);
}
NCCL_IR_DEVICE_INLINE void ncclIrBarrierSessionInitRail(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                        ncclIrGin const* net, uint32_t index) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  ::new (session) Session(coopImpl, ncclTeamTagRail{}, netImpl, index);
}
NCCL_IR_DEVICE_INLINE void ncclIrBarrierSessionSync(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                                    cuda::memory_order order, ncclGinFenceLevel fence) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  auto& sessionImpl = *reinterpret_cast<Session*>(session);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  sessionImpl.sync(coopImpl, order, fence);
}
NCCL_IR_DEVICE_INLINE void ncclIrBarrierSessionDestroy(ncclIrBarrierSession* session) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  reinterpret_cast<Session*>(session)->~Session();
}
