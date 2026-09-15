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

NCCL_DEVICE_INLINE size_t ncclLsaBarrierSession_C_size() {
  return sizeof(ncclLsaBarrierSession_C);
}
NCCL_DEVICE_INLINE size_t ncclGinBarrierSession_C_size() {
  return sizeof(ncclGinBarrierSession_C);
}
NCCL_DEVICE_INLINE size_t ncclBarrierSession_C_size() {
  return sizeof(ncclBarrierSession_C);
}

NCCL_DEVICE_INLINE void ncclLsaBarrierSessionInit(ncclLsaBarrierSession_C* session, ncclCoopAny coop,
                                                  ncclDevComm const& comm, ncclTeam team, ncclLsaBarrierHandle handle,
                                                  uint32_t index, bool multimem, ncclMultimemHandle mmHandle) {
  ::new (&(session->bar)) ncclLsaBarrierSession<ncclCoopAny>(coop, comm, team, handle, index, multimem, mmHandle);
}
NCCL_DEVICE_INLINE void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session, ncclCoopAny coop,
                                                    cuda::memory_order order) {
  session->bar.arrive(coop, order);
}
NCCL_DEVICE_INLINE void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session, ncclCoopAny coop,
                                                  cuda::memory_order order) {
  session->bar.wait(coop, order);
}
NCCL_DEVICE_INLINE void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session, ncclCoopAny coop,
                                                  cuda::memory_order order) {
  session->bar.sync(coop, order);
}
NCCL_DEVICE_INLINE void ncclLsaBarrierSessionDestroy(ncclLsaBarrierSession_C* session) {
  using Session = ncclLsaBarrierSession<ncclCoopAny>;
  session->bar.~Session();
}

NCCL_DEVICE_INLINE void ncclGinBarrierSessionInit(ncclGinBarrierSession_C* session, ncclCoopAny coop,
                                                  ncclGin_C const* net, ncclTeam team, ncclGinBarrierHandle handle,
                                                  uint32_t index) {
  ::new (&(session->bar))
    ncclGinBarrierSession<ncclCoopAny>(coop, reinterpret_cast<ncclGin const&>(*net), team, handle, index);
}

NCCL_DEVICE_INLINE void ncclGinBarrierSessionInitAllContexts(ncclGinBarrierSession_C* session, ncclCoopAny coop,
                                                             ncclDevComm const& comm, ncclTeam team,
                                                             ncclGinBarrierHandle handle, uint32_t index) {
  ::new (&(session->bar)) ncclGinBarrierSession<ncclCoopAny>(coop, ncclGinAllContexts(comm), team, handle, index);
}

NCCL_DEVICE_INLINE void ncclGinBarrierSessionSync(ncclGinBarrierSession_C* session, ncclCoopAny coop,
                                                  cuda::memory_order order, ncclGinFenceLevel fence) {
  session->bar.sync(coop, order, fence);
}
NCCL_DEVICE_INLINE void ncclGinBarrierSessionDestroy(ncclGinBarrierSession_C* session) {
  using Session = ncclGinBarrierSession<ncclCoopAny>;
  session->bar.~Session();
}

NCCL_DEVICE_INLINE void ncclBarrierSessionInit(ncclBarrierSession_C* session, ncclCoopAny coop, ncclTeam innerTeam,
                                               ncclTeam outerTeam, ncclGin_C const* net,
                                               ncclLsaBarrierHandle const innerBarHandle,
                                               ncclGinBarrierHandle const outerBarHandle, uint32_t index, bool multimem,
                                               ncclMultimemHandle const innerMmHandle) {
  ::new (&(session->bar))
    ncclBarrierSession<ncclCoopAny>(coop, innerTeam, outerTeam, reinterpret_cast<ncclGin const&>(*net), innerBarHandle,
                                    outerBarHandle, index, multimem, innerMmHandle);
}

NCCL_DEVICE_INLINE void ncclBarrierSessionSync(ncclBarrierSession_C* session, ncclCoopAny coop,
                                               cuda::memory_order order, ncclGinFenceLevel fence) {
  session->bar.sync(coop, order, fence);
}
NCCL_DEVICE_INLINE void ncclBarrierSessionDestroy(ncclBarrierSession_C* session) {
  using Session = ncclBarrierSession<ncclCoopAny>;
  session->bar.~Session();
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_BARRIER_H_
