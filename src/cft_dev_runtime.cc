/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime_internal.h"
#include "comm.h"
#include "nccl_device/core.h"
#include "device.h"
#include "bootstrap.h"
#include "argcheck.h"
#include "param.h"

NCCL_PARAM(CftEnable, "CFT_ENABLE", 1);

ncclResult_t ncclGpuCftSupport(struct ncclComm* comm, int* gpuCftSupport) {
  *gpuCftSupport = 0;
#if CUDA_VERSION >= 13030
  if (ncclParamCftEnable()) {
    int driverVersion;
    NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
    int unicastSupported = 0, multicastSupported = 0;
    if (driverVersion >= 13030) {
      CUCHECK(cuDeviceGetAttribute(&unicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                                   (CUdevice)comm->cudaDev));
      CUCHECK(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED,
                                   (CUdevice)comm->cudaDev));
    }
    if (unicastSupported && multicastSupported) *gpuCftSupport = std::min(CUDA_VERSION, driverVersion);
  }
#endif
  return ncclSuccess;
}

int computeCftSize(struct ncclComm* comm) {
  if (comm->devrState.bigSize != 0) return comm->devrState.cftSize;
  int res = 1;
  if (comm->gpuCftSupport >= 13030) res = computeLsaSize(comm);
  return res;
}

int computeCftMcSize(struct ncclComm* comm) {
  if (comm->devrState.bigSize != 0) return comm->devrState.cftMcSize;
  int res = 1;
  if (comm->gpuCftSupport >= 13030) res = computeLsaSize(comm);
  return res;
}

ncclResult_t symBindTeamLe(struct ncclComm* comm, struct ncclDevrMemory* mem, ncclCftLeId le) {
  if (le != NCCL_LE_ID_INVALID && !mem->globalHasSysmemSegment) {
#if CUDA_VERSION >= 13030
    CUCHECK(cuLogicalEndpointBindAddr(le, (CUdevice)comm->cudaDev, mem->bigOffset, mem->primaryAddr, mem->lsaMinSize,
                                      0));
#endif
  }
  return ncclSuccess;
}

ncclResult_t symUnbindTeamLe(struct ncclComm* comm, struct ncclDevrMemory* mem, ncclCftLeId le) {
  if (le != NCCL_LE_ID_INVALID && !mem->globalHasSysmemSegment) {
#if CUDA_VERSION >= 13030
    CUCHECKIGNORE(cuLogicalEndpointUnbind(le, (CUdevice)comm->cudaDev, mem->bigOffset, mem->lsaMinSize));
#endif
  }
  return ncclSuccess;
}

static ncclResult_t checkLeSize(CUlogicalEndpointProp prop, size_t bigSize) {
  ncclResult_t ret = ncclSuccess;
  cuuint64_t bindAlignment = 0, maxSize = 0;
  CUCHECK(cuLogicalEndpointGetLimits(&bindAlignment, &maxSize, &prop));
  if (bigSize > maxSize || (bindAlignment > 0 && bigSize % bindAlignment != 0)) {
    WARN("CFT logical endpoint: bigSize=%zu incompatible with hardware limits "
         "(maxSize=%llu, bindAlignment=%llu).",
         bigSize, (unsigned long long)maxSize, (unsigned long long)bindAlignment);
    ret = ncclInvalidArgument;
  }
  return ret;
}

static ncclResult_t waitLeReady(ncclCftLeId le, uint32_t leCount) {
  while (true) {
    int query = 0;
    CUCHECK(cuLogicalEndpointQuery(le, leCount, &query));
    if (query) break;
  }
  return ncclSuccess;
}

ncclResult_t symTeamObtainUcLe(struct ncclComm* comm, struct ncclDevrTeam* t, struct ncclDevrState* devr,
                               bool* /*needBarrier*/) {
  // No need to request a barrier: local completion of waitLeReady guarantees that the remote LE is ready.
  // Remote completion of the binding is guaranteed by the bootstrapAllGather.
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrStateCftUc* cftUc = &devr->le;

  // Always create the flat LE team to be stored in the devrState.
  ncclTeam_t flatTeam = ncclTeamCft(comm);
  int importedCount = 0, bindCount = 0, releaseCount = 0;
  ncclCftLeId leUcSelf = NCCL_LE_ID_INVALID;
  CUlogicalEndpointFabricHandle* leUcHandles = nullptr;
  int* cftRankList = nullptr;
  if (cftUc->baseId == NCCL_LE_ID_INVALID) {
    CUlogicalEndpointProp prop = {};
    prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    prop.flags = CU_LOGICAL_ENDPOINT_FLAG_NONE;
    prop.unicast.device = (CUdevice)comm->cudaDev;
    prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    prop.size = devr->bigSize;
    NCCLCHECK(checkLeSize(prop, devr->bigSize));
    NCCLCHECKGOTO(ncclCalloc(&leUcHandles, flatTeam.nRanks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&cftRankList, flatTeam.nRanks), ret, fail);

    CUCHECKGOTO(cuLogicalEndpointIdReserve(&cftUc->baseId, flatTeam.nRanks), ret, fail);
    releaseCount = flatTeam.nRanks;
    CUCHECKGOTO(cuLogicalEndpointCreate(cftUc->baseId + flatTeam.rank, &prop), ret, fail);
    // leUcSelf is set if creation was succesfull
    leUcSelf = cftUc->baseId + flatTeam.rank;
    NCCLCHECKGOTO(waitLeReady(leUcSelf, /*count=*/1), ret, fail);
    // bind all the windows already existing in the comm
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
      NCCLCHECKGOTO(symBindTeamLe(comm, mem, leUcSelf), ret, fail);
      bindCount++;
    }
    CUCHECKGOTO(cuLogicalEndpointExport(&leUcHandles[flatTeam.rank], leUcSelf,
                                        CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC),
                ret, fail);

    for (int r = 0; r < flatTeam.nRanks; r++) cftRankList[r] = ncclTeamRankToWorld(comm, flatTeam, r);
    NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, cftRankList, flatTeam.rank, flatTeam.nRanks, leUcHandles,
                                              sizeof(CUlogicalEndpointFabricHandle)),
                  ret, fail);
    for (int r = 0; r < flatTeam.nRanks; r++) {
      if (r == flatTeam.rank) continue;
      CUCHECKGOTO(cuLogicalEndpointImport(cftUc->baseId + r, &leUcHandles[r],
                                          CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC),
                  ret, fail);
      importedCount++;
    }
    NCCLCHECKGOTO(waitLeReady(cftUc->baseId, flatTeam.nRanks), ret, fail);
    // no need of barrier, see above.
  }
  // Devr team stores the base LE for that specific team
  t->ucLeId = cftUc->baseId + flatTeam.rank - t->team.rank * t->team.stride;

exit:
  free(cftRankList);
  free(leUcHandles);
  return ret;
fail:
  for (int r = 0; r < flatTeam.nRanks && importedCount > 0; r++) {
    if (r == flatTeam.rank) continue;
    CUCHECKIGNORE(cuLogicalEndpointDestroy(cftUc->baseId + r));
    importedCount--;
  }
  for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr && bindCount > 0; mem = mem->next) {
    symUnbindTeamLe(comm, mem, leUcSelf);
    bindCount--;
  }
  if (leUcSelf != NCCL_LE_ID_INVALID) CUCHECKIGNORE(cuLogicalEndpointDestroy(leUcSelf));
  if (releaseCount > 0) CUCHECKIGNORE(cuLogicalEndpointIdRelease(cftUc->baseId, releaseCount));
  cftUc->baseId = NCCL_LE_ID_INVALID;
  goto exit;
}

ncclResult_t symTeamObtainMcLe(struct ncclComm* comm, struct ncclDevrTeam* t, struct ncclDevrState* devr,
                               bool* needBarrier) {
  if (!comm->nvlsSupport) {
    WARN("CFT multicast support requested, but NVLS is disabled or unsupported.");
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  ncclCftLeId leMcId = NCCL_LE_ID_INVALID;
  int bindCount = 0, releaseCount = 0;

  CUlogicalEndpointProp prop = {};
  prop.type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
  prop.flags = CU_LOGICAL_ENDPOINT_FLAG_NONE;
  prop.multicast.numDevices = t->team.nRanks;
  prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
  prop.size = devr->bigSize;
  NCCLCHECK(checkLeSize(prop, devr->bigSize));
  CUCHECKGOTO(cuLogicalEndpointIdReserve(&leMcId, 1), ret, fail);
  releaseCount = 1;

  CUlogicalEndpointFabricHandle leMcHandle;
  if (t->team.rank == 0) {
    CUCHECKGOTO(cuLogicalEndpointCreate(leMcId, &prop), ret, fail);
    t->mcLeId = leMcId; // t->mcLeId is only set after creation is succesfull
    CUCHECKGOTO(cuLogicalEndpointExport(&leMcHandle, leMcId, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC), ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, t->team.rank, t->team.nRanks, 0,
                                              &leMcHandle, sizeof(CUlogicalEndpointFabricHandle)),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, t->team.rank, t->team.nRanks, 0,
                                              &leMcHandle, sizeof(CUlogicalEndpointFabricHandle)),
                  ret, fail);
    CUCHECKGOTO(cuLogicalEndpointImport(leMcId, &leMcHandle, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC), ret, fail);
    t->mcLeId = leMcId; // t->mcLeId is only set after import is succesfull
  }
  // LE is ready when all devices have joined the LE, it acts as a barrier
  CUCHECKGOTO(cuLogicalEndpointAddDevice(leMcId, (CUdevice)comm->cudaDev), ret, fail);
  NCCLCHECKGOTO(waitLeReady(leMcId, /*count=*/1), ret, fail);

  for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
    NCCLCHECKGOTO(symBindTeamLe(comm, mem, leMcId), ret, fail);
    bindCount++;
  }
  // Barrier needed to guarantee that no peer will start using the MC LE before binding completes.
  if (needBarrier) *needBarrier = true;

exit:
  return ret;
fail:
  for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr && bindCount > 0; mem = mem->next) {
    symUnbindTeamLe(comm, mem, leMcId);
    bindCount--;
  }
  if (t->mcLeId != NCCL_LE_ID_INVALID) CUCHECKIGNORE(cuLogicalEndpointDestroy(t->mcLeId));
  if (releaseCount > 0) CUCHECKIGNORE(cuLogicalEndpointIdRelease(leMcId, releaseCount));
  t->mcLeId = NCCL_LE_ID_INVALID;
  goto exit;
}

NCCL_API(ncclResult_t, ncclGetMultimemDeviceLeInfo, ncclWindow_t window, size_t offset, ncclCftLeId* leId,
         size_t* leOffset);
ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId, size_t* leOffset) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(leId, __func__, "leId"));
  NCCLCHECK(PtrCheck(leOffset, __func__, "leOffset"));

  ncclComm_t comm = nullptr;
  struct ncclDevrWindow* winHost = nullptr;
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));
  if (comm->gpuCftSupport == 0) {
    WARN("Using CFT query function without CFT support in the communicator.");
    return ncclInvalidArgument;
  }

  struct ncclDevrTeam* tm;
  bool needBarrier = false;
  ncclTeam flatTeam = ncclTeamCftMultimem(comm);
  NCCLCHECK(symTeamObtain(comm, flatTeam, /*multimem=*/false, /*uc=*/false, /*mc=*/true, &tm, &needBarrier));
  if (needBarrier) {
    NCCLCHECK(bootstrapIntraNodeBarrier(comm->bootstrap, tm->worldRankList, flatTeam.rank, flatTeam.nRanks, 0xbeef));
  }
  *leId = tm->mcLeId;
  *leOffset = winHost->bigOffset + offset;

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetCftDeviceLeInfo, ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam,
         ncclCftLeId* leId, size_t* leOffset);
ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam,
                                    ncclCftLeId* leId, size_t* leOffset) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(leId, __func__, "leId"));
  NCCLCHECK(PtrCheck(leOffset, __func__, "leOffset"));

  // Get the host version of the device window
  ncclComm_t comm = nullptr;
  struct ncclDevrWindow* winHost = nullptr;
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));

  if (comm->gpuCftSupport == 0) {
    WARN("Using CFT query function without CFT support in the communicator.");
    return ncclInvalidArgument;
  }

  ncclCftLeId baseLeId = comm->devrState.le.baseId;
  if (baseLeId == NCCL_LE_ID_INVALID) {
    WARN("Querying CFT LE before LE creation. Create a CFT-enabled devComm first or enable hostCftMode.");
    return ncclInvalidUsage;
  }

  ncclTeam_t flatTeam = ncclTeamCft(comm, NCCL_CFT_TEAM_FLAT);
  *leId = baseLeId + flatTeam.rank + (peerCft - cftTeam.rank) * cftTeam.stride;
  *leOffset = winHost->bigOffset + offset;

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetPeerDeviceLeInfo, ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId,
         size_t* leOffset);
ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId,
                                     size_t* leOffset) {
  NCCLCHECK(PtrCheck(window, __func__, "window"));
  NCCLCHECK(PtrCheck(leId, __func__, "leId"));
  NCCLCHECK(PtrCheck(leOffset, __func__, "leOffset"));

  // Get the host version of the device window
  ncclComm_t comm = nullptr;
  struct ncclDevrWindow* winHost = nullptr;
  NCCLCHECK(findCommAndHostWindowFromDeviceWindow(window, &comm, &winHost));

  if (comm->gpuCftSupport == 0) {
    WARN("Using CFT query function without CFT support in the communicator.");
    return ncclInvalidArgument;
  }

  ncclTeam_t flatTeam = ncclTeamCft(comm, NCCL_CFT_TEAM_FLAT);
  int flatStart = comm->rank - flatTeam.rank;
  if (peerWorld < flatStart || peerWorld >= flatStart + flatTeam.nRanks) {
    WARN("Peer %d is not within the flat CFT team boundaries [%d, %d).", peerWorld, flatStart,
         flatStart + flatTeam.nRanks);
    return ncclInvalidArgument;
  }

  ncclCftLeId baseLeId = comm->devrState.le.baseId;
  if (baseLeId == NCCL_LE_ID_INVALID) {
    WARN("Querying CFT LE before LE creation. Create a CFT-enabled devComm first or enable hostCftMode.");
    return ncclInvalidUsage;
  }

  *leId = baseLeId + flatTeam.rank + (peerWorld - comm->rank);
  *leOffset = winHost->bigOffset + offset;

  return ncclSuccess;
}
