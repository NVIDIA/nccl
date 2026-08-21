/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <algorithm>
#include "dev_runtime_internal.h"
#include "comm.h"
#include "nccl_device/core.h"
#include "device.h"
#include "bootstrap.h"
#include "argcheck.h"
#include "param.h"

NCCL_PARAM(CftEnable, "CFT_ENABLE", 1);

ncclResult_t ncclGpuCftSupport(struct ncclComm* comm, int* gpuCftSupport, bool* gpuCftMulticastSupport,
                               bool* gpuCftCountedSupport) {
  *gpuCftSupport = 0;
  *gpuCftMulticastSupport = false;
  *gpuCftCountedSupport = false;
#if CUDA_VERSION >= 13030
  if (ncclParamCftEnable()) {
    int driverVersion;
    NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
    int unicastSupported = 0, multicastSupported = 0, countedSupported = 0;
    if (driverVersion >= 13030) {
      CUCHECK(cuDeviceGetAttribute(&unicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                                   (CUdevice)comm->cudaDev));
      CUCHECK(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED,
                                   (CUdevice)comm->cudaDev));
    }
    if (unicastSupported) {
      *gpuCftSupport = std::min(CUDA_VERSION, driverVersion);
    }
    if (multicastSupported) {
      *gpuCftMulticastSupport = (bool)multicastSupported;
    }
    if (unicastSupported || multicastSupported) {
      CUCHECK(cuDeviceGetAttribute(&countedSupported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
                                   (CUdevice)comm->cudaDev));
      *gpuCftCountedSupport = (bool)countedSupported;
    }
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
  if (ncclDevrWinRegEnabled(mem->winFlags, ncclDevrRegisterCft) && le != NCCL_LE_ID_INVALID &&
      !mem->globalHasSysmemSegment) {
#if CUDA_VERSION >= 13030
    CUCHECK(cuLogicalEndpointBindAddr(le, (CUdevice)comm->cudaDev, mem->bigOffset, mem->primaryAddr, mem->lsaMinSize,
                                      0));
#endif
  }
  return ncclSuccess;
}

ncclResult_t symUnbindTeamLe(struct ncclComm* comm, struct ncclDevrMemory* mem, ncclCftLeId le) {
  if (ncclDevrWinRegEnabled(mem->winFlags, ncclDevrRegisterCft) && le != NCCL_LE_ID_INVALID &&
      !mem->globalHasSysmemSegment) {
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
                               bool* /*needBarrier*/, bool counted) {
  // No need to request a barrier: local completion of waitLeReady guarantees that the remote LE is ready.
  // Remote completion of the binding is guaranteed by the bootstrapAllGather.
  ncclResult_t ret = ncclSuccess;

  // Always create the flat LE team to be stored in the devrState.
  ncclTeam_t flatTeam = ncclTeamCft(comm);
  int importedCount = 0, bindCount = 0, releaseCount = 0;
  // Counted LEs are limited to 256GB in size
  const size_t leSize = counted ? std::min(size_t(256ULL << 30), devr->bigSize) : devr->bigSize;
  ncclCftLeId leUcSelf = NCCL_LE_ID_INVALID;
  CUlogicalEndpointFabricHandle* leUcHandles = nullptr;
  int* cftRankList = nullptr;
  struct ncclDevrStateCftUc* cftUc = devr->le;
  if (cftUc[counted].baseId == NCCL_LE_ID_INVALID) {
    CUlogicalEndpointProp prop = {};
    prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    prop.flags = counted ? CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS : CU_LOGICAL_ENDPOINT_FLAG_NONE;
    prop.unicast.device = (CUdevice)comm->cudaDev;
    prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    prop.size = leSize;
    NCCLCHECK(checkLeSize(prop, leSize));
    NCCLCHECKGOTO(ncclCalloc(&leUcHandles, flatTeam.nRanks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&cftRankList, flatTeam.nRanks), ret, fail);

    // UC LD IDs are reserved in contiguous pairs, such that in devComm: counted ucLeId = devComm->ucLeId + nRanks
    bool ucLeIdsCreated = cftUc[!counted].baseId != NCCL_LE_ID_INVALID;
    if (ucLeIdsCreated) {
      cftUc[counted].baseId = counted ? cftUc[0].baseId + flatTeam.nRanks : cftUc[1].baseId - flatTeam.nRanks;
    } else {
      ncclCftLeId ucLeIdBase;
      CUCHECKGOTO(cuLogicalEndpointIdReserve(&ucLeIdBase, flatTeam.nRanks * 2), ret, fail);
      cftUc[counted].baseId = ucLeIdBase + ncclCftLeId(counted) * flatTeam.nRanks;
      releaseCount = flatTeam.nRanks * 2;
    }
    CUCHECKGOTO(cuLogicalEndpointCreate(cftUc[counted].baseId + flatTeam.rank, &prop), ret, fail);
    // leUcSelf is set if creation was successful
    leUcSelf = cftUc[counted].baseId + flatTeam.rank;
    NCCLCHECKGOTO(waitLeReady(leUcSelf, /*count=*/1), ret, fail);
    // bind all the windows already existing in the comm
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
      if ((bool)(mem->winFlags & NCCL_WIN_CFT_COUNTED) != counted) continue;
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
      CUCHECKGOTO(cuLogicalEndpointImport(cftUc[counted].baseId + r, &leUcHandles[r],
                                          CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC),
                  ret, fail);
      importedCount++;
    }
    NCCLCHECKGOTO(waitLeReady(cftUc[counted].baseId, flatTeam.nRanks), ret, fail);
    // no need of barrier, see above.
  }
  // Devr team stores the base LE for that specific team
  t->ucLeId[counted] = cftUc[counted].baseId + flatTeam.rank - t->team.rank * t->team.stride;

exit:
  free(cftRankList);
  free(leUcHandles);
  return ret;
fail:
  for (int r = 0; r < flatTeam.nRanks && importedCount > 0; r++) {
    if (r == flatTeam.rank) continue;
    CUCHECKIGNORE(cuLogicalEndpointDestroy(cftUc[counted].baseId + r));
    importedCount--;
  }
  for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr && bindCount > 0; mem = mem->next) {
    NCCLCHECKIGNORE(symUnbindTeamLe(comm, mem, leUcSelf), ret);
    bindCount--;
  }
  if (leUcSelf != NCCL_LE_ID_INVALID) CUCHECKIGNORE(cuLogicalEndpointDestroy(leUcSelf));
  if (releaseCount > 0)
    CUCHECKIGNORE(cuLogicalEndpointIdRelease(cftUc[counted].baseId - ncclCftLeId(counted) * flatTeam.nRanks,
                                             releaseCount));
  cftUc[counted].baseId = NCCL_LE_ID_INVALID;
  goto exit;
}

ncclResult_t symTeamObtainMcLe(struct ncclComm* comm, struct ncclDevrTeam* t, struct ncclDevrState* devr,
                               bool* needBarrier, bool counted) {
  if (!comm->nvlsSupport) {
    WARN("CFT multicast support requested, but NVLS is disabled or unsupported.");
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  ncclCftLeId mcLeIdBase = NCCL_LE_ID_INVALID;
  int bindCount = 0, releaseCount = 0;
  // Counted LEs are limited to 256GB in size
  const size_t leSize = counted ? std::min(size_t(256ULL << 30), devr->bigSize) : devr->bigSize;

  CUlogicalEndpointProp prop = {};
  prop.type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
  prop.flags = counted ? CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS : CU_LOGICAL_ENDPOINT_FLAG_NONE;
  prop.multicast.numDevices = t->team.nRanks;
  prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
  prop.size = leSize;
  NCCLCHECK(checkLeSize(prop, leSize));
  // MC LD IDs are reserved in contiguous pairs, such that in devComm: counted mcLeId = devComm->mcLeId + 1
  bool mcLeIdsCreated = t->mcLeId[!counted] != NCCL_LE_ID_INVALID;
  if (mcLeIdsCreated) {
    mcLeIdBase = counted ? t->mcLeId[0] + 1 : t->mcLeId[1] - 1;
  } else {
    CUCHECKGOTO(cuLogicalEndpointIdReserve(&mcLeIdBase, 2), ret, fail);
    mcLeIdBase += ncclCftLeId(counted);
    releaseCount = 2;
  }

  CUlogicalEndpointFabricHandle leMcHandle;
  if (t->team.rank == 0) {
    CUCHECKGOTO(cuLogicalEndpointCreate(mcLeIdBase, &prop), ret, fail);
    CUCHECKGOTO(cuLogicalEndpointExport(&leMcHandle, mcLeIdBase, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC), ret,
                fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, t->team.rank, t->team.nRanks, 0,
                                              &leMcHandle, sizeof(CUlogicalEndpointFabricHandle)),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, t->team.rank, t->team.nRanks, 0,
                                              &leMcHandle, sizeof(CUlogicalEndpointFabricHandle)),
                  ret, fail);
    CUCHECKGOTO(cuLogicalEndpointImport(mcLeIdBase, &leMcHandle, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC), ret,
                fail);
  }
  // LE is ready when all devices have joined the LE, it acts as a barrier
  CUCHECKGOTO(cuLogicalEndpointAddDevice(mcLeIdBase, (CUdevice)comm->cudaDev), ret, fail);
  NCCLCHECKGOTO(waitLeReady(mcLeIdBase, /*count=*/1), ret, fail);

  for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
    if ((bool)(mem->winFlags & NCCL_WIN_CFT_COUNTED) != counted) continue;
    NCCLCHECKGOTO(symBindTeamLe(comm, mem, mcLeIdBase), ret, fail);
    bindCount++;
  }
  // Barrier needed to guarantee that no peer will start using the MC LE before binding completes.
  if (needBarrier) *needBarrier = true;
  t->mcLeId[counted] = mcLeIdBase; // t->mcLeId is only set after import is successful

exit:
  return ret;
fail:
  for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr && bindCount > 0; mem = mem->next) {
    if ((bool)(mem->winFlags & NCCL_WIN_CFT_COUNTED) != counted) continue;
    NCCLCHECKIGNORE(symUnbindTeamLe(comm, mem, mcLeIdBase), ret);
    bindCount--;
  }
  if (mcLeIdBase != NCCL_LE_ID_INVALID) CUCHECKIGNORE(cuLogicalEndpointDestroy(mcLeIdBase));
  if (releaseCount > 0) CUCHECKIGNORE(cuLogicalEndpointIdRelease(mcLeIdBase - ncclCftLeId(counted), releaseCount));
  t->mcLeId[counted] = NCCL_LE_ID_INVALID;
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
  if (!ncclDevrWinRegEnabled(winHost->winFlags, ncclDevrRegisterCft)) {
    WARN("CFT access is disabled because the window was not registered for CFT.");
    return ncclInvalidUsage;
  }
  if (comm->gpuCftSupport == 0) {
    WARN("Using CFT query function without CFT support in the communicator.");
    return ncclInvalidArgument;
  }

  struct ncclDevrTeam* tm;
  bool needBarrier = false;
  bool counted = winHost->winFlags & NCCL_WIN_CFT_COUNTED;
  ncclTeam flatTeam = ncclTeamCftMultimem(comm);
  NCCLCHECK(symTeamObtain(comm, flatTeam, /*multimem=*/false, counted, /*uc=*/false, /*mc=*/true, &tm, &needBarrier));
  if (needBarrier) {
    NCCLCHECK(bootstrapIntraNodeBarrier(comm->bootstrap, tm->worldRankList, flatTeam.rank, flatTeam.nRanks, 0xbeef));
  }
  *leId = tm->mcLeId[counted];
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
  if (!ncclDevrWinRegEnabled(winHost->winFlags, ncclDevrRegisterCft)) {
    WARN("CFT access is disabled because the window was not registered for CFT.");
    return ncclInvalidUsage;
  }

  if (comm->gpuCftSupport == 0) {
    WARN("Using CFT query function without CFT support in the communicator.");
    return ncclInvalidArgument;
  }

  bool counted = winHost->winFlags & NCCL_WIN_CFT_COUNTED;
  ncclCftLeId baseLeId = comm->devrState.le[counted].baseId;
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
  if (!ncclDevrWinRegEnabled(winHost->winFlags, ncclDevrRegisterCft)) {
    WARN("CFT access is disabled because the window was not registered for CFT.");
    return ncclInvalidUsage;
  }

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

  bool counted = winHost->winFlags & NCCL_WIN_CFT_COUNTED;
  ncclCftLeId baseLeId = comm->devrState.le[counted].baseId;
  if (baseLeId == NCCL_LE_ID_INVALID) {
    WARN("Querying CFT LE before LE creation. Create a CFT-enabled devComm first or enable hostCftMode.");
    return ncclInvalidUsage;
  }

  *leId = baseLeId + flatTeam.rank + (peerWorld - comm->rank);
  *leOffset = winHost->bigOffset + offset;

  return ncclSuccess;
}
