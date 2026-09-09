# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.31.2. Do not modify it directly.


# <<<< PREAMBLE CONTENT >>>>

cimport cython as _cyb_cython
from libc.stdint cimport (
    int16_t,
    int32_t,
    int64_t,
    int8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    uint8_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from ._internal cimport nccl as _nccl


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t ncclMemAlloc(void** ptr, size_t size) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclMemAlloc(ptr, size)


cdef ncclResult_t ncclMemFree(void* ptr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclMemFree(ptr)


cdef ncclResult_t ncclGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetVersion(version)


cdef ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetUniqueId(uniqueId)


cdef ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommInitRankConfig(comm, nranks, commId, rank, config)


cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommInitRank(comm, nranks, commId, rank)


cdef ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommInitAll(comm, ndev, devlist)


cdef ncclResult_t ncclCommFinalize(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommFinalize(comm)


cdef ncclResult_t ncclCommDestroy(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommDestroy(comm)


cdef ncclResult_t ncclCommAbort(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommAbort(comm)


cdef ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommRevoke(comm, revokeFlags)


cdef ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommSplit(comm, color, key, newcomm, config)


cdef ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommShrink(comm, excludeRanksList, excludeRanksCount, newcomm, config, shrinkFlags)


cdef ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommGetUniqueId(comm, uniqueId)


cdef ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommGrow(comm, nRanks, uniqueId, rank, newcomm, config)


cdef ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommInitRankScalable(newcomm, nranks, myrank, nId, commIds, config)


cdef const char* ncclGetErrorString(ncclResult_t result) except?NULL nogil:
    return _nccl._ncclGetErrorString(result)


cdef const char* ncclGetLastError(ncclComm_t comm) except?NULL nogil:
    return _nccl._ncclGetLastError(comm)


cdef ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommGetAsyncError(comm, asyncError)


cdef ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommCount(comm, count)


cdef ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommCuDevice(comm, device)


cdef ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommUserRank(comm, rank)


cdef ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommRegister(comm, buff, size, handle)


cdef ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommDeregister(comm, handle)


cdef ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommSuspend(comm, flags)


cdef ncclResult_t ncclCommResume(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommResume(comm)


cdef ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommMemStats(comm, stat, value)


cdef ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommWindowRegister(comm, buff, size, win, winFlags)


cdef ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommWindowDeregister(comm, win)


cdef ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void** outUserPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclWinGetUserPtr(comm, win, outUserPtr)


cdef ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)


cdef ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclRedOpDestroy(op, comm)


cdef ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)


cdef ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclBcast(buff, count, datatype, root, comm, stream)


cdef ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)


cdef ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)


cdef ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)


cdef ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)


cdef ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream)


cdef ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream)


cdef ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream)


cdef ncclResult_t ncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclAllReduceConfig(sendbuff, recvbuff, count, datatype, op, comm, stream, config)


cdef ncclResult_t ncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclBroadcastConfig(sendbuff, recvbuff, count, datatype, root, comm, stream, config)


cdef ncclResult_t ncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclReduceConfig(sendbuff, recvbuff, count, datatype, op, root, comm, stream, config)


cdef ncclResult_t ncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclAllGatherConfig(sendbuff, recvbuff, sendcount, datatype, comm, stream, config)


cdef ncclResult_t ncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclReduceScatterConfig(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, config)


cdef ncclResult_t ncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclAlltoAllConfig(sendbuff, recvbuff, count, datatype, comm, stream, config)


cdef ncclResult_t ncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGatherConfig(sendbuff, recvbuff, count, datatype, root, comm, stream, config)


cdef ncclResult_t ncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclScatterConfig(sendbuff, recvbuff, count, datatype, root, comm, stream, config)


cdef ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclSend(sendbuff, count, datatype, peer, comm, stream)


cdef ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclRecv(recvbuff, count, datatype, peer, comm, stream)


cdef ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclPutSignal(localbuff, count, datatype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)


cdef ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclSignal(peer, sigIdx, ctx, flags, comm, stream)


cdef ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclWaitSignal(nDesc, signalDescs, comm, stream)


cdef ncclResult_t ncclGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGroupStart()


cdef ncclResult_t ncclGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGroupEnd()


cdef ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGroupSimulateEnd(simInfo)


cdef ncclResult_t ncclParamBind(ncclParamHandle_t* out, const char* key) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamBind(out, key)


cdef ncclResult_t ncclParamGetI8(ncclParamHandle_t h, int8_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetI8(h, out)


cdef ncclResult_t ncclParamGetI16(ncclParamHandle_t h, int16_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetI16(h, out)


cdef ncclResult_t ncclParamGetI32(ncclParamHandle_t h, int32_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetI32(h, out)


cdef ncclResult_t ncclParamGetI64(ncclParamHandle_t h, int64_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetI64(h, out)


cdef ncclResult_t ncclParamGetU8(ncclParamHandle_t h, uint8_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetU8(h, out)


cdef ncclResult_t ncclParamGetU16(ncclParamHandle_t h, uint16_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetU16(h, out)


cdef ncclResult_t ncclParamGetU32(ncclParamHandle_t h, uint32_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetU32(h, out)


cdef ncclResult_t ncclParamGetU64(ncclParamHandle_t h, uint64_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetU64(h, out)


cdef ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char** out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetStr(h, out)


cdef ncclResult_t ncclParamGet(ncclParamHandle_t h, void* out, int maxLen, int* len) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGet(h, out, maxLen, len)


cdef ncclResult_t ncclParamGetParameter(const char* key, const char** value, int* valueLen) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetParameter(key, value, valueLen)


cdef ncclResult_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclParamGetAllParameterKeys(table, tableLen)


@_cyb_cython.show_performance_hints(False)
cdef void ncclParamDumpAll() except* nogil:
    _nccl._ncclParamDumpAll()


cdef ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclCommQueryProperties(comm, props)


cdef ncclResult_t ncclDevCommCreate(ncclComm_t comm, const ncclDevCommRequirements_t* reqs, ncclDevComm_t* outDevComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclDevCommCreate(comm, reqs, outDevComm)


cdef ncclResult_t ncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t* devComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclDevCommDestroy(comm, devComm)


cdef ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetLsaMultimemDevicePointer(window, offset, outPtr)


cdef ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetLsaDevicePointer(window, offset, lsaRank, outPtr)


cdef ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle_t multimem, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetMultimemDevicePointer(window, offset, multimem, outPtr)


cdef ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetPeerDevicePointer(window, offset, peer, outPtr)


cdef ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetMultimemDeviceLeInfo(window, offset, leId, leOffset)


cdef ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetCftDeviceLeInfo(window, offset, peerCft, cftTeam, leId, leOffset)


cdef ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetPeerDeviceLeInfo(window, offset, peerWorld, leId, leOffset)


cdef ncclTeam_t ncclTeamWorld(ncclComm_t comm) except* nogil:
    return _nccl._ncclTeamWorld(comm)


cdef ncclTeam_t ncclTeamLsa(ncclComm_t comm) except* nogil:
    return _nccl._ncclTeamLsa(comm)


cdef ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode) except* nogil:
    return _nccl._ncclTeamCft(comm, mode)


cdef ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm) except* nogil:
    return _nccl._ncclTeamCftMultimem(comm)


cdef ncclTeam_t ncclTeamRail(ncclComm_t comm) except* nogil:
    return _nccl._ncclTeamRail(comm)


cdef int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) except?-42 nogil:
    return _nccl._ncclTeamRankToWorld(comm, team, rank)


cdef int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) except?-42 nogil:
    return _nccl._ncclTeamRankToLsa(comm, team, rank)


cdef ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclLsaBarrierCreateRequirement(team, nBarriers, outHandle, outReq)


cdef ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGinBarrierCreateRequirement(comm, team, nBarriers, outHandle, outReq)


cdef ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclLLA2ACreateRequirement(nBlocks, nSlots, outHandle, outReq)


cdef int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) except?-42 nogil:
    return _nccl._ncclLLA2ACalcSlots(maxElts, maxEltSize)
