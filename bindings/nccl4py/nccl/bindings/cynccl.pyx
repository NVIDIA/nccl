# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.30.0. Do not modify it directly.

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


cdef ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    return _nccl._ncclGetPeerDevicePointer(window, offset, peer, outPtr)
