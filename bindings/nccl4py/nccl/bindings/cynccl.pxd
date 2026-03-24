# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.30.0. Do not modify it directly.


from libc.stdint cimport int64_t, uint8_t, uint32_t, uint64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum ncclResult_t "ncclResult_t":
    ncclSuccess "ncclSuccess" = 0
    ncclUnhandledCudaError "ncclUnhandledCudaError" = 1
    ncclSystemError "ncclSystemError" = 2
    ncclInternalError "ncclInternalError" = 3
    ncclInvalidArgument "ncclInvalidArgument" = 4
    ncclInvalidUsage "ncclInvalidUsage" = 5
    ncclRemoteError "ncclRemoteError" = 6
    ncclInProgress "ncclInProgress" = 7
    ncclNumResults "ncclNumResults" = 8
    _NCCLRESULT_T_INTERNAL_LOADING_ERROR "_NCCLRESULT_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum ncclCommMemStat_t "ncclCommMemStat_t":
    ncclStatGpuMemSuspend "ncclStatGpuMemSuspend" = 0
    ncclStatGpuMemSuspended "ncclStatGpuMemSuspended" = 1
    ncclStatGpuMemPersist "ncclStatGpuMemPersist" = 2
    ncclStatGpuMemTotal "ncclStatGpuMemTotal" = 3

ctypedef enum ncclRedOp_dummy_t "ncclRedOp_dummy_t":
    ncclNumOps_dummy "ncclNumOps_dummy" = 5

ctypedef enum ncclRedOp_t "ncclRedOp_t":
    ncclSum "ncclSum" = 0
    ncclProd "ncclProd" = 1
    ncclMax "ncclMax" = 2
    ncclMin "ncclMin" = 3
    ncclAvg "ncclAvg" = 4
    ncclNumOps "ncclNumOps" = 5
    ncclMaxRedOp "ncclMaxRedOp" = (0x7fffffff >> (32 - (8 * sizeof(ncclRedOp_dummy_t))))

ctypedef enum ncclDataType_t "ncclDataType_t":
    ncclInt8 "ncclInt8" = 0
    ncclChar "ncclChar" = 0
    ncclUint8 "ncclUint8" = 1
    ncclInt32 "ncclInt32" = 2
    ncclInt "ncclInt" = 2
    ncclUint32 "ncclUint32" = 3
    ncclInt64 "ncclInt64" = 4
    ncclUint64 "ncclUint64" = 5
    ncclFloat16 "ncclFloat16" = 6
    ncclHalf "ncclHalf" = 6
    ncclFloat32 "ncclFloat32" = 7
    ncclFloat "ncclFloat" = 7
    ncclFloat64 "ncclFloat64" = 8
    ncclDouble "ncclDouble" = 8
    ncclBfloat16 "ncclBfloat16" = 9
    ncclFloat8e4m3 "ncclFloat8e4m3" = 10
    ncclFloat8e5m2 "ncclFloat8e5m2" = 11
    ncclNumTypes "ncclNumTypes" = 12

ctypedef enum ncclScalarResidence_t "ncclScalarResidence_t":
    ncclScalarDevice "ncclScalarDevice" = 0
    ncclScalarHostImmediate "ncclScalarHostImmediate" = 1

ctypedef enum ncclGinType_t "ncclGinType_t":
    NCCL_GIN_TYPE_NONE "NCCL_GIN_TYPE_NONE" = 0
    NCCL_GIN_TYPE_PROXY "NCCL_GIN_TYPE_PROXY" = 2
    NCCL_GIN_TYPE_GDAKI "NCCL_GIN_TYPE_GDAKI" = 3

ctypedef enum ncclGinConnectionType_t "ncclGinConnectionType_t":
    NCCL_GIN_CONNECTION_NONE "NCCL_GIN_CONNECTION_NONE" = 0
    NCCL_GIN_CONNECTION_FULL "NCCL_GIN_CONNECTION_FULL" = 1
    NCCL_GIN_CONNECTION_RAIL "NCCL_GIN_CONNECTION_RAIL" = 2


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>
    """
    ctypedef void* cudaStream_t 'cudaStream_t'


ctypedef uint32_t ncclDevResourceHandle_t 'ncclDevResourceHandle_t'
ctypedef uint32_t ncclGinSignal_t 'ncclGinSignal_t'
ctypedef uint32_t ncclGinCounter_t 'ncclGinCounter_t'
ctypedef void* ncclComm_t 'ncclComm_t'
ctypedef void* ncclWindow_t 'ncclWindow_t'
ctypedef void* ncclDevCommWindowTable_t 'ncclDevCommWindowTable_t'
ctypedef void* ncclGinWindow_t 'ncclGinWindow_t'
ctypedef struct ncclUniqueId 'ncclUniqueId':
    char internal[128]

ctypedef struct ncclConfig_t 'ncclConfig_t':
    size_t size
    unsigned int magic
    unsigned int version
    int blocking
    int cgaClusterSize
    int minCTAs
    int maxCTAs
    char* netName
    int splitShare
    int trafficClass
    char* commName
    int collnetEnable
    int CTAPolicy
    int shrinkShare
    int nvlsCTAs
    int nChannelsPerNetPeer
    int nvlinkCentricSched
    int graphUsageMode
    int numRmaCtx
    int maxP2pPeers

ctypedef struct ncclSimInfo_t 'ncclSimInfo_t':
    size_t size
    unsigned int magic
    unsigned int version
    float estimatedTime

ctypedef struct ncclWaitSignalDesc_t 'ncclWaitSignalDesc_t':
    int opCnt
    int peer
    int sigIdx
    int ctx

ctypedef struct ncclCommProperties_t 'ncclCommProperties_t':
    size_t size
    unsigned int magic
    unsigned int version
    int rank
    int nRanks
    int cudaDev
    int nvmlDev
    uint8_t deviceApiSupport
    uint8_t multimemSupport
    ncclGinType_t ginType
    int nLsaTeams
    uint8_t hostRmaSupport
    ncclGinType_t railedGinType

ctypedef struct ncclTeam_t 'ncclTeam_t':
    int nRanks
    int rank
    int stride

ctypedef struct ncclMultimemHandle_t 'ncclMultimemHandle_t':
    void* mcBasePtr

ctypedef struct ncclLsaBarrierHandle_t 'ncclLsaBarrierHandle_t':
    ncclDevResourceHandle_t bufHandle
    int nBarriers

ctypedef struct ncclGinBarrierHandle_t 'ncclGinBarrierHandle_t':
    ncclGinSignal_t signal0
    ncclDevResourceHandle_t unused

ctypedef struct ncclDevResourceRequirements_t 'ncclDevResourceRequirements_t':
    void* next
    size_t bufferSize
    size_t bufferAlign
    ncclDevResourceHandle_t* outBufferHandle
    int ginSignalCount
    int ginCounterCount
    ncclGinSignal_t* outGinSignalStart
    ncclGinCounter_t* outGinCounterStart

ctypedef struct ncclWindow_vidmem_t 'ncclWindow_vidmem_t':
    void* winHost
    char* lsaFlatBase
    int lsaRank
    int worldRank
    uint32_t stride4G
    uint32_t mcOffset4K
    uint32_t ginOffset4K
    ncclGinWindow_t ginWins[4]

ctypedef struct ncclTeamRequirements_t 'ncclTeamRequirements_t':
    void* next
    ncclTeam_t team
    uint8_t multimem
    ncclMultimemHandle_t* outMultimemHandle

ctypedef struct ncclDevComm_t 'ncclDevComm_t':
    int rank
    int nRanks
    uint32_t nRanks_rcp32
    int lsaRank
    int lsaSize
    uint32_t lsaSize_rcp32
    ncclDevCommWindowTable_t windowTable
    ncclWindow_t resourceWindow
    ncclWindow_vidmem_t resourceWindow_inlined
    ncclMultimemHandle_t lsaMultimem
    ncclLsaBarrierHandle_t lsaBarrier
    ncclGinBarrierHandle_t railGinBarrier
    uint8_t ginConnectionCount
    uint8_t ginNetDeviceTypes[4]
    void* ginHandles[4]
    uint32_t ginSignalBase
    int ginSignalCount
    uint32_t ginCounterBase
    int ginCounterCount
    uint64_t* ginSignalShadows
    uint32_t ginContextCount
    uint32_t ginContextBase
    uint8_t ginIsRailed
    uint32_t* abortFlag
    ncclLsaBarrierHandle_t hybridLsaBarrier
    ncclGinBarrierHandle_t hybridRailGinBarrier
    ncclGinBarrierHandle_t worldGinBarrier

ctypedef struct ncclDevCommRequirements_t 'ncclDevCommRequirements_t':
    size_t size
    unsigned int magic
    unsigned int version
    ncclDevResourceRequirements_t* resourceRequirementsList
    ncclTeamRequirements_t* teamRequirementsList
    uint8_t lsaMultimem
    int barrierCount
    int lsaBarrierCount
    int railGinBarrierCount
    int lsaLLA2ABlockCount
    int lsaLLA2ASlotCount
    uint8_t ginForceEnable
    int ginContextCount
    int ginSignalCount
    int ginCounterCount
    ncclGinConnectionType_t ginConnectionType
    uint8_t ginExclusiveContexts
    int ginQueueDepth
    int worldGinBarrierCount



###############################################################################
# Functions
###############################################################################

cdef ncclResult_t ncclMemAlloc(void** ptr, size_t size) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclMemFree(void* ptr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetVersion(int* version) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommFinalize(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommDestroy(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommAbort(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commIds, ncclConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef const char* ncclGetErrorString(ncclResult_t result) except?NULL nogil
cdef const char* ncclGetLastError(ncclComm_t comm) except?NULL nogil
cdef ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommResume(ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void** outUserPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclDevCommCreate(ncclComm_t comm, const ncclDevCommRequirements_t* reqs, ncclDevComm_t* outDevComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t* devComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
