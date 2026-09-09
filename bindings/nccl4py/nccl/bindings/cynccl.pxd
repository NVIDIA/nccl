# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 2.31.2. Do not modify it directly.


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums

# <<<< PREAMBLE CONTENT >>>>

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

ctypedef enum ncclResult_t "ncclResult_t":
    ncclSuccess "ncclSuccess" = 0
    ncclUnhandledCudaError "ncclUnhandledCudaError" = 1
    ncclSystemError "ncclSystemError" = 2
    ncclInternalError "ncclInternalError" = 3
    ncclInvalidArgument "ncclInvalidArgument" = 4
    ncclInvalidUsage "ncclInvalidUsage" = 5
    ncclRemoteError "ncclRemoteError" = 6
    ncclInProgress "ncclInProgress" = 7
    ncclTimeout "ncclTimeout" = 8
    ncclNumResults "ncclNumResults" = 9
    _NCCLRESULT_T_INTERNAL_LOADING_ERROR "_NCCLRESULT_T_INTERNAL_LOADING_ERROR" = -42

ctypedef enum ncclHostCftMode_t "ncclHostCftMode_t":
    ncclHostCftDefault "ncclHostCftDefault" = -(2147483648)
    ncclHostCftEnable "ncclHostCftEnable" = 1
    ncclHostCftDisable "ncclHostCftDisable" = 2
    ncclHostCftFallback "ncclHostCftFallback" = 3

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
    NCCL_GIN_TYPE_GPI "NCCL_GIN_TYPE_GPI" = 4
    NCCL_GIN_TYPE_EFA_GDA "NCCL_GIN_TYPE_EFA_GDA" = 5
    NCCL_GIN_MAX_TYPES "NCCL_GIN_MAX_TYPES" = 6

ctypedef enum ncclGinConnectionType_t "ncclGinConnectionType_t":
    NCCL_GIN_CONNECTION_NONE "NCCL_GIN_CONNECTION_NONE" = 0
    NCCL_GIN_CONNECTION_FULL "NCCL_GIN_CONNECTION_FULL" = 1
    NCCL_GIN_CONNECTION_RAIL "NCCL_GIN_CONNECTION_RAIL" = 2
    NCCL_GIN_CONNECTION_CUSTOM_STRIDE "NCCL_GIN_CONNECTION_CUSTOM_STRIDE" = 3

ctypedef enum ncclCftTeamMode_t "ncclCftTeamMode_t":
    NCCL_CFT_TEAM_FLAT "NCCL_CFT_TEAM_FLAT" = 0
    NCCL_CFT_TEAM_HIER_MULTIMEM "NCCL_CFT_TEAM_HIER_MULTIMEM" = 1
    NCCL_CFT_TEAM_HIER_LSA "NCCL_CFT_TEAM_HIER_LSA" = 2


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

ctypedef uint32_t ncclCftLeId 'ncclCftLeId'

ctypedef void* ncclComm_t 'ncclComm_t'

ctypedef void* ncclWindow_t 'ncclWindow_t'

ctypedef void* ncclParamHandle_t 'ncclParamHandle_t'

ctypedef void* ncclDevCommWindowTable_t 'ncclDevCommWindowTable_t'

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
    int graphStreamOrdering
    int launchOrderImplicit
    int numRmaSig
    int rmaEagerInit
    int hostCftMode

ctypedef struct nccl_bindings_nccl__anon_pod0:
    int vendorId
    int optionId

ctypedef union nccl_bindings_nccl__anon_pod1:
    int i
    char* s
    void* raw

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
    uint64_t commHash
    int ginMinStride
    ncclGinConnectionType_t ginConnectionType
    uint8_t ginSupport[64]
    size_t devCommRuntimeVersionSize

ctypedef struct ncclTeam_t 'ncclTeam_t':
    int nRanks
    int rank
    int stride

ctypedef struct ncclMultimemHandle_t 'ncclMultimemHandle_t':
    void* mcBasePtr

ctypedef struct ncclResourceWindow_vidmem_t 'ncclResourceWindow_vidmem_t':
    char* lsaFlatBase
    uint32_t stride4G
    uint32_t mcOffset4K

ctypedef struct ncclLsaBarrierHandle_t 'ncclLsaBarrierHandle_t':
    ncclDevResourceHandle_t bufHandle
    int nBarriers

ctypedef struct ncclCftBarrierHandle_t 'ncclCftBarrierHandle_t':
    ncclDevResourceHandle_t bufHandle
    int nBarriers

ctypedef struct ncclLLA2AHandle_t 'ncclLLA2AHandle_t':
    ncclDevResourceHandle_t bufHandle
    uint32_t nSlots

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

cdef struct ncclConfigExt:
    ncclConfigExt* next
    nccl_bindings_nccl__anon_pod0 key
    nccl_bindings_nccl__anon_pod1 val
ctypedef ncclConfigExt ncclConfigExt_t

ctypedef struct ncclTeamRequirements_t 'ncclTeamRequirements_t':
    void* next
    ncclTeam_t team
    uint8_t multimem
    ncclMultimemHandle_t* outMultimemHandle

ctypedef struct ncclDevComm_t 'ncclDevComm_t':
    unsigned int magic
    unsigned int version
    int rank
    int nRanks
    uint32_t nRanks_rcp32
    int lsaRank
    int lsaSize
    uint32_t lsaSize_rcp32
    ncclDevCommWindowTable_t windowTable
    ncclWindow_t resourceWindow
    ncclResourceWindow_vidmem_t resourceWindow_inlined
    ncclGinBarrierHandle_t hybridDenseGinBarrier
    ncclMultimemHandle_t lsaMultimem
    ncclLsaBarrierHandle_t lsaBarrier
    ncclGinBarrierHandle_t railGinBarrier
    uint8_t ginConnectionCount
    uint8_t backendIndex
    uint8_t ginNetDeviceTypes[4]
    void* ginHandles[4]
    int ginSignalCount
    int ginCounterCount
    uint64_t* ginSignalShadows
    uint32_t ginContextCount
    int ginConnectionStride
    int ginContextStride
    uint8_t ginStrongLegacySignals
    uint32_t* abortFlag
    ncclLsaBarrierHandle_t hybridLsaBarrier
    ncclGinBarrierHandle_t hybridRailGinBarrier
    ncclGinBarrierHandle_t worldGinBarrier
    uint32_t ginConnectionStride_rcp32
    int cftRank
    int cftSize
    int cftMultimemRank
    int cftMultimemSize
    uint32_t cftMultimemSize_rcp32
    ncclCftLeId ucLeId
    ncclCftLeId mcLeId
    ncclCftBarrierHandle_t cftBarrier
    ncclCftBarrierHandle_t cftMultimemBarrier

ctypedef struct ncclCollConfig_t 'ncclCollConfig_t':
    size_t size
    unsigned int magic
    unsigned int version
    ncclConfigExt_t* ext
    int minCTAs
    int maxCTAs
    int nvlsCTAs
    int cgaClusterSize
    char* algSelection
    int forceAlgSelection
    int CTAPolicy
    uint64_t userProfilerTag

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
    int ginTrafficClass
    int worldGinBarrierCount
    uint8_t ginStrongSignalsRequired
    uint8_t ginVaSignalsRequired
    int ginCustomStride
    ncclGinType_t ginType
    uint8_t useRuntimeVersion
    int cftCaps
    int cftBarrierCount


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
cdef ncclResult_t ncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamBind(ncclParamHandle_t* out, const char* key) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetI8(ncclParamHandle_t h, int8_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetI16(ncclParamHandle_t h, int16_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetI32(ncclParamHandle_t h, int32_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetI64(ncclParamHandle_t h, int64_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetU8(ncclParamHandle_t h, uint8_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetU16(ncclParamHandle_t h, uint16_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetU32(ncclParamHandle_t h, uint32_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetU64(ncclParamHandle_t h, uint64_t* out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char** out) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGet(ncclParamHandle_t h, void* out, int maxLen, int* len) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetParameter(const char* key, const char** value, int* valueLen) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclParamGetAllParameterKeys(const char*** table, int* tableLen) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef void ncclParamDumpAll() except* nogil
cdef ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclDevCommCreate(ncclComm_t comm, const ncclDevCommRequirements_t* reqs, ncclDevComm_t* outDevComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t* devComm) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle_t multimem, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId, size_t* leOffset) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclTeam_t ncclTeamWorld(ncclComm_t comm) except* nogil
cdef ncclTeam_t ncclTeamLsa(ncclComm_t comm) except* nogil
cdef ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode) except* nogil
cdef ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm) except* nogil
cdef ncclTeam_t ncclTeamRail(ncclComm_t comm) except* nogil
cdef int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) except?-42 nogil
cdef int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) except?-42 nogil
cdef ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil
cdef int ncclLLA2ACalcSlots(int maxElts, int maxEltSize) except?-42 nogil
