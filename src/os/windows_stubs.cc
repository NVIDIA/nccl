/*************************************************************************
 * Windows stub implementations for symbols provided on Linux by
 * plugin/, ras/, and gin/ subdirectories. Built only when NCCL_OS_WINDOWS.
 *************************************************************************/

#if defined(NCCL_OS_WINDOWS)

#include "nccl.h"
#include "comm.h"
#include "checks.h"
#include "net.h"
#include "ras.h"
#include "profiler.h"
#include "env.h"
#include "tuner.h"
#include "gin/gin_host_win_stub.h"
#include "device.h"

#include <cstring>
#include <cstdlib>

/* --------------------------------------------------------------------------
 * RAS (Linux-only) stubs
 * -------------------------------------------------------------------------- */
ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank) {
  (void)comm;
  (void)myRank;
  return ncclSuccess;
}

ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks) {
  (void)ranks;
  (void)nranks;
  return ncclSuccess;
}

ncclResult_t ncclRasCommFini(const struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Net plugin stubs: use built-in socket transport without plugin layer
 * -------------------------------------------------------------------------- */
ncclResult_t ncclNetInit(struct ncclComm* comm) {
  comm->ncclNet = &ncclNetSocket;
  comm->ncclCollNet = nullptr;
  comm->netContext = nullptr;
  comm->netPluginIndex = -1;
  return ncclSuccess;
}

ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  comm->netContext = parent->netContext;
  comm->collNetContext = parent->collNetContext;
  comm->ncclNet = parent->ncclNet;
  comm->ncclCollNet = parent->ncclCollNet;
  comm->netPluginIndex = parent->netPluginIndex;
  return ncclSuccess;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  if (comm->ncclNet && comm->netContext)
    NCCLCHECK(comm->ncclNet->finalize(comm->netContext));
  return ncclSuccess;
}

ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport) {
  (void)comm;
  *gdrSupport = 0;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Env plugin stubs
 * -------------------------------------------------------------------------- */
bool ncclEnvPluginInitialized(void) {
  return false;
}

ncclResult_t ncclEnvPluginInit(void) {
  return ncclSuccess;
}

const char* ncclEnvPluginGetEnv(const char* name) {
  return std::getenv(name);
}

/* --------------------------------------------------------------------------
 * Net/CollNet dev count stubs (plugin provides these on Linux)
 * -------------------------------------------------------------------------- */
ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)netPluginIndex;
  *nPhysDev = 0;
  *nVirtDev = 0;
  return ncclSuccess;
}

ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDev) {
  (void)netPluginIndex;
  (void)nVirtDev;
  return ncclSuccess;
}

ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)netPluginIndex;
  *nPhysDev = 0;
  *nVirtDev = 0;
  return ncclSuccess;
}

ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDev) {
  (void)netPluginIndex;
  (void)nVirtDev;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * GIN additional stubs (getGlobalGinType, ConnectOnce, Register, etc.)
 * -------------------------------------------------------------------------- */
ncclResult_t getGlobalGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  (void)comm;
  *ginType = (ncclGinType_t)0;  /* NCCL_GIN_TYPE_NONE */
  return ncclSuccess;
}

/* GIN requirement/create stubs (implemented in gin_barrier.cc and gin_scratch.cc on Linux only) */
ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                            ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) {
  (void)comm;
  (void)team;
  (void)nBarriers;
  (void)outHandle;
  memset(outReq, 0, sizeof(*outReq));
  return ncclSuccess;
}

ncclResult_t ncclGinOutboxCreateRequirement(int nBlocks, int size_log2,
                                            ncclGinOutboxHandle* outHandle, ncclDevResourceRequirements_t* outReq) {
  (void)nBlocks;
  (void)size_log2;
  (void)outHandle;
  memset(outReq, 0, sizeof(*outReq));
  return ncclSuccess;
}

ncclResult_t ncclGinInboxA2ACreateRequirement(ncclTeam peers, int nBlocks, int size_log2,
                                              ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements_t* outReq) {
  (void)peers;
  (void)nBlocks;
  (void)size_log2;
  (void)outHandle;
  memset(outReq, 0, sizeof(*outReq));
  return ncclSuccess;
}

ncclResult_t getGlobalRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  (void)comm;
  *ginType = (ncclGinType_t)0;  /* NCCL_GIN_TYPE_NONE */
  return ncclSuccess;
}

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* devComm) {
  (void)comm;
  (void)reqs;
  (void)devComm;
  return ncclSuccess;
}

ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  (void)comm;
  (void)devComm;
  return ncclSuccess;
}

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags) {
  (void)comm;
  (void)address;
  (void)size;
  (void)ginHostWins;
  (void)ginDevWins;
  (void)winFlags;
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  (void)comm;
  (void)ginHostWins;
  return ncclSuccess;
}

ncclResult_t ncclGinGetDevCount(int ginPluginIndex, int* nPhysDev, int* nVirtDev) {
  (void)ginPluginIndex;
  *nPhysDev = 0;
  *nVirtDev = 0;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Param stubs (defined in socket.cc / net_ib / gin on Linux; not built on Windows)
 * -------------------------------------------------------------------------- */
int ncclParamPollTimeOut(void) {
  return 0;
}

int ncclParamRetryTimeOut(void) {
  return 100;
}

long ncclParamRetryCnt(void) {
  return 34;
}

int ncclParamSocketMaxRecvBuff(void) {
  return -1;
}

int ncclParamSocketMaxSendBuff(void) {
  return -1;
}

int64_t ncclParamIbDataDirect(void) {
  return 0;
}

int64_t ncclParamGinEnable(void) {
  return 0;
}

/* --------------------------------------------------------------------------
 * Profiler plugin stubs and globals
 * -------------------------------------------------------------------------- */
thread_local ncclProfilerApiState_t ncclProfilerApiState = {
  0, 0, ncclProfilerGroupApiStartStateReset, nullptr, nullptr, nullptr
};

int ncclProfilerEventMask = 0;

bool ncclProfilerPluginLoaded(void) {
  return false;
}

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  (void)info;
  (void)isGraphCaptured;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupApiEvent(void) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  (void)eState;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  (void)info;
  (void)isGraphCaptured;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopP2pApiEvent(void) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  (void)info;
  (void)isGraphCaptured;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCollApiEvent(void) {
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream) {
  (void)plan;
  (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  (void)plan;
  return ncclSuccess;
}

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  (void)op;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args) {
  (void)sub;
  (void)args;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args) {
  (void)sub;
  (void)args;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId) {
  (void)sub;
  (void)args;
  (void)stepId;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId) {
  (void)sub;
  (void)args;
  (void)stepId;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId) {
  (void)sub;
  (void)args;
  (void)stepId;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  (void)profilerContext;
  if (eHandle) *eHandle = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle) {
  (void)eHandle;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start) {
  (void)args;
  (void)s;
  (void)start;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop) {
  (void)args;
  (void)s;
  (void)stop;
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  (void)sub;
  (void)args;
  (void)eState;
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState) {
  (void)sub;
  (void)args;
  (void)stepId;
  (void)eState;
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  (void)eHandle;
  (void)appended;
  (void)eState;
  return ncclSuccess;
}

bool ncclProfilerNeedsProxy(struct ncclComm* comm, struct ncclProxyOp* op) {
  (void)comm;
  (void)op;
  return false;
}

/* CE profiler stubs (ncclCeCollArgs / ncclCeBatchOpsParams forward-declared in profiler.h) */
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  (void)comm;
  (void)args;
  (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  (void)comm;
  (void)args;
  (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream, void** ceSyncHandle) {
  (void)comm;
  (void)args;
  (void)stream;
  if (ceSyncHandle) *ceSyncHandle = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, void* ceSyncHandle, cudaStream_t stream) {
  (void)comm;
  (void)ceSyncHandle;
  (void)stream;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, struct ncclCeBatchOpsParams* params, cudaStream_t stream, void** ceBatchHandle) {
  (void)comm;
  (void)args;
  (void)params;
  (void)stream;
  if (ceBatchHandle) *ceBatchHandle = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, void* ceBatchHandle, cudaStream_t stream) {
  (void)comm;
  (void)ceBatchHandle;
  (void)stream;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * Tuner plugin stubs
 * -------------------------------------------------------------------------- */
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

/* --------------------------------------------------------------------------
 * GIN (Linux-only) stubs
 * -------------------------------------------------------------------------- */
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  (void)ginState;
  if (hasError) *hasError = false;
  return ncclSuccess;
}

ncclResult_t ncclGinInit(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  (void)comm;
  (void)parent;
  return ncclSuccess;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
  (void)comm;
  return ncclSuccess;
}

#endif /* NCCL_OS_WINDOWS */
