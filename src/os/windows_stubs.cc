/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Stub implementations for Linux-only subsystems: RAS, NVLS, CollNet,
 * IPC sockets, GDR, and IB/GIN transports.  These are never reachable at
 * runtime on Windows but must be defined so the DLL links cleanly.       */

#ifdef NCCL_OS_WINDOWS

#include "nccl.h"
#include "transport.h"
#include "ras.h"
#include "ipcsocket.h"
#include "gdrwrap.h"
#include "net.h"
#include "gin.h"
#include "nccl_common.h"          // ncclProfilerCallback_t
#include "plugin/nccl_net.h"      // ncclNetCommConfig_v11_t, ncclNet_v11_t
#include "plugin/nccl_gin.h"      // ncclGin_v12_t

// ─── RAS (Reliability / Availability / Serviceability) ─────────────────────
ncclResult_t ncclRasCommInit(struct ncclComm* /*comm*/, struct rasRankInit* /*myRank*/) { return ncclSuccess; }
ncclResult_t ncclRasCommFini(const struct ncclComm* /*comm*/) { return ncclSuccess; }
ncclResult_t ncclRasAddRanks(struct rasRankInit* /*ranks*/, int /*nranks*/) { return ncclSuccess; }

// ─── NVLS (NVLink Switch) ───────────────────────────────────────────────────
ncclResult_t ncclNvlsInit(struct ncclComm* /*comm*/) { return ncclSuccess; }
ncclResult_t ncclNvlsSetup(struct ncclComm* /*comm*/, struct ncclComm* /*parent*/) { return ncclSuccess; }
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* /*comm*/) { return ncclSuccess; }
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* /*comm*/) { return ncclSuccess; }
ncclResult_t ncclNvlsFree(struct ncclComm* /*comm*/) { return ncclSuccess; }
ncclResult_t ncclNvlsGroupCreate(struct ncclComm* /*comm*/, CUmulticastObjectProp* /*prop*/,
    int /*rank*/, unsigned int /*nranks*/, CUmemGenericAllocationHandle* /*mcHandle*/,
    char* /*shareableHandle*/) { return ncclSuccess; }
ncclResult_t ncclNvlsGroupConnect(struct ncclComm* /*comm*/, char* /*shareableHandle*/,
    int /*rank*/, CUmemGenericAllocationHandle* /*mcHandle*/) { return ncclSuccess; }
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* /*comm*/, struct ncclTaskColl* /*info*/,
    int* recChannels) { if (recChannels) *recChannels = 0; return ncclSuccess; }
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* /*comm*/, const void* /*sendbuff*/,
    void* /*recvbuff*/, size_t /*sendbuffSize*/, size_t /*recvbuffSize*/,
    int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv) {
  if (outRegBufUsed) *outRegBufUsed = 0;
  if (outRegBufSend) *outRegBufSend = nullptr;
  if (outRegBufRecv) *outRegBufRecv = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm* /*comm*/, const void* /*sendbuff*/,
    void* /*recvbuff*/, size_t /*sendbuffSize*/, size_t /*recvbuffSize*/,
    int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* /*cleanupQueue*/,
    int* nCleanupQueueElts) {
  if (outRegBufUsed) *outRegBufUsed = 0;
  if (outRegBufSend) *outRegBufSend = nullptr;
  if (outRegBufRecv) *outRegBufRecv = nullptr;
  if (nCleanupQueueElts) *nCleanupQueueElts = 0;
  return ncclSuccess;
}
ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* /*comm*/,
    CUmemGenericAllocationHandle* /*mcHandler*/, CUdeviceptr /*ptr*/,
    int /*dev*/, size_t /*ucsize*/, size_t /*mcsize*/) { return ncclSuccess; }

// ─── CollNet ────────────────────────────────────────────────────────────────
ncclResult_t ncclCollNetSetup(ncclComm_t /*comm*/, ncclComm_t /*parent*/,
    struct ncclTopoGraph* /*graphs*/[]) { return ncclSuccess; }
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t /*comm*/) { return ncclSuccess; }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t /*comm*/) { return ncclSuccess; }
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* /*comm*/, const void* /*userbuff*/,
    size_t /*buffSize*/, int /*type*/, int* outRegBufUsed, void** outHandle) {
  if (outRegBufUsed) *outRegBufUsed = 0;
  if (outHandle) *outHandle = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclCollnetGraphRegisterBuffer(struct ncclComm* /*comm*/, const void* /*userbuff*/,
    size_t /*buffSize*/, int /*type*/, int* outRegBufFlag, void** outHandle,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* /*cleanupQueue*/,
    int* nCleanupQueueElts) {
  if (outRegBufFlag) *outRegBufFlag = 0;
  if (outHandle) *outHandle = nullptr;
  if (nCleanupQueueElts) *nCleanupQueueElts = 0;
  return ncclSuccess;
}
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* /*comm*/,
    struct ncclProxyConnector* /*proxyconn*/, void* /*handle*/) { return ncclSuccess; }

// Stub transport object — canConnect always returns "cannot connect"
static ncclResult_t collNetCanConnect(int* ret, struct ncclComm*, struct ncclTopoGraph*,
    struct ncclPeerInfo*, struct ncclPeerInfo*) { *ret = 0; return ncclSuccess; }
struct ncclTransport collNetTransport = { "CollNet", collNetCanConnect, {}, {} };

// ─── IPC sockets (AF_UNIX; not used on Windows) ─────────────────────────────
ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket* /*handle*/, int /*rank*/,
    uint64_t /*hash*/, volatile uint32_t* /*abortFlag*/) { return ncclSuccess; }
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket* /*handle*/) { return ncclSuccess; }
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* /*handle*/, int* fd) {
  if (fd) *fd = -1; return ncclSuccess;
}
ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket* /*handle*/, void* /*hdr*/, int /*hdrLen*/,
    const int /*sendFd*/, int /*rank*/, uint64_t /*hash*/) { return ncclSystemError; }
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket* /*handle*/, void* /*hdr*/, int /*hdrLen*/,
    int* recvFd) { if (recvFd) *recvFd = -1; return ncclSystemError; }

// ─── GDR copy (Linux-only kernel module) ────────────────────────────────────
ncclResult_t wrap_gdr_symbols(void) { return ncclSystemError; }
gdr_t wrap_gdr_open(void) { return nullptr; }
ncclResult_t wrap_gdr_close(gdr_t /*g*/) { return ncclSystemError; }
ncclResult_t wrap_gdr_pin_buffer(gdr_t /*g*/, unsigned long /*mr_addr*/,
    size_t /*size*/, uint64_t /*p2p_token*/, uint32_t /*va_space*/,
    gdr_mh_t* /*handle*/) { return ncclSystemError; }
ncclResult_t wrap_gdr_unpin_buffer(gdr_t /*g*/, gdr_mh_t /*handle*/) { return ncclSystemError; }
ncclResult_t wrap_gdr_map(gdr_t /*g*/, gdr_mh_t /*handle*/, void** /*va*/,
    size_t /*size*/) { return ncclSystemError; }
ncclResult_t wrap_gdr_unmap(gdr_t /*g*/, gdr_mh_t /*handle*/, void* /*va*/,
    size_t /*size*/) { return ncclSystemError; }
ncclResult_t wrap_gdr_get_info(gdr_t /*g*/, gdr_mh_t /*handle*/,
    gdr_info_t* /*info*/) { return ncclSystemError; }
ncclResult_t wrap_gdr_driver_get_version(gdr_t /*g*/, int* major,
    int* minor) { if (major) *major=0; if (minor) *minor=0; return ncclSystemError; }
ncclResult_t wrap_gdr_runtime_get_version(int* major,
    int* minor) { if (major) *major=0; if (minor) *minor=0; return ncclSystemError; }

// ─── IB / GIN transport objects (IB hardware; not available on Windows) ─────
// NCCL calls ncclNetIb.init() unconditionally, then ncclNetIb.devices() to
// probe for hardware, and ncclNetIb.finalize() on the fail path when init
// succeeded but devices returned 0.  All three must be valid function pointers.
// Returning ncclSuccess from init / 0 devices causes NCCL to mark the plugin
// disabled and fall through to ncclNetSocket, which works on Windows.
//
// For ncclGinIb / ncclGinIbProxy the init is called via short-circuit:
//   if (init() != ncclSuccess || devices() <= 0) → disabled
// Returning ncclSystemError from init short-circuits the check; devices and
// finalize are never called on that path.

static ncclResult_t ibStubInit(void** /*ctx*/, uint64_t /*commId*/,
    ncclNetCommConfig_v11_t* /*cfg*/, ncclDebugLogger_t /*log*/,
    ncclProfilerCallback_t /*prof*/) { return ncclSuccess; }
static ncclResult_t ibStubDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
static ncclResult_t ibStubFinalize(void* /*ctx*/) { return ncclSuccess; }

ncclNet_t ncclNetIb = {
  /* name       */ "IB",
  /* init       */ ibStubInit,
  /* devices    */ ibStubDevices,
  /* getProperties */ nullptr,
  /* listen     */ nullptr,
  /* connect    */ nullptr,
  /* accept     */ nullptr,
  /* regMr      */ nullptr,
  /* regMrDmaBuf*/ nullptr,
  /* deregMr    */ nullptr,
  /* isend      */ nullptr,
  /* irecv      */ nullptr,
  /* iflush     */ nullptr,
  /* test       */ nullptr,
  /* closeSend  */ nullptr,
  /* closeRecv  */ nullptr,
  /* closeListen*/ nullptr,
  /* getDeviceMr*/ nullptr,
  /* irecvConsumed */ nullptr,
  /* makeVDevice*/ nullptr,
  /* finalize   */ ibStubFinalize,
  /* setNetAttr */ nullptr,
};

static ncclResult_t ginStubInit(void** /*ctx*/, uint64_t /*commId*/,
    ncclDebugLogger_t /*log*/) { return ncclSystemError; }

ncclGin_t ncclGinIb = {
  /* name */ "GIN-IB",
  /* init */ ginStubInit,
};
ncclGin_t ncclGinIbProxy = {
  /* name */ "GIN-IB-Proxy",
  /* init */ ginStubInit,
};

#endif // NCCL_OS_WINDOWS
