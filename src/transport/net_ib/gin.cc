/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#include "gin/gin_host.h"
#include "gin.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t ncclGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  if (ncclIbDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  *gdrSupport = false;
  INFO(NCCL_NET, "Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t ncclGinIbGdrGpuSupport(bool gdaki) {
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int dmaBufSupportOnDevice = 1;
  CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
  if (dmaBufSupportOnDevice == 1) return ncclSuccess;

  WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
  return ncclInvalidUsage;
}

NCCL_PARAM(GinType, "GIN_TYPE", -1);

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;
int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t ncclGinIbGdakiInitOnce() {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    int64_t ginType = ncclParamGinType();
    if (ginType != -1 && ginType != NCCL_GIN_TYPE_GDAKI) {
      ncclGinIbGdakiNDevs = 0;
      return ncclSuccess;
    }
    for (int i = 0; i < ncclNIbDevs; i++) {
      if (ncclIbDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        ncclGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    ncclGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}

// Initlialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t ncclGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int type) {
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  if (ncclNIbDevs == 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

  if (type == NCCL_GIN_TYPE_GDAKI) {
    NCCLCHECK(ncclGinIbGdakiInitOnce());
    if (ncclGinIbGdakiNDevs == 0) return ncclInternalError;
  }

  bool gdrSupport;
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, type == NCCL_GIN_TYPE_GDAKI));
  if (!gdrSupport) return ncclInternalError;

  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}

ncclResult_t ncclGinIbFinalize(void *ctx) {
  if (ctx) free(ctx);
  return ncclIbFinalizeDevices();
}

static ncclResult_t ncclGinIbAllGather(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void *rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void *)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void *)((uintptr_t)recvBuf + rpeer * len);
      tag = NCCL_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm,
                                      (void *)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &len,
                                      &tag, &rMhandle, NULL, &rrequest),
                      status, out);
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &done, NULL),
                      status, out);
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &done, NULL),
                      status, out);
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle)
    ncclNetIb.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle)
    ncclNetIb.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t ncclGinIbAllToAll(struct ncclGinIbCollComm *cComm, void *src_buf, void *recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void *tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void **)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void *)((uintptr_t)recv_buf + i * len), (void *)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  if (tmp_buf)
    free(tmp_buf);

  return status;
}

ncclResult_t ncclGinIbP2PBarrier(struct ncclGinIbCollComm *cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int *dummy;
  NCCLCHECK(ncclIbMalloc((void **)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t ncclGinIbConnect(void *ctx, void *handles[], int nranks, int rank,
                              void *listenComm, void **collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct ncclGinIbCollComm *cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void **)&cCommArray, sizeof(*cCommArray)));

  struct ncclGinIbCollComm *cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclIbConnectImpl(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL, /*nQpsPerDev*/ 1));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclIbAcceptImpl(lComm, &cComm->recvComm, NULL, /*nQpsPerDev*/ 1));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclIbGetGidIndex;
  cComm->dev = lComm->dev;

  cComm->ib.context = ncclIbDevs[cComm->dev].context;
  cComm->ib.pd = ncclIbDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cCommArray = (struct ncclGinIbCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct ncclGinIbCollComm *cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cCommArray);
  return ncclSuccess;
}

#include "gdaki/gin_host_gdaki.h"

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI);
}

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetGinProperties(ncclGinProperties_t* ginProps) {
  ginProps->supportsStrongSignals = true;
  ginProps->supportsVASignals = true;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (dev >= ncclGinIbGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev, ncclGinIbGdakiNDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(ncclIbGetPhysProperties(ncclGinIbGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  return ncclNetIb.listen(ctx, ncclGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ true));

  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, ncclGinConfig_t* config, void **ginCtx, ncclNetDeviceHandle_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  // GDAKI currently doesn't support the rankStride optimization.
  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters, config->nContexts, config->queueDepth, config->trafficClass, config->backendVersion, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm *)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct ncclGinIbCollComm *)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiProgress(void *collComm)
{
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t ncclGinIbGdakiQueryLastError(void *ginCtx, bool *hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdaki = {
  "GIN_IB_GDAKI",
  ncclGinIbGdakiInit,
  ncclGinIbGdakiDevices,
  ncclGinIbGdakiGetGinProperties,
  ncclGinIbGdakiGetProperties,
  ncclGinIbGdakiListen,
  ncclGinIbGdakiConnect,
  ncclGinIbGdakiCreateContext,
  ncclGinIbGdakiRegMrSym,
  NULL, // regMrSymDmaBuf
  ncclGinIbGdakiDeregMrSym,
  ncclGinIbGdakiDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};


struct ncclRmaIbProxyMrHandle {
  struct ncclIbMrHandle *mrHandle;
  uintptr_t *base_vas;
  uint32_t *rkeys;
};

ncclResult_t ncclRmaIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_PROXY);
}

ncclResult_t ncclRmaIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIb.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  return ncclSuccess;
}

struct ncclRmaIbProxyCtx {
  void**        fullRecvComm;
  void**        fullSendComm;
  int rank, nranks;
  int nContexts;
};

ncclResult_t ncclRmaIbProxyCreateContext(void* collComm, ncclRmaConfig_t* config, void** rmaCtx) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  // Make sure all QP we create use the provided traffic class.
  ncclIbSetTrafficClass(cComm->ctx, config->trafficClass);

  if (config->rankStride <= 0 || (cComm->nranks % config->rankStride) != 0) {
    WARN("Rma Proxy create context: invalid rank stride %d, must be >= 0 and nranks (%d) must be a multiple of the stride", config->rankStride, cComm->nranks);
    return ncclInternalError;
  }

  int nranks;
  struct ncclRmaIbProxyCtx* rmaProxyCtx = NULL;
  *rmaCtx = NULL;
  NCCLCHECK(ncclCalloc(&rmaProxyCtx, config->nContexts));
  rmaProxyCtx[0].nContexts = config->nContexts;
  rmaProxyCtx[0].nranks = nranks = cComm->nranks;

  void *lComm = NULL;
  char* handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE*cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE*cComm->rank;

  NCCLCHECKGOTO(ncclNetIb.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c=0; c<config->nContexts; c++) {
    struct ncclRmaIbProxyCtx* gc = rmaProxyCtx+c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void *) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void *) * nranks), ret, end);
    gc->rank = cComm->rank;

    for (int i = 0; i < nranks; i+=config->rankStride) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(ncclIbConnectImpl(cComm->ctx, cComm->dev, handles+NCCL_NET_HANDLE_MAXSIZE*connectPeer, &gc->fullSendComm[connectPeer], NULL, /*nQpsPerDev*/ 1), ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(ncclIbAcceptImpl(lComm, &gc->fullRecvComm[acceptPeer], NULL, /*nQpsPerDev*/ 1), ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) ||
          (gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(ncclGinIbP2PBarrier(cComm), ret, end);
    }
  }

end:
  free(handles);
  if (lComm) ncclNetIb.closeListen(lComm);
  if (ret != ncclSuccess) free(rmaProxyCtx);
  else *rmaCtx = rmaProxyCtx;
  return ret;
}

ncclResult_t ncclRmaIbProxyDestroyContext(void* rmaCtx) {
  struct ncclRmaIbProxyCtx* gc = (struct ncclRmaIbProxyCtx*)rmaCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c=0; c<nContexts; c++) {
    if (gc[c].fullRecvComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclRmaIbProxyMrHandle *rmaMrHandle;
  NCCLCHECK(ncclCalloc(&rmaMrHandle, 1));

  NCCLCHECKNOWARN(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags, (void **)&rmaMrHandle->mrHandle), NCCL_NET);

  NCCLCHECK(ncclCalloc(&rmaMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&rmaMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, rmaMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &rmaMrHandle->mrHandle->mrs[0]->rkey, rmaMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = rmaMrHandle;

  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle) {
  return ncclRmaIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle);
}

ncclResult_t ncclRmaIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclRmaIbProxyMrHandle *rmaMrHandle = (struct ncclRmaIbProxyMrHandle *)mhandle;

  NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, rmaMrHandle->mrHandle));
  free(rmaMrHandle->base_vas);
  free(rmaMrHandle->rkeys);
  free(rmaMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

static ncclResult_t ncclRmaIbProxyGetSendComm(struct ncclRmaIbProxyCtx* rmaProxyCtx, int rank, struct ncclIbSendComm** commPtr) {
  *commPtr = (struct ncclIbSendComm*)rmaProxyCtx->fullSendComm[rank];
  if (*commPtr == NULL) {
    WARN("RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
static ncclResult_t ncclRmaIbProxyGetRecvComm(struct ncclRmaIbProxyCtx* rmaProxyCtx, int rank, struct ncclIbRecvComm** commPtr) {
  *commPtr = (struct ncclIbRecvComm*)rmaProxyCtx->fullRecvComm[rank];
  if (*commPtr == NULL) {
    WARN("RMA: trying to send to non-connected peer %d", rank);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyIPut(void *rmaCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                void **request) {
  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];

  struct ncclRmaIbProxyMrHandle *srcMrHandle = (struct ncclRmaIbProxyMrHandle *)srcMhandle;
  struct ncclRmaIbProxyMrHandle *dstMrHandle = (struct ncclRmaIbProxyMrHandle *)dstMhandle;

  void *srcPtr = (void *)(srcMrHandle->base_vas[rmaProxyCtx->rank] + srcOff);
  void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_WRITE;
  wr.send_flags              = IBV_SEND_SIGNALED;
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)dstPtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)srcPtr;  // Local buffer address
  sge.length = size;  // Size of the transfer
  sge.lkey = lkey;  // Local key

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyIGet(void *rmaCtx, int context, uint64_t remoteOffset, void *remoteMhandle,
                                 size_t size, uint64_t localOffset, void *localMhandle, uint32_t rank,
                                 void **request) {
  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];

  struct ncclRmaIbProxyMrHandle *remoteMrHandle = (struct ncclRmaIbProxyMrHandle *)remoteMhandle;
  struct ncclRmaIbProxyMrHandle *localMrHandle = (struct ncclRmaIbProxyMrHandle *)localMhandle;

  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  void *remotePtr = (void *)(remoteMrHandle->base_vas[rank] + remoteOffset);
  void *localPtr = (void *)(localMrHandle->base_vas[rmaProxyCtx->rank] + localOffset);
  uint32_t rkey = remoteMrHandle->rkeys[rank];
  uint32_t lkey = localMrHandle->mrHandle->mrs[0]->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_READ;
  wr.send_flags              = IBV_SEND_SIGNALED; // TODO: Potentially optimize this?
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)remotePtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)localPtr;
  sge.length = size;
  sge.lkey = lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyIPutSignal(void *rmaCtx, int context, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void *signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, bool isStrongSignal, void **request) {
  (void)isStrongSignal;
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclRmaIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];

  struct ncclRmaIbProxyMrHandle *srcMrHandle = (struct ncclRmaIbProxyMrHandle *)srcMhandle;
  struct ncclRmaIbProxyMrHandle *dstMrHandle = (struct ncclRmaIbProxyMrHandle *)dstMhandle;
  struct ncclRmaIbProxyMrHandle *signalMrHandle = (struct ncclRmaIbProxyMrHandle *)signalMhandle;

  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetSendComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->rmaProxyCtx = rmaProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    void *srcPtr = (void *)(srcMrHandle->base_vas[rmaProxyCtx->rank] + srcOff);
    void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    // PUT
    wr[0].opcode                  = IBV_WR_RDMA_WRITE;
    wr[0].send_flags              = 0; // We only need the CQE from the signal
    wr[0].wr_id                   = req - comm->base.reqs;
    wr[0].next                    = &wr[1];
    wr[0].wr.rdma.remote_addr     = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey            = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;  // Local buffer address
    sge[0].length = size;  // Size of the transfer
    sge[0].lkey = lkey;  // Local key
  }

  void *signalPtr = (void *)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  // SIGNAL
  wr[1].opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags              = IBV_SEND_SIGNALED;
  wr[1].wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  wr[1].next                    = NULL;
  wr[1].wr.atomic.remote_addr   = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey          = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Send the put and the signal in one go
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, size > 0 ? &wr[0] : &wr[1], &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyTest(void* collComm, void *request, int *done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct ncclRmaIbProxyCtx* rmaProxyCtx = (struct ncclRmaIbProxyCtx*)req->rmaProxyCtx;
  int rank = req->iput.rank;
  *done = 0;

  if (req->events[0] == 0) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));
    return ncclSuccess;
  }
  int wrDone = 0;
  struct ibv_wc wc[4];

  ncclIbNetCommBase* commBase;
  ncclIbNetCommDevBase* devBase;
  if (req->type == NCCL_NET_IB_REQ_FLUSH) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)rmaProxyCtx->fullRecvComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
  } else {
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*)rmaProxyCtx->fullSendComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
  }
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, 4, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char* localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&commBase->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN+1];
      char *hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
          ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, ncclIbReqTypeStr[req->type],
          localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
      printIbWcStatusHint(wc[i].status);
      return ncclRemoteError;
    }

    struct ncclIbRequest* wcReq = commBase->reqs + wc[i].wr_id;

    wcReq->events[0]--;
    if (wcReq == req && wcReq->events[0] == 0) {
      *done = 1;
      NCCLCHECK(ncclIbFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaIbProxyIFlush(void *rmaCtx, int context, void* mhandle, uint32_t rank, void **request) {
  struct ncclRmaIbProxyCtx* rmaProxyCtx = &((struct ncclRmaIbProxyCtx*)rmaCtx)[context];
  struct ncclRmaIbProxyMrHandle *rmaMrHandle = (struct ncclRmaIbProxyMrHandle *)mhandle;
  struct ncclIbRecvComm* comm;
  NCCLCHECK(ncclRmaIbProxyGetRecvComm(rmaProxyCtx, rank, &comm));
  struct ncclIbQp *qp = &comm->devs[0].gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->rmaProxyCtx = rmaProxyCtx;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  void *flushPtr = (void *)(rmaMrHandle->base_vas[rank]);
  wr.wr.rdma.remote_addr = (uint64_t)flushPtr;
  wr.wr.rdma.rkey = rmaMrHandle->rkeys[rank];
  wr.sg_list = &comm->devs[qp->devIndex].gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  TIME_STOP(4);

  ncclIbAddEvent(req, qp->devIndex);

  TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);

  *request = req;
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclRma_t ncclRmaIbProxy = {
  "RMA_IB_PROXY",
  ncclRmaIbProxyInit,
  ncclIbDevices,
  ncclRmaIbProxyGetProperties,
  ncclIbListen,
  ncclRmaIbProxyConnect,
  ncclRmaIbProxyCreateContext,
  ncclRmaIbProxyRegMrSym,
  ncclRmaIbProxyRegMrSymDmaBuf,
  ncclRmaIbProxyDeregMrSym,
  ncclRmaIbProxyDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclRmaIbProxyIPut,
  ncclRmaIbProxyIPutSignal,
  ncclRmaIbProxyIGet,
  ncclRmaIbProxyIFlush,
  ncclRmaIbProxyTest,
  NULL,
  NULL,
  ncclGinIbFinalize
};
