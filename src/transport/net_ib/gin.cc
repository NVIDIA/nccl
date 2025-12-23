/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"

#include "gin/gin_host.h"
#include "gin.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

ncclResult_t ncclGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
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
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank * sizeof(int),
                               dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t ncclGinIbConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct ncclGinIbCollComm *cComm = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void **)&cComm, sizeof(*cComm)));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->fullSendComm, sizeof(void *) * nranks));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->fullRecvComm, sizeof(void *) * nranks));

  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclNetIb.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclIbGetGidIndex;
  cComm->dev = lComm->dev;

  for (int i = 0; i < nranks; i++)
  {
    int connectPeer = (cComm->rank + i) % nranks;
    int acceptPeer = (cComm->rank - i + nranks) % nranks;
    do
    {
      if (cComm->fullSendComm[connectPeer] == NULL)
        NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[connectPeer], &cComm->fullSendComm[connectPeer], NULL));
      if (cComm->fullRecvComm[acceptPeer] == NULL)
        NCCLCHECK(ncclNetIb.accept(lComm, &cComm->fullRecvComm[acceptPeer], NULL));
    } while ((cComm->fullSendComm[connectPeer] == NULL) || (cComm->fullRecvComm[acceptPeer] == NULL));
    NCCLCHECK(ncclGinIbP2PBarrier(cComm));
  }

  *collComm = cComm;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  if (!cComm) return ncclSuccess;

  if (cComm->fullRecvComm) {
    for (int i=0; i<cComm->nranks; i++) {
      NCCLCHECK(ncclNetIb.closeRecv(cComm->fullRecvComm[i]));
    }
    free(cComm->fullRecvComm);
    cComm->fullRecvComm = NULL;
  }

  if (cComm->fullSendComm) {
    for (int i=0; i<cComm->nranks; i++) {
      NCCLCHECK(ncclNetIb.closeSend(cComm->fullSendComm[i]));
    }
    free(cComm->fullSendComm);
    cComm->fullSendComm = NULL;
  }

  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cComm);
  return ncclSuccess;
}

#include "gdaki/gin_host_gdaki.h"

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;
int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  NCCLCHECK(ncclGinIbInit(ctx, commId, logFunction));
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
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

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
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

ncclResult_t ncclGinIbGdakiConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  NCCLCHECK(ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclGinIbGdakiGetProperties;
  cComm->ibvCtx = ncclIbDevs[ncclGinIbGdakiDevIndexes[cComm->dev]].context;

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, int nSignals, int nCounters, void **ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, nSignals, nCounters, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm *)collComm, data, size, type, mhandle, ginHandle);
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
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};


struct ncclIbGinProxyMrHandle {
  struct ncclIbMrHandle *mrHandle;
  uintptr_t *base_vas;
  uint32_t *rkeys;
};

ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIb.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle;
  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  NCCLCHECKNOWARN(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags, (void **)&ginMrHandle->mrHandle), NCCL_NET);

  NCCLCHECK(ncclCalloc(&ginMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&ginMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, ginMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &ginMrHandle->mrHandle->mrs[0]->rkey, ginMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;

  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;

  NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle));
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPut(void *collComm, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank, void **request)
{
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;

  void *srcPtr = (void *)(srcMrHandle->base_vas[cComm->rank] + srcOff);
  void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)cComm->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
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

ncclResult_t ncclGinIbProxyIPutSignal(void *collComm, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle,
                                      uint32_t rank, uint64_t signalOff, void *signalMhandle,
                                      uint64_t signalValue, uint32_t signalOp, void **request)
{
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;
  struct ncclIbGinProxyMrHandle *signalMrHandle = (struct ncclIbGinProxyMrHandle *)signalMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)cComm->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
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
    void *srcPtr = (void *)(srcMrHandle->base_vas[cComm->rank] + srcOff);
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

ncclResult_t ncclGinIbProxyTest(void *collComm, void *request, int *done) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  int rank = req->iput.rank;
  *done = 0;

  if (req->events[0] == 0) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));
    return ncclSuccess;
  }
  int wrDone = 0;
  struct ibv_wc wc[4];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)cComm->fullSendComm[rank];
  NCCLCHECK(wrap_ibv_poll_cq(comm->devs[0].base.cq, 4, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char* localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&req->devBases[i]->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&req->base->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN+1];
      char *hcaName = req->devBases[i]->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
          ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, ncclIbReqTypeStr[req->type],
          localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
      return ncclRemoteError;
    }

    struct ncclIbRequest* wcReq = comm->base.reqs + wc[i].wr_id;

    wcReq->events[0]--;
    if (wcReq == req && wcReq->events[0] == 0) {
      *done = 1;
      NCCLCHECK(ncclIbFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclGin_t ncclGinIbProxy = {
  "GIN_IB_PROXY",
  ncclGinIbInit,
  ncclIbDevices,
  ncclGinIbProxyGetProperties,
  ncclIbListen,
  ncclGinIbConnect,
  NULL,
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  NULL,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  ncclGinIbProxyTest,
  NULL,
  NULL,
  ncclGinIbFinalize
};
