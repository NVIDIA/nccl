/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "p2p_resiliency.h"

char ncclIbIfName[MAX_IF_NAME_SIZE+1];
union ncclSocketAddress ncclIbIfAddr;

int ncclNMergedIbDevs = -1;
int ncclNIbDevs = -1;
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclIbRelaxedOrderingEnabled = 0;

ncclProfilerCallback_t ncclProfilerFunction;

NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);
NCCL_PARAM(IbPrepostReceiveWorkRequests, "IB_PREPOST_RECEIVE_WORK_REQUESTS", 0);
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);

ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {
    WARN("communicator encountered a fatal error (detected in %s)", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend) {
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) {
    baseComm->qps[i].devIndex= -1;
    baseComm->qps[i].remDevIdx= -1;
    baseComm->activeQps[i] = &baseComm->qps[i];
  }
  baseComm->nqps = -1;
  baseComm->splitDataOnQps = ncclParamIbSplitDataOnQps();
  baseComm->nDataQps = -1;
  baseComm->isSend = isSend;
  baseComm->ready = 0;

  NCCLCHECK(ncclIbResiliencyInit(baseComm, &baseComm->resiliency));

  return ncclSuccess;
}

ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm) {
  NCCLCHECK(ncclIbBaseCommInit(&recvComm->base, false));
  recvComm->ibRecvWorkRequest = {
    .wr_id = NCCL_IB_RECV_WR_ID_DUMMY,
    .next = NULL,
    .sg_list = NULL,
    .num_sge = 0
  };
  if (recvComm->base.resiliency) {
    if (ncclParamIbPrepostReceiveWorkRequests() == 0) {
      WARN("NET/IB: %s: Resiliency requires pre-posted receive work requests. Enabling pre-posting.", __func__);
    }
    recvComm->prepostReceiveWorkRequests = true;
  } else {
    recvComm->prepostReceiveWorkRequests = (ncclParamIbPrepostReceiveWorkRequests() == 1);
  }
  INFO(NCCL_NET, "NET/IB: %s: Receive work requests will be %s", __func__, recvComm->prepostReceiveWorkRequests ? "pre-posted" : "posted on-demand");
  return ncclSuccess;
}

ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm) {
  NCCLCHECK(ncclIbBaseCommInit(&sendComm->base, true));
  return ncclSuccess;
}

std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      ncclIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      ncclIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      ncclIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_GID_CHANGE:
      WARN("NET/IB : %s:%d GID table changed", dev->devName, dev->portNum);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got non-fatal async event: %s(%d)", dev->devName, dev->portNum, str, event.event_type);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbRegMrDmaBuf,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  ncclIbMakeVDevice,
  ncclIbFinalize,
  ncclIbSetNetAttr,
};
