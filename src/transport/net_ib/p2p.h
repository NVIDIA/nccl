/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_H_
#define NET_IB_P2P_H_

#include "common.h"

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot);
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot);

static inline ncclResult_t ncclIbRecvCommGetQpForCts(struct ncclIbRecvComm* recvComm, uint32_t id, ncclIbQp** qp) {
  int devIndex = id % recvComm->base.vProps.ndevs;
  // CTS message is always posted the first QP on the device
  int qpIndex = 0;
  ncclIbCommBaseGetQpByIndex(&recvComm->base, devIndex, qpIndex, qp);
  assert(*qp != NULL);
  return ncclSuccess;
}

static inline ncclResult_t ncclIbRequestRetrieveAsIndex(ncclIbRequest* reqs, uint32_t reqIndex, ncclIbRequest** req) {
  if (reqIndex < 0 || reqIndex >= NET_IB_MAX_REQUESTS) {
    WARN("NET/IB: %s: Invalid request index %d. Not in the range [%d, %d). Cannot retrieve request.", __func__, reqIndex, 0, NET_IB_MAX_REQUESTS);
    return ncclInternalError;
  }
  *req = &reqs[reqIndex];
  return ncclSuccess;
}


#endif // NET_IB_P2P_H_
