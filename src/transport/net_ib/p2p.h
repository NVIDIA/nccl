/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_H_
#define NET_IB_P2P_H_

#include "common.h"

static inline ncclResult_t ncclIbRecvCommGetQpForCts(struct ncclIbRecvComm* recvComm, uint32_t id, ncclIbQp** qp) {
  int devIndex = id % recvComm->base.vProps.ndevs;
  // CTS message is always posted the first QP on the device
  int qpIndex = 0;
  ncclIbCommBaseGetQpByIndex(&recvComm->base, devIndex, qpIndex, qp);
  assert(*qp != NULL);
  return ncclSuccess;
}

#endif // NET_IB_P2P_H_
