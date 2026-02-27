/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_P2P_H_
#define NET_IB_P2P_H_

#include "common.h"

#define NCCL_IB_FLUSH_REQ_WR_ID_OFFSET 0x1000
static_assert(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET > NET_IB_MAX_REQUESTS, "wr_id offset for flush requests must be greater than NET_IB_MAX_REQUESTS");
static_assert(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET <= UINT64_MAX - NET_IB_MAX_REQUESTS, "wr_id for flush requests must fit in 64 bits since ibv_send_wr::wr_id is 64 bits");

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
