/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_CONNECT_H_
#define NET_IB_CONNECT_H_

#include "common.h"
#include "ibvwrap.h"

struct ncclIbQpCreateAttr {
  uint8_t ibPort;
  enum ibv_qp_type type;
  unsigned int accessFlags;
  struct ibv_cq* cq;
  struct ibv_pd* pd;
  uint32_t maxRecvWorkRequest;
  uint32_t maxSendWorkRequest;
};

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;

  // The index of the device on which the QP was created. Allows the sender and
  // receiver side to have asymmetric device configuration, meaning the sender
  // and receiver can use different number of devices.
  int devIndex;
};

struct ncclIbResiliencyInfo {
  // QPs used for probing of data transfers in case of QP/device failures.
  struct ncclIbQpInfo probingQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Structure used to hold information needed to establish the communication
// between the sender and receiver.
// The structure is populated during the connection establishment phase and
// populated by each side of the connection before being sent to the remote
// peer. The remote peer uses the information passed to it from its peer to
// create and initialize its local resources.
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbResiliencyInfo resiliencyInfo;
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  // An address for a registered memory to be accessed by the peer. The address
  // can be accessed using RDMA using the key specified in ncclIbDevInfo::rkey.
  // The sender side gets in this member, from the receiver, the address of the
  // memory to which the sender writes the sizes of the data transfers that
  // the sender sends.
  // The receiver side gets in this member, from the sender, the address of the
  // memory to which the receiver writes the CTS messages.
  uint64_t addr;
  int ndevs;
  int tc;
  int sl;
};

ncclResult_t ncclIbCreateQp(struct ncclIbQpCreateAttr* createQpAttrs, void* qp_context, struct ncclIbQp* qp);
ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbGidInfo* sGidInfo, uint32_t dest_qp_num, struct ncclIbDevInfo* info, bool fifoTc, int tc, int sl);
ncclResult_t ncclIbRtsQp(struct ibv_qp* qp);

#endif // NET_IB_CONNECT_H_
