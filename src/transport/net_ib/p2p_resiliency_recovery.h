/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_RECOVERY_H_
#define NET_IB_P2P_RESILIENCY_RECOVERY_H_

#include "nccl.h" // For ncclResult_t
#include "p2p_resiliency.h"

ncclResult_t ncclIbPortRecoveryThreadStart();
ncclResult_t ncclIbPortRecoveryThreadStop();
ncclResult_t ncclIbPortRecoveryInit(struct ncclIbResiliency* resCtx);
ncclResult_t ncclIbPortRecoveryClose(struct ncclIbResiliency* resCtx);

ncclResult_t ncclIbPortRecoveryDevInit(struct ncclIbResiliency* resCtx, int devIndex, ncclIbDev* ibDev);
ncclResult_t ncclIbPortRecoveryDevDestroy(struct ncclIbResiliency* resCtx, int devIndex);

ncclResult_t ncclIbPortRecoverySenderQpsCreate(struct ncclIbResiliency* resCtx, struct ncclIbQpInfo* localResiliencyInfo, int nQps);
ncclResult_t ncclIbPortRecoverySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, int nQps);

ncclResult_t ncclIbPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, struct ncclIbQpInfo* localPortRecoveryQpsInfo, int nQps);

ncclResult_t ncclIbPortRecoveryQpsDestroy(struct ncclIbResiliency* resCtx, int nQps);

ncclResult_t ncclIbPortRecoveryHandleFailure(struct ncclIbResiliency* resCtx, int devIndex);

#endif // NET_IB_P2P_RESILIENCY_RECOVERY_H_