/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_RAW_TASK_H_
#define NCCL_ENQUEUE_RAW_TASK_H_

#include "utils.h"
#include "device.h"
#include "sym_kernels.h"
#include "nccl.h"
#include "config/collconfig.h"

struct ncclComm;

enum ncclTaskKind {
  ncclTaskKindColl = 0,
  ncclTaskKindSendRecv = 1,
  ncclTaskKindRma = 2,
  ncclTaskKindAllGatherV = 3,
};

// Collective API inputs captured at enqueue time.
struct ncclRawTaskColl {
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;
  cudaStream_t stream;
  ncclCollConfig_t collConfig;
};

// AllGatherV inputs built by merging broadcast raw tasks during pre-tuning.
// Per-rank recvbuff and counts arrays are allocated with length nRanks.
// sendbuff is set only when this rank is the broadcast root.
struct ncclRawTaskAllGatherV {
  ncclFunc_t func;
  int nRanks;
  void const* sendbuff;
  void** recvbuff;
  size_t* counts;
  size_t maxCount;
  ncclDataType_t datatype;
  cudaStream_t stream;
};

// Point-to-point API inputs captured at enqueue time.
struct ncclRawTaskSendRecv {
  ncclFunc_t func;
  ncclFunc_t collAPI;
  void* buff;
  size_t count;
  ncclDataType_t datatype;
  int peer;
  size_t bytes;
  cudaStream_t stream;
};

// ncclPutSignal API inputs captured at enqueue time.
struct ncclRawTaskPutSignal {
  const void* localbuff;
  size_t count;
  ncclDataType_t datatype;
  int peer;
  ncclWindow_t peerWin;
  size_t peerWinOffset;
  int sigIdx;
  int ctx;
  unsigned int flags;
  cudaStream_t stream;
};

// ncclSignal API inputs captured at enqueue time.
struct ncclRawTaskSignal {
  int peer;
  int sigIdx;
  int ctx;
  unsigned int flags;
  cudaStream_t stream;
};

// ncclWaitSignal API inputs captured at enqueue time.
struct ncclRawTaskWaitSignal {
  int nDesc;
  ncclWaitSignalDesc_t* signalDescs;
  cudaStream_t stream;
};

// RMA API inputs captured at enqueue time.
struct ncclRawTaskRma {
  ncclFunc_t func;
  union {
    struct ncclRawTaskPutSignal putSignal;
    struct ncclRawTaskSignal signal;
    struct ncclRawTaskWaitSignal waitSignal;
  } rmaOp;
};

// Pool allocation backing for any raw task type.
struct ncclRawTask {
  struct ncclRawTask* next;
  ncclTaskKind kind;
  union {
    struct ncclRawTaskColl coll;
    struct ncclRawTaskSendRecv sendRecv;
    struct ncclRawTaskRma rma;
    struct ncclRawTaskAllGatherV allGatherV;
  };
};

struct ncclRawTaskQueue {
  struct ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next> genericQueue;
  struct ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next> bcastQueue;
};

#endif // NCCL_ENQUEUE_RAW_TASK_H_
