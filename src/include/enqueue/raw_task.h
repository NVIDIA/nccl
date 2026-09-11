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

#include <map>

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
  uint64_t algMask;
};

// Per-root contribution stored when merging broadcasts into an AllGatherV.
struct ncclRawTaskAllGatherVBcast {
  void* sendbuff;
  void* recvbuff;
  size_t count;
};

using ncclRawTaskAllGatherVBcastMap = std::map<int, ncclRawTaskAllGatherVBcast>;

// AllGatherV inputs built by merging broadcast raw tasks during pre-tuning.
// bcastTasks is heap-allocated so ncclRawTaskAllGatherV can live in the ncclRawTask union.
struct ncclRawTaskAllGatherV {
  size_t maxCount;
  cudaStream_t stream;
  uint64_t algMask;
  ncclRawTaskAllGatherVBcastMap* bcastTasks;
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
  cudaEvent_t launchCompletionEvent;
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

// ncclWaitSignal API inputs captured at enqueue time. A descriptor carries its own ctx
// and peer, so a multi-descriptor ncclWaitSignal is split into one task per descriptor.
struct ncclRawTaskWaitSignal {
  ncclWaitSignalDesc_t signalDesc;
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
  ncclCollConfig_t collConfig;
  union {
    struct ncclRawTaskColl coll;
    struct ncclRawTaskSendRecv sendRecv;
    struct ncclRawTaskRma rma;
    struct ncclRawTaskAllGatherV allGatherV;
  };
};

using ncclRawTaskBcastQueueMap = std::map<int, ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next>>;

struct ncclRawTaskQueue {
  struct ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next> genericQueue;
  ncclRawTaskBcastQueueMap* bcastQueue;
};

#endif // NCCL_ENQUEUE_RAW_TASK_H_
