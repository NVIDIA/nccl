/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RMA_SOCKET_H_
#define NCCL_RMA_SOCKET_H_

#include "core.h"
#include "gdrwrap.h"
#include "socket.h"
#include "rma.h"

#include <mutex>

#define NCCL_RMA_SOCKET_MAX_REQUESTS_PER_PEER NCCL_NET_MAX_REQUESTS
#define NCCL_RMA_SOCKET_CHUNK_SIZE (4 * 1024 * 1024)

#define NCCL_RMA_SOCKET_OP_HAS_SIGNAL (1U << 0)
#define NCCL_RMA_SOCKET_CONTROL_ACK_MAGIC 0x524d4141U

struct ncclRmaSocketProxyCtx;

enum ncclRmaSocketProxyMsgType {
  ncclRmaSocketProxyMsgTypeIput = 1,
  ncclRmaSocketProxyMsgTypeRequestGetData = 2,  // iGet caller -> callee: request to read the callee's memory
  ncclRmaSocketProxyMsgTypeRespondGetData = 3,  // iGet callee -> caller: respond with status and requested bytes
};

enum ncclRmaSocketProxyRequestType {
  ncclRmaSocketProxyRequestTypeFree = 0,
  ncclRmaSocketProxyRequestTypePut = 1,
  ncclRmaSocketProxyRequestTypeGet = 2,
};

enum ncclRmaSocketProxyRecvState {
  ncclRmaSocketProxyRecvStateControlMsg = 0,
  ncclRmaSocketProxyRecvStateControlAck = 1,
  ncclRmaSocketProxyRecvStatePayload = 2,
  ncclRmaSocketProxyRecvStateClosed = 3,
};

struct ncclRmaSocketProxyHeader {
  uint16_t type;
  uint32_t status;
  uint64_t requestId;
  uint32_t opFlags;
  uint32_t signalOp;
  uint64_t mrId;
  uint64_t dstOff;
  uint64_t size;
  uint64_t localMrId;
  uint64_t localOff;
  uint64_t signalMrId;
  uint64_t signalOff;
  uint64_t signalValue;
};

struct ncclRmaSocketProxyMrHandle {
  // Registration record: the symmetric MR the user registered.
  void* data;
  size_t size;
  int type;
  uint64_t mrId;
  int recvActiveOps;
  // GDRCopy mapping for CUDA MRs.
  struct {
    int mapped;
    void* map;
    void* barPtr;
    size_t mapSize;
    gdr_mh_t mh;
  } gdr;
  struct ncclRmaSocketProxyMrHandle* next;
};

/* Opaque API completion object. Wire-progress state belongs to SendTask. */
struct ncclRmaSocketProxyRequest {
  enum ncclRmaSocketProxyRequestType type;
  struct ncclRmaSocketProxyCtx* ctx;
  int peer;
  uint64_t globalRequestId;
};

/* Reusable queue slot for one control message and its optional payload. */
struct ncclRmaSocketProxySendTask {
  struct ncclRmaSocketProxyHeader controlMsg;
  void* srcPtr;
  struct ncclRmaSocketProxyMrHandle* srcHandle;
  size_t size;
  int srcType;
  int controlMsgOffset;
  uint32_t controlAck;
  int controlAckOffset;
  size_t payloadOffset;
  int chunkSize;
  int chunkOffset;
};

struct ncclRmaSocketProxyPeerSender {
  struct ncclSocket sock;
  void* buffer;
  uint64_t lastCompletedRequestGlobalId;  // rolling watermark; initially UINT64_MAX

  struct ncclRmaSocketProxySendTask* tasks;
  int capacity;
  int head;
  int tail;
  int count;
};

struct ncclRmaSocketProxyPeerReceiver {
  struct ncclSocket sock;
  void* buffer;
  uint64_t lastCompletedRequestGlobalId;  // rolling GET watermark; initially UINT64_MAX

  // State of the receiver
  enum ncclRmaSocketProxyRecvState recvState;

  // Last received / in-progress control message and its payload status
  struct ncclRmaSocketProxyHeader controlMsg;
  int controlMsgOffset;
  uint32_t controlAck;
  int controlAckOffset;
  size_t recvPayloadOffset;
  int recvChunkSize;
  int recvChunkOffset;

  // Local data target resolved from the active control message.
  void* dataPtr;
  int dataType;
  struct ncclRmaSocketProxyMrHandle* dataHandle;

  // Local signal target resolved from the active control message.
  void* signalPtr;
  int signalType;
  uint64_t signalDelta;    // Value added to the signal target.
  int hasSignal;
  struct ncclRmaSocketProxyMrHandle* signalHandle;
};

struct ncclRmaSocketProxyCollComm {
  int rank;
  int nranks;
  uint64_t nextGlobalRequestId;  // next collComm-local request ID; wraps naturally at UINT64_MAX
  struct ncclRmaSocketProxyPeerSender* peerSender;
  struct ncclRmaSocketProxyPeerReceiver* peerReceiver;
  int nContexts;
  uint64_t nextMrId;
  struct ncclRmaSocketProxyMrHandle* mrHead;
  ncclResult_t lastError;
};

struct ncclRmaSocketProxyCtx {
  struct ncclRmaSocketProxyCollComm* collComm;
  int nContexts;
  struct ncclRmaSocketProxyRequest* requests;
  int requestCount;
  int activeOps;
};

struct ncclRmaSocketProxyHandle {
  union ncclSocketAddress connectAddr;
  uint64_t magic;
};

/* Memory layer (rma_socket_mem.cc). */
bool ncclRmaSocketProxySupportedPtrType(int type);
ncclResult_t ncclRmaSocketProxyAllocChunkBuffer(void** ptr, size_t size);
ncclResult_t ncclRmaSocketProxyFreeChunkBuffer(void* ptr);
ncclResult_t ncclRmaSocketProxyFreeMr(struct ncclRmaSocketProxyMrHandle* handle);
ncclResult_t ncclRmaSocketProxyGdrPtr(struct ncclRmaSocketProxyMrHandle* handle, const void* ptr, size_t size,
                                      void** gdrPtr);
ncclResult_t ncclRmaSocketProxyCopy(void* dst, int dstType, struct ncclRmaSocketProxyMrHandle* dstHandle,
                                    const void* src, int srcType, struct ncclRmaSocketProxyMrHandle* srcHandle,
                                    size_t size);
ncclResult_t ncclRmaSocketProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                        void** mhandle);
ncclResult_t ncclRmaSocketProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset,
                                              int fd, uint64_t mrFlags, void** mhandle);
ncclResult_t ncclRmaSocketProxyDeregMrSym(void* collComm, void* mhandle);

/* Connection/context setup (rma_socket_setup.cc). */
ncclResult_t ncclRmaSocketProxyConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                       void** collComm);
ncclResult_t ncclRmaSocketProxyCreateContext(void* collComm, ncclRmaConfig_t* config, void** rmaCtx);
ncclResult_t ncclRmaSocketProxyDestroyContext(void* rmaCtx);
ncclResult_t ncclRmaSocketProxyCloseColl(void* collComm);

/* Plugin data-path callbacks (rma_socket.cc). */
ncclResult_t ncclRmaSocketProxyIPut(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request);
ncclResult_t ncclRmaSocketProxyIPutSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                          uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                          void* signalMhandle, uint64_t signalValue, uint32_t signalOp,
                                          bool isStrongSignal, void** request);
ncclResult_t ncclRmaSocketProxyIGet(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                                    uint64_t localOff, void* localMhandle, uint32_t rank, void** request);
ncclResult_t ncclRmaSocketProxyIFlush(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request);
ncclResult_t ncclRmaSocketProxyTest(void* collComm, void* request, int* done);
ncclResult_t ncclRmaSocketProxyProgress(void* rmaCtx);

ncclResult_t ncclRmaSocketProxyValidateRange(size_t objectSize, uint64_t offset, size_t size, const char* objectName);
ncclResult_t ncclRmaSocketProxyEnqueueRespondGetData(struct ncclRmaSocketProxyPeerSender* sender,
                                                     const struct ncclRmaSocketProxyHeader* requestMsg, uint32_t status,
                                                     void* srcPtr, int srcType,
                                                     struct ncclRmaSocketProxyMrHandle* srcHandle, size_t size);

/* Receive-side data-path progress (rma_socket_progress_recver.cc). */
ncclResult_t ncclRmaSocketProxyProgressPeerRecv(struct ncclRmaSocketProxyCollComm* comm, int peer);

#endif // NCCL_RMA_SOCKET_H_
