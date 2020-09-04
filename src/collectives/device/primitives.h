/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)

#define ROLE_SRC       0x01
#define ROLE_DST       0x02
#define ROLE_WAIT_RECV 0x04
#define ROLE_WAIT_SEND 0x08
#define ROLE_POST_SEND 0x10
#define ROLE_POST_RECV 0x20

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* conn = NULL;
  volatile int* connSizesFifoPtr = NULL;
  void** connPtrsFifoPtr = NULL;
  volatile uint64_t* connHeadPtr = NULL;
  volatile uint64_t* connTailPtr = NULL;
  uint64_t connTailCache; // Cache last seen value
  uint64_t connHeadCache; // Cache last seen value

  int index; // Peer index I'm responsible for
  int peer = -1;
  int role = 0;
  int group;
  uint64_t step;
  T* direct = NULL;
  T* buff;
  struct ncclDevComm* comm;

  const T** srcs;
  T** dsts;

  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads) barrier();
    else asm volatile ("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort() {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  template <int DIRECTPTR>
  inline __device__ T* directPtr(ssize_t directOffset) {
    return DIRECTPTR && direct ? direct+directOffset : buff+(step%NCCL_STEPS)*stepSize;
  }

  template <int DST, int DIRECTSEND>
  inline __device__ void waitSend(ssize_t directOffset, int nbytes) {
    spins = 0;
    while (connHeadCache + NCCL_STEPS < step + SLICESTEPS) {
      connHeadCache = *connHeadPtr;
      if (checkAbort()) break;
    }
    if (connSizesFifoPtr) {
      connSizesFifoPtr[step%NCCL_STEPS] = nbytes;
    }

    if (connPtrsFifoPtr) loadPtr(connPtrsFifoPtr+step%NCCL_STEPS, dsts[DST+index]);
    else dsts[DST+index] = directPtr<DIRECTSEND>(directOffset);
    step += SLICESTEPS;
  }

  template <int SRC, int DIRECTRECV>
  inline __device__ void waitRecv(ssize_t directOffset) {
    spins = 0;
    while (connTailCache < step + SLICESTEPS) {
      connTailCache = *connTailPtr;
      if (checkAbort()) break;
    }
    if (connPtrsFifoPtr) loadPtr(connPtrsFifoPtr+step%NCCL_STEPS, srcs[SRC+index]);
    else srcs[SRC+index] = directPtr<DIRECTRECV>(directOffset);
    step += SLICESTEPS;
  }

  inline __device__ void postRecv() {
    *connHeadPtr = step += SLICESTEPS;
  }

  inline __device__ void postSend() {
    *connTailPtr = step += SLICESTEPS;
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (tid < nworkers) {
        if (SRC && (role & ROLE_SRC)) srcs[0] = srcPtr+offset;
        if (RECV && (role & ROLE_WAIT_RECV)) waitRecv<SRC, DIRECTRECV>(directOffset+offset);
        if (DST && (role & ROLE_DST)) dsts[0] = dstPtr+offset;
        if (SEND && (role & ROLE_WAIT_SEND)) waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && srcs[0] == dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              // (1-SEND) is only there to avoid compilation errors in case NSEND=0 (and SEND=0).
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1-SEND)+NSEND>(tid, nworkers, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nworkers, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
      }
      barrier();
      if (SEND && (role & ROLE_POST_SEND) && realSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      if (SEND && (role & ROLE_POST_SEND)) postSend();
      if (RECV && (role & ROLE_POST_RECV)) postRecv();
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclChannel* channel, T* directBuff) {
    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) {
      conn = &channel->devPeers[peer].recv.conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_POST_RECV) {
        connHeadPtr = conn->head;
        // Return credits in case we rounded up.
        *connHeadPtr = step;
      }
      if (role & ROLE_WAIT_RECV) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          direct = directBuff;
          *conn->ptrExchange = directBuff;
        }
        connTailPtr = conn->tail;
        connTailCache = *connTailPtr;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclChannel* channel) {
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) {
      conn = &channel->devPeers[peer].send.conn;
      step = conn->step;
      step = ROUNDUP(step, SLICESPERCHUNK*SLICESTEPS);
      if (role & ROLE_POST_SEND) {
        connTailPtr = conn->tail;
      }
      if (role & ROLE_WAIT_SEND) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
          void* volatile* ptr = conn->ptrExchange;
          while ((direct = (T*)(*ptr)) == NULL);
          *ptr = NULL;
        }
        connHeadPtr = conn->head;
        connHeadCache = *connHeadPtr;
        connSizesFifoPtr = conn->sizesFifo;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void saveSync() {
    if (role & (ROLE_POST_SEND|ROLE_POST_RECV)) {
      conn->step = step;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nworkers, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, struct ncclShmemPtrs* ptrs, int group)
    : comm(comm), tid(tid), nworkers(nworkers), stepSize(stepSize), srcs((const T**)ptrs[group].srcs), dsts((T**)ptrs[group].dsts), group(group) {
    nthreads = nworkers;
    // For send operations, we need an extra warp to overlap the threadfence and the copy
    int postThreads = NSEND && nworkers >= 64 ? WARP_SIZE : 0;
    nthreads += postThreads;

    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV; i++) if (recvPeers[i] != -1) nrecv++;
    for (int i=0; i<NSEND; i++) if (sendPeers[i] != -1) nsend++;

    #define SYNC_GROUP 8
    static_assert(NSEND < SYNC_GROUP && NRECV < SYNC_GROUP, "Not enough threads to cover all peers");

    int g = tid / SYNC_GROUP;
    int ng = nthreads / SYNC_GROUP;
    index = tid % SYNC_GROUP;

    if (g == 0) {
      if (index < nrecv) role |= ROLE_WAIT_RECV;
      if (index == nrecv) role |= ROLE_SRC;
    } else if (g == 1) {
      if (index < nsend) role |= ROLE_WAIT_SEND;
      if (index == nsend) role |= ROLE_DST;
    } else if (g == ng - 2) {
      if (index < nrecv) role |= ROLE_POST_RECV;
    } else if (g == ng - 1) {
      if (index < nsend) role |= ROLE_POST_SEND;
    }

    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) peer = recvPeers[index];
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)) peer = sendPeers[index];

    loadRecvConn(channel, directBuff);
    loadSendConn(channel);
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#endif
