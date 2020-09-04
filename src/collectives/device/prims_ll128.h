/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "op128.h"

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLL128Primitives {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int warp;
  const bool flagThread;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  uint64_t* recvBuff[NRECV];
  uint64_t* sendBuff[NSEND];
  struct ncclDevComm* comm;

  volatile uint64_t* shmem;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

  inline __device__ void barrier() {
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
    } else {
      asm volatile ("bar.sync 2, %0;" :: "r"(nthreads));
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendStep[wid]%NCCL_STEPS] = nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += 1;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) { __threadfence(); *sendConnTailPtr = sendConnTail += 1; }
  }

  template <int ELEMS_PER_THREAD>
  inline __device__ void loadSrcToShmem128(int maxOffset, const uint64_t* src64Ptr) {
#if 0
    uint64_t v[ELEMS_PER_THREAD];
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      if (u*WARP_SIZE < maxOffset) load128(src64Ptr+u*WARP_SIZE, v[u], v[u+1]);
    }
    uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      storeShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
    }
#else
    uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      if (u*WARP_SIZE < maxOffset) {
        uint64_t v0, v1;
        load128(src64Ptr+u*WARP_SIZE, v0, v1);
        storeShmem128(shmemAsmPtr+u*WARP_SIZE, v0, v1);
      }
    }
#endif
  }

  inline __device__ void loadSrcToShmem(int start, int end, const T* srcPtr) {
    T* shmemPtr = (T*)(shmem-2*wid);
    for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
      shmemPtr[offset] = srcPtr[offset];
    }
  }

  template <int ELEMS_PER_THREAD>
  inline __device__ void storeShmemToDst128(int maxOffset, uint64_t* dst64Ptr) {
    uint64_t v[ELEMS_PER_THREAD];
    uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      loadShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
    }
    #pragma unroll
    for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
      if (u*WARP_SIZE < maxOffset) store128(dst64Ptr+u*WARP_SIZE, v[u], v[u+1]);
    }
  }

  inline __device__ void storeShmemToDst(int start, int end, T* dstPtr) {
    T* shmemPtr = (T*)(shmem-2*wid);
    for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
      dstPtr[offset] = shmemPtr[offset];
    }
  }

  #define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SRC, int DST>
  __device__ __forceinline__ void recvReduceSendCopy(int ll128Offset) {
    uint64_t v[ELEMS_PER_THREAD];

    /************* Data Loading : SHMEM -> REG **************/
    if (SRC) {
      volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u] = shmem64Ptr[u*(WARP_SIZE-2)];
        if (!flagThread) v[u+1] = shmem64Ptr[u*(WARP_SIZE-2)+1];
      }
    }
    /*********** End Data Loading : SHMEM -> REG ************/

    /************************ Recv **************************/
    if (RECV) {
      uint64_t flag = recvFlag(0);
      uint64_t* ptr = recvPtr(0)+ll128Offset;
      bool needReload;
      uint64_t v0, v1;
      do {
        needReload = false;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          needReload |= flagThread && (v1 != flag);
        }
      } while (__any_sync(WARP_MASK, needReload) && checkAbort(0, 0) == 0);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        load128(ptr+u*WARP_SIZE, v0, v1);
        v[u] = SRC ? MULTI<FUNC, T>()(v0, v[u]) : v0;
        v[u+1] = SRC ? MULTI<FUNC, T>()(v1, v[u+1]) : v1;
      }

      for (int i=1; i<NRECV && i<nrecv; i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128Offset;
        uint64_t v0, v1;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            needReload |= flagThread && (v1 != flag);
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(i, 0) == 0);
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          v[u] = MULTI<FUNC, T>()(v0, v[u]);
          v[u+1] = MULTI<FUNC, T>()(v1, v[u+1]);
        }
      }
    }
    /********************** End Recv ************************/

    /************************ Send **************************/
    if (SEND) {
      for (int i=1; i<NSEND && i<nsend; i++) {
        uint64_t flag = sendFlag(i);
        uint64_t* ptr = sendPtr(i)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      uint64_t flag = sendFlag(0);
      uint64_t* ptr = sendPtr(0)+ll128Offset;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
      }
    }
    /********************** End Send ************************/

    /************* Data Storing : REG -> SHMEM **************/
    if (DST) {
      volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        shmem64Ptr[u*(WARP_SIZE-2)] = v[u];
        if (!flagThread) shmem64Ptr[u*(WARP_SIZE-2)+1] = v[u+1];
      }
    }
    /*********** End data Storing : REG -> SHMEM ************/
  }

  #define LL128INC (WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD)
  #define ELEMINC (LL128INC-(LL128INC/NCCL_LL128_LINEELEMS))

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void GenericOp(const T* srcPtr, T* dstPtr, int nelem) {
    if (nelem <= 0) {
      // Don't move any data but still increase steps and sync with prev/next
      if (SEND) waitSend(0);
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
      return;
    }
    const int nelem64 = ((nelem*sizeof(T))/(2*sizeof(uint64_t)))*2;
    const uint64_t* src64Ptr = ((uint64_t*)srcPtr);
    uint64_t* dst64Ptr = ((uint64_t*)dstPtr);

    int ll128Offset = LL128INC*warp+2*wid;
    int elemOffset = ELEMINC*warp;
    const int nwarps = nthreads/WARP_SIZE;

    if (SEND) waitSend(DIVUP(nelem*sizeof(T), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
    barrier();

    while (elemOffset*(sizeof(uint64_t)/sizeof(T)) < nelem) {
      const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
      const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(T)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(T))));
      if (SRC) {
        int done = 0;
        if ((((uint64_t)srcPtr)&0xf) == 0) {
          loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, src64Ptr+elemOffset+2*wid);
          done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
        }
        loadSrcToShmem(done, maxOffset, (T*)(src64Ptr+elemOffset));
      }
      __syncwarp();
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST>(ll128Offset);
      __syncwarp();
      if (DST) {
        int done = 0;
        if ((((uint64_t)dstPtr)&0xf) == 0) {
          storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, dst64Ptr+elemOffset+2*wid);
          done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
        }
        storeShmemToDst(done, maxOffset, (T*)(dst64Ptr+elemOffset));
      }
      __syncwarp();
      ll128Offset += LL128INC*nwarps;
      elemOffset += ELEMINC*nwarps;
    }

    barrier();
    FOR_SEND(incSend); if (SEND) postSend();
    FOR_RECV(incRecv); if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifoPtr = sendConn->sizesFifo;
    }
    if (tid >= nthreads-WARP_SIZE && wid<nsend) {
      if (sendConn->sizesFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConn->step = recvConnHead;
      __threadfence_block();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      __threadfence_block();
    }
  }

 public:
  __device__ __forceinline__
  ncclLL128Primitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE), flagThread((tid%8)==7), stepSize(stepSize), shmem(ncclShmem->data+(threadIdx.x/WARP_SIZE)*NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE+2*wid) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  __device__ void send(const T* src, int nelem) {
    return GenericOp<0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recv(T* dst, int nelem) {
    return GenericOp<1, 0, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return GenericOp<1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    return GenericOp<1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void copySend(const T* src, T* dst, int nelem) {
    return GenericOp<0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySend(T* dst, int nelem) {
    return GenericOp<1, 1, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem) {
    return GenericOp<1, 1, 1, 1>(src, dst, nelem);
  }

  __device__ __forceinline__ ~ncclLL128Primitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
