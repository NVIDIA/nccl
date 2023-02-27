/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "op128.h"

#define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

template<typename T, typename RedOp, typename Fan, int Direct, int P2p>
class Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p>:
  public PrimitivesWithoutDirect<Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p>> {

  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  RedOp redOp;
  const int tid; // thread index in primitives group
  const int nthreads; // thread count in primitives group
  const int wid; // lane index in warp
  const int stepSize;
  const int warp; // warp index in primitives group
  const int warpInBlock; // warp index in thread block
  const bool flagThread;
  const int group;
  Fan fan;
  T *userBufs[2];
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

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  uint64_t* recvBuff[MaxRecv];
  uint64_t* sendBuff[MaxSend];

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
  inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

  inline __device__ void barrier() {
    asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15-group));
  }

  uint32_t abort = 0;

  inline __device__ int checkAbort(int &spins, int i, int send) {
    spins++;
    if (abort == 0 && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      abort = *ncclShmem.comm.abortFlag;
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    if (sendConnHeadPtr) {
      int spins = 0;
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(spins, wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendStep[wid]%NCCL_STEPS] = nbytes;
      }
      sendConnHead += 1;
    }
  }

  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) {
#if __CUDA_ARCH__ >= 900
      __threadfence_system();
#else
      __threadfence();
#endif
      *sendConnTailPtr = sendConnTail += 1;
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsBegin(uint64_t(&regs)[WordPerThread], T const *src, int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    if(reinterpret_cast<uintptr_t>(src)%16 == 0) {
      /* We are aligned to 16 bytes, so load directly to registers no shmem.
       * Flag threads load half as much data which gets shuffled to the even
       * registers during Finish. The point of splitting into two phases is to
       * defer that shuffle, which incurs a dependency stall, until after other
       * memops are launched by the caller.
       */
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
        int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
        if(!flagThread || g%2==0) {
          if(ix*EltPer16B < eltN)
            load128((uint64_t*)(src + ix*EltPer16B), regs[2*g+0], regs[2*g+1]);
        }
      }
    }
    else {
      // Not aligned. Stage the smallest 16 byte aligned region subsuming the
      // buffer into shmem.
      int misalignment = reinterpret_cast<uintptr_t>(src) % 16;
      uint64_t *src8 = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(src) & -uintptr_t(16));
      uint64_t *shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++)
        if((g*WARP_SIZE + wid)*16 < misalignment + eltN*sizeof(T))
          load128(src8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++)
        storeShmem128(shm8 + 2*(g*WARP_SIZE + wid), regs[2*g+0], regs[2*g+1]);

      __syncwarp();

      // Now load from shmem stage to regs. Preserve the same pre-shuffled layout
      // as the aligned case since Finish() will be applied regardless.
      T *shm = (T*)shm8 + misalignment/sizeof(T);
      #pragma unroll
      for(int g=0; g < WordPerThread/2; g++) {
        int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
        if(!flagThread || g%2==0) {
          if(ix*EltPer16B < eltN)
            loadShmemMisaligned128(shm + ix*EltPer16B, regs[2*g+0], regs[2*g+1]);
        }
      }
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void loadRegsFinish(uint64_t(&regs)[WordPerThread]) {
    // Move data out of flag registers into the vacant registers.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g] = regs[2*g-1];
    }
  }

  template<int WordPerThread>
  __device__ __forceinline__ void storeRegs(T *dst, uint64_t(&regs)[WordPerThread], int eltN) {
    constexpr int EltPer16B = 16/sizeof(T);
    // Reverse Finish() register permuatation.
    #pragma unroll
    for (int g=1; g < WordPerThread/2; g+=2) {
      if (flagThread) regs[2*g-1] = regs[2*g];
    }
    // Write to dst if 16-byte aligned, shmem otherwise.
    int misalignment = reinterpret_cast<uintptr_t>(dst)%16;
    uint64_t *shm8 = shmemCvtPtr((uint64_t*)ncclScratchForWarp(warpInBlock));
    #pragma unroll
    for(int g=0; g < WordPerThread/2; g++) {
      int ix = g*WARP_SIZE - 4*(g/2) + wid - (g%2)*(wid/8);
      if (!flagThread || g%2==0) {
        if(misalignment == 0 && (ix+1)*EltPer16B <= eltN)
          store128((uint64_t*)(dst + ix*EltPer16B), regs[2*g+0], regs[2*g+1]);
        else
          storeShmem128(shm8+2*ix, regs[2*g+0], regs[2*g+1]);
      }
    }
    __syncwarp();
    // Write rest from shmem to dst. No need to coalesce stores to 16-bytes,
    // the hardware keeps up fine.
    T *shm = (T*)ncclScratchForWarp(warpInBlock);
    int skip = misalignment == 0 ? eltN & -EltPer16B : 0;
    for(int i=skip+wid; i < eltN; i += WARP_SIZE)
      dst[i] = shm[i];
  }

  #define WARP_MASK 0xffffffff

  template <int ELEMS_PER_THREAD, int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void recvReduceSendCopy(uint64_t(&v)[ELEMS_PER_THREAD], int ll128Offset, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    uint64_t vr[ELEMS_PER_THREAD];

    __syncwarp();
    /************************ Wait first recv ********************/
    if (RECV) {
      uint64_t* ptr = recvPtr(0)+ll128Offset;
      uint64_t flag = recvFlag(0);
      bool needReload;
      int spins = 0;
      do {
        needReload = false;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
          needReload |= flagThread && (vr[u+1] != flag);
        }
        needReload &= (0 == checkAbort(spins, 0, 0));
      } while (__any_sync(WARP_MASK, needReload));

      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2)
        load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
    }

    /************* Finish register load **************/
    if (SRC) {
      // By deferring register shuffle here we've overlapped spinning on first
      // peer's data with memory loads of src data.
      loadRegsFinish(v);
      if (SrcBuf == Input) {
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = applyPreOp(redOp, v[u]);
          if (!flagThread)
            v[u+1] = applyPreOp(redOp, v[u+1]);
        }
      }
    }

    /************************ Recv rest *********************/
    if (RECV) {
      { // Consume data from first recv
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u]   = SRC ? applyReduce(redOp, vr[u], v[u]) : vr[u];
          v[u+1] = SRC ? applyReduce(redOp, vr[u+1], v[u+1]) : vr[u+1];
        }
      }

      for (int i=1; i<MaxRecv && i<fan.nrecv(); i++) {
        uint64_t flag = recvFlag(i);
        uint64_t* ptr = recvPtr(i)+ll128Offset;
        bool needReload;
        int spins = 0;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);
            needReload |= flagThread && (vr[u+1] != flag);
          }
          needReload &= (0 == checkAbort(spins, i, 0));
        } while (__any_sync(WARP_MASK, needReload));

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2)
          load128(ptr+u*WARP_SIZE, vr[u], vr[u+1]);

        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u]   = applyReduce(redOp, vr[u], v[u]);
          v[u+1] = applyReduce(redOp, vr[u+1], v[u+1]);
        }
      }
    }
    /********************** End Recv ************************/

    if (postOp) {
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        v[u]   = applyPostOp(redOp, v[u]);
        v[u+1] = applyPostOp(redOp, v[u+1]);
      }
    }

    /************************ Send **************************/
    if (SEND) {
      for (int i=1; i<MaxSend && i<fan.nsend(); i++) {
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
  }

  static constexpr int WireWordPerSlice = WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD;
  static constexpr int DataEltPerSlice = (WireWordPerSlice - WireWordPerSlice/NCCL_LL128_LINEELEMS)*(sizeof(uint64_t)/sizeof(T));

  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;
    constexpr int DST = DstBuf != -1 ? 1 : 0;
    T const *srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T       *dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    if (SEND) waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
    barrier();
    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(regs, wireOffset, postOp);
      if (DST) storeRegs(dstPtr, regs, eltInSlice);

      wireOffset += WireWordPerSlice*nwarps;
      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      nelem -= DataEltPerSlice*nwarps;
    }

    barrier();
    if (SEND) for (int i=0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV) for (int i=0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv()) {
      recvConnHeadPtr = recvConn->head;
      recvConnHead = recvConn->step;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (uint64_t*)conn->buffs[NCCL_PROTO_LL128];
    sendStep[i] = conn->step;
    if (wid == i) sendConn = conn;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < fan.nsend()) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnHead = sendConn->step;
      sendConnFifoPtr = sendConn->sizesFifo;
    }
    if (tid >= nthreads-WARP_SIZE && wid<fan.nsend()) {
      if (sendConn->sizesFifo) {
        sendConnTailPtr = sendConn->tail;
        sendConnTail = sendConn->step;
      }
    }
  }

public:
  __device__ Primitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, int group=0
    ):
    redOp(redOpArg),
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE),
    warpInBlock(threadIdx.x/WARP_SIZE),
    flagThread((tid%8)==7), group(group&(uint16_t)0xFFFF),
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS/sizeof(uint64_t)) {
    int connIndex = group >> 16;
    auto *channel = &ncclShmem.channel;
    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] >= 0) {
      loadRecvConn(&channel->peers[recvPeers[nrecv]].recv[connIndex], nrecv);
      nrecv++;
    }
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      loadSendConn(&channel->peers[sendPeers[nsend]].send[connIndex], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    loadRecvSync();
    loadSendSync();
    setDataPtrs(inputBuf, outputBuf);
  }

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (tid >= nthreads-WARP_SIZE && wid < fan.nrecv())
      recvConn->step = recvConnHead;
    if (tid < fan.nsend())
      sendConn->step = sendConnHead;
    // Ensure all steps written back
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf) {
    userBufs[Input] = (T*)inputBuf;
    userBufs[Output] = (T*)outputBuf;
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    userBufs[Input] += delta;
    userBufs[Output] += delta;
  }

  __device__ void send(intptr_t inpIx, int eltN) {
    return GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void sendFromOutput(intptr_t outIx, int eltN) {
    return GenericOp<0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceSend(intptr_t inpIx, int eltN) {
    return GenericOp<1, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return GenericOp<1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
};
