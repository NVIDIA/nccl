/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "network/unpack/unpack.h"
#include <cassert>

enum primsMode {
  primsModeDefault = 0,
  primsModePatRs = 1,
  primsModePatAg = 2
};

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p, int MultimemSrcs, int MultimemDsts>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p
  > {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       NetRegMode = 0x80,
                       ConnFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       PatMode = 0x800,
                       NvlsMinPolling = 0x1000,
                       NetDeviceUnpack = 0x2000,
                       AnyNetDeviceUnpack = 0x4000,
                       NvlsDirectRead = 0x8000,
                       NvlsDirectWrite = 0x10000,
                       IpcWrite = 0x20000,
                       IpcRead = 0x40000;
  const int tid, tidInBlock;
  const int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  struct ncclConnInfo* conn = NULL;
  struct ncclConnFifo* connFifo = NULL;
  T* connEltsFifo;
  T* directBuff = NULL;
  uint64_t *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)
  int      connStepSize; // Connection step size
  void*    netDeviceHandle;
  uint64_t accSize; // Accumulated size. Used by PAT operations

  // Don't use barrier 0 as it's used by the final sync
  __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group;
      barrier_sync(bar, nthreads);
    }
  }
  __device__ void subBarrier() {
    if (nworkers == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group - (nworkers!=nthreads ? 1 : 0);
      barrier_sync(bar, nworkers);
    }
  }

  __device__ bool barrierAny(int vote) {
    if (nthreads == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group;
      return barrier_red_or(vote, name, nthreads);
    }
  }
  __device__ bool subBarrierAny(int vote) {
    if (nworkers == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group - (nworkers!=nthreads ? 1 : 0);
      return barrier_red_or(vote, name, nworkers);
    }
  }

  inline __device__ bool checkAbort(int &spins) {
    spins++;
    if (!(flags & Aborted) && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      if (*ncclShmem.comm.abortFlag) {
        flags |= Aborted;
        ncclShmem.aborted = 1;
      }
      spins = 0;
    }
    return flags & Aborted;
  }

  inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
    #if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    if (flags & NvlsMinPolling) {
      uint64_t ans;
      asm volatile("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
      return ans;
    }
    #endif
    // volatile is faster than acquire but not as correct. Make sure reduceCopy
    // loads data using volatile so it doesn't see stale data in L1.
    return ld_volatile_global(ptr);
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t srcIx, intptr_t dstIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    const bool noRecvWait = DirectRecv && Src && (flags & (DirectRead | IpcRead));        // no wait when directly reading from remote input
    const bool noSendWait = DirectSend && (flags & (DirectRead|DirectWrite)); // no wait in empty send (e.g. directScatter) or direct remote write
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    if (((flags & (Recv*RoleWaitRecv)) && !noRecvWait) ||
        ((flags & (Send*RoleWaitSend)) && !noSendWait)) {
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(spins)) break;
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
      }
    }

    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      if (flags & ConnFifoEnabled)
        connFifo[step%NCCL_STEPS].size = nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
      if (flags & NetRegMode) {
         // Do nothing
      } else if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
        ptrs[index] = connEltsFifo + loadInt(&connFifo[step%NCCL_STEPS].offset)/sizeof(T);
      } else if (isSendNotRecv && DirectSend) {
        if (flags & (DirectWrite | NvlsDirectWrite | IpcWrite)) {
          ptrs[index] = directBuff + dstIx + offset;
        } else if ((flags & DirectRead) || (flags & IpcRead)) {  // empty send
          ptrs[index] = nullptr;
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
        }
      } else if (!isSendNotRecv && DirectRecv) {
        if (flags & (DirectRead | NvlsDirectRead | IpcRead)) {
          ptrs[index] = directBuff + srcIx + offset;
        } else if ((flags & DirectWrite) || (flags & IpcWrite)) {
          ptrs[index] = directBuff + dstIx + offset;  // send to next from my output buffer
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
        }
      }
      else {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
      }
      if (flags & NetDeviceUnpack) {
        ncclNetDeviceIncrementHead(group, index);
      }
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer(bool dataStored) {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && (dataStored||(flags&ConnFifoEnabled))) {
        fence_acq_rel_sys();
      }
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem && ((flags & NetRegMode) == 0)) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      #if __CUDA_ARCH__ < 700
        // Above doesn't matter on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
        subBarrier();
        /* if user abort the kernel, we don't need to actually perform copy/reduce; just set size
         * to 0 to avoid unnecessary workload. */
        int workSize = ncclShmem.aborted ? 0 : sliceSize;
        if (flags & AnyNetDeviceUnpack) {
          ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
          // Sync here to make sure all workers are reading from the updated srcs)
          subBarrier();
        }

        if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
            /* NVLS can have srcs[0] == dsts[0], but we cannot enter this "if branch",
             * so we need to check whether MultimemSrcs and MultimemDsts are 0. */
            && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/0>
              (tid, nworkers, /*redArg*/0, /*preOpArgs*/nullptr, /*postOp*/false,
               1, ncclShmem.groups[group].srcs,
               fan.nsend(), ncclShmem.groups[group].dsts+1,
               workSize);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // For broadcast in CollNet to do empty send
          reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/0>
            (tid, nworkers, ncclShmem.redOpArgs[0],  nullptr, postOp,
             Recv, ncclShmem.groups[group].srcs,
             Dst, ncclShmem.groups[group].dsts,
             workSize);
        } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                    DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          reduceCopy<Unroll, RedOp, T,
            MultimemSrcs, Recv+Src, Recv*MaxRecv+Src,
            MultimemDsts, Send+Dst, Send*MaxSend+Dst, PreOpSrcs>
            (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
             Recv*fan.nrecv()+Src, ncclShmem.groups[group].srcs,
             Send*fan.nsend()+Dst, ncclShmem.groups[group].dsts,
             workSize);
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < sliceSize);
        offset += sliceSize;
        slice += 1;
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, 0);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      postPeer<Recv, Send>(0 < sliceSize);
      offset += sliceSize;
      slice += 1;
    }
  }

public:
  static inline __device__ void sendPeerNotify(int peer, int connIndex, int steps) {
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->send[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->send[connIndex].tail, peerPtr->send[connIndex].step);
  }

  static inline __device__ void recvPeerNotify(int peer, int connIndex, int steps) {
    int spins = 0;
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->recv[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->recv[connIndex].head, peerPtr->recv[connIndex].step);
    while (ld_volatile_global(peerPtr->recv[connIndex].tail) < peerPtr->recv[connIndex].step) {
      if (spins++ == NCCL_SPINS_BEFORE_CHECK_ABORT) {
        if (*ncclShmem.comm.abortFlag) {
          ncclShmem.aborted = 1;
          break;
        }
        spins = 0;
      }
    }
  }

  template<int Recv, int Send, typename Fn>
  __device__ __forceinline__ void process(Fn &&fn, uint32_t sendDirectFlag, uint32_t recvDirectFlag) {
    #pragma unroll 1
    for (int slice=0; slice < SlicePerChunk; slice++) {
      if (tid < nworkers) {
        int nsend, nrecv;
        if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
          const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
          int spins = 0;
          while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
            connStepCache = loadStepValue(connStepPtr);
            if (checkAbort(spins)) break;
          }
          void **ptrs = isSendNotRecv ? ncclShmem.groups[group].dsts
                                      : ncclShmem.groups[group].srcs;
          if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
            int offset = loadInt(&connFifo[step%NCCL_STEPS].offset);
            ptrs[index] = connEltsFifo + offset/sizeof(T);
          } else if (Direct && fn.work->regUsed) {
            if (isSendNotRecv) {
              if (flags & (DirectWrite | IpcWrite)) {
                ptrs[index] = directBuff;
              } else if (flags & (DirectRead | IpcRead)) {  // empty send
                ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
              }
            } else {
              if (flags & (DirectRead | IpcRead)) {
                ptrs[index] = directBuff;
              } else if (flags & (DirectWrite | IpcWrite)) {
                if (Send)
                  ptrs[index] = directBuff;  // send to next from my output buffer
                else
                  ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
              }
            }
          } else {
            ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
          }
        }
        subBarrier();
        if (Recv == 0 || ncclShmem.groups[group].srcs[0] == nullptr) {
          nrecv = 0;
        } else {
          nrecv = fan.nrecv();
        }

        if (Send == 0 || ncclShmem.groups[group].dsts[0] == nullptr) {
          nsend = 0;
        } else {
          nsend = fan.nsend();
        }
        fn.template operator() < SlicePerChunk, 0, Recv*MaxRecv, 0, Send*MaxSend >
          (tid, nworkers, slice, stepSize * StepPerSlice,
            nrecv, ncclShmem.groups[group].srcs,
            nsend, ncclShmem.groups[group].dsts, ncclShmem.groups[group].dstSizes, sendDirectFlag, recvDirectFlag);
      }
      barrier();
      int32_t dstSize = 0;
      if (flags & Send*RolePostSend) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_begin]
        dstSize = ncclShmem.groups[group].dstSizes[index];
        ncclShmem.groups[group].dstSizes[index] = 0;
        if (flags & ConnFifoEnabled) connFifo[step%NCCL_STEPS].size = dstSize*sizeof(T);
      }
      barrier();
      if (flags & (Recv*(RoleWaitRecv|RolePostRecv) | Send*(RoleWaitSend|RolePostSend))) {
        step += StepPerSlice;
      }
      if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
        if (Send && (!Recv || (flags & RolePostSend)) && (dstSize!=0 || (flags&ConnFifoEnabled))) {
          fence_acq_rel_sys();
        }
        st_relaxed_sys_global(connStepPtr, step);
      }
    }
  }

private:
  // Scatter/Gather generic op
  // skip: my own rank order in the buffer chunks
  // shift: peer offset to avoid all ranks sending to or receiving from same peer
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      ssize_t realSize = max(0, min(dataSize, peerElem-offset));
      bool fenceNeeded = false;
      if (tid < nworkers) {
        if (Send) {
          // Scatter pre-scales data of input buffer only in non-Direct case
          constexpr int PreOpSrcs = DirectSend ? 0 : 1;
          if (tid==0) ncclShmem.groups[group].srcs[0] = (T*)ncclShmem.groups[group].userInput + inpIx + offset;
          // realSize is not accurate here; but intra-node does not rely on sizes FIFO
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          #pragma unroll
          // Loop over peers
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            ssize_t pOffset = i*peerOffset;
            // Skip the data I am responsible of reducing myself
            if (skip >= 0 && i >= skip) pOffset += peerElem;
            void* src0 = (T*)ncclShmem.groups[group].srcs[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, PreOpSrcs>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, false, 1, &src0, 1, ncclShmem.groups[group].dsts+i, realPeerSize);
              // Mark for threadfence at the end
              fenceNeeded |= true;
            }
          }
        } else if (Recv) {
          if (tid==0) ncclShmem.groups[group].dsts[0] = (T*)ncclShmem.groups[group].userOutput + outIx + offset;
          ssize_t pOffset = index*peerOffset;
          if (skip >= 0 && index >= skip) pOffset += peerElem;
          // Adjust remote index with peer offset in case we are directly pulling from peer's output buffer
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx+pOffset, outIx+pOffset, offset, realSize);
          subBarrier();
          #pragma unroll
          for (int j=0; j<fan.nrecv(); j++) {
            int i = (j+shift)%fan.nrecv();
            pOffset = i*peerOffset;
            if (skip >= 0 && i >= skip) pOffset += peerElem;
            void* dst0 = (T*)ncclShmem.groups[group].dsts[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (DirectRecv && ncclShmem.groups[group].srcs[i] == dst0) realPeerSize = 0;
            if (realPeerSize > 0) reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp, 1, ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      fenceNeeded = barrierAny(fenceNeeded);
      postPeer<Recv, Send>(fenceNeeded);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int regFlag) {
    conn = &peer->recv[connIndex];
    if (conn->netDeviceHandle.netDeviceType == NCCL_NET_DEVICE_UNPACK) {
      // handle must be a device ptr
      netDeviceHandle = conn->netDeviceHandle.handle;
      // Cache the handle
      ncclNetDeviceUnpackSetup(netDeviceHandle, group, index);
      flags |= NetDeviceUnpack;
    }
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);
    if (flags & RolePostRecv) {
      connStepPtr = conn->head;
      *connStepPtr = step; // Return credits in case we rounded up.
    }
    if (flags & RoleWaitRecv) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->tail;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (conn->connFifo != nullptr) {
        flags |= ConnFifoEnabled;
        connFifo = conn->connFifo;
      } else if (Direct && regFlag) {
        // User buffers have been registered
        if (conn->flags & (NCCL_IPC_READ | NCCL_IPC_WRITE)) {
          if (P2p) {
            flags |= conn->flags & NCCL_IPC_WRITE ? IpcWrite : IpcRead;
          } else if (connIndex == 1 && direct) {
            flags |= IpcRead;
          } else {
            flags |= direct & NCCL_DIRECT_READ ? IpcRead : IpcWrite;
          }
        } else if (conn->flags & (NCCL_DIRECT_WRITE | NCCL_DIRECT_READ)) {
          if (P2p) {
            flags |= conn->flags & NCCL_DIRECT_WRITE ? DirectWrite : DirectRead;
          } else if (connIndex == 1 && direct) {
            flags |= DirectRead;  // scatter-reduce use direct pull
          } else {
            flags |= direct & NCCL_DIRECT_READ ? DirectRead : DirectWrite;
          }
        } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
          /* NVLS direct */
          flags |= NvlsDirectRead;
        }
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int regFlag) {
    conn = &peer->send[connIndex];
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);

    connFifo = conn->connFifo;
    if (connFifo != nullptr) flags |= ConnFifoEnabled;

    if (flags & RolePostSend) {
      connStepPtr = conn->tail;
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
    }
    if (flags & RoleWaitSend) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->head;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (connFifo == nullptr && Direct && regFlag) {
        // User buffers have been registered
        if (conn->flags & (NCCL_IPC_READ | NCCL_IPC_WRITE)) {
          if (P2p) {
            flags |= conn->flags & NCCL_IPC_WRITE ? IpcWrite : IpcRead;
          } else if (connIndex == 1 && direct) {
            flags |= IpcRead;
          } else {
            flags |= direct & NCCL_DIRECT_READ ? IpcRead : IpcWrite;
          }
        } else if (conn->flags & (NCCL_DIRECT_WRITE | NCCL_DIRECT_READ)) {
          if (P2p) {
            flags |= conn->flags & NCCL_DIRECT_WRITE ? DirectWrite : DirectRead;
          } else if (connIndex == 1 && direct) {
            flags |= DirectRead;  // scatter-reduce use direct pull
          } else {
            flags |= direct & NCCL_DIRECT_READ ? DirectRead : DirectWrite;
          }
        } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
          /* NVLS direct */
          flags |= NvlsDirectWrite;
        }
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* e = nullptr,
      bool ipcReg = false, bool netReg = false, int stepSize_ = 0, int mode = primsModeDefault
    ):
    tid(tid), nthreads(nthreads), tidInBlock(threadIdx.x), group(group),
    stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T) : stepSize_) {

    // For send operations, we need an extra warp to overlap the threadfence and the copy
    this->nworkers = nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);

    int peer = -1;
    flags = 0;
    index = -1;
    if (mode == primsModeDefault) { // Connect to ranks in sendPeers/recvPeers
      int nrecv=0, nsend=0;
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
      // coverity[dead_error_line]
      while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
      this->fan = Fan(nrecv, nsend);

      constexpr int ThreadPerSync =
        MaxSend >= 16 || MaxRecv >= 16 ? 32 : // NVLS may have an arity > 8. In that case increase the size of the groups
        MaxSend >= 8 || MaxRecv >= 8 ? 16 :
        8; // Allows for all roles (WaitRecv/WaitSend/PostRecv/PostSend) within a single warp
      static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

      assert(2*(nrecv+nsend) <= nthreads); // Ensure no thread is assigned more than one role.
      // Coverity assumes that index will equal tid based on the line below, but it doesn't consider the setting
      // of flags.  This results in multiple false positive overruns being reported here and in all_reduce.h.
      // Unfortunately, we've been unsuccessful in trying to silence them with a single directive here so
      // instead it's being done at the callers.
      // coverity[assignment:FALSE]
      if      (tid < nrecv)                 { flags |= RoleWaitRecv; index = tid; }
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      else if (tid < nrecv+nsend)           { flags |= RoleWaitSend; index = tid-nrecv; }
      else if (nthreads-nsend <= tid)       { flags |= RolePostSend; index = tid-(nthreads-nsend); }
      else if (nthreads-nrecv-nsend <= tid) { flags |= RolePostRecv; index = tid-(nthreads-nrecv-nsend); }

      if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
      if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];
    } else if (mode == primsModePatRs || mode == primsModePatAg) { // Connect to all ranks +/- 2^n
      flags |= PatMode;
      accSize = 0;
      int nranks = ncclShmem.comm.nRanks;
      int rank = ncclShmem.comm.rank;
      // A thread is responsible for rank +/- 2 ^ (tid%32). That should be fine as long as rank is a 32-bits integer.
      index = tid % 32;
      uint32_t delta = 1 << index;
      const int roles[4] = { RoleWaitRecv, RoleWaitSend, RolePostSend, RolePostRecv};
      int block = tid / 32;
      if (block < 4 && delta < nranks) {
        int role = roles[block];
        if (mode == primsModePatRs) {
          if (role & (RoleWaitRecv|RolePostRecv)) peer = (rank - delta + nranks) % nranks;
          if (role & (RoleWaitSend|RolePostSend)) peer = (rank + delta) % nranks;
        } else if (mode == primsModePatAg) {
          if (role & (RoleWaitSend|RolePostSend)) peer = (rank - delta + nranks) % nranks;
          if (role & (RoleWaitRecv|RolePostRecv)) peer = (rank + delta) % nranks;
        }
        flags |= role;
      } else if (tid == 128) {
        flags |= RoleInput | RoleOutput; // Only one will be used depending on the operation
      }
    }

    // Coverity thinks that index could be -1 here but that's not actually the case.
    // coverity[negative_returns:FALSE]
    if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(ncclShmem.channel.peers[peer], connIndexRecv, e ? e->direct : 0, e ? e->regUsed : ipcReg);
    // coverity[negative_returns:FALSE]
    if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(ncclShmem.channel.peers[peer], connIndexSend, e ? e->direct : 0, e ? e->regUsed : ipcReg);

    if (netReg) flags |= NetRegMode;

    if (barrierAny(flags & NetDeviceUnpack)) {
      flags |= AnyNetDeviceUnpack;
      // RoleWaitRecv starts at tid=0, so this creates the bitmask of which recv peers
      // have NetDeviceUnpack.
      uint32_t mask = __ballot_sync(~0u, ((flags & RoleWaitRecv) && (flags & NetDeviceUnpack)) ? 1 : 0);
      if (tid == 0) {
        ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
      }
    }

    // coverity[negative_returns:FALSE]
    setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclDevWorkCollReg*)e, (uint8_t)(e ? e->regUsed : ipcReg), peer);
  }

  __device__ ~Primitives() {
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) conn->step = step;
    if ((flags & NetRegMode) && (flags & RoleWaitSend)) {
      // Make sure we wait until the proxy has sent data before we return.
      // We don't want the next CUDA kernel to overwrite the send buffer which
      // was accessed directly.
      uint64_t prevStep = step - StepPerSlice;
      volatile ssize_t* ptr = &(connFifo[prevStep%NCCL_STEPS].size);
      int spins = 0;
      while (*ptr != -1) if (checkAbort(spins)) break;
    }

    if (flags & NetDeviceUnpack) {
      ncclNetDeviceSaveHead(netDeviceHandle, group, index);
    }

    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclDevWorkCollReg* work, uint8_t ipcReg, int peer) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
    }

    if (Direct && ipcReg) {
      bool recvProvider = (flags & RoleWaitRecv) && (flags & DirectWrite || flags & IpcWrite);
      bool sendAcceptor = (flags & RoleWaitSend) && (flags & DirectWrite || flags & IpcWrite || flags & NvlsDirectWrite);
      bool sendProvider = (flags & RoleWaitSend) && (flags & DirectRead || flags & IpcRead); // sender provides direct buffer (to be fetched)
      bool recvAcceptor = (flags & RoleWaitRecv) && (flags & DirectRead || flags & IpcRead || flags & NvlsDirectRead); // receiver accepts direct buffer
      if (recvProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        // Wait for consumer to consume previous value before trampling it.
        if (slot) {
          T* exchgPtr;
          directBuff = (T*)outputBuf;
          while (*slot != nullptr && !checkAbort(spins));
          if (P2p) {
            exchgPtr = (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }
          *slot = reinterpret_cast<void*>(exchgPtr);
        }
      }
      if (sendAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          directBuff = (T*)work->dnOutputs[index];
        }
      }
      if (sendProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        volatile uint64_t* argSlot0 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange;
        volatile uint64_t* argSlot1 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange + 1;
        // Wait for consumer to consume previous value before trampling it.
        if (slot && argSlot0 && argSlot1) {
          T* exchgPtr;
          while ((*slot != nullptr || *argSlot0 != 0 || *argSlot1 != 0) && !checkAbort(spins));
          // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
          // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
          directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          if (P2p) {
            exchgPtr = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            if (MaxRecv == 0)
              exchgPtr = (T*)(work->coll.sendbuffOffset + work->coll.sendbuffRmtAddrs[localPeer]);
            else
              exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }

          // Exchange pre-scalers for use in direct pull
          *argSlot0 = (uint64_t(1) << 32) | (uint32_t)redOpArg;
          *argSlot1 = (uint64_t(1) << 32) | (uint32_t)(redOpArg >> 32);
          *slot = reinterpret_cast<T*>(exchgPtr);
        }
      }
      if (recvAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        volatile uint64_t* argSlot0 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange;
        volatile uint64_t* argSlot1 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange + 1;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(spins)) break;
        }

        if (slot && argSlot0 && argSlot1) {
          directBuff = reinterpret_cast<T*>(ptr);
          if (MaxSend != 0) { // reduce group rather than gather group
            // Store scalers for remote inputs
            uint64_t arg0, arg1;
            while (true) {
              arg0 = *argSlot0;
              arg1 = *argSlot1;
              if ((arg0 != 0 && arg1 != 0) || checkAbort(spins)) break;
            }
            ncclShmem.redOpArgs[1 + index] = ((arg1 & 0xffffffff) << 32) | (arg0 & 0xffffffff);
          }
          *argSlot0 = 0; *argSlot1 = 0;
          *slot = nullptr;
        } else {
          // Coverity complains about work being possibly NULL below.  However, slot
          // being NULL means that the NVLS buffer is registered (regUsed == 1)
          // so work can't be NULL in this code path.
          // coverity[var_deref_op]
          directBuff = (T*)work->dnInputs[index];
        }
      }
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (T*)ncclShmem.groups[group].userInput + delta;
      ncclShmem.groups[group].userOutput = (T*)ncclShmem.groups[group].userOutput + delta;
    }
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, outIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopy(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvSend(int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, -1>(-1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopyDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<1, 1, 1, 1, -1, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void patReduce(int recvPow2, int sendPow2, intptr_t inpIx, intptr_t outIx, int recvOffset, int sendOffset, int sendStepOffset, int nelem, int postRecv, int postSend) {
    nelem = nelem < 0 ? 0 : nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    if (recvPow2 >= 0 && recvPow2 == index && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = (T*)(connEltsFifo + (step%NCCL_STEPS)*connStepSize) + recvOffset;
      int spins = 0;
      while (connStepCache < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(spins)) break;
      }
      if (postRecv) step += StepPerSlice;
    }
    if (sendPow2 >= 0 && sendPow2 == index && (flags & RoleWaitSend)) {
      int spins = 0;
      while (connStepCache + NCCL_STEPS < step + sendStepOffset + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = (T*)(connEltsFifo + ((step+sendStepOffset)%NCCL_STEPS)*connStepSize) + sendOffset;
      if (accSize < sendOffset + nelem + (step+sendStepOffset)*connStepSize) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + inpIx;
        accSize = sendOffset + nelem + (step+sendStepOffset)*connStepSize;
        if (flags & ConnFifoEnabled)
          connFifo[(step+sendStepOffset)%NCCL_STEPS].size = (sendOffset + nelem)*sizeof(T);
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
      if (postSend) step += StepPerSlice;
    }
    if (sendPow2 < 0 && (flags & RoleOutput)) { // Destination is our own local buffer
      ncclShmem.groups[group].dsts[0] = userOutput + outIx;
      if (accSize < outIx + nelem) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + inpIx;
        accSize = outIx + nelem;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    barrier();
    int nSrcs = 2;
    void** srcs = ncclShmem.groups[group].srcs;
    if (recvPow2 < 0) { srcs++; nSrcs--; } // No peer to receive from, remove one source

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 2, 0, 1, 1, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.redOpArgs[0],  nullptr, /*postOp=*/false,
       nSrcs, srcs, 1, ncclShmem.groups[group].dsts, workSize);

    barrier();
    if (postRecv && recvPow2 >= 0 && recvPow2 == index && (flags & RolePostRecv)) postPeer<1, 0>(0 < nelem);
    if (postSend && sendPow2 >= 0 && sendPow2 == index && (flags & RolePostSend)) postPeer<0, 1>(0 < nelem);
  }

  __device__ __forceinline__ void patCopy(int recvPow2, int sendPow2, intptr_t inpIx, intptr_t outIx, int recvOffset, int sendOffset, int recvStepOffset, int nelem, int postRecv, int postSend) {
    nelem = nelem < 0 ? 0 : nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    if (recvPow2 >= 0 && recvPow2 == index && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = (T*)(connEltsFifo + ((step+recvStepOffset)%NCCL_STEPS)*connStepSize) + recvOffset;
      int spins = 0;
      while (connStepCache < step + recvStepOffset + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(spins)) break;
      }
      if (accSize < recvOffset + nelem + (step+recvStepOffset)*connStepSize) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + outIx;
        accSize = recvOffset + nelem + (step+recvStepOffset)*connStepSize;
      } else {
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0]; // Already done
      }
      if (postRecv) step += StepPerSlice;
    }
    if (sendPow2 >= 0 && sendPow2 == index && (flags & RoleWaitSend)) {
      int spins = 0;
      while (connStepCache + NCCL_STEPS < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = (T*)(connEltsFifo + (step%NCCL_STEPS)*connStepSize) + sendOffset;
      if (postSend) {
        if (flags & ConnFifoEnabled)
          connFifo[step%NCCL_STEPS].size = (sendOffset + nelem)*sizeof(T);
        step += StepPerSlice;
      }
    }
    if (recvPow2 < 0 && (flags & RoleInput)) { // Source is our own local buffer
      ncclShmem.groups[group].srcs[0] = userInput + inpIx;
      if (accSize < inpIx + nelem) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + outIx;
        accSize = inpIx + nelem;
      } else {
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0]; // Already done
      }
    }
    barrier();
    int nDsts = 2;
    void** dsts = ncclShmem.groups[group].dsts;
    if (sendPow2 < 0) { dsts++; nDsts--; } // No peer to send to, remove one dest
    if (ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[1]) nDsts--; // In-place or already done.

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 2, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.redOpArgs[0],  nullptr, /*postOp=*/false,
       1, ncclShmem.groups[group].srcs, nDsts, dsts, workSize);

    barrier();
    if (postRecv && recvPow2 >= 0 && recvPow2 == index && (flags & RolePostRecv)) postPeer<1, 0>(0 < nelem);
    if (postSend && sendPow2 >= 0 && sendPow2 == index && (flags & RolePostSend)) postPeer<0, 1>(0 < nelem);
  }

};
