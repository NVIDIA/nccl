/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p, bool NVLS>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, NVLS>, P2p
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
                       OffsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       ThreadsSynced = 0x800,
                       NvlsMinPolling = 0x1000,
                       NvlsRecv = 0x2000;
  const int tid, tidInBlock;
  int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  int *connOffsFifoPtr;   // (flags & OffsFifoEnabled)
  union {
    T *userBuff;            // (flags & (RoleInput|RoleOutput))
    T *connEltsFifo;        // !(flags & (RoleInput|RoleOutput))
  };
  union {
    int volatile *connSizesFifoPtr; //  (flags & SizesFifoEnabled)
    T *directBuff;                  // !(flags & SizesFifoEnabled)
  };
  uint64_t *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)

  // Don't use barrier 0 as it's used by the final sync
  __device__ void barrier() {
    flags |= ThreadsSynced;
    if (nthreads == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group;
      asm volatile("bar.sync %0, %1;" :: "r"(bar), "r"(nthreads) : "memory");
    }
  }
  __device__ void subBarrier() {
    if (nworkers == WARP_SIZE) __syncwarp();
    else {
      int bar = (nworkers==nthreads ? 15 : 8) - group;
      asm volatile("bar.sync %0, %1;" :: "r"(bar), "r"(nworkers) : "memory");
    }
  }

  __device__ bool barrierAny(int vote) {
    flags |= ThreadsSynced;
    if (nthreads == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int ans, bar = 15-group;
      asm volatile(
        "{ .reg .pred p;"
        "  setp.ne.s32 p, %1, 0;"
        "  bar.red.or.pred p, %2, %3, p; "
        "  selp.s32 %0, 1, 0, p; }"
        : "=r"(ans) : "r"(vote), "r"(bar), "r"(nthreads) : "memory");
      return ans != 0;
    }
  }
  __device__ bool subBarrierAny(int vote) {
    if (nworkers == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int ans, bar = (nworkers==nthreads ? 15 : 8) - group;
      asm volatile(
        "{ .reg .pred p;"
        "  setp.ne.s32 p, %1, 0;"
        "  bar.red.or.pred p, %2, %3, p; "
        "  selp.s32 %0, 1, 0, p; }"
        : "=r"(ans) : "r"(vote), "r"(bar), "r"(nworkers) : "memory");
      return ans != 0;
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
    if (NVLS && (flags & NvlsMinPolling)) {
      uint64_t ans;
      asm("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)));
      return ans;
    }
    #endif
    // volatile is faster than acquire but not as correct. Make sure ReduceOrCopyMulti
    // loads data using volatile so it doesn't see stale data in L1.
    return ld_volatile_global(ptr);
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t dstIx, intptr_t remoteIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    const bool noRecvWait = DirectRecv && Src && (flags & DirectRead);        // no wait when directly reading from remote input
    const bool noSendWait = DirectSend && (flags & (DirectRead|DirectWrite)); // no wait in empty send (e.g. directScatter) or direct remote write
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
      if (isSendNotRecv && (flags & SizesFifoEnabled))
        connSizesFifoPtr[step%NCCL_STEPS] = nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
      if (flags & OffsFifoEnabled)
        ptrs[index] = connEltsFifo + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
      else if (isSendNotRecv && DirectSend) {
        if (flags & DirectWrite) {
          ptrs[index] = directBuff + remoteIx + offset;
        } else if (flags & DirectRead) {  // empty send
          ptrs[index] = nullptr;
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
        }
      } else if (!isSendNotRecv && DirectRecv) {
        if (flags & DirectRead) {
          ptrs[index] = directBuff + remoteIx + offset;
        } else if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;  // send to next from my output buffer
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
        }
      }
      else {
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
      }
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer(bool dataStored) {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && dataStored) fence_acq_rel_sys();
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteIx, int nelem, bool postOp
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

    if (tid < nworkers && offset < nelem) {
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
        if (Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem.groups[group].srcs[0] = userBuff + srcIx + offset;
        if (Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput)))
          ncclShmem.groups[group].dsts[0] = userBuff + dstIx + offset;
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(dstIx, remoteIx, offset, sliceSize);
        subBarrier();
        /* if user abort the kernel, we don't need to actually perform copy/reduce; just set size
         * to 0 to avoid unnecessary workload. */
        int workSize = ncclShmem.aborted ? 0 : sliceSize;
        if (NVLS && ncclShmem.groups[group].nvlsRecv) {
          void* src = ncclShmem.groups[group].srcs[0];
          void* dst = ncclShmem.groups[group].dsts[0];
          copyMultimemMultimem<RedOp>(tid, nworkers, ncclShmem.redOpArgs[0], postOp, src, dst, workSize,
          cvta_to_shared(ncclScratchForWarp(tidInBlock/WARP_SIZE)));
        } else if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {
            ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, MaxSend, /*PreOpSrcs*/0>
              (tid, nworkers, /*redArg*/0, /*preOpArgs*/nullptr, /*postOp*/false,
               1, ncclShmem.groups[group].srcs,
               fan.nsend(), ncclShmem.groups[group].dsts+1,
               workSize);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // For broadcast in CollNet to do empty send
          ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, /*PreOpSrcs*/0>
            (tid, nworkers, ncclShmem.redOpArgs[0],  nullptr, postOp,
             Recv, ncclShmem.groups[group].srcs,
             Dst, ncclShmem.groups[group].dsts,
             workSize);
        } else {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                    DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          ReduceOrCopyMulti<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst, PreOpSrcs>
            (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
             Recv*fan.nrecv()+Src, ncclShmem.groups[group].srcs,
             Send*fan.nsend()+Dst, ncclShmem.groups[group].dsts,
             workSize);
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < sliceSize);
        offset += sliceSize;
        slice += 1;
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

  // Scatter/Gather generic op
  // skip: my own rank order in the buffer chunks
  // shift: peer offset to avoid all ranks sending to or receiving from same peer
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, int peerOffset, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      bool fenceNeeded = false;
      if (tid < nworkers) {
        if (Send) {
          // Scatter pre-scales data of input buffer only in non-Direct case
          constexpr int PreOpSrcs = DirectSend ? 0 : 1;
          if (flags & RoleInput) ncclShmem.groups[group].srcs[0] = userBuff + inpIx + offset;
          // realSize is not accurate here; but intra-node does not rely on sizes FIFO
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          #pragma unroll
          // Loop over peers
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            int pOffset = i*peerOffset;
            // Skip the data I am responsible of reducing myself
            if (skip >= 0 && i >= skip) pOffset += peerElem;
            void* src0 = (T*)ncclShmem.groups[group].srcs[0] + pOffset;
            int realPeerSize = min(realSize, totalElem-pOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, PreOpSrcs>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, false, 1, &src0, 1, ncclShmem.groups[group].dsts+i, realPeerSize);
              // Mark for threadfence at the end
              fenceNeeded |= true;
            }
          }
        } else if (Recv) {
          if (flags & RoleOutput) ncclShmem.groups[group].dsts[0] = userBuff + outIx + offset;
          int pOffset = index*peerOffset;
          if (skip >= 0 && index >= skip) pOffset += peerElem;
          // Adjust remote index with peer offset in case we are directly pulling from peer's output buffer
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx, outIx+pOffset, offset, realSize);
          subBarrier();
          if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]) {
            // Since waitPeer sets srcs[0] to output buffer + offset, we are doing a direct-write based recv
            // Do nothing
          } else {
            #pragma unroll
            for (int j=0; j<fan.nrecv(); j++) {
              int i = (j+shift)%fan.nrecv();
              pOffset = i*peerOffset;
              if (skip >= 0 && i >= skip) pOffset += peerElem;
              void* dst0 = (T*)ncclShmem.groups[group].dsts[0] + pOffset;
              int realPeerSize = min(realSize, totalElem-pOffset);
              if (realPeerSize > 0) ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, /*PreOpSrcs=*/0>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp, 1, ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
            }
          }
        }
      }
      fenceNeeded = barrierAny(fenceNeeded);
      postPeer<Recv, Send>(fenceNeeded);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *peer, int connIndex, struct ncclWorkElem* e) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      auto *conn = &peer->recv[connIndex];
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostRecv) {
        connStepPtr = conn->head;
        *connStepPtr = step; // Return credits in case we rounded up.
      }
      if (flags & RoleWaitRecv) {
        ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
        if ((index == 0) && (flags & RoleWaitRecv)) {
          if (conn->flags & NCCL_NVLS_MIN_POLL) {
            flags |= NvlsMinPolling;
            ncclShmem.groups[group].nvlsRecv = 1;
          } else {
            ncclShmem.groups[group].nvlsRecv = 0;
          }
        }
        connStepPtr = conn->tail;
        connStepCache = loadStepValue(connStepPtr);
        flags |= (conn->offsFifo != nullptr) ? OffsFifoEnabled : 0;
        if (Direct) {
          // User buffers have been registered
          if ((conn->flags & (NCCL_IPC_READ|NCCL_IPC_WRITE)) && e != nullptr && e->regUsed) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= (e->direct & NCCL_DIRECT_WRITE) ? DirectWrite :
                       (e->direct & NCCL_DIRECT_READ)  ? DirectRead  : 0;
            }
          } else if (conn->flags & (NCCL_DIRECT_WRITE|NCCL_DIRECT_READ)) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              // direct read not allowed in non-register case
              // otherwise, in one-to-multi send, we could mix empty send and intermediate send
              flags |= (conn->flags & NCCL_DIRECT_WRITE) ? DirectWrite : 0;
            }
          }
        }
        if (flags & OffsFifoEnabled)
          connOffsFifoPtr = conn->offsFifo;
        connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer *peer, int connIndex, struct ncclWorkElem* e) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      auto *conn = &peer->send[connIndex];
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostSend) {
        connStepPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
        flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
        connStepPtr = conn->head;
        connStepCache = loadStepValue(connStepPtr);
        flags |= (conn->offsFifo != nullptr) ? OffsFifoEnabled : 0;
        if (flags & OffsFifoEnabled)
          connOffsFifoPtr = conn->offsFifo;
        connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        if (conn->sizesFifo != nullptr) {
          flags |= SizesFifoEnabled;
          connSizesFifoPtr = conn->sizesFifo;
        } else if (Direct) {
          // User buffers have been registered
          if ((conn->flags & (NCCL_IPC_READ|NCCL_IPC_WRITE)) && e != nullptr && e->regUsed) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= (e->direct & NCCL_DIRECT_WRITE) ? DirectWrite :
                       (e->direct & NCCL_DIRECT_READ)  ? DirectRead  : 0;
            }
          } else if (conn->flags & (NCCL_DIRECT_WRITE|NCCL_DIRECT_READ)) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              // direct read not allowed in non-register case
              // otherwise, in one-to-multi send, we could mix empty send and intermediate send
              flags |= (conn->flags & NCCL_DIRECT_WRITE) ? DirectWrite : 0;
            }
          }
        }
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint32_t group=0, struct ncclWorkElem* e = nullptr
    ):
    tid(tid), tidInBlock(threadIdx.x),
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)) {

    // For send operations, we need an extra warp to overlap the threadfence and the copy
    this->nthreads = nthreads;
    this->nworkers = nthreads - (MaxSend > 0 && nthreads-WARP_SIZE >= 64 ? WARP_SIZE : 0);
    this->group = group & (uint16_t)0xFFFF;
    int connIndex = group >> 16;

    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
    while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
    this->fan = Fan(nrecv, nsend);

    constexpr int ThreadPerSync = 8;
    static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

    int g = tid / ThreadPerSync;
    int ng = nthreads / ThreadPerSync;
    index = tid % ThreadPerSync;
    flags = 0;
    if (g == 0) {
      if (index < nrecv) flags |= RoleWaitRecv;
      if (index == nrecv) flags |= RoleInput;
    } else if (g == 1) {
      if (index < nsend) flags |= RoleWaitSend;
      if (index == nsend) flags |= RoleOutput;
    } else if (g == ng - 2) {
      if (index < nrecv) flags |= RolePostRecv;
    } else if (g == ng - 1) {
      if (index < nsend) flags |= RolePostSend;
    }

    int peer = 0;
    if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
    if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];

    loadRecvConn(&ncclShmem.channel.peers[peer], connIndex, e);
    loadSendConn(&ncclShmem.channel.peers[peer], connIndex, e);

    setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclWorkElemReg*)e);
  }

  __device__ ~Primitives() {
    // Ensure ncclShmem.groups[].send/recvConns are available
    if (!(flags & ThreadsSynced))
      barrier();
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) {
      auto *conns = (flags & RolePostSend) ? ncclShmem.groups[group].sendConns : ncclShmem.groups[group].recvConns;
      conns[index]->step = step;
    }
    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclWorkElemReg* e) {
    if (flags & RoleInput) {
      userBuff = (T*)inputBuf;
      ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
    }
    if (flags & RoleOutput) userBuff = (T*)outputBuf;
    bool recvProvider = flags == (flags|RoleWaitRecv|DirectWrite);
    bool sendAcceptor = flags == (flags|RoleWaitSend|DirectWrite);
    bool sendProvider = flags == (flags|RoleWaitSend|DirectRead); // sender provides direct buffer (to be fetched)
    bool recvAcceptor = flags == (flags|RoleWaitRecv|DirectRead); // receiver accepts direct buffer
    int regUsed = e != nullptr ? e->elem.regUsed : 0;

    if (Direct && recvProvider) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
      // Wait for consumer to consume previous value before trampling it.
      while (*slot != nullptr && !checkAbort(spins));
      directBuff = (T*)outputBuf;
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
    }
    if (Direct && sendAcceptor) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
      void *ptr;
      while (true) {
        ptr = *slot;
        if (ptr != nullptr || checkAbort(spins)) break;
      }
      directBuff = regUsed ? (T*)(e->dnOutputs[index]) :
                   reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));
      *slot = nullptr;
    }
    if (Direct && sendProvider) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
      volatile uint64_t* argSlot0 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange;
      volatile uint64_t* argSlot1 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange+1;
      // Wait for consumer to consume previous value before trampling it.
      while ((*slot != nullptr || *argSlot0 != 0 || *argSlot1 !=0) && !checkAbort(spins));
      // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
      // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
      directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
      // Exchange pre-scalers for use in direct pull
      *argSlot0 = (uint64_t(1)<<32) | (uint32_t)redOpArg;
      *argSlot1 = (uint64_t(1)<<32) | (uint32_t)(redOpArg>>32);
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
    }
    if (Direct && recvAcceptor) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
      volatile uint64_t* argSlot0 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange;
      volatile uint64_t* argSlot1 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange+1;
      void *ptr;
      while (true) {
        ptr = *slot;
        if (ptr != nullptr || checkAbort(spins)) break;
      }
      directBuff = regUsed ? (T*)(MaxSend == 0 ? e->upOutputs[index] : e->dnInputs[index]) :
                   reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));
      if (MaxSend != 0) { // reduce group rather than gather group
        // Store scalers for remote inputs
        uint64_t arg0, arg1;
        while (true) {
          arg0 = *argSlot0;
          arg1 = *argSlot1;
          if ((arg0 != 0 && arg1 != 0) || checkAbort(spins)) break;
        }
        ncclShmem.redOpArgs[1+index] = ((arg1 & 0xffffffff)<<32) | (arg0 & 0xffffffff);
      }
      *argSlot0 = 0; *argSlot1 = 0;
      *slot = nullptr;
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (flags & (RoleInput|RoleOutput))
      userBuff += delta;
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, -1, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, -1, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvSend(int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, -1>(-1, -1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopySend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<1, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, intptr_t remoteInpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, remoteInpIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, int peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, int totalElem, int peerElem, int peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, int peerOffset, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, int totalElem, int peerElem, int peerOffset, int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
};
