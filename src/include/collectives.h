/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "nccl.h"
#include "nccl_common.h"
#include "device.h"

#define NCCL_MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above
#define NCCL_MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

const char* ncclFuncToString(ncclFunc_t op);
const char* ncclDevRedOpToString(ncclDevRedOp_t op);
const char* ncclDatatypeToString(ncclDataType_t type);
const char* ncclAlgoToString(int algo);
const char* ncclProtoToString(int proto);

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

#include <sys/types.h>

#define NCCL_MODE_NORMAL 0
#define NCCL_MODE_OFFSET 1
#define NCCL_MODE_PTR    2
struct ncclConnFifo {
  int mode;
  int offset;
  ssize_t size;
  void* ptr;
};

#include <stdio.h>

class RingAlgorithm {
protected:
  int refCount;
  int nRanks;
  int nStepsPerLoop;
  int chunkSteps;
  int sliceSteps;
  ssize_t sliceSize;
  ssize_t loopSize;
  ssize_t channelSize;
  uint8_t *sendbuff;
  uint8_t *recvbuff;
  void *sendMhandle;
  void *recvMhandle;
  void *srecvMhandle;
public:
  // this ring class is used by proxy thread to retrieve the send and recv buffer, size as well as corresponding
  // mem handle based on the current step of the proxy args. The derived ring algo class is AR, AG, and BC which
  // would be allocated during enqueue stage and copied to proxy side through shared memory. For each copy, we will
  // increase the refCount by incRefCount() since the same ring algo object can be referenced multiple times for send
  // and recv progress. After all steps are done, we decrease the refCount and only delete the ring object when
  // refCount == 0.
  virtual void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) = 0;
  virtual void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) = 0;
  int incRefCount() {
    return __atomic_add_fetch(&refCount, 1, __ATOMIC_RELAXED);
  }
  int decRefCount() {
    return __atomic_sub_fetch(&refCount, 1, __ATOMIC_RELEASE);
  }
  RingAlgorithm() { refCount = 0; }
  virtual ~RingAlgorithm() {};
};

class RingARAlgorithm : public RingAlgorithm {
private:
  int ringIndex;
  int elemSize;
  ssize_t chunkSize;
  int slicePerChunk;
public:
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int curLoopStage = (curStep % nStepsPerLoop) / chunkSteps;
    int chunkStage = curLoopStage % nRanks;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t remSize = channelSize - elemOffset;
    ssize_t chunkOffset;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t curChunkSize;
    ssize_t size;
    ssize_t nelem;
    int chunkId;

    if (remSize < loopSize) {
      curChunkSize = alignUp(divUp(remSize / elemSize, nRanks), 16 / elemSize) * elemSize;
    } else {
      curChunkSize = chunkSize;
    }
    chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;
    chunkOffset = chunkId * curChunkSize;
    nelem = std::min(remSize - chunkOffset, curChunkSize);
    curSliceSize = std::max(divUp(nelem / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;

    if (nelem <= sliceOffset) {
      *sendbuffOut = sendbuff;
      *mhandleOut = sendMhandle;
    } else {
      if (curLoopStage == 0) {
        *sendbuffOut = sendbuff + elemOffset + chunkOffset + sliceOffset;
        *mhandleOut = sendMhandle;
      } else {
        *sendbuffOut = recvbuff + elemOffset + chunkOffset + sliceOffset;
        *mhandleOut = srecvMhandle;
      }
    }
    size = std::min(curSliceSize, nelem - sliceOffset);
    *sizeOut = size < 0 ? 0 : size;
    return;
  }

  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int curLoopStage = ((curStep + chunkSteps) % nStepsPerLoop) / chunkSteps;
    int chunkStage = curLoopStage % nRanks;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t remSize = channelSize - elemOffset;
    ssize_t chunkOffset;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t curChunkSize;
    ssize_t size;
    ssize_t nelem;
    int chunkId;

    if (remSize < loopSize) {
      curChunkSize = alignUp(divUp(remSize / elemSize, nRanks), 16 / elemSize) * elemSize;
    } else {
      curChunkSize = chunkSize;
    }

    if (curLoopStage == 0) {
      chunkId = (ringIndex + 1) % nRanks;
    } else {
      chunkId = (ringIndex + nRanks - 1 - chunkStage) % nRanks;
    }

    chunkOffset = chunkId * curChunkSize;
    nelem = std::min(remSize - chunkOffset, curChunkSize);
    curSliceSize = std::max(divUp(nelem / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;
    if (nelem <= sliceOffset) {
      *recvbuffOut = recvbuff;
    } else {
      *recvbuffOut = recvbuff + elemOffset + chunkOffset + sliceOffset;
    }
    if (sizeOut) {
      size = std::min(curSliceSize, nelem - sliceOffset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
    return;
  }

  RingARAlgorithm(const void *sendbuff, void *recvbuff, int nRanks, int ringIndex, int chunkSteps, int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, int elemSize, void *sendMhandle, void *recvMhandle, void *srecvMhandle) {
    this->ringIndex = ringIndex;
    this->nRanks = nRanks;
    this->nStepsPerLoop = 2 * (nRanks - 1) * chunkSteps;
    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->chunkSize = chunkSize;
    this->sliceSize = sliceSize;
    this->loopSize = nRanks * chunkSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->channelSize = channelSize;
    this->elemSize = elemSize;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
    this->slicePerChunk = chunkSteps / sliceSteps;
  }
  ~RingARAlgorithm() {}
};

class RingAGAlgorithm : public RingAlgorithm {
private:
  int *ringRanks;
  int elemSize;
  ssize_t sendSize;
  int slicePerChunk;
public:
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int chunkStage = (curStep % nStepsPerLoop) / chunkSteps;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t chunkSize = std::min(loopSize, channelSize - elemOffset);
    ssize_t size;
    int rankDest;
    uint8_t *buff;
    void *mhandle;

    curSliceSize = std::max(divUp(chunkSize / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;
    if (chunkStage == 0) {
      rankDest = ringRanks[0];
      offset = elemOffset + sliceOffset;
      buff = sendbuff + offset;
      mhandle = sendMhandle;
    } else {
      rankDest = ringRanks[nRanks - chunkStage];
      offset = elemOffset + rankDest * sendSize + sliceOffset;
      buff = recvbuff + offset;
      mhandle = srecvMhandle;
    }
    *sendbuffOut = buff;
    size = std::min(curSliceSize, channelSize - elemOffset - sliceOffset);
    *sizeOut = size < 0 ? 0 : size;
    *mhandleOut = mhandle;
    return;
  }

  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int chunkStage = ((curStep + chunkSteps) % nStepsPerLoop) / chunkSteps;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset;
    ssize_t curSliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t chunkSize = std::min(loopSize, channelSize - elemOffset);
    ssize_t size;
    int rankDest;

    curSliceSize = std::max(divUp(chunkSize / elemSize, 16 * slicePerChunk) * 16, sliceSize / elemSize / 32) * elemSize;
    sliceOffset = sliceStage * curSliceSize;
    if (chunkStage == 0) {
      rankDest = ringRanks[1];
    } else {
      rankDest = ringRanks[nRanks - chunkStage];
    }
    offset = elemOffset + rankDest * sendSize + sliceOffset;
    *recvbuffOut = recvbuff + offset;
    if (sizeOut) {
      size = std::min(sliceSize, channelSize - elemOffset - sliceOffset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
  }

  RingAGAlgorithm(const void *sendbuff, void *recvbuff, int nRanks, int *ringRanks, int chunkSteps, int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, int elemSize, size_t sendSize, void *sendMhandle, void *recvMhandle, void *srecvMhandle) {
    this->ringRanks = ringRanks;
    this->nRanks = nRanks;
    this->nStepsPerLoop = (nRanks - 1) * chunkSteps;
    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->elemSize = elemSize;
    this->sliceSize = sliceSize;
    this->loopSize = chunkSize;
    this->sendSize = sendSize;
    this->channelSize = channelSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
    this->slicePerChunk = chunkSteps / sliceSteps;
  }
  ~RingAGAlgorithm() {}
};

class RingBCAlgorithm : public RingAlgorithm {
private:
  int root;
  int rank;
  int nextRank;
public:
  void getNextSendAddr(int curStep, uint8_t **sendbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset = sliceStage * sliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t size;
    uint8_t *buff;
    void *mhandle;

    offset = elemOffset + sliceOffset;
    if (offset >= channelSize) {
      buff = sendbuff;
      mhandle = sendMhandle;
    } else if (rank == root) {
      buff = sendbuff + offset;
      mhandle = sendMhandle;
    } else {
      buff = recvbuff + offset;
      mhandle = srecvMhandle;
    }
    *sendbuffOut = buff;
    size = std::min(sliceSize, channelSize - offset);
    *sizeOut = size < 0 ? 0 : size;
    *mhandleOut = mhandle;
    return;
  }

  void getNextRecvAddr(int curStep, uint8_t **recvbuffOut, size_t *sizeOut, void **mhandleOut) {
    int curLoop = curStep / nStepsPerLoop;
    int sliceStage = (curStep % chunkSteps) / sliceSteps;
    ssize_t sliceOffset = sliceStage * sliceSize;
    ssize_t offset;
    ssize_t elemOffset = curLoop * loopSize;
    ssize_t size;
    offset = elemOffset + sliceOffset;
    if (offset >= channelSize) {
      *recvbuffOut = recvbuff;
    } else {
      *recvbuffOut = recvbuff + offset;
    }
    if (sizeOut) {
      size = std::min(sliceSize, channelSize - offset);
      *sizeOut = size < 0 ? 0 : size;
    }
    *mhandleOut = recvMhandle;
    return;
  }

  RingBCAlgorithm(const void* sendbuff, void* recvbuff, int rank, int root, int nRanks, int *ringRanks, int chunkSteps, int sliceSteps, size_t chunkSize, size_t sliceSize, size_t gridOffset, size_t channelSize, void *sendMhandle, void *recvMhandle, void *srecvMhandle) {
    this->root = root;
    this->rank = rank;
    this->nextRank = ringRanks[1];
    this->nStepsPerLoop = chunkSteps;
    this->chunkSteps = chunkSteps;
    this->sliceSteps = sliceSteps;
    this->sliceSize = sliceSize;
    this->loopSize = chunkSize;
    this->channelSize = channelSize;
    this->sendbuff = (uint8_t*)sendbuff + gridOffset;
    this->recvbuff = (uint8_t*)recvbuff + gridOffset;
    this->sendMhandle = sendMhandle;
    this->recvMhandle = recvMhandle;
    this->srecvMhandle = srecvMhandle;
  }
  ~RingBCAlgorithm() {}
};

#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#include <cuda/atomic>
#endif

// Need a power of two to ensure it divides by parallelFactor (which is also a power of two)
#define NCCL_PAT_NWORKERS 512

static constexpr int PatUsed = 0x1,
                     PatSkipped = 0x2;

struct ncclPatStep {
  int recvDim, sendDim, recvOffset, sendOffset, stepOffset, postRecv, postSend, nelem, last, flags;
  size_t inpIx, outIx;
};

struct ncclPatPeer {
    uint64_t step;
    struct ncclConnInfo* conn;
    struct ncclConnFifo* connFifo;
    void* buff;
    uint64_t *headPtr;
    uint64_t *tailPtr;
    uint64_t stepCache;
    long long int accSize;
    int connStepSize;
};

#define NCCL_SHMEM_PAT_STEPS 32
struct ncclPatShmem {
  struct ncclPatStep patSteps[NCCL_SHMEM_PAT_STEPS];
  int parallelFactor;
  long long int localAccSize;
  struct ncclPatPeer sendDims[32]; // Should cover 2^32 ranks
  struct ncclPatPeer recvDims[32];
};

template<typename T>
class PatRSAlgorithm{
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as; // aggregated steps
  int a; // step inside aggregated step
  int sendSkipped; // number of skipped steps during aggregation
  int stepOffset;
  int aggDelta;
  int scale;
  int phase;

  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a<b)?a:b;
  }

  __device__ __host__ int getNelem() {
    return min(chunkCount, end-offset);
  }

  __device__ __host__ int mirrorInvert(int i, int max) {
    int ret = 0;
    for (int mask=1, imask=max/2; mask<max; mask<<=1, imask>>=1) {
      if ((i&mask) == 0) ret += imask;
    }
    return ret;
  }

  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs =
#ifdef __CUDA_ARCH__
      __ffs(i);
#else
      __builtin_ffs(i);
#endif
    return ffs ? ffs-1 : max;
  }

  __device__ __host__ void resetA() {
    a = 0;
    sendSkipped = stepOffset = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2*scale;
    if (phase == 4) lastA = 1;
  }

  __device__ __host__ void reset() {
    nelem = getNelem();
    phase = 0;
    scale = 1;
    as = aggDelta - 1;
    resetA();
  }

  __device__ __host__ int nBitsSet(int i) {
    int nbits =
#ifdef __CUDA_ARCH__
      __popc(i);
#else
      __builtin_popcount(i);
#endif
    return nbits;
  }

  // Return 1 when only upper bits are set. For example, if nrpow2==16 we'll return 1 for 8, 12, 14, 15.
  // A number being in the form of 1111000 implies that the complementary is 0000111 meaning it's a power of 2 minus 1.
  __device__ __host__ int newPeer(int i, int pow2) {
    //printf("New peer %d/%d -> %d\n", i, pow2, nBitsSet((i ^ (pow2-1)) + 1) == 1 ? 1 : 0);
    return nBitsSet((i ^ (pow2-1)) + 1) == 1 ? 1 : 0;
  }

public:
   __device__ __host__ PatRSAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end, size_t count, int chunkCount, int rank, int nranks):
     offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1<<log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end-offset;
    while (stepSize / (channelSize*sizeof(T)*aggFactor) >= 2 && aggFactor < nranks/2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks/2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    reset();
  }

  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->outIx = offset;
    ps->stepOffset = stepOffset;
    int skip = 0;
    if (a >= lastA) {
      skip = 1;
    } else if (phase == 0) {
      int s = mirrorInvert(a, lastA)*aggDelta + as;
      if (s >= nranks) skip = 1;
      int sendDataRank = (rank + s) % nranks;
      ps->inpIx = sendDataRank * count + offset;
      ps->recvDim = -1;
      ps->sendDim = 0;
      ps->outIx = 0;
      ps->recvOffset = -1;
      ps->sendOffset = (a%postFreq) * nelem;
      if (((a%postFreq) + 1 >= postFreq) || (a == lastA-1)) {
        ps->postSend = 1;
      } else {
        ps->postSend = 0;
      }
      ps->postRecv = 0;
    } else if (phase == 1) {
      int s = mirrorInvert(a, lastA)*aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->sendOffset = (a%postFreq)*nelem;
      ps->recvOffset = (a%postFreq)*nelem;
      ps->postSend = 0;
      if (ps->recvDim == 0 && (((a%postFreq) + 1 >= postFreq) || (a == lastA-1))) ps->postSend = 1;
      if (((a%postFreq) + 1 >= postFreq) || (a == lastA-1)) {
        ps->postRecv = 1;
      } else {
        ps->postRecv = 0;
      }
      s -= (1<<ps->recvDim);
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->sendDim == -1) {
        ps->sendOffset = -1;
      } else if (as - (1<<ps->recvDim) == 0) {
        if (newPeer(a, aggFactor)) { sendSkipped = a; ps->stepOffset = stepOffset = 0; }
        int foffset = a - sendSkipped;
        ps->sendOffset = (foffset%postFreq)*nelem;
      }
      int recvDim = ps->recvDim;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (recvDim > 0 && (((a-sendSkipped)%postFreq) + 1 >= postFreq) && skip == 0) stepOffset++;
    } else if (phase == 2) {
      int s = (2*mirrorInvert(a, lastA)+1)*scale*aggDelta + 1;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = 0;
      ps->postSend = a == lastA-1 ? 1 : 0;
      s -= 1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        skip = 0;
      } else if (!skip) {
        int foffset = a + aggFactor - aggFactor/scale;
        ps->postRecv |= ((foffset+1)%postFreq) == 0 ? 1 : 0;
        ps->recvOffset = (foffset%postFreq) * nelem;
      }
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      int foffset = a;
      ps->postSend |= ((foffset+1)%postFreq) == 0 ? 1 : 0;
      ps->sendOffset = (foffset%postFreq) * nelem;
    } else if (phase == 3) {
      int s = (2*mirrorInvert(a, lastA)+1)*scale*aggDelta;
      ps->postRecv = a == lastA-1 ? 1 : 0;
      if (s >= nranks) skip = 1;
      ps->recvDim = firstBitSet(s, nrPow2);
      ps->postSend = 0;
      s -= (1<<ps->recvDim);
      int foffset = a;
      ps->postRecv |= (foffset+1)%postFreq == 0 ? 1 : 0;
      ps->recvOffset = (foffset%postFreq) * nelem;
      int recvDataRank = (rank + nranks + s) % nranks;
      ps->inpIx = recvDataRank * count + offset;
      ps->sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (s < nranks && skip) {
        ps->recvDim = -1;
        ps->recvOffset = -1;
        ps->postRecv = 0;
        skip = 0;
      }
      if (newPeer(a, aggFactor/(2*scale))) { sendSkipped = a; ps->stepOffset = stepOffset = 0; }
      foffset = a - sendSkipped;
      if ((foffset%postFreq) + 1 >= postFreq && skip == 0) stepOffset++;
      ps->sendOffset = ps->sendDim >= 0 ? (foffset%postFreq) * nelem : -1;
    } else if (phase == 4) {
      ps->recvDim = 0;
      ps->sendDim = -1;
      ps->inpIx = rank * count + offset;
      ps->recvOffset = ((aggFactor-1)%postFreq) * nelem;
      ps->sendOffset = -1;
      ps->postRecv = 1;
      ps->postSend = 0;
      offset += chunkCount;
    }
    a++;
    if (a >= lastA && a >= parallelFactor) {
      int p = phase;
      if (p == 1) as--;
      if (p == 3) scale *= 2;
      phase =
        p == 0 ? as == 1 ? (aggFactor > 1 ? 2 : 4) : 1 :
        p == 1 ? as % 2 == 1 ? 0 : 1 :
        p == 2 ? 3 :
        p == 3 ? scale < aggFactor ? 2 : 4 :
        5;
      if (p == 4) {
        if (offset >= end) {
          ps->last = 2;
        } else {
          reset();
        }
      } else {
        resetA();
      }
    } else if (phase == 4 && offset >= end) {
      ps->last = 1;
    }
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if __CUDA_ARCH__ >= 600
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    ps->flags = flags;
#endif
  }
};

template<typename T>
class PatAGAlgorithm{
  size_t offset;
  size_t end;
  size_t count;
  int chunkCount;
  int nelem;
  int rank;
  int nranks;
  int nrPow2;
  int postFreq;
  int lastA;
  int parallelFactor;
  int aggFactor;
  int as; // aggregated steps
  int a; // step inside aggregated step
  int aggDelta;
  int scale;
  int phase;

  // AS computation
  int asDim;
  int v;
  int bitCount[32];
  int bitZeroStep[32];

  __device__ __host__ ssize_t min(ssize_t a, ssize_t b) {
    return (a<b)?a:b;
  }

  __device__ __host__ int getNelem() {
    return min(chunkCount, end-offset);
  }

  __device__ __host__ int mirror(int i, int max) {
    int ret = 0;
    for (int mask=1, imask=max/2; mask<max; mask<<=1, imask>>=1) {
      if ((i&mask)) ret += imask;
    }
    return ret;
  }

  __device__ __host__ int firstBitSet(int i, int max) {
    int ffs =
#ifdef __CUDA_ARCH__
      __ffs(i);
#else
      __builtin_ffs(i);
#endif
    return ffs ? ffs-1 : max;
  }

  __device__ __host__ void resetA() {
    a = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2*scale;
  }

  __device__ __host__ void reset() {
    nelem = getNelem();
    scale = aggFactor/2;
    phase = scale ? 2 : 1;
    v = 0;
    for (int i = 0; i<asDim; i++) {
      bitCount[i] = asDim-i;
      bitZeroStep[i] = 1;
    }
    as = nextAs();
    resetA();
  }

  __device__ __host__ int nextAs() {
    for (int d=0; d<asDim; d++) {
      int p = 1<<d;
      bitCount[d]--;
      if (bitCount[d] == 0) {
        v ^= p;
        bitCount[d] = p;
        if ((v&p) == 0) {
          bitCount[d] += firstBitSet(bitZeroStep[d], asDim) - 1;
          if (bitCount[d] == 0) {
            v ^= p;
            bitCount[d] = p;
          }
          bitZeroStep[d]++;
        }
      }
    }
    return v;
  }


public:
   __device__ __host__ PatAGAlgorithm(int stepSize, int stepDepth, int maxParallelFactor, size_t offset, size_t end, size_t count, int chunkCount, int rank, int nranks):
     offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    parallelFactor = maxParallelFactor;
    aggDelta = nrPow2 = (1<<log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end-offset;
    while (stepSize / (channelSize*sizeof(T)*aggFactor) >= 2 && aggFactor < nranks/2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks/2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    asDim = log2Up(aggDelta);
    reset();
  }

  __device__ __host__ int getParallelFactor() {
    return parallelFactor;
  }

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    ps->last = 0;
    ps->nelem = nelem;
    ps->inpIx = offset;
    int skip = 0;
    if (a >= lastA) {
      skip = 1;
    } else if (phase == 0) {
      int s = a*aggDelta + as;
      if (s >= nranks) skip = 1;
      int recvDataRank = (rank + s) % nranks;
      ps->outIx = recvDataRank * count + offset;
      ps->sendDim = -1;
      ps->recvDim = 0;
      ps->inpIx = 0;
      ps->sendOffset = -1;
      ps->recvOffset = (a % postFreq) * nelem;
      ps->stepOffset = 0;
      ps->postRecv = (a % postFreq == postFreq-1) || ((a+1)*aggDelta+as >= nranks) ? 1 : 0;
      ps->postSend = 0;
   } else if (phase == 1) {
      int s = a*aggDelta + as;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      s -= (1<<ps->sendDim);
      int sendDataRank = (rank + nranks + s) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = s ? firstBitSet(s, nrPow2) : -1;
      ps->sendOffset = ps->recvOffset = (a % postFreq) * nelem;
      ps->postSend = (a % postFreq == postFreq-1) || ((a+1)*aggDelta+as >= nranks) ? 1 : 0;
      ps->postRecv = (ps->sendDim == 0) && ((a % postFreq == postFreq-1) || ((a+1)*aggDelta+as-1 >= nranks)) ? 1 : 0;
      ps->stepOffset = (ps->sendDim == 0) ? 0 : a/postFreq;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
        ps->postRecv = 0;
      } else if (as - (1<<ps->sendDim) == 0) {
        int foffset = (a*aggDelta) >> (ps->recvDim+1);
        ps->recvOffset = (foffset%postFreq)*nelem;
        ps->postRecv = (ps->sendDim == 0) && ((foffset % postFreq == postFreq-1) || ((((foffset+1)*2)+1)<<ps->recvDim) >= nranks) ? 1 : 0;
        ps->stepOffset = (ps->sendDim == 0) ? 0 : foffset/postFreq;
      }
      if (s < nranks && ps->sendDim == 0 && skip) {
        // Don't forget to receive at least once even if we don't send afterwards
        ps->sendDim = -1;
        ps->sendOffset = -1;
        ps->postSend = 0;
        skip = 0;
      }
    } else if (phase == 2) {
      int s = (2*a+1)*scale*aggDelta;
      ps->postSend = (a % postFreq == postFreq-1) || ((2*(a+1)+1)*scale*aggDelta >= nranks) ? 1 : 0;
      ps->postRecv = 0;
      if (s >= nranks) skip = 1;
      ps->sendDim = firstBitSet(s, nrPow2);
      s -= (1<<ps->sendDim);
      ps->sendOffset = (a%postFreq) * nelem;
      ps->stepOffset = a / postFreq;
      int sendDataRank = (rank + nranks + s) % nranks;
      ps->outIx = sendDataRank * count + offset;
      ps->recvDim = s ? firstBitSet(s, nrPow2) : -1;
      if (ps->recvDim == -1) {
        ps->recvOffset = -1;
      } else {
        s -= (1<<ps->recvDim);
        int foffset = (a*2*scale*aggDelta) >> (ps->recvDim+1);
        ps->recvOffset = (foffset%postFreq)*nelem;
        ps->stepOffset = foffset / postFreq;
      }
    }
    a++;
    if (a >= lastA && a >= parallelFactor) {
      int p = phase;
      if (p == 2) scale /= 2;
      phase =
        p == 2 ? scale ? 2 : 1 :
        p == 1 ? as % 2 == 1 ? 0 : 1 :
        1;
      if (p == 0 || (p == 1 && as % 2 == 0)) as = nextAs();
      if (p == 0 && as == aggDelta/2) {
        offset += chunkCount;
        if (offset >= end) {
          ps->last = 2;
        } else {
          reset();
        }
      } else {
        resetA();
      }
    } else if (phase == 0 && as == 1 && offset + chunkCount >= end && a-1 >= ((lastA-1) / parallelFactor) * parallelFactor) {
      ps->last = 1;
    }
    int flags = PatUsed | (skip ? PatSkipped : 0);
#if __CUDA_ARCH__ >= 600
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    ps->flags = flags;
#endif
  }
};
#endif
