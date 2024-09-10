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

const char* ncclFuncToString(ncclFunc_t op);
const char* ncclDevRedOpToString(ncclDevRedOp_t op);
const char* ncclDatatypeToString(ncclDataType_t type);
const char* ncclAlgoToString(int algo);
const char* ncclProtoToString(int proto);

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
    return 1;
  case ncclFloat16:
  #if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
  #endif
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

  int aggFactor;
  int as; // aggregated steps
  int a; // step inside aggregated step
  int sendSkipped; // number of skipped steps during aggregation
  int recvSkipped; // number of skipped steps during aggregation
  int phase2recv;  // receive offset for phase 2
  int aggDelta;
  int scale;
  int phase;

  __device__ __host__ int min(int a, int b) {
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
    sendSkipped = recvSkipped = 0;
    lastA = aggFactor;
    if (phase >= 2) lastA /= 2*scale;
  }

  __device__ __host__ void reset() {
    nelem = getNelem();
    phase = 0;
    scale = 1;
    phase2recv = 0;
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
   __device__ __host__ PatRSAlgorithm(int stepSize, int stepDepth, size_t offset, size_t end, size_t count, int chunkCount, int rank, int nranks):
     offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    aggDelta = nrPow2 = (1<<log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end-offset;
    while (stepSize / (channelSize*sizeof(T)*aggFactor) >= 2 && aggFactor < nranks/2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks/2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }

    reset();
  }

  __device__ __host__ void getNextOp(int &recvDim, int &sendDim, size_t &inpIx, size_t &outIx, int &recvOffset, int &sendOffset, int &sendStepOffset, int &nelemOut, int &postRecv, int &postSend, int &last) {
restart:
    last = 0;
    nelemOut = nelem;
    outIx = offset;
    int skip = 0;
    //printf("Phase %d as %d/%d a %d/%d scale %d\n", phase, as, aggDelta, a, lastA, scale);
    if (phase == 0) {
      int s = mirrorInvert(a, lastA)*aggDelta + as;
      if (s >= nranks) skip = 1;
      int sendDataRank = (rank + s) % nranks;
      inpIx = sendDataRank * count + offset;
      recvDim = -1;
      sendDim = 0;
      outIx = 0;
      recvOffset = -1;
      sendOffset = ((a - sendSkipped)%postFreq) * nelem;
      sendStepOffset = 0;
      if ((((a - sendSkipped)%postFreq) + 1 >= postFreq) || (a == lastA-1)) {
        postSend = 1;
      } else {
        postSend = 0;
      }
      postRecv = 0;
      if (skip) sendSkipped++;
      if (++a == lastA) {
        phase = as == 1 ? (aggFactor > 1 ? 2 : 4) : 1; // If as == 1, switch to phase 2
        resetA();
      }
      if (skip == 0) return;
    } else if (phase == 1) {
      int s = mirrorInvert(a, lastA)*aggDelta + as;
      if (s >= nranks) skip = 1;
      recvDim = firstBitSet(s, nrPow2);
      sendOffset = ((a - sendSkipped)%postFreq)*nelem;
      recvOffset = ((a - recvSkipped)%postFreq)*nelem;
      postSend = 0;
      if (recvDim == 0) {
        if ((((a - sendSkipped)%postFreq) + 1 >= postFreq) || (a == lastA-1)) postSend = 1;
        sendStepOffset = 0;
      } else {
        sendStepOffset = (a - sendSkipped)/postFreq;
      }
      if ((((a - recvSkipped)%postFreq) + 1 >= postFreq) || (a == lastA-1)) {
        postRecv = 1;
      } else {
        postRecv = 0;
      }
      s -= (1<<recvDim);
      int recvDataRank = (rank + nranks + s) % nranks;
      inpIx = recvDataRank * count + offset;
      sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (sendDim == -1) {
        sendOffset = -1;
        sendStepOffset = 0;
      } else if (as - (1<<recvDim) == 0) {
        if (newPeer(a, aggFactor)) sendSkipped = a;
        int foffset = a - sendSkipped;
        sendStepOffset = recvDim == 0 ? 0 : foffset/postFreq;
        sendOffset = (foffset%postFreq)*nelem;
      }
      if (s < nranks && skip) {
        recvDim = -1;
        recvOffset = -1;
        postRecv = 0;
        skip = 0;
      }
      if (skip || recvDim == -1) recvSkipped++;
      if (skip) sendSkipped++;
      if (++a == lastA) {
        as--;
        phase = as % 2 == 1 ? 0 : 1;
        resetA();
      }
      if (skip == 0) return;
    } else if (phase == 2) {
      int s = (2*mirrorInvert(a, lastA)+1)*scale*aggDelta + 1;
      postRecv = 0;
      if (s >= nranks) skip = 1;
      recvDim = 0;
      postSend = a == lastA-1 ? 1 : 0;
      s -= 1;
      if (s < nranks && skip) {
        recvDim = -1;
        recvOffset = -1;
        skip = 0;
      } else if (!skip) {
        int foffset = phase2recv;
        phase2recv++;
        postRecv |= ((foffset+1)%postFreq) == 0 ? 1 : 0;
        recvOffset = (foffset%postFreq) * nelem;
      }
      int recvDataRank = (rank + nranks + s) % nranks;
      inpIx = recvDataRank * count + offset;
      sendDim = s ? firstBitSet(s, nrPow2) : -1;
      int foffset = a - sendSkipped;
      postSend |= ((foffset+1)%postFreq) == 0 ? 1 : 0;
      sendStepOffset = 0;
      sendOffset = (foffset%postFreq) * nelem;
      if (skip || sendDim == -1) sendSkipped++;
      if (++a == lastA) {
        phase = 3;
        resetA();
      }
      if (skip == 0) return;
    } else if (phase == 3) {
      int s = (2*mirrorInvert(a, lastA)+1)*scale*aggDelta;
      postRecv = a == lastA-1 ? 1 : 0;
      if (s >= nranks) skip = 1;
      recvDim = firstBitSet(s, nrPow2);
      postSend = 0;
      s -= (1<<recvDim);
      int foffset = a - recvSkipped;
      postRecv |= (foffset+1)%postFreq == 0 ? 1 : 0;
      recvOffset = (foffset%postFreq) * nelem;
      int recvDataRank = (rank + nranks + s) % nranks;
      inpIx = recvDataRank * count + offset;
      sendDim = s ? firstBitSet(s, nrPow2) : -1;
      if (s < nranks && skip) {
        recvDim = -1;
        recvOffset = -1;
        postRecv = 0;
        skip = 0;
      }
      if (newPeer(a, aggFactor/(2*scale))) sendSkipped = a;
      foffset = a - sendSkipped;
      sendStepOffset = foffset / postFreq; // Accumulate on next steps
      sendOffset = sendDim >= 0 ? (foffset%postFreq) * nelem : -1;
      if (skip || recvDim == -1) recvSkipped++;
      if (skip) sendSkipped++;
      if (++a == lastA) {
        scale *= 2;
        phase = scale < aggFactor ? 2 : 4;
        resetA();
      }
      if (skip == 0) return;
    } else if (phase == 4) {
      recvDim = 0;
      sendDim = -1;
      inpIx = rank * count + offset;
      recvOffset = (phase2recv%postFreq) * nelem;
      sendStepOffset = 0;
      sendOffset = -1;
      postRecv = 1;
      postSend = 0;
      offset += chunkCount;
      if (offset >= end) {
        last = 1;
      } else {
        reset();
      }
      return;
    }
    goto restart;
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

  __device__ __host__ int min(int a, int b) {
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
   __device__ __host__ PatAGAlgorithm(int stepSize, int stepDepth, size_t offset, size_t end, size_t count, int chunkCount, int rank, int nranks):
     offset(offset), end(end), count(count), chunkCount(chunkCount), rank(rank), nranks(nranks) {
    aggDelta = nrPow2 = (1<<log2Up(nranks));

    aggFactor = 1;
    size_t channelSize = end-offset;
    while (stepSize / (channelSize*sizeof(T)*aggFactor) >= 2 && aggFactor < nranks/2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks/2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }
    //printf("AggFactor %d PostFreq %d AggDelta %d\n", aggFactor, postFreq, aggDelta);

    asDim = log2Up(aggDelta);
    reset();
  }

  __device__ __host__ void getNextOp(int &recvDim, int &sendDim, size_t &inpIx, size_t &outIx, int &recvOffset, int &sendOffset, int &recvStepOffset, int &nelemOut, int &postRecv, int &postSend, int &last) {
restart:
    //printf("Phase %d as %d/%d a %d/%d scale %d\n", phase, as, aggDelta, a, lastA, scale);
    last = 0;
    nelemOut = nelem;
    inpIx = offset;
    int skip = 0;
    if (phase == 0) {
      int s = a*aggDelta + as;
      if (s >= nranks) skip = 1;
      int nextSkip = (a+1)*aggDelta + as >= nranks ? 1 : 0;
      int recvDataRank = (rank + s) % nranks;
      outIx = recvDataRank * count + offset;
      sendDim = -1;
      recvDim = 0;
      inpIx = 0;
      sendOffset = -1;
      recvOffset = (a % postFreq) * nelem;
      recvStepOffset = 0;
      postRecv = (a % postFreq == postFreq-1) || ((a+1)*aggDelta+as >= nranks) ? 1 : 0;
      postSend = 0;
      a++;
      if (nextSkip) {
        as = nextAs();
        if (as == aggDelta/2) {
          offset += chunkCount;
          if (offset >= end) {
            last = 1;
          } else {
            reset();
          }
          return;
        }
        phase = 1;
        resetA();
      }
      if (skip == 0) return;
   } else if (phase == 1) {
      int s = a*aggDelta + as;
      if (s >= nranks) skip = 1;
      sendDim = firstBitSet(s, nrPow2);
      s -= (1<<sendDim);
      int sendDataRank = (rank + nranks + s) % nranks;
      outIx = sendDataRank * count + offset;
      recvDim = s ? firstBitSet(s, nrPow2) : -1;
      sendOffset = recvOffset = (a % postFreq) * nelem;
      postSend = (a % postFreq == postFreq-1) || ((a+1)*aggDelta+as >= nranks) ? 1 : 0;
      postRecv = (sendDim == 0) && ((a % postFreq == postFreq-1) || ((a+1)*aggDelta+as-1 >= nranks)) ? 1 : 0;
      recvStepOffset = (sendDim == 0) ? 0 : a/postFreq;
      if (recvDim == -1) {
        recvOffset = -1;
        postRecv = 0;
      } else if (as - (1<<sendDim) == 0) {
        int foffset = (a*aggDelta) >> (recvDim+1);
        recvOffset = (foffset%postFreq)*nelem;
        postRecv = (sendDim == 0) && ((foffset % postFreq == postFreq-1) || ((((foffset+1)*2)+1)<<recvDim) >= nranks) ? 1 : 0;
        recvStepOffset = (sendDim == 0) ? 0 : foffset/postFreq;
      }
      if (s < nranks && sendDim == 0 && skip) {
        // Don't forget to receive at least once even if we don't send afterwards
        sendDim = -1;
        sendOffset = -1;
        postSend = 0;
        skip = 0;
      }
      if (++a == lastA) {
        if (as % 2 == 1) {
          phase = 0;
        } else {
          as = nextAs();
        }
        resetA();
      }
      if (skip == 0) return;
    } else if (phase == 2) {
      int s = (2*a+1)*scale*aggDelta;
      postSend = (a % postFreq == postFreq-1) || ((2*(a+1)+1)*scale*aggDelta >= nranks) ? 1 : 0;
      postRecv = 0;
      if (s >= nranks) skip = 1;
      sendDim = firstBitSet(s, nrPow2);
      s -= (1<<sendDim);
      sendOffset = (a%postFreq) * nelem;
      recvStepOffset = a / postFreq;
      int sendDataRank = (rank + nranks + s) % nranks;
      outIx = sendDataRank * count + offset;
      recvDim = s ? firstBitSet(s, nrPow2) : -1;
      s -= (1<<recvDim);
      if (recvDim == -1) {
        recvOffset = -1;
      } else {
        int foffset = (a*2*scale*aggDelta) >> (recvDim+1);
        recvOffset = (foffset%postFreq)*nelem;
        recvStepOffset = foffset / postFreq;
      }
      if (++a == lastA) {
        scale /= 2;
        phase = scale ? 2 : 1;
        resetA();
      }
      if (skip == 0) return;
    }
    goto restart;
  }
};
#endif
