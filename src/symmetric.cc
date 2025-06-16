#include "symmetric.h"
#include "comm.h"
#include "device.h"
#include <cmath>

constexpr char const* kernelName[] = {
  // Must align with enum ncclSymKernelId definition in src/include/symmetric.h
  "AllReduce_AGxLL_R",
  "AllReduce_AGxLLMC_R",
  "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC",
  "AllGather_LL",
  "AllGather_LLMC",
  "AllGather_ST",
  "AllGather_STMC",
  "ReduceScatter_LL",
  "ReduceScatter_LD",
  "ReduceScatter_LDMC"
};

constexpr uint32_t kernelMask_STMC = 1<<ncclSymKernelId_AllGather_LLMC |
                                     1<<ncclSymKernelId_AllGather_STMC |
                                     1<<ncclSymKernelId_AllReduce_AGxLLMC_R |
                                     1<<ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymKernelId_ReduceScatter_LDMC;

constexpr uint32_t kernelMask_LDMC = 1<<ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymKernelId_ReduceScatter_LDMC;

constexpr uint32_t kernelMask_LL = 1<<ncclSymKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymKernelId_AllGather_LL |
                                   1<<ncclSymKernelId_AllGather_LLMC |
                                   1<<ncclSymKernelId_ReduceScatter_LL;

constexpr uint32_t kernelMask_AG = 1<<ncclSymKernelId_AllGather_LL |
                                   1<<ncclSymKernelId_AllGather_LLMC |
                                   1<<ncclSymKernelId_AllGather_ST |
                                   1<<ncclSymKernelId_AllGather_STMC;

constexpr uint32_t kernelMask_AR = 1<<ncclSymKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                   1<<ncclSymKernelId_AllReduce_RSxLD_AGxST;

constexpr uint32_t kernelMask_RS = 1<<ncclSymKernelId_ReduceScatter_LD |
                                   1<<ncclSymKernelId_ReduceScatter_LDMC |
                                   1<<ncclSymKernelId_ReduceScatter_LL;

static uint32_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather: return kernelMask_AG;
  case ncclFuncAllReduce: return kernelMask_AR;
  case ncclFuncReduceScatter: return kernelMask_RS;
  default: return 0;
  }
}

static uint32_t kernelMask_user() {
  static uint32_t cache = -1u;
  uint32_t got = __atomic_load_n(&cache, __ATOMIC_RELAXED);
  if (got == -1u) {
    // TODO: Enhance this to be a pattern match. I like regex's but we also have
    // the parseList() used by NCCL_ALGO/PROTO.
    char const* name = ncclGetEnv("NCCL_SYM_KERNEL");
    if (name == nullptr || strcmp(name, "^") == 0) {
      static_assert((int)ncclSymKernelId_Count < 32, "Use more than 32 bits");
      got = (1<<(int)ncclSymKernelId_Count)-1;
    } else {
      got = 0;
      for (int k=0; k < (int)ncclSymKernelId_Count; k++) {
        if (strcmp(kernelName[k], name) == 0) {
          __atomic_store_n(&cache, 1<<k, __ATOMIC_RELAXED);
          got = 1<<k;
          break;
        }
      }
    }
    __atomic_store_n(&cache, got, __ATOMIC_RELAXED);
  }
  return got;
}

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)

static double softmin(double x, double ceiling, double softness) {
  // looks like a smooth version of: min(x, ceiling)
  return ceiling - softness*std::log1p((std::exp(ceiling/softness) - 1)*std::exp(-x/softness));
}

static double softplus(double x, double softness) {
  // looks like a smooth version of: max(0, x)
  double z = x/softness;
  return 100.0 <= z ? x : softness*std::log1p(std::exp(z));
}

static double model(double busBytes, double baseLat, int nSMs, double smBw, double busMultiplier, double peakBw) {
  double bw = softmin(nSMs*smBw*busMultiplier, peakBw, smBw);
  return baseLat + softplus(busBytes/bw - 1, 1);
}

// Given the kernel and bytes, return the minimum number of blocks to run on such that
// perf is 99% of running at max blocks, and return the estimate runtime for that
// block count.
static void queryModel(struct ncclComm* comm, ncclSymKernelId k, size_t nBytes, float* timeUs, int* nBlocks) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling

  int nRanks = comm->nRanks;
  int nMaxBlocks = ncclSymMaxBlocks;
  int nMaxBlocksNvls = divUp((comm->cudaArch < 1000 ? 16 : 32), nRanks);
  size_t busBytes; // max(bytes sent, bytes received)
  double busMultiplier = 1;

  switch (k) {
  default:
    busBytes = size_t(1)<<50;
    break;

  case ncclSymKernelId_AllReduce_AGxLL_R:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymKernelId_AllReduce_AGxLLMC_R:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymKernelId_AllReduce_RSxLD_AGxST:
    busBytes = 2*nBytes*(nRanks-1)/nRanks;
    break;
  case ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC:
    busBytes = nBytes/nRanks + nBytes;
    busMultiplier = nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;

  case ncclSymKernelId_AllGather_LL:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymKernelId_AllGather_LLMC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymKernelId_AllGather_ST:
    busBytes = (nRanks-1)*nBytes;
    break;
  case ncclSymKernelId_AllGather_STMC:
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55*nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;

  case ncclSymKernelId_ReduceScatter_LL:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymKernelId_ReduceScatter_LD:
    busBytes = (nRanks-1)*nBytes;
    break;
  case ncclSymKernelId_ReduceScatter_LDMC:
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55*nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;
  }

  nMaxBlocks = std::min<int>(nMaxBlocks, comm->config.maxCTAs);
  int nMinBlocks = comm->config.minCTAs;

  int nUserCTAs = std::min<int>(ncclSymMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  bool isLL = kernelMask_LL>>k & 1;
  bool isAG = kernelMask_AG>>k & 1;
  bool isAR = kernelMask_AR>>k & 1;
  constexpr double GBps = (1<<30)/1.e6;
  double baseLat, smBw, peakBw;
  if (comm->cudaArch < 1000) {
    baseLat = isLL ? 4.5 : 7.8;
    smBw = isAR ? 65*GBps : 44*GBps;
    peakBw = k == ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480*GBps : 320*GBps;
  } else {
    baseLat = isLL ? (isAG ? 8.5 : 11) : (isAR ? 19.5 : 13.0);
    smBw = 55*GBps;
    peakBw = k == ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000*GBps : 600*GBps;
  }
  *nBlocks = nMaxBlocks;
  *timeUs = model(busBytes, baseLat, nMaxBlocks, smBw, busMultiplier, peakBw);
  // Use least number of blocks that puts us within a tolerance of peak performance.
  for (int bn = nMinBlocks; bn < nMaxBlocks; bn++) {
    double time = model(busBytes, baseLat, bn, smBw, busMultiplier, peakBw);
    if (time <= 1.025*(*timeUs)) {
      *nBlocks = bn;
      *timeUs = time;
      break;
    }
  }
}

bool ncclSymImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  bool isFloat;
  switch (ty) {
  case ncclFloat64:
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    isFloat = true;
    break;
  default:
    isFloat = false;
    break;
  }

  switch (coll) {
  case ncclFuncAllGather:
    return true;
  case ncclFuncAllReduce:
  case ncclFuncReduceScatter:
    return red == ncclDevSum && isFloat && ty != ncclFloat64;
  default:
    return false;
  }
}

ncclResult_t ncclSymPickKernel(
    struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, size_t nElts,
    float* estTimeUs, ncclSymKernelId* kernelId, int* nBlocks, int* nWarps
  ) {
  uint32_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();

  bool hasSTMC = comm->nvlsSupport;
  bool hasLDMC = false;
  if (comm->nvlsSupport) {
    switch (ty) {
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclBfloat16:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      break;
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      hasLDMC &= comm->compCap >= 100;
      break;
    case ncclFloat:
    case ncclDouble:
      hasLDMC = red == ncclDevSum;
      break;
    default: break;
    }
  }
  if (!hasSTMC) kmask &= ~kernelMask_STMC;
  if (!hasLDMC) kmask &= ~kernelMask_LDMC;

  size_t nBytes = nElts*ncclTypeSize(ty);
  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks)*nBytes;
  // LL kernels use 32-bit ints to track element counts and indices.
  if (nBusBytes >= (size_t(2)<<30)) kmask &= ~kernelMask_LL;
  // Any kernel might use 32-bit int to track unrolled loop chunks (which are going
  // to be at least 32 bytes per chunk)
  if (nBusBytes >= 32*(size_t(2)<<30)) kmask = 0;

  ncclSymKernelId bestKernel = ncclSymKernelId_Count;
  float bestTime = 1.e30f;
  int bestBlocks = 999;

  constexpr float smPenalty = .025f; // 2.5% percent increase in time per SM
  uint32_t kmaskRemain = kmask;
  while (kmaskRemain != 0) {
    ncclSymKernelId k = (ncclSymKernelId)popFirstOneBit(&kmaskRemain);
    float kTime;
    int kBlocks;
    queryModel(comm, k, nBytes, &kTime, &kBlocks);
    if (kTime*(1.0f + smPenalty*kBlocks) < bestTime*(1.0f + smPenalty*bestBlocks)) {
      bestKernel = k;
      bestTime = kTime;
      bestBlocks = kBlocks;
    }
  }

  *kernelId = bestKernel;
  *estTimeUs = kmask==0 || kernelMask_user() == (1<<ncclSymKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;
  *nWarps = 16;
  return ncclSuccess;
}

const char* ncclSymKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymKernelId_Count) {
    return "Unknown";
  }
  return kernelName[kernelId];
}
