#include "symmetric.h"
#include "comm.h"
#include "device.h"

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

constexpr uint32_t kernelMask_nvls = 1<<ncclSymKernelId_AllGather_LLMC |
                                     1<<ncclSymKernelId_AllGather_STMC |
                                     1<<ncclSymKernelId_AllReduce_AGxLLMC_R |
                                     1<<ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymKernelId_ReduceScatter_LDMC;

constexpr uint32_t kernelMask_LL = 1<<ncclSymKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymKernelId_AllGather_LL |
                                   1<<ncclSymKernelId_AllGather_LLMC |
                                   1<<ncclSymKernelId_ReduceScatter_LL;

static uint32_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncAllGather:
    return 1<<ncclSymKernelId_AllGather_LL |
           1<<ncclSymKernelId_AllGather_LLMC |
           1<<ncclSymKernelId_AllGather_ST |
           1<<ncclSymKernelId_AllGather_STMC;
  case ncclFuncAllReduce:
    return 1<<ncclSymKernelId_AllReduce_AGxLLMC_R |
           1<<ncclSymKernelId_AllReduce_AGxLL_R |
           1<<ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC |
           1<<ncclSymKernelId_AllReduce_RSxLD_AGxST;
  case ncclFuncReduceScatter:
    return 1<<ncclSymKernelId_ReduceScatter_LD |
           1<<ncclSymKernelId_ReduceScatter_LDMC |
           1<<ncclSymKernelId_ReduceScatter_LL;
  default:
    return 0;
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

NCCL_PARAM(SymSMs, "SYM_SMS", 0)

static float calcTime_sm100(ncclSymKernelId kernelId, int nRanks, size_t nBytes);
static float calcTime_sm90(ncclSymKernelId kernelId, int nRanks, size_t nBytes);

// Given the kernel and bytes, return the minimum number of blocks to run on such that
// perf is 99% of running at max blocks, and return the estimate runtime for that
// block count.
static void queryModel(struct ncclComm* comm, ncclSymKernelId k, size_t nBytes, float* timeUs, int* nBlocks) {
  size_t totalBytes = nBytes*((kernelMask_coll(ncclFuncAllReduce)>>k & 1) ? 1 : comm->nRanks);
  if (comm->cudaArch >= 1000) {
    *timeUs = calcTime_sm100(k, comm->nRanks, totalBytes);
  } else {
    *timeUs = calcTime_sm90(k, comm->nRanks, totalBytes);
  }

  int nSM = ncclParamSymSMs();
  if (nSM != 0) *nBlocks = nSM;
  else {
    bool isLL = kernelMask_LL>>k & 1;
    *nBlocks = std::min<size_t>(divUp(nBytes, isLL ? 512*8 : 16<<10), comm->nChannels);
    if (kernelMask_nvls>>k & 1) {
      int needs = 0;
      if (comm->cudaArch >= 900) needs = 16;
      if (comm->cudaArch >= 1000) needs = 32;
      *nBlocks = std::min(divUp(needs, comm->nRanks), *nBlocks);
    }
  }

  *nBlocks = std::min(ncclSymMaxBlocks, *nBlocks);
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
  bool hasNvls = comm->nvlsSupport && (ncclNvlsSupported((ncclDevRedOp_t)red, ty) || coll == ncclFuncAllGather);
  size_t nBytes = nElts*ncclTypeSize(ty);

  uint32_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();
  if (!hasNvls) kmask &= ~kernelMask_nvls;

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

namespace {
  struct Sample { size_t x; float y; };
}

static float interp(size_t x, int n, Sample const samps[]) {
  if (n == 0) return 1.e30f;
  if (x <= samps[0].x) return samps[0].y;
  for (int i=1; i < n; i++) {
    if (x < samps[i].x) {
      float w = float(x - samps[i-1].x)/float(samps[i].x - samps[i-1].x);
      return (1.0f - w)*samps[i-1].y + w*samps[i].y;
    }
  }
  return float(x)*samps[n-1].y/float(samps[n-1].x);
}

template<int n>
static float interp(size_t x, Sample const(&samps)[n]) {
  return interp(x, n, samps);
}

float calcTime_sm90(ncclSymKernelId kernelId, int nRanks, size_t nBytes) {
  switch (kernelId) {
  default: return 1.e30;
  case ncclSymKernelId_AllReduce_AGxLL_R: {
    static const Sample samps_r2[] = {
      {256, 4.20},
      {512, 4.21},
      {1024, 4.22},
      {2048, 4.4},
      {4096, 4.62},
      {8192, 4.87},
      {16384, 5.01},
      {32768, 5.19},
      {65536, 5.32},
      {131072, 7.29},
      {262144, 9.36},
      {524288, 15.08},
      {1048576, 25.58},
      {2097152, 46.8},
      {4194304, 87.21},
      {8388608, 168.9},
      {16777216, 331.7},
      {33554432, 658}
    };
    static const Sample samps_r4[] = {
      {256, 4.45},
      {512, 4.46},
      {1024, 4.53},
      {2048, 4.77},
      {4096, 5.38},
      {8192, 5.56},
      {16384, 5.65},
      {32768, 5.92},
      {65536, 6.87},
      {131072, 9.09},
      {262144, 12.94},
      {524288, 22.02},
      {1048576, 39.11},
      {2097152, 73.72},
      {4194304, 140.9},
      {8388608, 278.7},
      {16777216, 544.2},
      {33554432, 1089}
    };
    static const Sample samps_r8[] = {
      {256, 4.40},
      {512, 4.42},
      {1024, 4.95},
      {2048, 5.3},
      {4096, 6.2},
      {8192, 6.08},
      {16384, 6.22},
      {32768, 6.45},
      {65536, 7.93},
      {131072, 12.01},
      {262144, 18.88},
      {524288, 33.35},
      {1048576, 61.63},
      {2097152, 116.7},
      {4194304, 224.7},
      {8388608, 456.2},
      {16777216, 879.3},
      {33554432, 1767.4}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllReduce_AGxLLMC_R: {
    static const Sample samps_r2[] = {
    };
    static const Sample samps_r4[] = {
      {256, 4.35},
      {512, 4.38},
      {1024, 4.55},
      {2048, 4.53},
      {4096, 4.92},
      {8192, 4.91},
      {16384, 5.08},
      {32768, 7.25},
      {65536, 11.27},
      {131072, 18.51},
      {262144, 33.64},
      {524288, 65.15},
      {1048576, 130.9},
      {2097152, 259.4},
      {4194304, 514.8},
      {8388608, 1025.1},
      {16777216, 2045.1},
      {33554432, 4089.9}
    };
    static const Sample samps_r8[] = {
      {256, 4.41},
      {512, 4.47},
      {1024, 4.54},
      {2048, 4.63},
      {4096, 4.86},
      {8192, 4.93},
      {16384, 7.04},
      {32768, 10.74},
      {65536, 18},
      {131072, 32.05},
      {262144, 58.84},
      {524288, 117.9},
      {1048576, 243.7},
      {2097152, 485.7},
      {4194304, 970.1},
      {8388608, 1931.3},
      {16777216, 3868.2},
      {33554432, 7732.3}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllReduce_RSxLD_AGxST: {
    static const Sample samps_r2[] = {
      {256, 7.13},
      {1024, 7.14},
      {4096, 7.32},
      {16384, 7.46},
      {65536, 7.92},
      {262144, 8.57},
      {1048576, 12.61},
      {4194304, 23.49},
      {16777216, 61.81},
      {67108864, 210.7},
      {268435456, 800},
      {1073741824, 3151.5},
      {4294967296, 12544}
    };
    static const Sample samps_r4[] = {
      {256, 7.50},
      {1024, 7.55},
      {4096, 7.59},
      {16384, 8.76},
      {65536, 9.44},
      {262144, 10.13},
      {1048576, 15.87},
      {4194304, 30.92},
      {16777216, 92.08},
      {67108864, 331.7},
      {268435456, 1280.9},
      {1073741824, 5071.3},
      {4294967296, 20234}
    };
    static const Sample samps_r8[] = {
      {256, 7.94},
      {1024, 7.94},
      {4096, 7.98},
      {16384, 10.21},
      {65536, 11.08},
      {262144, 11.6},
      {1048576, 15.96},
      {4194304, 33.49},
      {16777216, 106.9},
      {67108864, 409.9},
      {268435456, 1610.6},
      {1073741824, 6386.6},
      {4294967296, 25450}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC: {
    static const Sample samps_r2[] = {
    };
    static const Sample samps_r4[] = {
      {256, 7.42},
      {1024, 7.48},
      {4096, 7.59},
      {16384, 7.7},
      {65536, 8.14},
      {262144, 8.65},
      {1048576, 12.26},
      {4194304, 25.75},
      {16777216, 75.95},
      {67108864, 277.9},
      {268435456, 1083.5},
      {1073741824, 4295.5},
      {4294967296, 17124}
    };
    static const Sample samps_r8[] = {
      {256, 7.63},
      {1024, 7.66},
      {4096, 7.62},
      {16384, 7.76},
      {65536, 8.3},
      {262144, 8.87},
      {1048576, 12.16},
      {4194304, 24.51},
      {16777216, 70.43},
      {67108864, 254.6},
      {268435456, 991.5},
      {1073741824, 3917.3},
      {4294967296, 15586}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllGather_LL: {
    static const Sample samps_r2[] = {
      {256, 4.18},
      {512, 4.18},
      {1024, 4.18},
      {2048, 4.22},
      {4096, 4.28},
      {8192, 4.45},
      {16384, 4.53},
      {32768, 4.68},
      {65536, 4.92},
      {131072, 5.37},
      {262144, 7.06},
      {524288, 8.94},
      {1048576, 14.55},
      {2097152, 23.44},
      {4194304, 42.25},
      {8388608, 77.76},
      {16777216, 150.5},
      {33554432, 295}
    };
    static const Sample samps_r4[] = {
      {256, 4.29},
      {512, 4.29},
      {1024, 4.29},
      {2048, 4.33},
      {4096, 5.7},
      {8192, 5.89},
      {16384, 5.96},
      {32768, 5.07},
      {65536, 5.25},
      {131072, 5.44},
      {262144, 6.09},
      {524288, 8.73},
      {1048576, 12.49},
      {2097152, 20.93},
      {4194304, 36.55},
      {8388608, 66.51},
      {16777216, 131.5},
      {33554432, 259.4}
    };
    static const Sample samps_r8[] = {
      {256, 4.38},
      {512, 4.39},
      {1024, 4.42},
      {2048, 4.44},
      {4096, 4.5},
      {8192, 4.53},
      {16384, 6.08},
      {32768, 6.44},
      {65536, 6.58},
      {131072, 6.81},
      {262144, 7.2},
      {524288, 8.68},
      {1048576, 13.01},
      {2097152, 21.03},
      {4194304, 36.43},
      {8388608, 64.52},
      {16777216, 120.1},
      {33554432, 237.6}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllGather_LLMC: {
    static const Sample samps_r2[] = {
    };
    static const Sample samps_r4[] = {
      {256, 4.23},
      {512, 4.23},
      {1024, 4.23},
      {2048, 4.33},
      {4096, 4.44},
      {8192, 4.41},
      {16384, 4.6},
      {32768, 4.71},
      {65536, 4.94},
      {131072, 6.84},
      {262144, 10.52},
      {524288, 17.84},
      {1048576, 32.18},
      {2097152, 60.4},
      {4194304, 115.3},
      {8388608, 224.7},
      {16777216, 443.8},
      {33554432, 881.8}
    };
    static const Sample samps_r8[] = {
      {256, 4.48},
      {512, 4.49},
      {1024, 4.51},
      {2048, 4.54},
      {4096, 4.6},
      {8192, 4.62},
      {16384, 4.64},
      {32768, 5.53},
      {65536, 5.75},
      {131072, 8.32},
      {262144, 13.33},
      {524288, 23.37},
      {1048576, 42.84},
      {2097152, 81.64},
      {4194304, 158.8},
      {8388608, 311.3},
      {16777216, 618.2},
      {33554432, 1235.6}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllGather_ST: {
    static const Sample samps_r2[] = {
      {256, 6.22},
      {1024, 6.23},
      {4096, 6.23},
      {16384, 6.59},
      {65536, 6.96},
      {262144, 7.29},
      {1048576, 10},
      {4194304, 14.83},
      {16777216, 32.57},
      {67108864, 102.5},
      {268435456, 380.1},
      {1073741824, 1492.3},
      {4294967296, 5924.3}
    };
    static const Sample samps_r4[] = {
      {256, 6.83},
      {1024, 6.85},
      {4096, 7.61},
      {16384, 6.68},
      {65536, 7.84},
      {262144, 8.08},
      {1048576, 9.4},
      {4194304, 17.89},
      {16777216, 45.87},
      {67108864, 152.9},
      {268435456, 581.2},
      {1073741824, 2305.2},
      {4294967296, 9210.4}
    };
    static const Sample samps_r8[] = {
      {256, 6.66},
      {1024, 6.72},
      {4096, 6.73},
      {16384, 6.77},
      {65536, 7.83},
      {262144, 9.56},
      {1048576, 10.43},
      {4194304, 20.32},
      {16777216, 54.51},
      {67108864, 186},
      {268435456, 702.7},
      {1073741824, 2791},
      {4294967296, 11177}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_AllGather_STMC: {
    static const Sample samps_r2[] = {
    };
    static const Sample samps_r4[] = {
      {256, 6.55},
      {1024, 6.57},
      {4096, 6.66},
      {16384, 6.74},
      {65536, 6.8},
      {262144, 7.67},
      {1048576, 9.49},
      {4194304, 18.25},
      {16777216, 52.16},
      {67108864, 187.9},
      {268435456, 726.8},
      {1073741824, 2884.3},
      {4294967296, 11503}
    };
    static const Sample samps_r8[] = {
      {256, 6.88},
      {1024, 6.92},
      {4096, 6.94},
      {16384, 8.18},
      {65536, 7.17},
      {262144, 7.68},
      {1048576, 9.99},
      {4194304, 18.84},
      {16777216, 52.46},
      {67108864, 187.4},
      {268435456, 724.9},
      {1073741824, 2877.6},
      {4294967296, 11479}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_ReduceScatter_LL: {
    static const Sample samps_r2[] = {
      {256, 4.36},
      {512, 4.38},
      {1024, 4.39},
      {2048, 4.59},
      {4096, 4.69},
      {8192, 4.99},
      {16384, 5.14},
      {32768, 5.26},
      {65536, 5.51},
      {131072, 5.63},
      {262144, 7.77},
      {524288, 10.48},
      {1048576, 17.71},
      {2097152, 29.38},
      {4194304, 53.87},
      {8388608, 101.3},
      {16777216, 196.9},
      {33554432, 387.5}
    };
    static const Sample samps_r4[] = {
      {256, 4.88},
      {512, 4.89},
      {1024, 4.9},
      {2048, 4.97},
      {4096, 5.22},
      {8192, 5.32},
      {16384, 5.82},
      {32768, 6.07},
      {65536, 6.27},
      {131072, 6.39},
      {262144, 7.01},
      {524288, 10.74},
      {1048576, 15.11},
      {2097152, 25.31},
      {4194304, 42.54},
      {8388608, 78.82},
      {16777216, 154.2},
      {33554432, 304}
    };
    static const Sample samps_r8[] = {
      {256, 4.47},
      {512, 4.49},
      {1024, 4.5},
      {2048, 5.25},
      {4096, 5.32},
      {8192, 5.44},
      {16384, 5.43},
      {32768, 6.45},
      {65536, 6.72},
      {131072, 6.9},
      {262144, 7.35},
      {524288, 8.64},
      {1048576, 15.46},
      {2097152, 21.58},
      {4194304, 38.19},
      {8388608, 67.43},
      {16777216, 124.9},
      {33554432, 240}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_ReduceScatter_LD: {
    static const Sample samps_r2[] = {
      {256, 5.2},
      {1024, 5.89},
      {4096, 5.96},
      {16384, 6.17},
      {65536, 6.4},
      {262144, 6.63},
      {1048576, 10.1},
      {4194304, 16.31},
      {16777216, 35.95},
      {67108864, 112.5},
      {268435456, 414.5},
      {1073741824, 1620.8},
      {4294967296, 6447.1}
    };
    static const Sample samps_r4[] = {
      {256, 6.1},
      {1024, 6.12},
      {4096, 6.13},
      {16384, 7.56},
      {65536, 9.91},
      {262144, 10.23},
      {1048576, 11.15},
      {4194304, 20.08},
      {16777216, 49.86},
      {67108864, 165.5},
      {268435456, 618.6},
      {1073741824, 2428.7},
      {4294967296, 9673}
    };
    static const Sample samps_r8[] = {
      {256, 6.76},
      {1024, 6.39},
      {4096, 6.43},
      {16384, 6.58},
      {65536, 10.84},
      {262144, 12.71},
      {1048576, 13.78},
      {4194304, 22.84},
      {16777216, 58.08},
      {67108864, 193.9},
      {268435456, 726.8},
      {1073741824, 2858.8},
      {4294967296, 11384}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  case ncclSymKernelId_ReduceScatter_LDMC: {
    static const Sample samps_r2[] = {
    };
    static const Sample samps_r4[] = {
      {256, 5.2},
      {1024, 5.86},
      {4096, 5.9},
      {16384, 6.04},
      {65536, 6.42},
      {262144, 7.65},
      {1048576, 9.68},
      {4194304, 18.82},
      {16777216, 55.06},
      {67108864, 197.8},
      {268435456, 767.4},
      {1073741824, 3033.5},
      {4294967296, 12093}
    };
    static const Sample samps_r8[] = {
      {256, 6.02},
      {1024, 6.08},
      {4096, 6.12},
      {16384, 6.26},
      {65536, 6.42},
      {262144, 7.36},
      {1048576, 10},
      {4194304, 18.85},
      {16777216, 54.25},
      {67108864, 193.7},
      {268435456, 747.3},
      {1073741824, 2958},
      {4294967296, 11781}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)}
    });
  }
  }
}

float calcTime_sm100(ncclSymKernelId kernelId, int nRanks, size_t nBytes) {
  switch (kernelId) {
  default: return 1.e30;
  case ncclSymKernelId_AllReduce_AGxLL_R: {
    static const Sample samps_r2[] = {
      {256, 8.41},
      {512, 8.47},
      {1024, 8.59},
      {2048, 9.35},
      {4096, 9.32},
      {8192, 9.69},
      {16384, 9.59},
      {32768, 10.01},
      {65536, 10.05},
      {131072, 9.9},
      {262144, 13.37},
      {524288, 19.35},
      {1048576, 31.15},
      {2097152, 54.58},
      {4194304, 101.5},
      {8388608, 194.6},
      {16777216, 380},
      {33554432, 750.5}
    };
    static const Sample samps_r4[] = {
      {256, 8.9},
      {512, 8.99},
      {1024, 9.31},
      {2048, 9.64},
      {4096, 10.3},
      {8192, 10.41},
      {16384, 10.54},
      {32768, 10.62},
      {65536, 10.69},
      {131072, 11.22},
      {262144, 15.1},
      {524288, 23.07},
      {1048576, 38.11},
      {2097152, 68.35},
      {4194304, 128.8},
      {8388608, 249.8},
      {16777216, 491.2},
      {33554432, 972.9}
    };
    static const Sample samps_r8[] = {
      {256, 9.44},
      {512, 9.45},
      {1024, 9.95},
      {2048, 10.15},
      {4096, 11.16},
      {8192, 11.46},
      {16384, 11.46},
      {32768, 11.71},
      {65536, 11.81},
      {131072, 12.96},
      {262144, 18.2},
      {524288, 28.86},
      {1048576, 49.92},
      {2097152, 90.3},
      {4194304, 172.9},
      {8388608, 342},
      {16777216, 675.9},
      {33554432, 1341.7}
    };
    static const Sample samps_r12[] = {
      {256, 10.21},
      {512, 10.23},
      {1024, 10.54},
      {2048, 11.26},
      {4096, 12.71},
      {8192, 13.02},
      {16384, 13.12},
      {32768, 13.17},
      {65536, 13.76},
      {131072, 16.76},
      {262144, 25.92},
      {524288, 43.33},
      {1048576, 76.59},
      {2097152, 142.6},
      {4194304, 269.8},
      {8388608, 517.8},
      {16777216, 1018.8},
      {33554432, 2008}
    };
    static const Sample samps_r16[] = {
      {256, 10.37},
      {512, 10.38},
      {1024, 10.73},
      {2048, 11.54},
      {4096, 13.41},
      {8192, 13.72},
      {16384, 13.78},
      {32768, 13.98},
      {65536, 13.97},
      {131072, 16.65},
      {262144, 25.75},
      {524288, 43.49},
      {1048576, 76.7},
      {2097152, 143},
      {4194304, 269.6},
      {8388608, 519.8},
      {16777216, 1019.8},
      {33554432, 2003.5}
    };
    static const Sample samps_r24[] = {
      {256, 11.02},
      {512, 11.05},
      {1024, 11.65},
      {2048, 12.99},
      {4096, 15.89},
      {8192, 16.08},
      {16384, 16.2},
      {32768, 16.45},
      {65536, 18.25},
      {131072, 24.05},
      {262144, 39.5},
      {524288, 69.75},
      {1048576, 128.6},
      {2097152, 246},
      {4194304, 483},
      {8388608, 959.6},
      {16777216, 1905.9},
      {33554432, 3781.1}
    };
    static const Sample samps_r36[] = {
      {256, 12.81},
      {512, 13.05},
      {1024, 13.81},
      {2048, 15.7},
      {4096, 20.17},
      {8192, 20.33},
      {16384, 20.34},
      {32768, 20.77},
      {65536, 25.85},
      {131072, 36.52},
      {262144, 63.8},
      {524288, 115.7},
      {1048576, 214.6},
      {2097152, 416},
      {4194304, 818.5},
      {8388608, 1610.5},
      {16777216, 3203.6},
      {33554432, 6364.2}
    };
    static const Sample samps_r48[] = {
      {256, 13.08},
      {512, 13.12},
      {1024, 14.74},
      {2048, 17.45},
      {4096, 22.85},
      {8192, 22.99},
      {16384, 23.13},
      {32768, 23.38},
      {65536, 27.62},
      {131072, 40.2},
      {262144, 69.87},
      {524288, 127.84},
      {1048576, 241.9},
      {2097152, 477.07},
      {4194304, 942.7},
      {8388608, 1878.98},
      {16777216, 3764.12},
      {33554432, 7511.31}
    };
    static const Sample samps_r64[] = {
      {256, 14.57},
      {512, 15.14},
      {1024, 16.75},
      {2048, 20.74},
      {4096, 27.76},
      {8192, 37.22},
      {16384, 27.93},
      {32768, 28.25},
      {65536, 31.35},
      {131072, 43.61},
      {262144, 78.57},
      {524288, 142.12},
      {1048576, 275.44},
      {2097152, 544.14},
      {4194304, 1062.36},
      {8388608, 2077.7},
      {16777216, 4110.1},
      {33554432, 8205.42}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllReduce_AGxLLMC_R: {
    static const Sample samps_r2[] = {
      {256, 13.25},
      {512, 13.25},
      {1024, 13.33},
      {2048, 13.35},
      {4096, 13.77},
      {8192, 14.42},
      {16384, 14.52},
      {32768, 14.68},
      {65536, 14.8},
      {131072, 15},
      {262144, 15.06},
      {524288, 15.86},
      {1048576, 18.72},
      {2097152, 25.54},
      {4194304, 49.47},
      {8388608, 54.29},
      {16777216, 68.78},
      {33554432, 97.51}
    };
    static const Sample samps_r4[] = {
      {256, 9.45},
      {512, 9.45},
      {1024, 9.47},
      {2048, 9.67},
      {4096, 9.79},
      {8192, 10.1},
      {16384, 10.17},
      {32768, 10.2},
      {65536, 13.27},
      {131072, 19.52},
      {262144, 31.92},
      {524288, 56.37},
      {1048576, 105.2},
      {2097152, 203.2},
      {4194304, 399.9},
      {8388608, 790.1},
      {16777216, 1571.4},
      {33554432, 3115}
    };
    static const Sample samps_r8[] = {
      {256, 9.4},
      {512, 9.48},
      {1024, 9.88},
      {2048, 9.72},
      {4096, 10.15},
      {8192, 10.42},
      {16384, 10.36},
      {32768, 13.51},
      {65536, 19.47},
      {131072, 31.17},
      {262144, 55.32},
      {524288, 102.4},
      {1048576, 198.4},
      {2097152, 391.1},
      {4194304, 776.8},
      {8388608, 1544.5},
      {16777216, 3079.3},
      {33554432, 6125.3}
    };
    static const Sample samps_r12[] = {
      {256, 10.24},
      {512, 10.26},
      {1024, 10.47},
      {2048, 10.71},
      {4096, 11.04},
      {8192, 11.26},
      {16384, 14.96},
      {32768, 19.11},
      {65536, 30.79},
      {131072, 50.51},
      {262144, 93.8},
      {524288, 175.3},
      {1048576, 344.8},
      {2097152, 681.3},
      {4194304, 1356.9},
      {8388608, 2698.4},
      {16777216, 5389.6},
      {33554432, 10771}
    };
    static const Sample samps_r16[] = {
      {256, 9.88},
      {512, 9.97},
      {1024, 10.22},
      {2048, 10.39},
      {4096, 10.88},
      {8192, 11.09},
      {16384, 14.99},
      {32768, 22.72},
      {65536, 38.12},
      {131072, 68.17},
      {262144, 127.5},
      {524288, 249.5},
      {1048576, 489.6},
      {2097152, 974.6},
      {4194304, 1944.1},
      {8388608, 3874.3},
      {16777216, 7731.1},
      {33554432, 15417}
    };
    static const Sample samps_r24[] = {
      {256, 10.15},
      {512, 10.93},
      {1024, 11.03},
      {2048, 11.34},
      {4096, 11.78},
      {8192, 12.05},
      {16384, 17.02},
      {32768, 26.62},
      {65536, 45.59},
      {131072, 82.83},
      {262144, 154.9},
      {524288, 305.7},
      {1048576, 602.1},
      {2097152, 1209.2},
      {4194304, 2416.3},
      {8388608, 4803.5},
      {16777216, 9601},
      {33554432, 19195}
    };
    static const Sample samps_r36[] = {
      {256, 12.38},
      {512, 12.4},
      {1024, 12.61},
      {2048, 12.94},
      {4096, 13.57},
      {8192, 19.85},
      {16384, 32.66},
      {32768, 58.29},
      {65536, 109.6},
      {131072, 211.4},
      {262144, 413.6},
      {524288, 817.9},
      {1048576, 1630.5},
      {2097152, 3264},
      {4194304, 6531.8},
      {8388608, 13052},
      {16777216, 26077},
      {33554432, 52143}
    };
    static const Sample samps_r48[] = {
      {256, 12.25},
      {512, 12.28},
      {1024, 12.84},
      {2048, 13.04},
      {4096, 13.79},
      {8192, 20.44},
      {16384, 33.93},
      {32768, 61.12},
      {65536, 115.14},
      {131072, 222.75},
      {262144, 436.96},
      {524288, 864.39},
      {1048576, 1723.16},
      {2097152, 3458.33},
      {4194304, 6929.8},
      {8388608, 13844.2},
      {16777216, 27665.6},
      {33554432, 55325.8}
    };
    static const Sample samps_r64[] = {
      {256, 13.43},
      {512, 13.82},
      {1024, 14.12},
      {2048, 14.31},
      {4096, 15.1},
      {8192, 23.3},
      {16384, 39.99},
      {32768, 73.03},
      {65536, 139.14},
      {131072, 270.59},
      {262144, 532.98},
      {524288, 1057.48},
      {1048576, 2109.51},
      {2097152, 4227.08},
      {4194304, 8465.07},
      {8388608, 16909.9},
      {16777216, 33796.2},
      {33554432, 67589}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllReduce_RSxLD_AGxST: {
    static const Sample samps_r2[] = {
      {256, 13.44},
      {1024, 13.53},
      {4096, 13.64},
      {16384, 13.91},
      {65536, 14.09},
      {262144, 14.52},
      {1048576, 16.57},
      {4194304, 22.56},
      {16777216, 45.77},
      {67108864, 132.8},
      {268435456, 472.6},
      {1073741824, 1823.6},
      {4294967296, 7215.3}
    };
    static const Sample samps_r4[] = {
      {256, 14.11},
      {1024, 14.15},
      {4096, 14.16},
      {16384, 16.32},
      {65536, 16.68},
      {262144, 18.18},
      {1048576, 19.94},
      {4194304, 24.91},
      {16777216, 54.33},
      {67108864, 172.7},
      {268435456, 640.3},
      {1073741824, 2508.4},
      {4294967296, 9934.6}
    };
    static const Sample samps_r8[] = {
      {256, 14.6},
      {1024, 14.6},
      {4096, 14.65},
      {16384, 18.85},
      {65536, 19.28},
      {262144, 21.52},
      {1048576, 22.41},
      {4194304, 29.84},
      {16777216, 63.04},
      {67108864, 214.8},
      {268435456, 813.1},
      {1073741824, 3158.1},
      {4294967296, 12504}
    };
    static const Sample samps_r12[] = {
      {256, 17.09},
      {1024, 17.11},
      {4096, 17.23},
      {16384, 17.38},
      {65536, 18.02},
      {262144, 19.93},
      {1048576, 29.99},
      {4194304, 48.4},
      {16777216, 86.59},
      {67108864, 251.5},
      {268435456, 920.7},
      {1073741824, 3569.3},
      {4294967296, 14084}
    };
    static const Sample samps_r16[] = {
      {256, 17.21},
      {1024, 17.23},
      {4096, 17.25},
      {16384, 17.49},
      {65536, 18.2},
      {262144, 20.24},
      {1048576, 28.02},
      {4194304, 39.39},
      {16777216, 66.54},
      {67108864, 229.9},
      {268435456, 918.7},
      {1073741824, 3654.4},
      {4294967296, 14506}
    };
    static const Sample samps_r24[] = {
      {256, 19.65},
      {1024, 19.7},
      {4096, 19.86},
      {16384, 20},
      {65536, 20.71},
      {262144, 22.93},
      {1048576, 24.19},
      {4194304, 45.69},
      {16777216, 94.83},
      {67108864, 263},
      {268435456, 1004.6},
      {1073741824, 4004},
      {4294967296, 15908}
    };
    static const Sample samps_r36[] = {
      {256, 24.39},
      {1024, 24.39},
      {4096, 24.62},
      {16384, 24.71},
      {65536, 25.42},
      {262144, 27.81},
      {1048576, 29.41},
      {4194304, 57.04},
      {16777216, 127.7},
      {67108864, 337.3},
      {268435456, 1190.4},
      {1073741824, 4515},
      {4294967296, 17396}
    };
    static const Sample samps_r48[] = {
      {256, 30.95},
      {1024, 31.42},
      {4096, 31.69},
      {16384, 31.86},
      {65536, 32.59},
      {262144, 35.62},
      {1048576, 37.6},
      {4194304, 79.51},
      {16777216, 172.29},
      {67108864, 374.87},
      {268435456, 1211.24},
      {1073741824, 4371.21},
      {4294967296, 16784.8}
    };
    static const Sample samps_r64[] = {
      {256, 37.38},
      {1024, 37.47},
      {4096, 37.66},
      {16384, 37.88},
      {65536, 38.46},
      {262144, 41.51},
      {1048576, 44.19},
      {4194304, 73.87},
      {16777216, 139.19},
      {67108864, 245.9},
      {268435456, 975.23},
      {1073741824, 3940.8},
      {4294967296, 18089.7}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC: {
    static const Sample samps_r2[] = {
      {256, 13.56},
      {1024, 13.71},
      {4096, 14.03},
      {16384, 14.87},
      {65536, 14.96},
      {262144, 15.38},
      {1048576, 19.66},
      {4194304, 49.91},
      {16777216, 70.36},
      {67108864, 165.5},
      {268435456, 551},
      {1073741824, 2041.3},
      {4294967296, 7820.9}
    };
    static const Sample samps_r4[] = {
      {256, 13.81},
      {1024, 13.82},
      {4096, 13.85},
      {16384, 14.29},
      {65536, 14.76},
      {262144, 15.72},
      {1048576, 17.53},
      {4194304, 24.7},
      {16777216, 52.17},
      {67108864, 162.4},
      {268435456, 603.7},
      {1073741824, 2363.2},
      {4294967296, 9374.9}
    };
    static const Sample samps_r8[] = {
      {256, 14.52},
      {1024, 14.54},
      {4096, 14.79},
      {16384, 14.87},
      {65536, 15.32},
      {262144, 15.5},
      {1048576, 16.94},
      {4194304, 23.83},
      {16777216, 50.47},
      {67108864, 157},
      {268435456, 578.4},
      {1073741824, 2259.8},
      {4294967296, 8976.5}
    };
    static const Sample samps_r12[] = {
      {256, 14.71},
      {1024, 14.72},
      {4096, 15.1},
      {16384, 15.13},
      {65536, 15.61},
      {262144, 15.87},
      {1048576, 17.41},
      {4194304, 24.55},
      {16777216, 52.48},
      {67108864, 156.7},
      {268435456, 571.6},
      {1073741824, 2227.2},
      {4294967296, 8867.7}
    };
    static const Sample samps_r16[] = {
      {256, 14.82},
      {1024, 14.88},
      {4096, 15.25},
      {16384, 15.21},
      {65536, 15.6},
      {262144, 15.97},
      {1048576, 17.45},
      {4194304, 24.75},
      {16777216, 50.85},
      {67108864, 154.6},
      {268435456, 565.6},
      {1073741824, 2202.7},
      {4294967296, 8753.2}
    };
    static const Sample samps_r24[] = {
      {256, 15.15},
      {1024, 15.16},
      {4096, 15.39},
      {16384, 15.41},
      {65536, 15.85},
      {262144, 16.3},
      {1048576, 18.04},
      {4194304, 26.05},
      {16777216, 51.08},
      {67108864, 153.6},
      {268435456, 560.8},
      {1073741824, 2184.2},
      {4294967296, 8674.2}
    };
    static const Sample samps_r36[] = {
      {256, 15.36},
      {1024, 15.37},
      {4096, 15.39},
      {16384, 15.69},
      {65536, 15.88},
      {262144, 16.14},
      {1048576, 18.12},
      {4194304, 26.27},
      {16777216, 53.47},
      {67108864, 155.8},
      {268435456, 565.7},
      {1073741824, 2203.1},
      {4294967296, 8755.3}
    };
    static const Sample samps_r48[] = {
      {256, 17.8},
      {1024, 17.82},
      {4096, 17.87},
      {16384, 18.05},
      {65536, 18.51},
      {262144, 18.54},
      {1048576, 19.72},
      {4194304, 29.34},
      {16777216, 54.78},
      {67108864, 157.36},
      {268435456, 564.69},
      {1073741824, 2198.35},
      {4294967296, 8776.86}
    };
    static const Sample samps_r64[] = {
      {256, 17.9},
      {1024, 17.92},
      {4096, 17.99},
      {16384, 18.21},
      {65536, 18.69},
      {262144, 18.74},
      {1048576, 19.94},
      {4194304, 27},
      {16777216, 51.9},
      {67108864, 153.42},
      {268435456, 554.66},
      {1073741824, 2159.3},
      {4294967296, 8586.91}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllGather_LL: {
    static const Sample samps_r2[] = {
      {256, 8.22},
      {512, 8.29},
      {1024, 8.38},
      {2048, 8.79},
      {4096, 8.5},
      {8192, 9.06},
      {16384, 9.22},
      {32768, 9.26},
      {65536, 9.49},
      {131072, 9.45},
      {262144, 9.79},
      {524288, 12.61},
      {1048576, 18.15},
      {2097152, 28.78},
      {4194304, 49.97},
      {8388608, 92.12},
      {16777216, 176.2},
      {33554432, 343.7}
    };
    static const Sample samps_r4[] = {
      {256, 8.38},
      {512, 8.44},
      {1024, 8.52},
      {2048, 8.54},
      {4096, 8.85},
      {8192, 9.09},
      {16384, 9.86},
      {32768, 10.15},
      {65536, 10.22},
      {131072, 10.31},
      {262144, 10.4},
      {524288, 10.96},
      {1048576, 14.63},
      {2097152, 21.56},
      {4194304, 35.72},
      {8388608, 63.54},
      {16777216, 118.5},
      {33554432, 228.3}
    };
    static const Sample samps_r8[] = {
      {256, 8.83},
      {512, 8.84},
      {1024, 8.85},
      {2048, 8.97},
      {4096, 9.1},
      {8192, 9.39},
      {16384, 10.11},
      {32768, 11.47},
      {65536, 11.8},
      {131072, 12.07},
      {262144, 12.29},
      {524288, 12.08},
      {1048576, 13.15},
      {2097152, 18.93},
      {4194304, 30.24},
      {8388608, 52.82},
      {16777216, 97.15},
      {33554432, 185.6}
    };
    static const Sample samps_r12[] = {
      {192, 9.02},
      {384, 9.07},
      {960, 9.17},
      {1920, 9.28},
      {4032, 9.4},
      {8064, 9.78},
      {16320, 10.64},
      {32640, 12.08},
      {65472, 12.52},
      {130944, 13.83},
      {262080, 13.93},
      {524160, 14.19},
      {1048512, 16.03},
      {2097024, 21.58},
      {4194240, 33.51},
      {8388480, 56.75},
      {16777152, 99.2},
      {33554304, 184.4}
    };
    static const Sample samps_r16[] = {
      {256, 9.46},
      {512, 9.48},
      {1024, 9.54},
      {2048, 9.58},
      {4096, 9.63},
      {8192, 9.85},
      {16384, 10.59},
      {32768, 12.21},
      {65536, 15.02},
      {131072, 15.2},
      {262144, 15.29},
      {524288, 15.48},
      {1048576, 15.39},
      {2097152, 17.94},
      {4194304, 28.17},
      {8388608, 48.25},
      {16777216, 85.87},
      {33554432, 160.5}
    };
    static const Sample samps_r24[] = {
      {384, 9.52},
      {768, 9.59},
      {1920, 9.61},
      {3840, 9.64},
      {8064, 10.07},
      {16128, 10.66},
      {32640, 12.75},
      {65280, 15.68},
      {130944, 19.53},
      {261888, 18.45},
      {524160, 18.69},
      {1048320, 18.9},
      {2097024, 23.12},
      {4194048, 33.35},
      {8388480, 56.37},
      {16776960, 98.96},
      {33554304, 178.7}
    };
    static const Sample samps_r36[] = {
      {576, 9.86},
      {1728, 9.87},
      {4032, 10.07},
      {8064, 10.26},
      {16128, 10.98},
      {32256, 12.58},
      {65088, 16.08},
      {130752, 22.19},
      {262080, 22.88},
      {524160, 23.2},
      {1048320, 23.27},
      {2096640, 28.99},
      {4193856, 37.14},
      {8388288, 65.66},
      {16777152, 115},
      {33554304, 207.4}
    };
    static const Sample samps_r48[] = {
      {768, 10.3},
      {1536, 10.31},
      {3840, 10.46},
      {7680, 10.66},
      {16128, 11.29},
      {32256, 13.06},
      {65280, 16.81},
      {130560, 22.39},
      {261888, 23.35},
      {523776, 27.45},
      {1048320, 27.71},
      {2096640, 29.23},
      {4194048, 40.41},
      {8388096, 58.15},
      {16776960, 104.97},
      {33553920, 190.98}
    };
    static const Sample samps_r64[] = {
      {1024, 10.51},
      {2048, 10.53},
      {4096, 10.75},
      {8192, 10.79},
      {16384, 11.51},
      {32768, 13.09},
      {65536, 16.36},
      {131072, 22.19},
      {262144, 33.22},
      {524288, 33.82},
      {1048576, 34.14},
      {2097152, 34.14},
      {4194304, 35.82},
      {8388608, 49.52},
      {16777216, 88.25},
      {33554432, 165.7}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllGather_LLMC: {
    static const Sample samps_r2[] = {
      {256, 11.71},
      {512, 11.74},
      {1024, 11.79},
      {2048, 11.96},
      {4096, 12.15},
      {8192, 12.99},
      {16384, 14.61},
      {32768, 14.95},
      {65536, 14.96},
      {131072, 15.02},
      {262144, 15.07},
      {524288, 15.57},
      {1048576, 18.37},
      {2097152, 23.9},
      {4194304, 36.83},
      {8388608, 41.85},
      {16777216, 48.23},
      {33554432, 64.21}
    };
    static const Sample samps_r4[] = {
      {256, 8.62},
      {512, 8.65},
      {1024, 8.69},
      {2048, 8.84},
      {4096, 8.97},
      {8192, 9.11},
      {16384, 9.45},
      {32768, 9.84},
      {65536, 9.85},
      {131072, 9.92},
      {262144, 13.05},
      {524288, 19.05},
      {1048576, 30.5},
      {2097152, 52.76},
      {4194304, 97.58},
      {8388608, 187},
      {16777216, 365.6},
      {33554432, 722.7}
    };
    static const Sample samps_r8[] = {
      {256, 9.08},
      {512, 9.11},
      {1024, 9.23},
      {2048, 9.23},
      {4096, 9.35},
      {8192, 9.3},
      {16384, 9.76},
      {32768, 10.78},
      {65536, 10.78},
      {131072, 10.98},
      {262144, 14.97},
      {524288, 22.59},
      {1048576, 37.16},
      {2097152, 66.02},
      {4194304, 123.5},
      {8388608, 238},
      {16777216, 466.2},
      {33554432, 921.6}
    };
    static const Sample samps_r12[] = {
      {192, 9.21},
      {384, 9.24},
      {960, 9.31},
      {1920, 9.42},
      {4032, 9.46},
      {8064, 9.62},
      {16320, 9.73},
      {32640, 10.83},
      {65472, 11.2},
      {130944, 12.02},
      {262080, 16.48},
      {524160, 24.84},
      {1048512, 42.01},
      {2097024, 73.3},
      {4194240, 136},
      {8388480, 261.2},
      {16777152, 509.8},
      {33554304, 999.2}
    };
    static const Sample samps_r16[] = {
      {256, 9.16},
      {512, 9.17},
      {1024, 9.18},
      {2048, 9.22},
      {4096, 9.26},
      {8192, 9.52},
      {16384, 9.65},
      {32768, 10.57},
      {65536, 12.32},
      {131072, 12.62},
      {262144, 17.84},
      {524288, 28.56},
      {1048576, 49.23},
      {2097152, 90.12},
      {4194304, 172.1},
      {8388608, 335.5},
      {16777216, 660.9},
      {33554432, 1313}
    };
    static const Sample samps_r24[] = {
      {384, 9.25},
      {768, 9.33},
      {1920, 9.43},
      {3840, 9.56},
      {8064, 9.86},
      {16128, 9.89},
      {32640, 10.83},
      {65280, 12.66},
      {130944, 13.26},
      {261888, 18.17},
      {524160, 27.1},
      {1048320, 46.27},
      {2097024, 82.19},
      {4194048, 156.1},
      {8388480, 302.4},
      {16776960, 596.8},
      {33554304, 1183.2}
    };
    static const Sample samps_r36[] = {
      {576, 9.37},
      {1728, 9.4},
      {4032, 9.67},
      {8064, 9.86},
      {16128, 10.05},
      {32256, 10.91},
      {65088, 12.76},
      {130752, 15.65},
      {262080, 23.85},
      {524160, 39.91},
      {1048320, 71.87},
      {2096640, 134.1},
      {4193856, 264.4},
      {8388288, 519.4},
      {16777152, 1030.9},
      {33554304, 2004.6}
    };
    static const Sample samps_r48[] = {
      {768, 9.82},
      {1536, 9.89},
      {3840, 9.97},
      {7680, 10.26},
      {16128, 10.33},
      {32256, 11.3},
      {65280, 12.92},
      {130560, 15.35},
      {261888, 23.69},
      {523776, 36.62},
      {1048320, 67.73},
      {2096640, 121.89},
      {4194048, 243.13},
      {8388096, 463.64},
      {16776960, 945.93},
      {33553920, 1830.64}
    };
    static const Sample samps_r64[] = {
      {1024, 9.87},
      {2048, 9.9},
      {4096, 10.08},
      {8192, 10.39},
      {16384, 10.44},
      {32768, 11.07},
      {65536, 12.4},
      {131072, 15.02},
      {262144, 20.53},
      {524288, 34.02},
      {1048576, 61.17},
      {2097152, 115.09},
      {4194304, 222.66},
      {8388608, 438.09},
      {16777216, 868.96},
      {33554432, 1731.25}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllGather_ST: {
    static const Sample samps_r2[] = {
      {256, 11.3},
      {1024, 11.47},
      {4096, 11.57},
      {16384, 11.92},
      {65536, 12.49},
      {262144, 13.13},
      {1048576, 14.06},
      {4194304, 16.74},
      {16777216, 29.8},
      {67108864, 77.62},
      {268435456, 267},
      {1073741824, 1023.2},
      {4294967296, 4041.7}
    };
    static const Sample samps_r4[] = {
      {256, 11.71},
      {1024, 11.72},
      {4096, 11.73},
      {16384, 11.73},
      {65536, 13.07},
      {262144, 13.61},
      {1048576, 14.04},
      {4194304, 17.47},
      {16777216, 33.6},
      {67108864, 94.75},
      {268435456, 342.1},
      {1073741824, 1332.8},
      {4294967296, 5298.1}
    };
    static const Sample samps_r8[] = {
      {256, 12.1},
      {1024, 12.11},
      {4096, 12.13},
      {16384, 12.15},
      {65536, 13.44},
      {262144, 15.5},
      {1048576, 16.01},
      {4194304, 18.65},
      {16777216, 36.97},
      {67108864, 108},
      {268435456, 400.2},
      {1073741824, 1562.2},
      {4294967296, 6290.7}
    };
    static const Sample samps_r12[] = {
      {192, 12.61},
      {960, 12.67},
      {4032, 12.92},
      {16320, 15.17},
      {65472, 16.41},
      {262080, 18.25},
      {1048512, 19.46},
      {4194240, 25.63},
      {16777152, 45.3},
      {67108800, 130.5},
      {268435392, 478.1},
      {1073741760, 1878.3},
      {4294967232, 7494}
    };
    static const Sample samps_r16[] = {
      {256, 12.71},
      {1024, 12.89},
      {4096, 12.92},
      {16384, 13.11},
      {65536, 13.6},
      {262144, 18.86},
      {1048576, 19.59},
      {4194304, 20.72},
      {16777216, 39.86},
      {67108864, 114.9},
      {268435456, 427.8},
      {1073741824, 1661.6},
      {4294967296, 6711.1}
    };
    static const Sample samps_r24[] = {
      {768, 13.24},
      {3840, 13.3},
      {16128, 14.65},
      {65280, 15.92},
      {261888, 21.99},
      {1048320, 24.37},
      {4194048, 29.94},
      {16776960, 48.29},
      {67108608, 133.7},
      {268435200, 487.1},
      {1073741568, 1897.3},
      {4294967040, 7609.5}
    };
    static const Sample samps_r36[] = {
      {576, 13.57},
      {4032, 13.66},
      {16128, 14.32},
      {65088, 19.32},
      {262080, 24.76},
      {1048320, 30.28},
      {4193856, 32.67},
      {16777152, 62.75},
      {67108608, 133.5},
      {268435008, 508.9},
      {1073741760, 1955.1},
      {4294967040, 7479.7}
    };
    static const Sample samps_r48[] = {
      {768, 15.2},
      {3840, 15.22},
      {16128, 15.98},
      {65280, 20.46},
      {261888, 26.63},
      {1048320, 32.89},
      {4194048, 39.19},
      {16776960, 68.36},
      {67108608, 149.05},
      {268435200, 511.99},
      {1073741568, 1991.23},
      {4294967040, 7975.08}
    };
    static const Sample samps_r64[] = {
      {1024, 16.03},
      {4096, 16.05},
      {16384, 16.08},
      {65536, 18.91},
      {262144, 20.93},
      {1048576, 41.25},
      {4194304, 42.21},
      {16777216, 44.51},
      {67108864, 118.81},
      {268435456, 432.12},
      {1073741824, 1692.16},
      {4294967296, 6831.91}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_AllGather_STMC: {
    static const Sample samps_r2[] = {
      {256, 11.9},
      {1024, 11.9},
      {4096, 12.18},
      {16384, 14.37},
      {65536, 14.95},
      {262144, 15.32},
      {1048576, 18.03},
      {4194304, 36.42},
      {16777216, 49.1},
      {67108864, 100.4},
      {268435456, 303.8},
      {1073741824, 1146.8},
      {4294967296, 4519.9}
    };
    static const Sample samps_r4[] = {
      {256, 12.45},
      {1024, 12.5},
      {4096, 13.12},
      {16384, 12.82},
      {65536, 13.06},
      {262144, 13.33},
      {1048576, 14.07},
      {4194304, 19.01},
      {16777216, 36.7},
      {67108864, 107.5},
      {268435456, 388.4},
      {1073741824, 1546.6},
      {4294967296, 6168.1}
    };
    static const Sample samps_r8[] = {
      {256, 12.68},
      {1024, 12.69},
      {4096, 12.93},
      {16384, 13.02},
      {65536, 13.21},
      {262144, 13.78},
      {1048576, 14.6},
      {4194304, 19.26},
      {16777216, 37.25},
      {67108864, 107.9},
      {268435456, 388.7},
      {1073741824, 1508.8},
      {4294967296, 5986.8}
    };
    static const Sample samps_r12[] = {
      {192, 12.94},
      {960, 12.95},
      {4032, 13.08},
      {16320, 14.35},
      {65472, 14.32},
      {262080, 14.44},
      {1048512, 14.88},
      {4194240, 19.9},
      {16777152, 39.64},
      {67108800, 117.5},
      {268435392, 425},
      {1073741760, 1648.8},
      {4294967232, 6655.6}
    };
    static const Sample samps_r16[] = {
      {256, 12.84},
      {1024, 12.88},
      {4096, 12.99},
      {16384, 13.71},
      {65536, 13.31},
      {262144, 13.41},
      {1048576, 14.81},
      {4194304, 19.53},
      {16777216, 37.56},
      {67108864, 108.4},
      {268435456, 389.9},
      {1073741824, 1510.2},
      {4294967296, 6984.3}
    };
    static const Sample samps_r24[] = {
      {768, 13.43},
      {3840, 13.46},
      {16128, 13.91},
      {65280, 16.43},
      {261888, 16.65},
      {1048320, 16.05},
      {4194048, 19.9},
      {16776960, 38.73},
      {67108608, 113.7},
      {268435200, 407.6},
      {1073741568, 1579.6},
      {4294967040, 6755.6}
    };
    static const Sample samps_r36[] = {
      {576, 13.46},
      {4032, 13.47},
      {16128, 13.68},
      {65088, 15.54},
      {262080, 17.1},
      {1048320, 15.19},
      {4193856, 20.42},
      {16777152, 40.21},
      {67108608, 113.1},
      {268435008, 428.4},
      {1073741760, 1655.4},
      {4294967040, 6236.2}
    };
    static const Sample samps_r48[] = {
      {768, 15.33},
      {3840, 15.35},
      {16128, 15.37},
      {65280, 16.63},
      {261888, 16.57},
      {1048320, 17.24},
      {4194048, 21.5},
      {16776960, 40.87},
      {67108608, 118.33},
      {268435200, 426.58},
      {1073741568, 1648.88},
      {4294967040, 7468.57}
    };
    static const Sample samps_r64[] = {
      {1024, 15.29},
      {4096, 15.33},
      {16384, 15.72},
      {65536, 16.22},
      {262144, 15.78},
      {1048576, 16.27},
      {4194304, 21.04},
      {16777216, 38.9},
      {67108864, 110.03},
      {268435456, 391.44},
      {1073741824, 1513.24},
      {4294967296, 8415.06}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_ReduceScatter_LL: {
    static const Sample samps_r2[] = {
      {256, 9.11},
      {512, 9.12},
      {1024, 9.17},
      {2048, 9.67},
      {4096, 9.91},
      {8192, 9.9},
      {16384, 10.24},
      {32768, 10.24},
      {65536, 10.43},
      {131072, 10.52},
      {262144, 10.63},
      {524288, 14.29},
      {1048576, 21.16},
      {2097152, 34.68},
      {4194304, 61.5},
      {8388608, 115.2},
      {16777216, 222.4},
      {33554432, 437.4}
    };
    static const Sample samps_r4[] = {
      {256, 9.2},
      {512, 9.21},
      {1024, 9.21},
      {2048, 9.49},
      {4096, 9.64},
      {8192, 10.04},
      {16384, 10.42},
      {32768, 10.92},
      {65536, 10.98},
      {131072, 11.2},
      {262144, 11.23},
      {524288, 11.58},
      {1048576, 16.32},
      {2097152, 25.03},
      {4194304, 42.41},
      {8388608, 76.69},
      {16777216, 144.4},
      {33554432, 279.8}
    };
    static const Sample samps_r8[] = {
      {256, 9.38},
      {512, 9.4},
      {1024, 9.41},
      {2048, 9.68},
      {4096, 9.95},
      {8192, 10.26},
      {16384, 10.72},
      {32768, 12.01},
      {65536, 12.31},
      {131072, 12.49},
      {262144, 12.75},
      {524288, 12.94},
      {1048576, 13.26},
      {2097152, 19.2},
      {4194304, 30.5},
      {8388608, 53.01},
      {16777216, 97.04},
      {33554432, 183.6}
    };
    static const Sample samps_r12[] = {
      {192, 10.4},
      {384, 10.41},
      {960, 10.49},
      {1920, 10.65},
      {4032, 10.85},
      {8064, 11.05},
      {16320, 11.56},
      {32640, 12.83},
      {65472, 13.35},
      {130944, 14.32},
      {262080, 14.7},
      {524160, 15.39},
      {1048512, 15.99},
      {2097024, 23.38},
      {4194240, 34.41},
      {8388480, 58.19},
      {16777152, 99.43},
      {33554304, 187.1}
    };
    static const Sample samps_r16[] = {
      {256, 10.1},
      {512, 10.13},
      {1024, 10.15},
      {2048, 10.41},
      {4096, 10.48},
      {8192, 10.63},
      {16384, 11.22},
      {32768, 12.2},
      {65536, 15.06},
      {131072, 15.32},
      {262144, 15.52},
      {524288, 16},
      {1048576, 15.86},
      {2097152, 16.68},
      {4194304, 25.98},
      {8388608, 44.14},
      {16777216, 79.53},
      {33554432, 147.6}
    };
    static const Sample samps_r24[] = {
      {384, 10.77},
      {768, 10.87},
      {1920, 10.93},
      {3840, 11.22},
      {8064, 11.45},
      {16128, 11.72},
      {32640, 12.7},
      {65280, 15.88},
      {130944, 16.39},
      {261888, 18.18},
      {524160, 19.19},
      {1048320, 19.53},
      {2097024, 21.98},
      {4194048, 33.68},
      {8388480, 51.92},
      {16776960, 92.79},
      {33554304, 163.4}
    };
    static const Sample samps_r36[] = {
      {576, 12.11},
      {1728, 12.14},
      {4032, 12.38},
      {8064, 12.75},
      {16128, 13},
      {32256, 13.96},
      {65088, 17.55},
      {130752, 23.46},
      {262080, 23.98},
      {524160, 24.44},
      {1048320, 24.6},
      {2096640, 26.42},
      {4193856, 36.88},
      {8388288, 63.11},
      {16777152, 108.4},
      {33554304, 202.3}
    };
    static const Sample samps_r48[] = {
      {768, 12.6},
      {1536, 12.62},
      {3840, 12.63},
      {7680, 12.94},
      {16128, 13.38},
      {32256, 14.55},
      {65280, 18},
      {130560, 23.15},
      {261888, 24.6},
      {523776, 29.22},
      {1048320, 29.57},
      {2096640, 29.46},
      {4194048, 37.95},
      {8388096, 56.98},
      {16776960, 96.42},
      {33553920, 178.25}
    };
    static const Sample samps_r64[] = {
      {1024, 13.83},
      {2048, 13.86},
      {4096, 13.5},
      {8192, 13.58},
      {16384, 13.93},
      {32768, 15.71},
      {65536, 18.86},
      {131072, 24.76},
      {262144, 36.5},
      {524288, 39.13},
      {1048576, 38.86},
      {2097152, 36.84},
      {4194304, 35.93},
      {8388608, 46.79},
      {16777216, 82.4},
      {33554432, 149.79}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_ReduceScatter_LD: {
    static const Sample samps_r2[] = {
      {256, 11.42},
      {1024, 11.5},
      {4096, 11.62},
      {16384, 11.93},
      {65536, 12.29},
      {262144, 12.24},
      {1048576, 12.37},
      {4194304, 17.08},
      {16777216, 36.75},
      {67108864, 112.6},
      {268435456, 411.2},
      {1073741824, 1596.2},
      {4294967296, 6321.7}
    };
    static const Sample samps_r4[] = {
      {256, 11.89},
      {1024, 11.89},
      {4096, 11.91},
      {16384, 14.54},
      {65536, 18.68},
      {262144, 19.12},
      {1048576, 19.1},
      {4194304, 18.09},
      {16777216, 37.35},
      {67108864, 109.4},
      {268435456, 392},
      {1073741824, 1517.3},
      {4294967296, 6007.8}
    };
    static const Sample samps_r8[] = {
      {256, 12.5},
      {1024, 12.51},
      {4096, 12.7},
      {16384, 12.88},
      {65536, 20.67},
      {262144, 23.92},
      {1048576, 24.56},
      {4194304, 26.08},
      {16777216, 38.95},
      {67108864, 112.3},
      {268435456, 403.1},
      {1073741824, 1557.1},
      {4294967296, 6146.6}
    };
    static const Sample samps_r12[] = {
      {192, 14.76},
      {960, 14.79},
      {4032, 14.95},
      {16320, 15.07},
      {65472, 24.7},
      {262080, 39.1},
      {1048512, 49.04},
      {4194240, 50.84},
      {16777152, 92.4},
      {67108800, 170.8},
      {268435392, 449.1},
      {1073741760, 1708.6},
      {4294967232, 6601.9}
    };
    static const Sample samps_r16[] = {
      {256, 14.99},
      {1024, 15},
      {4096, 15.03},
      {16384, 15.27},
      {65536, 20.42},
      {262144, 49.52},
      {1048576, 50.82},
      {4194304, 51.8},
      {16777216, 61.98},
      {67108864, 117.1},
      {268435456, 422},
      {1073741824, 1646.3},
      {4294967296, 6512.5}
    };
    static const Sample samps_r24[] = {
      {768, 17.61},
      {3840, 17.63},
      {16128, 17.73},
      {65280, 25.11},
      {261888, 53.92},
      {1048320, 69.51},
      {4194048, 71.3},
      {16776960, 98.58},
      {67108608, 265.7},
      {268435200, 593.2},
      {1073741568, 1779.9},
      {4294967040, 6961.9}
    };
    static const Sample samps_r36[] = {
      {576, 22.22},
      {4032, 22.23},
      {16128, 22.29},
      {65088, 22.68},
      {262080, 57.37},
      {1048320, 103.7},
      {4193856, 106.6},
      {16777152, 113.4},
      {67108608, 372.7},
      {268435008, 772},
      {1073741760, 1957.8},
      {4294967040, 7221.9}
    };
    static const Sample samps_r48[] = {
      {768, 28.26},
      {3840, 28.4},
      {16128, 28.58},
      {65280, 28.97},
      {261888, 64.26},
      {1048320, 117.98},
      {4194048, 156.23},
      {16776960, 165.08},
      {67108608, 435.4},
      {268435200, 1211.85},
      {1073741568, 2559.13},
      {4294967040, 7523.29}
    };
    static const Sample samps_r64[] = {
      {1024, 33.94},
      {4096, 33.98},
      {16384, 34.5},
      {65536, 34.73},
      {262144, 57.9},
      {1048576, 194.48},
      {4194304, 197.38},
      {16777216, 207.89},
      {67108864, 411.71},
      {268435456, 834.49},
      {1073741824, 1798.31},
      {4294967296, 7189.09}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  case ncclSymKernelId_ReduceScatter_LDMC: {
    static const Sample samps_r2[] = {
      {256, 12.03},
      {1024, 12.11},
      {4096, 12.18},
      {16384, 13.58},
      {65536, 13.75},
      {262144, 14.11},
      {1048576, 16.15},
      {4194304, 32.05},
      {16777216, 51.71},
      {67108864, 126.4},
      {268435456, 426.1},
      {1073741824, 1521},
      {4294967296, 5841.6}
    };
    static const Sample samps_r4[] = {
      {256, 12.02},
      {1024, 12.03},
      {4096, 12.05},
      {16384, 12.4},
      {65536, 12.58},
      {262144, 14.14},
      {1048576, 14.65},
      {4194304, 20.01},
      {16777216, 40.45},
      {67108864, 116.5},
      {268435456, 419.2},
      {1073741824, 1629.9},
      {4294967296, 6471.3}
    };
    static const Sample samps_r8[] = {
      {256, 12.41},
      {1024, 12.42},
      {4096, 12.43},
      {16384, 12.59},
      {65536, 12.78},
      {262144, 13.51},
      {1048576, 15.25},
      {4194304, 20.56},
      {16777216, 41.66},
      {67108864, 117.1},
      {268435456, 420.4},
      {1073741824, 1632.2},
      {4294967296, 6472.4}
    };
    static const Sample samps_r12[] = {
      {192, 12.62},
      {960, 12.63},
      {4032, 12.75},
      {16320, 12.86},
      {65472, 15.48},
      {262080, 16.05},
      {1048512, 18.13},
      {4194240, 23.49},
      {16777152, 44.87},
      {67108800, 120.5},
      {268435392, 422.6},
      {1073741760, 1638.4},
      {4294967232, 6508.1}
    };
    static const Sample samps_r16[] = {
      {256, 12.15},
      {1024, 12.84},
      {4096, 12.88},
      {16384, 13.13},
      {65536, 13.14},
      {262144, 13.32},
      {1048576, 15.61},
      {4194304, 21.05},
      {16777216, 42.92},
      {67108864, 119.1},
      {268435456, 424.7},
      {1073741824, 1647.7},
      {4294967296, 6543.7}
    };
    static const Sample samps_r24[] = {
      {768, 12.91},
      {3840, 12.95},
      {16128, 13.07},
      {65280, 15.71},
      {261888, 18.56},
      {1048320, 17.82},
      {4194048, 24.25},
      {16776960, 42.51},
      {67108608, 114.8},
      {268435200, 397.7},
      {1073741568, 1532.4},
      {4294967040, 6077.5}
    };
    static const Sample samps_r36[] = {
      {576, 13.22},
      {4032, 13.23},
      {16128, 13.29},
      {65088, 13.51},
      {262080, 18.74},
      {1048320, 17.45},
      {4193856, 24.32},
      {16777152, 49.48},
      {67108608, 121.4},
      {268435008, 430.4},
      {1073741760, 1667.2},
      {4294967040, 6447.6}
    };
    static const Sample samps_r48[] = {
      {768, 14.85},
      {3840, 14.9},
      {16128, 15.05},
      {65280, 15.22},
      {261888, 18.9},
      {1048320, 19.73},
      {4194048, 28},
      {16776960, 51.2},
      {67108608, 130.25},
      {268435200, 446.7},
      {1073741568, 1714.66},
      {4294967040, 6766.97}
    };
    static const Sample samps_r64[] = {
      {1024, 15.06},
      {4096, 15.07},
      {16384, 15.28},
      {65536, 15.37},
      {262144, 15.56},
      {1048576, 16.54},
      {4194304, 21.26},
      {16777216, 40.51},
      {67108864, 110.84},
      {268435456, 384.94},
      {1073741824, 1478.74},
      {4294967296, 5849.41}
    };
    return interp(nRanks, {
      {2, interp(nBytes, sizeof(samps_r2)/sizeof(Sample), samps_r2)},
      {4, interp(nBytes, sizeof(samps_r4)/sizeof(Sample), samps_r4)},
      {8, interp(nBytes, sizeof(samps_r8)/sizeof(Sample), samps_r8)},
      {12, interp(nBytes, sizeof(samps_r12)/sizeof(Sample), samps_r12)},
      {16, interp(nBytes, sizeof(samps_r16)/sizeof(Sample), samps_r16)},
      {24, interp(nBytes, sizeof(samps_r24)/sizeof(Sample), samps_r24)},
      {36, interp(nBytes, sizeof(samps_r36)/sizeof(Sample), samps_r36)},
      {48, interp(nBytes, sizeof(samps_r48)/sizeof(Sample), samps_r48)},
      {64, interp(nBytes, sizeof(samps_r64)/sizeof(Sample), samps_r64)}
    });
  }
  }
}
