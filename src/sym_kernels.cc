/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sym_kernels.h"
#include "comm.h"
#include "device.h"
#include "nccl_device/core.h"
#include "transport.h"
#include <cmath>
#include <cfloat>

constexpr char const* kernelName[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R",
  "AllReduce_AGxLLMC_R",
  "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC",
  "AllGather_LL",
  "AllGather_LLMC",
  "AllGather_ST",
  "AllGather_STMC",
  "AllGather_RailRing_LsaSTMC",
  "ReduceScatter_LL",
  "ReduceScatter_LD",
  "ReduceScatter_LDMC",
  "ReduceScatter_RailA2A_LsaLD",
  "ReduceScatter_RailA2A_LsaLDMC"
};

constexpr uint32_t kernelMask_STMC = 1<<ncclSymkKernelId_AllGather_LLMC |
                                     1<<ncclSymkKernelId_AllGather_STMC |
                                     1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                     1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymkKernelId_ReduceScatter_LDMC |
                                     1<<ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr uint32_t kernelMask_LDMC = 1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1<<ncclSymkKernelId_ReduceScatter_LDMC |
                                     1<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;

constexpr uint32_t kernelMask_LL = 1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymkKernelId_AllGather_LL |
                                   1<<ncclSymkKernelId_AllGather_LLMC |
                                   1<<ncclSymkKernelId_ReduceScatter_LL;

constexpr uint32_t kernelMask_AG = 1<<ncclSymkKernelId_AllGather_LL |
                                   1<<ncclSymkKernelId_AllGather_LLMC |
                                   1<<ncclSymkKernelId_AllGather_ST |
                                   1<<ncclSymkKernelId_AllGather_STMC |
                                   1<<ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr uint32_t kernelMask_AR = 1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                   1<<ncclSymkKernelId_AllReduce_RSxLD_AGxST;

constexpr uint32_t kernelMask_RS = 1<<ncclSymkKernelId_ReduceScatter_LD |
                                   1<<ncclSymkKernelId_ReduceScatter_LDMC |
                                   1<<ncclSymkKernelId_ReduceScatter_LL |
                                   1<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                   1<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;

constexpr uint32_t kernelMask_LSA = 1<<ncclSymkKernelId_AllReduce_AGxLL_R |
                                    1<<ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                    1<<ncclSymkKernelId_AllReduce_RSxLD_AGxST |
                                    1<<ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                    1<<ncclSymkKernelId_AllGather_LL |
                                    1<<ncclSymkKernelId_AllGather_LLMC |
                                    1<<ncclSymkKernelId_AllGather_ST |
                                    1<<ncclSymkKernelId_AllGather_STMC |
                                    1<<ncclSymkKernelId_ReduceScatter_LL |
                                    1<<ncclSymkKernelId_ReduceScatter_LD |
                                    1<<ncclSymkKernelId_ReduceScatter_LDMC;

constexpr uint32_t kernelMask_Gin = 1<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                    1<<ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
                                    1<<ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr uint32_t kernelMask_DynamicSmem = kernelMask_Gin & kernelMask_RS;

int ncclSymkLLKernelMask() {
  return kernelMask_LL;
}
int ncclSymkDynamicSmemKernelMask() {
  return kernelMask_DynamicSmem;
};

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
  uint32_t got = COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed);
  if (got == -1u) {
    // TODO: Enhance this to be a pattern match. I like regex's but we also have
    // the parseList() used by NCCL_ALGO/PROTO.
    char const* name = ncclGetEnv("NCCL_SYM_KERNEL");
    if (name == nullptr || strcmp(name, "^") == 0) {
      static_assert((int)ncclSymkKernelId_Count < 32, "Use more than 32 bits");
      got = (1<<(int)ncclSymkKernelId_Count)-1;
    } else {
      got = 0;
      for (int k=0; k < (int)ncclSymkKernelId_Count; k++) {
        if (strcmp(kernelName[k], name) == 0) {
          COMPILER_ATOMIC_STORE(&cache, 1<<k, std::memory_order_relaxed);
          got = 1<<k;
          break;
        }
      }
    }
    COMPILER_ATOMIC_STORE(&cache, got, std::memory_order_relaxed);
  }
  return got;
}

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)
NCCL_PARAM(SymGinKernelsEnable, "SYM_GIN_KERNELS_ENABLE", 0)

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
static void queryModel_gin(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks);
static void queryModel_lsa(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks);

static void queryModel(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks) {
  if (kernelMask_Gin>>k & 1) {
    queryModel_gin(comm, k, nBytes, timeUs, nBlocks);
  } else {
    queryModel_lsa(comm, k, nBytes, timeUs, nBlocks);
  }
}

#define NCCL_NVLINK_BW_IDX_HOPPER 0
#define NCCL_NVLINK_BW_IDX_BLACKWELL 1
#define NCCL_NVLINK_BW_IDX_NUM 2

// NVLS max bws NCCL can achieve
static const float nvlinkBws[NCCL_NVLINK_BW_IDX_NUM] = {
  360.0f, // Hopper
  720.0f, // Blackwell
};

static double getLsaBw(struct ncclComm* comm) {
  int compCapIndex = comm->minCompCap >= 100 ? NCCL_NVLINK_BW_IDX_BLACKWELL : NCCL_NVLINK_BW_IDX_HOPPER;
  return (/*byte/sec*/1.e9)*nvlinkBws[compCapIndex];
}

static double getGinLat(struct ncclComm* comm) {
  return (/*sec/usec*/1.e-6)*comm->tunerConstants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
}

static double getGinBw(struct ncclComm* comm) {
  return (/*byte/sec*/1.e9)*comm->minNetBw;
}

// Bus multipliers count number of times data is sent through that widget.
static void getBusMul_ReduceScatter_RailA2A(
    struct ncclComm* comm, bool ldmc,
    // Bus multipliers per bottleneck
    double* out_smMul, double* out_lsaMul, double* out_ginMul
  ) {
  int lsaRanks = ncclTeamLsa(comm).nRanks;
  int railRanks = ncclTeamRail(comm).nRanks;
  // LSA
  *out_lsaMul = std::max(
    /*inbound*/(ldmc ? lsaRanks : lsaRanks-1)*railRanks,
    /*outbound*/(lsaRanks-1)*railRanks
  );
  // GIN
  *out_ginMul = railRanks-1; // inbound == outbound
  // SM. Inbound (reads) only because it dominates outbound (writes).
  *out_smMul =
    /*stage 0*/(lsaRanks == 1 ? 0 : (ldmc ? 1 : lsaRanks)*(railRanks-1)) +
    /*stage 1*/(ldmc ? 1 : lsaRanks) + (railRanks-1);
}

static double getSmBw_ReduceScatter_RailA2A(struct ncclComm* comm, bool ldmc) {
  // Empirically calculated as effbw/nctas where effbw is reported by TUNING
  // debug logging (from getRequirements_gin()) and nctas is the number of ctas
  // that appear to saturate bandwidth.
  if (100 <= comm->minCompCap) {
    return ldmc ? 2.25e9 : 5.0e9;
  } else {
    return ldmc ? 9.85e9 : 14.5e9;
  }
}

static double getSmLat_ReduceScatter_RailA2A(struct ncclComm* comm, bool ldmc) {
  // Processing delay. Larger value means bigger network buffers.
  return 10.e-6;
}

// Calculate saturation block count:
static int calcSatBlocks_ReduceScatter_RailA2A(struct ncclComm* comm, bool ldmc) {
  double lsaBw = getLsaBw(comm);
  double ginBw = getGinBw(comm);
  double smBw = getSmBw_ReduceScatter_RailA2A(comm, ldmc);
  double smMul, lsaMul, ginMul;
  getBusMul_ReduceScatter_RailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
  // Effective Bandwidth: EffBw = Bw/Mul
  // Let smsEffBw = smEffBw*nBlocks
  // Set smsEffBw = min(lsaEffBw, ginEffBw)
  // Solve for nBlocks:
  double minLsaGinEffBw = std::min(lsaBw/lsaMul, ginBw/ginMul);
  return std::ceil(std::min(double(1<<30), minLsaGinEffBw/(smBw/smMul)));
}

static void getRequirements_gin(struct ncclComm* comm, int* out_nBlocks, size_t* out_bufSize ) {
  *out_nBlocks = 0;
  *out_bufSize = 0;
  for (int ldmc = 0; ldmc <= 1; ldmc++) {
    double lsaBw = getLsaBw(comm);
    double ginBw = getGinBw(comm);
    double ginLat = getGinLat(comm);
    double smLat = getSmLat_ReduceScatter_RailA2A(comm, ldmc);
    double smMul, lsaMul, ginMul;
    getBusMul_ReduceScatter_RailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
    // GIN could be throttled by LSA work
    double ginBwRenorm = std::min(lsaBw/lsaMul, ginBw/ginMul)*ginMul;
    size_t bufSize = ginBwRenorm*(ginLat + smLat);
    int nBlocks = calcSatBlocks_ReduceScatter_RailA2A(comm, ldmc);
    if (comm->rank == 0) {
      double minLsaGinEffBw = std::min(lsaBw/lsaMul, ginBw/ginMul);
      INFO(NCCL_TUNING, "ReduceScatter_RailA2A_Lsa%s : satblocks=%d bufsize=%d effbw=%g\n", ldmc ? "LDMC" : "LD", nBlocks, (int)bufSize, minLsaGinEffBw*smMul);
    }
    *out_nBlocks = std::max(*out_nBlocks, nBlocks);
    *out_bufSize = std::max(*out_bufSize, bufSize);
  }
}

static void queryModel_gin(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks) {
  struct ncclSymkState* symk = &comm->symkState;
  //ncclTeam world = ncclTeamWorld(comm);
  //ncclTeam lsa = ncclTeamLsa(comm);
  ncclTeam rail = ncclTeamRail(comm);
  double lsaBw = getLsaBw(comm);
  double ginLat = getGinLat(comm);
  double ginBw = getGinBw(comm);
  int nMaxBlocks = std::min<int>(comm->config.maxCTAs, ncclSymkMaxBlocks);
  if (k == ncclSymkKernelId_AllGather_RailRing_LsaSTMC) {
    nMaxBlocks = std::min<int>(nMaxBlocks, divUp((comm->cudaArch < 1000 ? 16 : 32), comm->nvlsResources->nHeads));
  }
  int nMinBlocks = comm->config.minCTAs;
  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  *timeUs = FLT_MAX;
  *nBlocks = 0;
  switch (k) {
  case ncclSymkKernelId_AllGather_RailRing_LsaSTMC: {
      constexpr int railChunkSize = ncclSymkAllGather_RailRing_ChunkSize;
      int requiredBlocks = DIVUP(nBytes, railChunkSize);
      float intraBw = lsaBw;
      float interBw = ginBw;
      float intraTime = (float)(nBytes * comm->nRanks) / intraBw;
      float interTime = (float)(nBytes * (rail.nRanks - 1)) / interBw;
      uint32_t steps = DIVUP(nBytes, railChunkSize) * (rail.nRanks - 1);
      *timeUs = steps * ginLat + std::max(intraTime, interTime);
      *nBlocks = std::max(nMinBlocks, std::min(nMaxBlocks, requiredBlocks));
    } break;
  case ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD:
  case ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC: {
      bool ldmc = k == ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;
      nMaxBlocks = std::min(nMaxBlocks, symk->maxGinInboxBlocks);
      nMaxBlocks = std::min(nMaxBlocks, calcSatBlocks_ReduceScatter_RailA2A(comm, ldmc));
      constexpr int chunkSize = 64<<10;
      double smBw = getSmBw_ReduceScatter_RailA2A(comm, ldmc);
      double smMul, lsaMul, ginMul;
      getBusMul_ReduceScatter_RailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
      *nBlocks = divUp(nBytes, chunkSize);
      // max against nMinBlocks last since we may have nMaxBlocks < nMinBlocks
      *nBlocks = std::max(nMinBlocks, std::min(nMaxBlocks, *nBlocks));
      double effBw = (*nBlocks)*(smBw/smMul);
      effBw = std::min(effBw, lsaBw/lsaMul);
      effBw = std::min(effBw, ginBw/ginMul);
      double time = nBytes/effBw;
      // Delayed by LSA processing of first chunk.
      time += std::min<size_t>(nBytes, chunkSize*(*nBlocks))*(lsaMul/lsaBw + ginMul/ginBw);
      // Delay by GIN latency of first chunk.
      time += ginLat;
      *timeUs = (/*usec/sec=*/1.e6)*time;
    } break;
  default: break;
  }
}

static void queryModel_lsa(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, float* timeUs, int* nBlocks) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling

  int nRanks = comm->nRanks;
  int nMaxBlocks = ncclSymkMaxBlocks;
  int nMaxBlocksNvls = divUp((comm->cudaArch < 1000 ? 16 : 32), nRanks);
  size_t busBytes; // max(bytes sent, bytes received)
  double busMultiplier = 1;

  switch (k) {
  default:
    busBytes = size_t(1)<<50;
    break;

  case ncclSymkKernelId_AllReduce_AGxLL_R:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllReduce_AGxLLMC_R:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:
    busBytes = 2*nBytes*(nRanks-1)/nRanks;
    break;
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
    busBytes = nBytes/nRanks + nBytes;
    busMultiplier = nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;

  case ncclSymkKernelId_AllGather_LL:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_AllGather_LLMC:
    busBytes = nRanks*nBytes*LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllGather_ST:
    busBytes = (nRanks-1)*nBytes;
    break;
  case ncclSymkKernelId_AllGather_STMC:
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55*nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;

  case ncclSymkKernelId_ReduceScatter_LL:
    busBytes = nRanks*nBytes*LL_BusFactor;
    break;
  case ncclSymkKernelId_ReduceScatter_LD:
    busBytes = (nRanks-1)*nBytes;
    break;
  case ncclSymkKernelId_ReduceScatter_LDMC:
    busBytes = (nRanks-1)*nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55*nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;
  }

  nMaxBlocks = std::min<int>(nMaxBlocks, comm->config.maxCTAs);
  int nMinBlocks = comm->config.minCTAs;

  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  bool isLL = kernelMask_LL>>k & 1;
  bool isAG = kernelMask_AG>>k & 1;
  bool isAR = kernelMask_AR>>k & 1;
  constexpr double GBps = (1<<30)/1.e6;
  double baseLat, smBw, peakBw;
  if (comm->cudaArch < 1000) {
    baseLat = isLL ? 4.5 : 7.8;
    smBw = isAR ? 65*GBps : 44*GBps;
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480*GBps : 320*GBps;
  } else {
    baseLat = isLL ? (isAG ? 8.5 : 11) : (isAR ? 19.5 : 13.0);
    smBw = 55*GBps;
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000*GBps : 600*GBps;
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

ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) {
  // ncclTeamLsa() below calls this internally but drops the error code so we do it here.
  NCCLCHECK(ncclDevrInitOnce(comm));

  struct ncclSymkState* symk = &comm->symkState;
  if (!symk->initialized) {
    symk->initialized = true;
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    symk->hasLsaMultimem = comm->nvlsSupport && ncclTeamLsa(comm).nRanks > 2;
    reqs.lsaMultimem = symk->hasLsaMultimem;
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;

    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(
      ncclSymkMaxBlocks, ncclLLA2ACalcSlots(ncclTeamLsa(comm).nRanks*ncclSymkMaxThreads, ncclSymkLLMaxEltSize),
      &symk->kcomm.lsaLLA2A, &lla2aReq
    );
    lla2aReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &lla2aReq;

    struct ncclDevResourceRequirements ginInboxRailReq = {};
    struct ncclDevResourceRequirements ginOutboxReq = {};
    struct ncclDevResourceRequirements railSignalReq = {};
    if (ncclParamSymGinKernelsEnable() && ncclTeamLsa(comm).nRanks < comm->nRanks) {
      int maxBlocks;
      size_t bufSize;
      getRequirements_gin(comm, &maxBlocks, &bufSize);

      maxBlocks = std::max(maxBlocks, comm->config.minCTAs);
      maxBlocks = std::min(maxBlocks, comm->config.maxCTAs);
      if (ncclParamSymCTAs() >= 1) maxBlocks = ncclParamSymCTAs();
      maxBlocks = std::min(maxBlocks, ncclSymkMaxBlocks);
      symk->maxGinInboxBlocks = maxBlocks;

      ncclGinInboxA2ACreateRequirement(
        ncclTeamRail(comm), maxBlocks, log2Up(bufSize),
        &symk->kcomm.ginInboxRail, &ginInboxRailReq
      );
      ginInboxRailReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &ginInboxRailReq;

      ncclGinOutboxCreateRequirement(
        maxBlocks, log2Up(bufSize),
        &symk->kcomm.ginOutbox, &ginOutboxReq
      );
      ginOutboxReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &ginOutboxReq;

      uint32_t railSignalCount = ncclTeamRail(comm).nRanks * ncclSymkMaxBlocks;

      railSignalReq.bufferSize = 0;
      railSignalReq.bufferAlign = 0;
      railSignalReq.outBufferHandle = nullptr;
      railSignalReq.ginSignalCount = railSignalCount;
      railSignalReq.outGinSignalStart = &symk->kcomm.ginSyncHandle.railSignals;
      railSignalReq.ginCounterCount = ncclSymkMaxBlocks;
      railSignalReq.outGinCounterStart = &symk->kcomm.ginCounterPerBlock;
      railSignalReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &railSignalReq;
      reqs.railGinBarrierCount = ncclSymkMaxBlocks;

      reqs.ginConnectionType = comm->globalGinSupport;
    }

    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm, true));
  }
  return ncclSuccess;
}

ncclResult_t ncclSymkFinalize(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->kcomm.devComm));
  }
  return ncclSuccess;
}

static bool ncclSymkImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
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

static uint32_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, size_t nElts) {
  uint32_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();

  bool hasSTMC = comm->symkState.hasLsaMultimem;
  bool hasLDMC = false;
  if (comm->symkState.hasLsaMultimem) {
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

  bool hasGin = ncclParamSymGinKernelsEnable() != 0;
  if (!hasGin) kmask &= ~kernelMask_Gin;
  bool needGin = ncclTeamLsa(comm).nRanks < comm->nRanks;
  kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
  return kmask;
}

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts) {
  if (!comm->isAllDirectNvlink)
    return false;
  if (!ncclSymkImplemented(coll, red, ty))
    return false;

  return (ncclSymkMask(comm, coll, red, ty, nElts) != 0);
}

ncclResult_t ncclSymkPickKernel(
    struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
    size_t nEltsTotal, size_t nEltsMax, int nWorks, ncclSymRegType_t winRegType,
    float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, bool* forced
  ) {
  uint32_t kmask = ncclSymkMask(comm, coll, red, ty, nEltsMax);

  *forced = !(kernelMask_user() == (1<<(int)ncclSymkKernelId_Count)-1);
  // We currently don't support grouping for LL kernels.
  if (nWorks > 1)
    kmask &= ~kernelMask_LL;

  if (coll == ncclFuncAllReduce) {
    if (winRegType != ncclSymSendRegRecvReg) kmask &= kernelMask_LL;
  } else if (coll == ncclFuncAllGather) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) kmask &= kernelMask_LL;
    if (winRegType != ncclSymSendRegRecvReg && comm->nNodes > 1) kmask &= ~kernelMask_Gin;
  } else if (coll == ncclFuncReduceScatter) {
    if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendRegRecvNonreg) kmask &= kernelMask_LL;
  }

  ncclSymkKernelId bestKernel = ncclSymkKernelId_Count;
  float bestTime = 1.e30f;
  int bestBlocks = 999;
  size_t nBytes = nEltsTotal*ncclTypeSize(ty);

  constexpr float smPenalty = .025f; // 2.5% percent increase in time per SM
  uint32_t kmaskRemain = kmask;
  while (kmaskRemain != 0) {
    ncclSymkKernelId k = (ncclSymkKernelId)popFirstOneBit(&kmaskRemain);
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
  *estTimeUs = kmask==0 || kernelMask_user() == (1<<ncclSymkKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;
  *nWarps = 16;
  return ncclSuccess;
}

const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) {
    return "Unknown";
  }
  return kernelName[kernelId];
}

int ncclSymkMaxChunkElts(struct ncclComm* comm, ncclSymkKernelId kernelId, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  bool isReduce = 1 & ((kernelMask_AR|kernelMask_RS) >> (int)kernelId);
  int eltSize = ncclTypeSize(ty);
  int accMult = !isReduce ? 1 : eltSize < 4 ? 2 : 1;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, red, ty);
  return ncclSymkKernelMaxDynamicSmem[kernelIndex]/(eltSize*accMult);
}

/* this function fills in the devWork except nextWorkOffset */
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;
  outDevWork->redOpArg = task->opDev.scalarArg;
  outDevWork->nElts = task->count;
  outDevWork->inputWin = task->sendWin ? task->sendWin->vidmem : nullptr;
  outDevWork->inputOff = task->sendWin ? (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr : (size_t)task->sendbuff;
  outDevWork->outputWin = task->recvWin ? task->recvWin->vidmem : nullptr;
  outDevWork->outputOff = task->recvWin ? (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr : (size_t)task->recvbuff;
  outDevWork->sChannelId = 0xffff;
  outDevWork->nChannels = 0;
  return ncclSuccess;
}

ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType) {
  bool isSendSymmReg = false;
  bool isRecvSymmReg = false;
  if (sendWin && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isSendSymmReg = true;
  if (recvWin && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isRecvSymmReg = true;
  // determine the registration type
  if (!isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvNonreg;
  } else if (isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvNonreg;
  } else if (!isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvReg;
  } else if (isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvReg;
  }
  return ncclSuccess;
}
