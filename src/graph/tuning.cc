/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "device.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t parseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
       {  6.8, 14.0,    0 }, {  6.6, 14.0,  8.4 },  // Tree, Ring
       {    0,    0,    0 }, {    0,    0,    0 },  // Collnet Direct, Chain
       {    0,    0,    0 }, {    0,    0,    0 }}; // NVLS, NVLS Tree

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
static float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree (LL/LL128/Simple)*/ { .6, 1.25, 28 }, /* Ring (LL/LL128/Simple)*/ { .6, 1.9, 3.4 },
    /* CollNetDirect (Simple)*/ { 0, 0, 3.7 }, /* CollNetChain (Simple)*/ { 0, 0, 2.8 },
    /* NVLS */ { 0, 0, 23 }, /* NVLSTree */ { 0, 0, 23 } },
  /* PCI */
  { /* Tree (LL/LL128/Simple)*/ { 1.0, 1.9, 28 }, /* Ring (LL/LL128/Simple)*/ { 1.0, 2.5, 5.7 },
    /* CollNetDirect (Simple)*/ { 0, 0, 3.7 }, /* CollNetChain (Simple)*/ { 0, 0, 2.8 },
    /* NVLS */ { 0, 0, 0 }, /* NVLSTree */ { 0, 0, 0 } },
  /* NET */
  { /* Tree (LL/LL128/Simple)*/ { 5.0, 8.5, 28 }, /* Ring (LL/LL128/Simple)*/ { 2.7, 4.0, 14.0 },
    /* CollNetDirect (Simple)*/ { 0, 0, 31 }, /* CollNetChain (Simple)*/ { 0, 0, 30 },
    /* NVLS */ { 0, 0, 18 }, /* NVLSTree */ { 0, 0, 14 } }
};

/* Array indexes used below */
#define VOLTA_COMPCAP_IDX 0
#define AMPERE_COMPCAP_IDX 1
#define HOPPER_COMPCAP_IDX 2

// LL128 max BW per channel
static const double llMaxBws[3][3] = {
  /* Volta-N1/Intel-N2/Intel-N4) */ {39.0, 39.0, 20.4},
  /* Ampere-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0},
  /* Hopper-N1/AMD-N2/AMD-N4) */ {87.7, 22.5 /*avg of ring & tree*/, 19.0}
};

static const double perChMaxRingLL128Bws[3][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 36.7},
};
static const double perChMaxTreeLL128Bws[3][3] = {
  /* Volta (N1/N2/N4) */  {20.0, 20.0, 20.0},
  /* Ampere (N1/N2/N4) */ {20.0, 20.0, 20.0},
  /* Hopper (N1/N2/N4) */ {36.7, 36.7, 29.0},
};
static const double perChMaxTreeBws[3][3] = {
  /* Volta (N1/N2/N4) */  {26.5, 18.5, 10.0},
  /* Ampere (N1/N2/N4) */ {24.0, 23.6, 17.8},
  /* Hopper (N1/N2/N4) */ {38.7, 41.4, 36.0},
};

// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM(NetOverhead, "NET_OVERHEAD", -2);

static float getNetOverhead(struct ncclComm* comm) {
  if (ncclParamNetOverhead() != -2) return ncclParamNetOverhead() * .001;
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  if (cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL) return 1.0;
  if (cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD) return 2.0;
  else return 1.0;
}

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) {
  int simpleDefaultThreads = (graphs[NCCL_ALGO_RING]->bwIntra*graphs[NCCL_ALGO_RING]->nChannels <= PCI_BW) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] =
    comm->maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = NCCL_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] =
    getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS/4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);

  // MNNVL support - treat as a single NVLink connected node
  int nNodes = comm->MNNVL ? 1 : comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;

  int compCapIndex = minCompCap >= 90 ? HOPPER_COMPCAP_IDX : minCompCap >= 80 ? AMPERE_COMPCAP_IDX : VOLTA_COMPCAP_IDX;
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = nNodes == 1 ? compCapIndex : cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD ? 1 : 0;
  double llMaxBw = llMaxBws[index1][index2];
  double perChMaxTreeBw = perChMaxTreeBws[compCapIndex][index2];
  double perChMaxRingLL128Bw = perChMaxRingLL128Bws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = perChMaxTreeLL128Bws[compCapIndex][index2];
  // De-penalize Tree/Simple latency on Power systems to favor Tree than Ring
  if (cpuArch == NCCL_TOPO_CPU_ARCH_POWER) hwLat[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = hwLat[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
  float ppn = (float)nRanks / nNodes; // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount

  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;
    int nInterSteps = coll == ncclFuncAllReduce ? (nNodes > 1 ? 2*nNodes :0) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nNodes-1 :
      nNodes;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      if (coll == ncclFuncBroadcast && a != NCCL_ALGO_RING) continue;
      if (coll == ncclFuncReduce && a != NCCL_ALGO_RING) continue;
      if (coll == ncclFuncReduceScatter && a != NCCL_ALGO_RING && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;
      if (coll == ncclFuncAllGather && a != NCCL_ALGO_RING && a != NCCL_ALGO_NVLS && a != NCCL_ALGO_COLLNET_DIRECT) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && p != NCCL_PROTO_SIMPLE) continue;
        int collnet = (a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) ? 1 : 0;
        float bw = nNodes <= 2 || collnet ? graphs[a]->bwIntra : graphs[a]->bwInter;
        if (a == NCCL_ALGO_NVLS) bw = std::min(graphs[a]->bwIntra, graphs[a]->bwInter);
        if (a == NCCL_ALGO_NVLS_TREE) bw = std::min(graphs[a]->bwIntra, nNodes <= 2 ? graphs[a]->bwInter : graphs[a]->bwInter/2);
        float busBw = graphs[a]->nChannels * bw;

        // Various model refinements
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) { busBw = std::min(llMaxBw, busBw * ((nNodes > 1 || coll == ncclFuncAllReduce || coll == ncclFuncReduce) ? 1.0/4.0 : 1.0/3.0)); }
        if (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (ppn < 2 ? 0.7 : 0.92 /*120.0/128.0*/), graphs[a]->nChannels*perChMaxRingLL128Bw);
        if (a == NCCL_ALGO_TREE) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) busBw = std::min(busBw*1.0/3.8, llMaxBw);
        if (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL128) busBw = std::min(busBw * (nNodes == 1 ? 7.0/9.0 : 120.0/128.0), graphs[a]->nChannels*perChMaxTreeLL128Bw);
        if (a == NCCL_ALGO_TREE && graphs[a]->pattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;
        if (a == NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_CHAIN && p != NCCL_PROTO_SIMPLE) busBw = 0;  // Not used
        if (a == NCCL_ALGO_COLLNET_DIRECT && p == NCCL_PROTO_SIMPLE) {
          if (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter) {
            busBw = ppn * bw;
            // AllGather/ReduceScatter requires 1:1 GPU:NIC
            int nicPerNode = comm->collNetHeadsUniqueNum;
            if (coll == ncclFuncAllGather && comm->nNodes > 1) {
              if (!comm->ncclCollNet || !comm->ncclCollNet->iallgather || ppn > nicPerNode) busBw = 0;
            }
            if (coll == ncclFuncReduceScatter && comm->nNodes > 1) {
              if (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter || ppn > nicPerNode) busBw = 0;
            }
            // Measured corrective ratio needed at 1 ppn and 8ppn. Here we hackishly
            // interpolate the two.
            float w = (ppn-1)/(8-1);
            busBw *= w*0.85 + (1-w)*0.95;
          } else {
            // Collnet+Direct requires all GPUs to have a local NIC to work at full speed
            float factor = ppn / (1.0*graphs[a]->nChannels); // GPU/NIC ratio
            factor -= (factor-1)/2;
            busBw /= factor;
            if (minCompCap >= 90) busBw *= .85;
          }
        }

        // Convert bus BW to algorithm BW
        if (!(a == NCCL_ALGO_COLLNET_DIRECT && (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter))) {
          float ratio = 1.0f;
          if (a == NCCL_ALGO_RING) ratio *= (1.0 * nRanks) / nsteps;
          else if (a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) ratio *= 5.0/6.0;
          else ratio *= .5;
          busBw *= ratio;
        }
        comm->bandwidths[coll][a][p] = busBw;
        /* Ring bandwidth backup */
        if (a == NCCL_ALGO_RING)
          comm->ringbdw[coll][p] = comm->bandwidths[coll][NCCL_ALGO_RING][p];

        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = hwLat[intraHw[a]][a][p];
        float interLat = hwLat[NCCL_HW_NET][a][p] + graphs[a]->latencyInter;
        // Also add the flush extra latency
        if (p == NCCL_PROTO_SIMPLE) interLat += graphs[a]->latencyInter;

        if (a == NCCL_ALGO_RING) {
          float lat = hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (graphs[a]->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == NCCL_PROTO_SIMPLE) lat = hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            // Inter-node rings still have to launch nsteps * net overhead.
            float netOverhead = 0.0;
            if (nNodes > 1) {
              netOverhead = getNetOverhead(comm);
              if (p == NCCL_PROTO_SIMPLE) netOverhead *= 3;
            }
            intraLat = std::max(intraLat, netOverhead);
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          comm->latencies[coll][a][p] +=
            2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
        } else if (a == NCCL_ALGO_COLLNET_DIRECT) {
          comm->latencies[coll][a][p] +=
            2 * (std::min(1, (nRanks/nNodes-1)) * intraLat + (nRanks/nNodes-1) * 0.4) + interLat;  // Add 0.4 us arity serialization latency
        } else if (a == NCCL_ALGO_COLLNET_CHAIN) {
          comm->latencies[coll][a][p] += 2 * (nRanks/nNodes-1) * intraLat + interLat;
        } else if (a == NCCL_ALGO_NVLS) {
          comm->latencies[coll][a][p] = intraLat;
          if (nNodes > 1) comm->latencies[coll][a][p] += interLat;
        } else if (a == NCCL_ALGO_NVLS_TREE) {
          comm->latencies[coll][a][p] += intraLat + 2 * log2i(nNodes) * interLat;
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  int protoEnable[NCCL_NUM_PROTOCOLS] = { 1, 2, 1 };
  int algoEnable[NCCL_NUM_ALGORITHMS] = { 1, 1, 1, 1, 1, 1 };

  const char *protoStr = ncclGetEnv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = ncclGetEnv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  // MNNVL: NVLS not yet supported
  if (comm->nNodes == 1 || comm->MNNVL) algoEnable[NCCL_ALGO_NVLS_TREE] = 0;

  // Disable CollNet if it is not supported
  if (comm->collNetSupport == 0) {
    algoEnable[NCCL_ALGO_COLLNET_DIRECT] = 0;
    algoEnable[NCCL_ALGO_COLLNET_CHAIN] = 0;
    // MNNVL: NVLS not yet supported
    if (comm->nNodes > 1 || comm->MNNVL) algoEnable[NCCL_ALGO_NVLS] = 0;
    // If user has hard set NCCL_ALGO=COLLNET, ignore it
    if (algoEnable[NCCL_ALGO_RING] == 0 && algoEnable[NCCL_ALGO_TREE] == 0 &&
        algoEnable[NCCL_ALGO_NVLS] == 0 && algoEnable[NCCL_ALGO_NVLS_TREE] == 0) {
      algoEnable[NCCL_ALGO_RING] = algoEnable[NCCL_ALGO_TREE] = 1;
    }
  } else {
    // Disable CollNet+Direct if not on an NVSwitch system
    int nvsCount = 0;
    NCCLCHECK(ncclTopoGetNvsCount(comm->topo, &nvsCount));
    if (nvsCount == 0) algoEnable[NCCL_ALGO_COLLNET_DIRECT] = 0;
  }

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[p];
    if (pEnable == 2 && p == NCCL_PROTO_LL128) {
      // Enable LL128 by default only on Volta/Ampere/Hopper+NVLink. Other cases are not tested and may cause silent data corruption.
      pEnable = 1;
      pEnable &= (graphs[a]->typeInter <= PATH_PXB || (minCompCap >= 90 && graphs[a]->typeInter <= PATH_PXN));
      pEnable &= (graphs[a]->typeIntra <= PATH_NVB);
      pEnable &= (minCompCap == maxCompCap);
      switch (minCompCap) {
      case 70: pEnable &= 1; break;
      case 80: pEnable &= 1; break;
      case 90: pEnable &= !(CUDART_VERSION == 11080 && c == ncclFuncAllReduce && a == NCCL_ALGO_RING && comm->nRanks == 2); break;
      default: pEnable &= 0; break;
      }
    }
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    if (algoEnable[a] == 0) comm->bandwidths[c][a][p] = 0;
  }

  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    bool available = false;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
        if (comm->bandwidths[c][a][p] != 0) {
          available = true;
          goto check_avail;
        }
  check_avail:
    if (available == false) {
      /* at least set ring algo available */
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
        comm->bandwidths[c][NCCL_ALGO_RING][p] = comm->ringbdw[c][p];
    }
  }

  if (comm->rank == 0) {
    char line[1024];
    for (int block=0; block<2; block++) {
      sprintf(line, "  Algorithm   |");
      for (int ba=0; ba<NCCL_NUM_ALGORITHMS/2; ba++) {
	int a = block*NCCL_NUM_ALGORITHMS/2+ba;
        sprintf(line+strlen(line), " %14s   %14s   %14s |", "", ncclAlgoStr[a], "");
      }
      INFO(NCCL_TUNING, "%s", line);
      sprintf(line, "  Protocol    |");
      for (int ba=0; ba<NCCL_NUM_ALGORITHMS/2; ba++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), " %14s |", ncclProtoStr[p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      sprintf(line, " Max NThreads |");
      for (int ba=0; ba<NCCL_NUM_ALGORITHMS/2; ba++) {
	int a = block*NCCL_NUM_ALGORITHMS/2+ba;
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), " %14d |", comm->maxThreads[a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
      for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
        sprintf(line, "%13s |", ncclFuncStr[c]);
        for (int ba=0; ba<NCCL_NUM_ALGORITHMS/2; ba++) {
	  int a = block*NCCL_NUM_ALGORITHMS/2+ba;
          for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
            sprintf(line+strlen(line), "%8.1f/%6.1f |", comm->latencies[c][a][p], comm->bandwidths[c][a][p]);
          }
        }
        INFO(NCCL_TUNING, "%s", line);
      }
    }
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }
  comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks;
  comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 512;
  comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] = 512;

  // Override defaults with user env
  const char* str = ncclGetEnv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[2][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2 }};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<2; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld/%ld/%ld | %ld/%ld/%ld | %ld | %ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .6,  .6,  .6,  .8,  .9,  .9,  .9,  .9, 1.0, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};

ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo* info, int algorithm, int protocol, int numPipeOps, float* time, bool* backup) {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];

  if (backup) {
    *backup = false;
    if (algorithm == NCCL_ALGO_RING && bw == 0.0f) {
      /* try back up RING algorithm */
      bw = info->comm->ringbdw[info->coll][protocol];
      *backup = true;
    }
  }

  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(info->nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && (!info->comm->MNNVL && info->comm->nNodes > 1)
      && info->coll == ncclFuncAllReduce && info->nBytes/(info->comm->nChannels*info->comm->nRanks) >= 64) {
    lat *= info->comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
  // Tree pipelining saves latency in aggregation cases
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_WORK_ELEMENTS);
  *time = lat * latCount + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}
