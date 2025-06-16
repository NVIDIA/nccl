/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
#include "xml.h"
#include "net.h"

#define LOC_BW 5000.0
#define SM60_NVLINK_BW 18.0
#define SM70_NVLINK_BW 20.0
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define SM86_NVLINK_BW 12.0
#define SM100_NVLINK_BW 40.0
#define PCI_BW 12.0           // PCI Gen3 x16
#define AMD_BW 16.0
#define BDW_QPI_BW 6.0
#define SKL_QPI_BW 10.0
#define SRP_QPI_BW 22.0
#define ERP_QPI_BW 40.0
#define ZPI_BW 6.0
#define YONGFENG_ZPI_BW 9.0
#define P9_BW 32.0
#define ARM_BW 6.0
#define NET_BW 12.0           // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P_OVERHEAD(bw) (bw*6/5)

#define NCCL_TOPO_NODE_TYPES 6
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
extern const char* topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB
#define LINK_C2C 3
#define LINK_PCI 4
// Skipping 5 for PATH_PXB
// Skipping 6 for PATH_PXN
// Skipping 7 for PATH_P2C
// Skipping 8 for PATH_PHB
#define LINK_SYS 9
#define LINK_NET 10
extern const char* topoLinkTypeStr[];

// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection through C2C
#define PATH_C2C 3

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 4

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 5

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
#define PATH_PXN 6

// Connection between a GPU and a NIC using the C2C connection to the CPU and the PCIe connection to the NIC
#define PATH_P2C 7

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 8

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 9

// Connection through the network
#define PATH_NET 10

// New type of path which should precede PATH_PIX
#define PATH_PORT PATH_NVL

// Disconnected
#define PATH_DIS 11
extern const char* topoPathTypeStr[];

struct ncclTopoNode;
struct ncclTopoLink {
  int type;
  float bw;
  struct ncclTopoNode* remNode;
};
// Allows for up to 32 NICs per node on GB200-NVL72
#define NCCL_TOPO_MAX_LINKS 576
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  float bw;
  int type;
};

#define NCCL_TOPO_UNDEF (-1)

#define NCCL_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff
#define NCCL_TOPO_ID_SYSTEM_ID(id) (id >> 56)
#define NCCL_TOPO_ID_LOCAL_ID(id) (id & NCCL_TOPO_ID_LOCAL_ID_MASK)
#define NCCL_TOPO_LOCAL_NIC_ID(numaid, busid) (((int64_t)numaid << 56) + busid)
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & NCCL_TOPO_ID_LOCAL_ID_MASK))

struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int cudaCompCap;
      int gdrSupport;
    }gpu;
    struct {
      int dev; // Plugin dev number
      uint64_t asic;
      int port;
      float bw;
      float latency;
      int gdrSupport;
      int collSupport;
      int maxChannels;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
    struct {
      uint64_t device;
    }pci;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  int systemId;
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxBw;
  float totalBw;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank);
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min);
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max);
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink);

struct ncclTopoNetState {
  int nVirtualNics;
  int nPhysicalNics;
  const char* name;
};
ncclResult_t ncclTopoProcessNet(ncclXml* xml, int coll, const char* dumpXmlFile, ncclTopoNetState* state, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*devices)(int*), const char* netName, bool dmaBufSupport);

#define NCCL_TOPO_XML_MAX_NODES 256
#define NCCL_GRAPH_XML_MAX_NODES 4096
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, uint64_t localHostHash);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml);

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax);

static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index, bool showWarn) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
      *index = i;
      return ncclSuccess;
    }
  }
  if (showWarn) WARN("ncclTopoRankToIndex could not find rank %d", rank);
  return ncclInternalError;
}

static ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int dev, int* rank) {
  *rank = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id) != system->systemId) continue; // Only consider GPUs on our node
    if (system->nodes[GPU].nodes[i].gpu.dev == dev) {
      *rank = system->nodes[GPU].nodes[i].gpu.rank;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev) {
  *netDev = -1;
  for (int i=0; i<system->nodes[NET].count; i++) {
    if (system->nodes[NET].nodes[i].id == id) {
      *netDev = system->nodes[NET].nodes[i].net.dev;
      return ncclSuccess;
    }
  }
  WARN("Could not find NET with id %lx", id);
  return ncclInternalError;
}

// Returns NVLink bw in GB/s
static float ncclTopoNVLinkBw(int cudaCompCap) {
  return
    cudaCompCap >= 100 ? SM100_NVLINK_BW :
    cudaCompCap >= 90 ? SM90_NVLINK_BW :
    cudaCompCap == 86 ? SM86_NVLINK_BW :
    cudaCompCap >= 80 ? SM80_NVLINK_BW :
    cudaCompCap >= 70 ? SM70_NVLINK_BW :
    cudaCompCap >= 60 ? SM60_NVLINK_BW :
    SM80_NVLINK_BW;
}

// Mirror bits
static bool isPow2(int val) {
  return (val & (val-1)) == 0;
}
static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b=1, mb=(pow2>>1); b<pow2; b<<=1, mb>>=1) if (val & b) mirror |= mb;
  return mirror;
}
#endif
