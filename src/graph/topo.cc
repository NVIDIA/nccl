/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nvmlwrap.h"
#include "net.h"
#include <sys/stat.h>
#include <fcntl.h>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* pathDists[] = { "PIX", "PXB", "PHB", "NODE", "SYS" };

const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };
const char* topoLinkTypeStr[] = { "LOC", "NVL", "PCI", "QPI", "NET" };

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/
static int getNumaId(char *path) {
  char npath[PATH_MAX];
  snprintf(npath, PATH_MAX, "%s/numa_node", path);
  npath[PATH_MAX-1] = '\0';

  int numaId = -1;
  FILE *file = fopen(npath, "r");
  if (file == NULL) return -1;
  if (fscanf(file, "%d", &numaId) == EOF) { fclose(file); return -1; }
  fclose(file);

  return numaId;
}

static ncclResult_t getPciPath(char* busId, char** path) {
  for (int i=0; i<BUSID_SIZE; i++) busId[i] = tolower(busId[i]);
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpy(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  char* str = path+offset;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  NCCLCHECK(busIdToInt64(str, id));
  return ncclSuccess;
}

static ncclResult_t idToIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].id == id) {
      *index = i;
    }
  }
  return ncclSuccess;
}


static ncclResult_t getPath(int64_t id, char** path) {
  char busId[] = "0000:00:00.0";
  NCCLCHECK(int64ToBusId(id, busId));
  NCCLCHECK(getPciPath(busId, path));
  return ncclSuccess;
}

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path) {
  char busId[BUSID_SIZE];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, BUSID_SIZE, cudaDev));
  NCCLCHECK(getPciPath(busId, path));
  return ncclSuccess;
}


int interCpuWidth = 0;
int cpuPciWidth = 0;

static ncclResult_t getCpuWidths() {
  // Check if already detected
  if (interCpuWidth + cpuPciWidth) return ncclSuccess;

  // Defaults
  char cpu[256];
  sprintf(cpu, "Generic");
  cpuPciWidth = interCpuWidth = PCI_WIDTH;

#ifdef __PPC__
  sprintf(cpu, "ppc64");
  interCpuWidth = P9_WIDTH;
#endif
#ifdef __x86_64__
  sprintf(cpu, "x86_64");
  union {
    struct {
      // CPUID 0 String register order
      uint32_t ebx;
      uint32_t edx;
      uint32_t ecx;
    };
    char vendor[12];
  } cpuid0;

  asm volatile("cpuid" : "=b" (cpuid0.ebx), "=c" (cpuid0.ecx), "=d" (cpuid0.edx) : "a" (0));
  if (strncmp(cpuid0.vendor, "GenuineIntel", 12) == 0) sprintf(cpu, "Intel");

  if (strcmp(cpu, "Intel") == 0) {
    union {
      struct {
        int steppingId:4;
        int model:4;
        int familyId:4;
        int processorType:2;
        int resv0:2;
        int extModelId:4;
        int modelId:8;
        int resv1:4;
      };
      uint32_t val;
    } cpuid1;
    asm volatile("cpuid" : "=a" (cpuid1.val) : "a" (1));
    if (cpuid1.familyId == 6 && cpuid1.modelId >= 0x55) { // Skylake
      sprintf(cpu, "Intel/Skylake (or later)");
      interCpuWidth = SKL_QPI_WIDTH;
    } else {
      interCpuWidth = QPI_WIDTH;
    }
  }
#endif
  INFO(NCCL_GRAPH, "%s CPU (PCI %d, InterCpu %d)", cpu, cpuPciWidth, interCpuWidth);
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetInterCpuWidth(int* width) {
  NCCLCHECK(getCpuWidths());
  *width = interCpuWidth;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetCpuPciP2pWidth(int* width) {
  NCCLCHECK(getCpuWidths());
  *width = cpuPciWidth;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetPciWidth(int* width) {
  *width = PCI_WIDTH;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetNetWidth(int* width) {
  *width = NET_WIDTH;
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

static ncclResult_t ncclDeviceType(const char* busId, enum ncclNvLinkDeviceType* type) {
  char classPath[] =  "/sys/bus/pci/devices/0000:00:00.0/class";
  memcpy(classPath+sizeof("/sys/bus/pci/devices/")-1, busId, sizeof("0000:00:00.0")-1);
  char* rPath = realpath(classPath, NULL);
  int fd;
  if ((fd = open(rPath, O_RDONLY)) == -1) {
    // Could not find device. It might be because we're in a VM and
    // we don't see the whole machine. This is handled silently so
    // we don't want to print an INFO error.
    TRACE(NCCL_INIT, "Open of %s failed : %s\n", rPath, strerror(errno));
    return ncclSystemError;
  }
  free(rPath);
  char pciClass[9];
  strncpy(pciClass, "0x000000", 9);
  int len;
  SYSCHECKVAL(read(fd, pciClass, 8), "read", len);
  SYSCHECK(close(fd), "close");
  if (strcmp(pciClass, "0x068000") == 0) {
    // PCI device is of type "Bridge / Other Bridge Device" (NVswitch)
    *type = ncclNvLinkDeviceSwitch;
  } else if (strcmp(pciClass, "0x068001") == 0) {
    // PCI device is of type "Bridge: IBM Device 04ea"
    *type = ncclNvLinkDeviceBridge;
  } else if (strcmp(pciClass, "0x030200") == 0 // "3D Controller" (Tesla)
      || strcmp(pciClass, "0x030000") == 0) {  // "VGA Controller" (GeForce)
    *type = ncclNvLinkDeviceGpu;
  } else {
    *type = ncclNvLinkDeviceUnknown;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectCpu(struct ncclTopoSystem* system, int numaId, struct ncclTopoNode* node, int linkType, int linkWidth) {
  struct ncclTopoNode* cpuNode = NULL;
  for (int c=0; c<system->nodes[CPU].count; c++) {
    if (system->nodes[CPU].nodes[c].id == numaId) cpuNode = system->nodes[CPU].nodes+c;
  }
  if (cpuNode == NULL) { // Create CPU
    NCCLCHECK(ncclTopoCreateNode(system, &cpuNode, CPU, numaId));
  }
  NCCLCHECK(ncclTopoConnectNodes(node, cpuNode, linkType, linkWidth));
  NCCLCHECK(ncclTopoConnectNodes(cpuNode, node, linkType, linkWidth));
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNVLink(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system) {
  struct ncclTopoNode* nvsNode = NULL;

  int minNvlinks = 6, minWidth = VOLTA_NVLINK_WIDTH;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    int cudaMajor, cudaMinor;
    NCCLCHECK(wrapNvmlDeviceGetCudaComputeCapability(nvmlDevs[g], &cudaMajor, &cudaMinor));
    int maxNvLinks, width;
    if (cudaMajor < 6) {
      maxNvLinks = 0;
      width = 0;
    } else if (cudaMajor == 6) {
      maxNvLinks = 4;
      width = PASCAL_NVLINK_WIDTH;
    } else {
      maxNvLinks = 6;
      width = VOLTA_NVLINK_WIDTH;
    }

    int nvlinks = 0;
    for (int l=0; l<maxNvLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDevs[g], l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive;
      if ((wrapNvmlDeviceGetNvLinkState(nvmlDevs[g], l, &isActive) != ncclSuccess) || (isActive != NVML_FEATURE_ENABLED)) continue;

      // Try to figure out what's on the other side of the NVLink
      nvmlPciInfo_t remoteProc;
      if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevs[g], l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        lowerId[c] = tolower(p[c]);
        if (p[c] == 0) break;
      }

      enum ncclNvLinkDeviceType type;
      NCCLCHECK(ncclDeviceType(lowerId, &type));
      if (type == ncclNvLinkDeviceGpu) {
        int64_t remoteId;
        NCCLCHECK(busIdToInt64(lowerId, &remoteId));
        int peer;
        NCCLCHECK(idToIndex(system, remoteId, &peer));
        if (peer != -1) {
          NCCLCHECK(ncclTopoConnectNodes(gpu, system->nodes[GPU].nodes+peer, LINK_NVL, width));
          nvlinks++;
        }
      } else if (type == ncclNvLinkDeviceBridge) {
        // Nvlink between GPU and CPU (PPC)
        // Since the remote bridge does not have a valid numa_node, assume we
        // are connected to the closest CPU.
        char* path;
        NCCLCHECK(getPath(gpu->id, &path));
        int numaId = getNumaId(path);
        free(path);
        NCCLCHECK(ncclTopoConnectCpu(system, numaId, gpu, LINK_NVL, width));
        nvlinks++;
      } else { // Nvswitch
        if (type == ncclNvLinkDeviceUnknown) {
          // The NVLink is up but we couldn't find the PCI device on the other
          // side. Assume it's an NVswitch outside a VM.
          if (l == 0) INFO(NCCL_INIT, "%d/%d -> %s : Assuming NVLink is connected to NVswitch", g, l, lowerId);
        }
        if (nvsNode == NULL) { // Create nvswitch
          NCCLCHECK(ncclTopoCreateNode(system, &nvsNode, NVS, 0));
        }
        NCCLCHECK(ncclTopoConnectNodes(gpu, nvsNode, LINK_NVL, VOLTA_NVLINK_WIDTH));
        NCCLCHECK(ncclTopoConnectNodes(nvsNode, gpu, LINK_NVL, VOLTA_NVLINK_WIDTH));
        nvlinks++;
      }
    }
    minNvlinks = std::min(minNvlinks, nvlinks);
    minWidth = std::min(minWidth, width);
  }
  int pciWidth;
  NCCLCHECK(ncclTopoGetPciWidth(&pciWidth));
  system->maxSpeed = minNvlinks ? minNvlinks*minWidth : pciWidth;
  system->maxWidth = minNvlinks ? minWidth : pciWidth;
  return ncclSuccess;
}

ncclResult_t ncclTopoCreatePciPath(struct ncclTopoSystem* system, struct ncclTopoNode* endNode, char* path) {
  struct ncclTopoNode* lastNode = endNode;
  int pciWidth;
  NCCLCHECK(ncclTopoGetPciWidth(&pciWidth));
  // Find intermediate PCI switches
  int slashCount = 0;
  int offsetRC = 0;
  while (offsetRC < strlen(path)) {
    if (path[offsetRC] == '/') slashCount++;
    if (slashCount == 4) break;
    offsetRC++;
  }
  int offset = strlen(path);
  slashCount = 0;
  while (--offset > offsetRC) {
    if (path[offset] == '/') {
      slashCount++;
      // Find if already existing
      if ((slashCount%2) == 0) {
        int64_t pciId;
        NCCLCHECK(pciPathToInt64(path, offset, offsetRC, &pciId));
        for (int p=0; p<system->nodes[PCI].count; p++) {
          if (system->nodes[PCI].nodes[p].id == pciId) {
            // Found our PCI switch. Attach and stop since the rest should already
            // be connected
            NCCLCHECK(ncclTopoConnectNodes(system->nodes[PCI].nodes+p, lastNode, LINK_PCI, pciWidth));
            NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[PCI].nodes+p, LINK_PCI, pciWidth));
            return ncclSuccess;
          }
        }
        struct ncclTopoNode* pciNode;
        NCCLCHECK(ncclTopoCreateNode(system, &pciNode, PCI, pciId));
        NCCLCHECK(ncclTopoConnectNodes(pciNode, lastNode, LINK_PCI, pciWidth));
        NCCLCHECK(ncclTopoConnectNodes(lastNode, pciNode, LINK_PCI, pciWidth));
        lastNode = pciNode;
      }
    }
  }
  // Then attach to a CPU node
  int numaId = getNumaId(path);
  int width;
  NCCLCHECK(ncclTopoGetCpuPciP2pWidth(&width));
  NCCLCHECK(ncclTopoConnectCpu(system, numaId, lastNode, LINK_PCI, width));
  return ncclSuccess;
}

// Try to detect if IB cards are in fact the same physical NIC, hence sharing ports.
#include <glob.h>
#define IB_GUID_PATH "%s/infiniband/mlx5_*/sys_image_guid"
uint64_t getIbGuid(char* path) {
  uint64_t guid = 0ULL;
  char guidPath[PATH_MAX];
  snprintf(guidPath, PATH_MAX, IB_GUID_PATH, path);
  // PATH has a wildcard in it so use glob()
  glob_t globbuf;
  glob(guidPath, 0, NULL, &globbuf);
  if (globbuf.gl_pathc > 0)
    strncpy(guidPath, globbuf.gl_pathv[0], PATH_MAX);
  globfree(&globbuf);
  guidPath[PATH_MAX-1] = '\0';
  FILE *file = fopen(guidPath, "r");
  if (file != NULL) {
    uint64_t a, b, c, d;
    if (fscanf(file, "%04lx:%04lx:%04lx:%04lx", &a, &b, &c, &d) != EOF) {
      guid = (a << 48) + (b << 32) + (c<<16) + d;
      TRACE(NCCL_GRAPH, "Opened %s guid %lx", guidPath, guid);
    }
    fclose(file);
  }
  return guid;
}

struct netInfo {
  char* path;
  int64_t nic;
  uint64_t asic;
  int port;
  int net;
};

ncclResult_t ncclTopoComputeNetInfo(struct netInfo* netInfos, int ndev) {
  for (int n=0; n<ndev; n++) {
    struct netInfo* info = netInfos+n;
    uint64_t ibGuid;
    info->nic = n;
    info->asic = n;
    info->port = 0;
    info->net = n;
    if (info->path && (ibGuid = getIbGuid(info->path)) != 0) {
      info->asic = ibGuid;

      // Ignore PCI subdevice when computing the ID to merge multi-port cards
      // and make them use the same PCI link.
      char* path = strdup(info->path);
      path[strlen(path)-1]='0';
      NCCLCHECK(pciPathToInt64(path, strlen(path), 0, &info->nic));
      free(path);

      // Same PCI path -> different ports of the same NIC
      for (int i=0; i<n; i++) if (netInfos[i].nic == info->nic) info->port++;

      // Same GUID -> same network links as the other NIC
      for (int i=0; i<n; i++) if (netInfos[i].asic == info->asic && netInfos[i].port == info->port) info->net = netInfos[i].net;
    }
    INFO(NCCL_GRAPH, "%s -> %x/%lx/%d/%d", info->path, info->nic, info->asic, info->port, info->net);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectPCI(struct ncclTopoSystem* system) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    char* path;
    NCCLCHECK(getPath(gpu->id, &path));
    NCCLCHECK(ncclTopoCreatePciPath(system, gpu, path));
    free(path);
  }

  // Connect the NICs
  int netDevCount;
  NCCLCHECK(ncclNetDevices(&netDevCount));
  int netWidth;
  NCCLCHECK(ncclTopoGetNetWidth(&netWidth));

  struct netInfo* netInfos;
  NCCLCHECK(ncclCalloc(&netInfos, netDevCount));

  for (int n=0; n<netDevCount; n++) {
    ncclResult_t res = ncclNetPciPath(n, &netInfos[n].path);
    if (res != ncclSuccess) netInfos[n].path = NULL;
  }

  NCCLCHECK(ncclTopoComputeNetInfo(netInfos, netDevCount));

  for (int n=0; n<netDevCount; n++) {
    struct netInfo* info = netInfos+n;
    // Create NIC and attach it to the PCI tree
    struct ncclTopoNode* nicNode = NULL;
    for (int i=0; i<system->nodes[NIC].count; i++) {
      if (system->nodes[NIC].nodes[i].id == info->nic) {
        nicNode = system->nodes[NIC].nodes+i;
        break;
      }
    }
    if (!nicNode) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, NIC, info->nic));
      if (info->path) {
        // Create the PCI path
        NCCLCHECK(ncclTopoCreatePciPath(system, nicNode, info->path));
      } else {
        // This is probably a virtual NIC. Just attach it directly to CPU 0
        int width;
        NCCLCHECK(ncclTopoGetCpuPciP2pWidth(&width));
        NCCLCHECK(ncclTopoConnectCpu(system, 0, nicNode, LINK_PCI, width));
      }
    }
    free(info->path);

    // Create the network side
    struct ncclTopoNode* netNode;
    NCCLCHECK(ncclTopoCreateNode(system, &netNode, NET, n));

    // Use rank to store the net information
    netNode->rank = info->net;

    NCCLCHECK(ncclTopoConnectNodes(nicNode, netNode, LINK_NET, netWidth));
    NCCLCHECK(ncclTopoConnectNodes(netNode, nicNode, LINK_NET, netWidth));
  }
  free(netInfos);

  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    for (int p=0; p<system->nodes[CPU].count; p++) {
      if (n == p) continue;
      int width;
      NCCLCHECK(ncclTopoGetInterCpuWidth(&width));
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, system->nodes[CPU].nodes+p, LINK_QPI, width));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%lX (%d)", topoNodeTypeStr[node->type], node->id, node->rank);
  } else {
    sprintf(line+offset, "%s/%lX", topoNodeTypeStr[node->type], node->id);
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) continue;
    if (link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2d] - ", topoLinkTypeStr[link->type], link->width);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line+nextOffset, "%s/%lX (%d)", topoNodeTypeStr[link->remNode->type], link->remNode->id, link->remNode->rank);
        } else {
          sprintf(line+nextOffset, "%s/%lX", topoNodeTypeStr[link->remNode->type], link->remNode->id);
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxWidth %2d maxSpeed %2d ===", s->maxWidth, s->maxSpeed);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. QPI (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system) {
  struct ncclTopoSystem* s;
  NCCLCHECK(ncclCalloc(&s, 1));
  nvmlDevice_t* nvmlDevs;
  int g = 0;
  NCCLCHECK(ncclCalloc(&nvmlDevs, comm->nRanks));
  for (int r=0; r<comm->nRanks; r++) {
    if (comm->peerInfo[r].hostHash == comm->peerInfo[comm->rank].hostHash) {
      // Consider the GPU as outside of our node if we can't see it through NVML.
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      NCCLCHECK(int64ToBusId(comm->peerInfo[r].busId, busId));
      if (wrapNvmlDeviceGetHandleByPciBusId(busId, nvmlDevs+g) != ncclSuccess) continue;
      g++;
      struct ncclTopoNode* gpuNode;
      NCCLCHECK(ncclTopoCreateNode(s, &gpuNode, GPU, comm->peerInfo[r].busId));
      gpuNode->rank = r;
    }
  }

  NCCLCHECK(ncclTopoConnectNVLink(nvmlDevs, s));
  NCCLCHECK(ncclTopoConnectPCI(s));

  free(nvmlDevs);
  NCCLCHECK(ncclTopoSortSystem(s));
  *system = s;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvlink(struct ncclTopoSystem* system, int64_t busId1, int64_t busId2, int* nvlink) {
  int g1, g2;
  NCCLCHECK(idToIndex(system, busId1, &g1));
  NCCLCHECK(idToIndex(system, busId2, &g2));
  *nvlink = g1 != -1 && g2 != -1 && system->nodes[GPU].nodes[g1].paths[GPU][g2].type == LINK_NVL;
  return ncclSuccess;
}

ncclResult_t ncclTopoHasNvlink(struct ncclTopoSystem* system, int64_t busId, int* nvlink) {
  int g;
  NCCLCHECK(idToIndex(system, busId, &g));
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (i == g) continue;
    if (system->nodes[GPU].nodes[g].paths[GPU][i].type == LINK_NVL) {
      *nvlink = 1;
      return ncclSuccess;
    }
  }
  *nvlink = 0;
  return ncclSuccess;
}

static int pathDistance(struct ncclTopoLinkList* links) {
  int distance = PATH_PIX;
  if (links->count > 2) distance = PATH_PXB;
  for (int l=0; l<links->count; l++) {
    // PHB if we go through 1 CPU, SYS if we go through 2 CPUs
    if (links->list[l]->remNode->type == CPU) distance = (distance == PATH_PHB) ? PATH_SYS : PATH_PHB;
  }
  return distance;
}

ncclResult_t ncclTopoGpuDistance(struct ncclTopoSystem* system, int64_t busId1, int64_t busId2, int* distance) {
  int g1, g2;
  NCCLCHECK(idToIndex(system, busId1, &g1));
  NCCLCHECK(idToIndex(system, busId2, &g2));
  *distance = pathDistance(system->nodes[GPU].nodes[g1].paths[GPU]+g2);
  return ncclSuccess;
}

ncclResult_t ncclTopoNetDistance(struct ncclTopoSystem* system, int64_t busId, int netDev, int* distance) {
  int g;
  NCCLCHECK(idToIndex(system, busId, &g));
  *distance = pathDistance(system->nodes[GPU].nodes[g].paths[NET]+netDev);
  return ncclSuccess;
}

ncclResult_t ncclTopoCpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[CPU].count;
  return ncclSuccess;
}
