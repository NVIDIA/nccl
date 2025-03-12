/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nvmlwrap.h"
#include "net.h"
#include "coll_net.h"
#include "transport.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "xml.h"
#include "cpuset.h"
#include "bootstrap.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };
const char* topoLinkTypeStr[] = { "LOC", "NVL", "",    "C2C", "PCI",    "",    "",    "", "SYS", "NET" };
const char* topoPathTypeStr[] = { "LOC", "NVL", "NVB", "C2C", "PIX", "PXB", "PXN", "PHB", "SYS", "NET", "DIS" };

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  char* str = path+offset;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  int64_t numid;
  NCCLCHECK(busIdToInt64(str, &numid));
  // Ignore subdevice because those should use the same PCI link so we want to merge nodes.
  numid -= numid & 0xf;
  *id = numid;
  return ncclSuccess;
}

static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from) {
  *cpu = NULL;
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  for (int l=0; l<node->nlinks; l++) {
    // Go up the PCI tree to find the CPU. Follow only PCI switches.
    if (node->links[l].type == LINK_PCI
	&& node->links[l].remNode != from
	&& (node->links[l].remNode->type == PCI
	    || node->links[l].remNode->type == CPU)) {
      NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu, node));
    }
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuBw = 0;
int cpuPciBw = 0;

static ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
  *bw = LOC_BW;
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *bw = P9_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *bw = ARM_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *bw =
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_ERP ? ERP_QPI_BW :
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SRP ? SRP_QPI_BW :
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SKL ? SKL_QPI_BW :
      BDW_QPI_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    *bw = AMD_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *bw = cpu->cpu.model ==  NCCL_TOPO_CPU_MODEL_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    free(delNode->paths[t]);
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      if (node == delNode) continue;
      for (int l=0; l<node->nlinks; l++) {
        while (l<node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (l<node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
  // Aggregate links into higher bw for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link - node->links == NCCL_TOPO_MAX_LINKS) {
    WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
    return ncclInternalError;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;

  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link-1)->bw >= linkSave.bw) break;
    memcpy(link, link-1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
int getBcmGen(uint64_t id, int level) {
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000) return 4;
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level*0x1000)) return 5;
  return 0;
}
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  ncclResult_t ret = ncclSuccess;
  for (int s=0; s<system->nodes[PCI].count; s++) {
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes+s;
    int gen = getBcmGen(pciSwitch->pci.device, 0);
    // Flatten Gen4 PEX switches in base mode
    if (gen) {
      // Find sub switches with the same device ID.
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l=0; l<pciSwitch->nlinks; l++) {
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen) continue;
        // Save sub switch for later
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        memmove(pciSwitch->links+l, pciSwitch->links+l+1, (pciSwitch->nlinks-l-1)*(sizeof(struct ncclTopoLink)));
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        l--;
      }

      for (int s=0; s<subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), ret, fail);
        struct ncclTopoNode* sub = system->nodes[PCI].nodes+index;
        // Connect all sub PCI devices to the parent switch
        for (int l=0; l<sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;
          // Add link from parent PCI switch -> PCI device
          if (pciSwitch->nlinks == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            ret = ncclInternalError;
            goto fail;
          }
          memcpy(pciSwitch->links+pciSwitch->nlinks, sub->links+l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          for (int rl=0; rl<remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), ret, fail);
      }
      // Set subdevice to 0xffff to make sure we don't merge this switch again.
      pciSwitch->pci.device |= 0xffff;
      free(subSwIds);
      // Restart, as system->nodes[PCI].nodes has changed.
      s = 0;
      continue;
fail:
      free(subSwIds);
      return ret;
    }
  }
  return ret;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes+n;
    for (int p=0; p<system->nodes[CPU].count; p++) {
      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes+p;
      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      float bw;
      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%lx-%lx (%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->gpu.rank);
  } else if (node->type == CPU) {
    sprintf(line+offset, "%s/%lx-%lx (%d/%d/%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line+offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else {
    sprintf(line+offset, "%s/%lx-%lx", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id), NCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) {
      sprintf(line+offset, "+ %s[%2.1f] - %s/%lx-%lx", topoLinkTypeStr[link->type], link->bw, topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
      INFO(NCCL_GRAPH, "%s", line);
    } else if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line+nextOffset, "%s/%lx-%lx (%d/%lx/%d/%f)", topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id), link->remNode->net.collSupport, link->remNode->net.asic, link->remNode->net.port, link->remNode->net.bw);
        } else {
          sprintf(line+nextOffset, "%s/%lx-%lx", topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
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
// 4. SYS (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, NCCL_TOPO_ID(systemId, dev)));
  net->net.dev = dev;
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  if (str) sscanf(str, "0x%lx", &net->net.asic);
  else net->net.asic = dev;

  ncclDebugNoWarn = NCCL_GRAPH;
  int mbps;
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0));
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;
  if (xmlGetAttrFloat(xmlNet, "latency", &net->net.latency) != ncclSuccess) net->net.latency = 0;
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0));
  ncclDebugNoWarn = 0;

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  for (int s=0; s<xmlNic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    // This means that the "dev" attribute wasn't set on this net xml node. That means it should not be added to the system topology graph
    if (index == -1) continue;
    NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  // Do not go any further, nvlinks will be added in a second pass
  return ncclSuccess;
}

struct kvDict kvDictPciClass[] = { { "0x060400", PCI }, { "0x068000", NVS }, { "0x068001", CPU }, { "0x03", GPU }, { "0x02", NIC }, { NULL, PCI /* Default fallback value */ } };
struct kvDict kvDictPciGen[] = {
  { "2.5 GT/s", 15 }, { "5 GT/s", 30 }, { "8 GT/s", 60 }, { "16 GT/s", 120 }, { "32 GT/s", 240 }, /* Kernel 5.6 and earlier */
  { "2.5 GT/s PCIe", 15 }, { "5.0 GT/s PCIe", 30 }, { "8.0 GT/s PCIe", 60 }, { "16.0 GT/s PCIe", 120 }, { "32.0 GT/s PCIe", 240 }, { "64.0 GT/s PCIe", 480 },
  { NULL, 60 /* Default fallback */ } }; // x100 Mbps per lane
ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId, int numaId) {
  const char* str;

  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlGpu = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
  if (xmlGpu != NULL) {
    type = GPU;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    if (index == -1) return ncclSuccess;
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, node));
  }
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    struct ncclTopoNode* nicNode = NULL;
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
      node = nicNode; // Connect it to parent later on
    }
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
  } else if (type == PCI) {
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 48;
    NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 32;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 16;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0);

    for (int s=0; s<xmlPci->nSubs; s++) {
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      if (strcmp(xmlSubPci->name, "pcilink") != 0) { // PCI links will be added later
        NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
      }
    }
  }

  if (node) {
    int width, speed;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));

    // Manage cases where speed was not indicated in /sys
    if (width == 0) width = 16;
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)

    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, width*speed/80.0));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, width*speed/80.0));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuArch[] = { { "x86_64", NCCL_TOPO_CPU_ARCH_X86 }, { "arm64", NCCL_TOPO_CPU_ARCH_ARM }, { "ppc64", NCCL_TOPO_CPU_ARCH_POWER }, { NULL, 0 } };
struct kvDict kvDictCpuVendor[] = { { "GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL }, { "AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD }, { "CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { "  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { NULL, 0 } };

ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  int systemId;
  for (systemId=0; systemId<system->nHosts; systemId++) if (system->hostHashes[systemId] == hostHash) break;
  if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;
  *systemIdPtr = systemId;
  return ncclSuccess;
}


ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  int systemId;
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.model =
        (familyId == 6 && modelId >= 0xCF) ? NCCL_TOPO_CPU_MODEL_INTEL_ERP :
        (familyId == 6 && modelId >= 0x8F) ? NCCL_TOPO_CPU_MODEL_INTEL_SRP :
        (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_MODEL_INTEL_SKL :
        NCCL_TOPO_CPU_MODEL_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_MODEL_YONGFENG;
    }
  }
  for (int s=0; s<xmlCpu->nSubs; s++) {
    struct ncclXmlNode* node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    if (strcmp(node->name, "nic") == 0) {
      struct ncclTopoNode* nic = NULL;
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, 0);
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "nvlink") == 0) {
    struct ncclTopoNode* gpu = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    int count;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    const char* targetClass;
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    int targetType;
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    struct ncclTopoNode* remote = NULL;
    if (targetType == GPU) {
      // NVL P2P connection to another GPU
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      NCCLCHECK(ncclTopoGetNode(system, &remote, GPU, NCCL_TOPO_ID(systemId, busId)));
    } else if (targetType == CPU) {
      // NVL connection to the local CPU
      NCCLCHECK(findLocalCpu(gpu, &remote, NULL));
    } else {
      if (system->nodes[NVS].count == 0) {
        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
      } else {
        remote = system->nodes[NVS].nodes;
      }
    }
    if (remote) {
      float nvlBw = ncclTopoNVLinkBw(gpu->gpu.cudaCompCap);
      NCCLCHECK(ncclTopoConnectNodes(gpu, remote, LINK_NVL, count*nvlBw));
      if (remote->type != GPU) {
        NCCLCHECK(ncclTopoConnectNodes(remote, gpu, LINK_NVL, count*nvlBw));
      }
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddPciLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "pcilink") == 0) {
    struct ncclTopoNode* pci = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &pci, PCI, pBusId));
    if (pci == NULL) {
      WARN("Add PCI Link error : could not find PCI SW %lx", pBusId);
      return ncclInternalError;
    }
    struct ncclTopoNode* remote = NULL;
    const char* target;
    NCCLCHECK(xmlGetAttrStr(node, "target", &target));
    int64_t busId;
    NCCLCHECK(busIdToInt64(target, &busId));
    NCCLCHECK(ncclTopoGetNode(system, &remote, PCI, NCCL_TOPO_ID(systemId, busId)));
    if (remote) NCCLCHECK(ncclTopoConnectNodes(pci, remote, LINK_LOC, LOC_BW));
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddPciLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}


ncclResult_t ncclTopoAddC2c(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "c2c") == 0) {
    struct ncclTopoNode* gpu = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    int count = 0;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    int bw = 0;
    NCCLCHECK(xmlGetAttrInt(node, "bw", &bw));
    double c2cBw = (bw*count)/1000.0;
    struct ncclTopoNode* cpu = NULL;
    NCCLCHECK(findLocalCpu(gpu, &cpu, NULL));
    if (cpu == NULL) return ncclSuccess;
    NCCLCHECK(ncclTopoConnectNodes(gpu, cpu, LINK_C2C, c2cBw));
    NCCLCHECK(ncclTopoConnectNodes(cpu, gpu, LINK_C2C, c2cBw));
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddC2c(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, const uint64_t localHostHash) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));
  struct ncclTopoSystem* system = *topoSystem;
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }
  for (int systemId=0; systemId<system->nHosts; systemId++) if (system->hostHashes[systemId] == localHostHash) system->systemId = systemId;

  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddC2c(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoAddPciLinks(topNode, *topoSystem, NULL, 0));

  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// Only set values if not already set
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoRefreshBcmP2pLinks(void) {
  //refresh the switch topology by reading the link below
  FILE *fp = fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r");
  if (fp != NULL) {
    int tmp;
    size_t r = fread(&tmp, sizeof(tmp), 1, fp);
    if (r != 1)
      INFO(NCCL_GRAPH, "Failed to read refresh_switch_toplogy");
    fclose(fp);
  }
  return ncclSuccess;
}

// This is just checking for direct descendence
int ncclTopoCheckPix(ncclXmlNode* common, ncclXmlNode** nodes, int nNodes) {
  const char* tempBusId;
  // If the common parent isn't a pci switch, then this isn't PIX
  NCCLCHECK(xmlGetAttrStr(common, "busid", &tempBusId));
  if (tempBusId == NULL) return 0;
  TRACE(NCCL_GRAPH, "Checking pix for busid=%s", tempBusId);

  // All the nodes must have a "nic" which is a parent, and then a pci node (busid) which must be a child of the "common"
  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* node = nodes[i];
    if (strcmp(node->name, "net") == 0) {
      node = node->parent;
      if (node == NULL) return 0;
      if (strcmp(node->name, "nic") == 0) {
        node = node->parent;
        if (node == NULL) return 0;
        // All nodes must descend from the same first level pci switch
        if (strcmp(node->name, "pci") == 0) {
          TRACE(NCCL_GRAPH, "Comparing parent of node=%p to common=%p", node->parent, common);
          if (node->parent != common) return 0;
        }
      }
    }
  }

  return 1;
}

#define NCCL_TOPO_XML_DEPTH_MAX 256
typedef struct xmlNodeStack {
  ncclXmlNode* elems[NCCL_TOPO_XML_DEPTH_MAX];
  int tail;

  ncclXmlNode* top() {
    if (!empty()) {
      return elems[tail - 1];
    } else {
      return NULL;
    }
  }

  ncclXmlNode* pop() {
    ncclXmlNode* node = top();
    if (node) {
      tail--;
    }
    return node;
  }

  void push(ncclXmlNode* node) {
    if (tail < NCCL_TOPO_XML_DEPTH_MAX) {
      elems[tail++] = node;
    }
  }

  bool empty() {
    return tail == 0;
  }

} xmlNodeStack;

// 1. Find the common parent xmlNode between the given set of nodes
ncclResult_t ncclTopoGetPath(ncclXmlNode** nodes, int nNodes, int* path, ncclXmlNode** parent) {
  // Track a stack of parents per-net node being merged
  xmlNodeStack* parents;
  NCCLCHECK(ncclCalloc(&parents, nNodes));
  // Find the common parent
  ncclXmlNode* common = NULL;

  if (nNodes == 1) {
    common = nodes[0];
    *path = PATH_LOC;
    goto out;
  }

  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* temp;
    temp = nodes[i];
    while (temp) {
      parents[i].push(temp);
      temp = strcmp(temp->name, "system") == 0 ? NULL : temp->parent;
    }
  }

  common = NULL;
  int c;
  c = 1;
  while (c && !parents[0].empty()) {
    ncclXmlNode* temp = parents[0].top();
    for (int i = 1; i < nNodes; i++) {
      if (!parents[i].empty()) {
        c &= (temp == parents[i].top());
      } else {
        c = 0;
        break;
      }
    }

    if (c) {
      common = temp;
      if (common == NULL) TRACE(NCCL_GRAPH, "COMMON IS NULL");
      for (int i = 0; i < nNodes; i++) {
        parents[i].pop();
      }
    // Check multi-port while we still have the mismatched parents
    // For multi-port to be true, all parents (peers) must have the busId attribute with all but the last character matching
    } else {
      int multiPort = 1;
      const char* tempBusId;

      NCCLCHECK(xmlGetAttr(temp, "busid", &tempBusId));
      if (tempBusId) {
        for (int i = 1; i < nNodes; i++) {
          if (!parents[i].empty()) {
            const char* busId;
            NCCLCHECK(xmlGetAttr(parents[i].top(), "busid", &busId));
            if (busId) {
              if (strlen(busId) != strlen(tempBusId)) {
                multiPort = 0;
                break;
              }
              if (strncmp(busId, tempBusId, strlen(busId)-1) != 0) {
                multiPort = 0;
                break;
              }
            } else {
              multiPort = 0;
              break;
            }
          }
        }
      } else {
        multiPort = 0;
      }

      if (multiPort) {
        *path = PATH_PORT;
        goto out;
      }
    }
  }

  if (common == NULL) {
    *path = PATH_DIS;
  } else if (strcmp(common->name,"system") == 0) {
    *path = PATH_SYS;
  } else if (strcmp(common->name, "cpu") == 0) {
    *path = PATH_PHB;
  } else if (strcmp(common->name, "nic") == 0) {
    *path = PATH_PORT;
  } else if (strcmp(common->name, "net") == 0) {
    *path = PATH_PORT;
  } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
    *path = PATH_PIX;
  } else {
    *path = PATH_PXB;
  }

out:
  *parent = common;
  free(parents);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeUniqueBusId(struct ncclXml* xml, char* busId, struct ncclXmlNode** pciNode, struct ncclXmlNode* parent) {
  int i = 0;
  int64_t rBusId;
  NCCLCHECK(busIdToInt64(busId, &rBusId));
  // Try to find an unused busid - NCCL expects leaf busid to be unique
  while (i < 100) {
    rBusId++;
    TRACE(NCCL_GRAPH, "Trying to make new busId %lx", rBusId);
    int64ToBusId(rBusId, busId);
    struct ncclXmlNode* temp = NULL;
    NCCLCHECK(xmlFindTagKv(xml, "pci", &temp, "busid", busId));
    if (temp == NULL) {
      NCCLCHECK(xmlAddNode(xml, parent, "pci", pciNode));
      NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
      TRACE(NCCL_GRAPH, "Made new busId %lx", rBusId);
      return ncclSuccess;
    }
    TRACE(NCCL_GRAPH, "Conflicting busId %lx", rBusId);
    i++;
  }

  WARN("TOPO/NET : Couldn't generate unique busId after %d tries", i);
  return ncclInternalError;
}

ncclResult_t ncclTopoMakePciParent(struct ncclXml* xml, struct ncclXmlNode** parent, struct ncclXmlNode* physNetNode) {
  struct ncclXmlNode* newBusId = NULL;
  struct ncclXmlNode* pci = physNetNode->parent;
  if (pci) {
    pci = pci->parent;
    if (pci) {
      if (strcmp(pci->name, "pci") == 0) {
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        memset(busId, 0, sizeof(busId));
        const char* originalBusId;
        // Seed busId with the current NIC 0's busId to make discovering a unique hash quicker
        NCCLCHECK(xmlGetAttrStr(pci, "busid", &originalBusId));
        snprintf(busId, sizeof(busId), "%s", originalBusId);
        NCCLCHECK(ncclTopoMakeUniqueBusId(xml, busId, &newBusId, *parent));
        for (int i = 0; i < pci->nAttrs; i++) {
          NCCLCHECK(xmlSetAttr(newBusId, pci->attrs[i].key, pci->attrs[i].value));
        }
        NCCLCHECK(xmlSetAttr(newBusId, "busid", busId));
        *parent = newBusId;
      }
    }
  }

  if (newBusId == NULL) {
    const char* name;
    NCCLCHECK(xmlGetAttr(physNetNode, "name", &name));
    WARN("TOPO/NET : Can't find busId of child 0 %s", name);
    return ncclInternalError;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVnic(ncclComm_t comm, struct ncclXml* xml, ncclNetVDeviceProps_t* vProps,
struct ncclXmlNode** physNetNodes, struct ncclXmlNode** netNode, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*)) {
  if (vProps->ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
    WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps->ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
    return ncclInternalError;
  }

  // Trigger the merge, then get the new device's properties
  int vDevIndex = 0;
  ncclResult_t ret = makeVDevice(&vDevIndex, vProps);
  if (ret != ncclSuccess) {
    INFO(NCCL_GRAPH|NCCL_INIT|NCCL_NET, "TOPO/NET : Tried merging multiple devices together and failed. vProps={ndevs=%d, devs=[%d %d %d %d]}. Set NCCL_NET_MERGE_LEVEL=LOC to disable NIC fusion.",
      vProps->ndevs, vProps->devs[0], vProps->devs[1], vProps->devs[2], vProps->devs[3]);
    return ret;
  }

  INFO(NCCL_GRAPH, "TOPO/NET : Made vNic %d", vDevIndex);
  return ncclSuccess;
}

ncclResult_t ncclTopoForceMerge(ncclComm_t comm, struct ncclXml* xml, const char* str, int* placedDevs, ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*)) {
  ncclResult_t ret = ncclSuccess;
  INFO(NCCL_ENV|NCCL_NET, "TOPO/NET : Force-fusing NICs using NCCL_NET_FORCE_MERGE=%s", str);
  char* ncStr;
  NCCLCHECK(ncclCalloc(&ncStr, strlen(str)+1));
  strcpy(ncStr, str);
  char* semi_token;
  char* semi = strtok_r(ncStr, ";", &semi_token);
  while (semi) {
    TRACE(NCCL_NET, "Fusing %s", semi);
    struct netIf userIfs[NCCL_NET_MAX_DEVS_PER_NIC];
    int nUserIfs = parseStringList(semi, userIfs, NCCL_NET_MAX_DEVS_PER_NIC);
    if (nUserIfs == 0) {
      INFO(NCCL_NET, "NET/IB : Invalid NCCL_NET_FORCE_MERGE specified %s. Couldn't parse substring %s. Please provide a semicolon-delimited list of comma-delimited NIC groups.",
        ncStr, semi);
      continue;
    }

    ncclNetVDeviceProps_t vProps = {0};
    for (int d = 0; d < nPhysDevs; d++) {
      if (matchIfList(propsList[d].name, propsList[d].port, userIfs, nUserIfs, 1)) {
        vProps.devs[vProps.ndevs++] = d;
      }
    }

    if (vProps.ndevs != nUserIfs) {
      WARN("TOPO/NET : Only matched %d devices, %d requested from %s",
        vProps.ndevs, nUserIfs, semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    if (vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
      WARN("Specified fused NIC %s which has too many devices (%d). Max %d", semi, vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
      ret = ncclInvalidUsage;
      goto fail;
    }

    struct ncclXmlNode* netNode;
    ret = ncclTopoMakeVnic(comm, xml, &vProps, physNetNodes, &netNode, makeVDevice);
    if (ret == ncclSuccess) {
      // Only set that a device is "placed" after successfully making a vNic (it's possible to exit before this)
      for (int i = 0; i < vProps.ndevs; i++) {
        placedDevs[vProps.devs[i]] = 1;
      }
    } else {
      WARN("TOPO/NET : Could not force merge NICs %s. Please specify a valid NCCL_NET_FORCE_MERGE string.", semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    semi = strtok_r(NULL, ";", &semi_token);;
  }

exit:
  free(ncStr);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoAutoMerge(ncclComm_t comm, struct ncclXml* xml, int mergeLevel, int* placedDevs, ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*)) {
  // Compute the path type between each device
  int* paths = NULL;
  ncclResult_t res = ncclSuccess;
  ncclCalloc(&paths, nPhysDevs*nPhysDevs);
  TRACE(NCCL_GRAPH, "Allocated %d paths", nPhysDevs*nPhysDevs);
  for (int i = 0; i < nPhysDevs; i++) {
    for (int j = 0; j < nPhysDevs; j++) {
      struct ncclXmlNode* nodes[2];
      nodes[0] = physNetNodes[i];
      nodes[1] = physNetNodes[j];
      struct ncclXmlNode* parent;
      NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i*nPhysDevs + j], &parent), res, out);
    }
  }

  // Place all remaining physical devices into a virtual device given the mergeLevel criteria
  for (int i = 0; i < nPhysDevs; i++) {
    // Select the first unplaced device "i" as the root
    if (placedDevs[i] == 0) {
      // Init a new vDevice
      ncclNetVDeviceProps_t vProps;
      vProps = {0};
      vProps.devs[vProps.ndevs++] = i;
      placedDevs[i] = 1;
      TRACE(NCCL_GRAPH, "Placed dev %d", i);

      // Select each unplaced device "j" which is at most "mergeLevel" distance from "i", but not equal to "i"
      // (Don't merge the same device with itself)
      for (int j = 0; j < nPhysDevs; j++) {
        if (paths[i*nPhysDevs + j] <= mergeLevel &&
        placedDevs[j] == 0 && j != i) {
          vProps.devs[vProps.ndevs++] = j;
          placedDevs[j] = 1;
          TRACE(NCCL_GRAPH, "Placed dev %d path=%d", j, paths[i*nPhysDevs + j] );
        }
        if (vProps.ndevs == NCCL_NET_MAX_DEVS_PER_NIC) break;
      }

      if (vProps.ndevs > NCCL_NET_MAX_DEVS_PER_NIC) {
        WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps.ndevs, NCCL_NET_MAX_DEVS_PER_NIC);
        return ncclInternalError;
      }

      struct ncclXmlNode* netNode;
      ncclResult_t ret = ncclTopoMakeVnic(comm, xml, &vProps, physNetNodes, &netNode, makeVDevice);

      // Merging failed.
      // Mark all as unplaced and increase their distance to disconnected (PATH_DIS)
      // Set i to 0 to restart the automatic merging process and ensure all are placed
      if (ret != ncclSuccess) {
        INFO(NCCL_GRAPH|NCCL_INIT|NCCL_NET, "Marking physical devices as unplaced, increasing distance and restarting search.");
        placedDevs[i] = 0;
        TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, keeping distance -> self as PATH_LOC", i);
        for (int k = 1; k < vProps.ndevs; k++) {
          int dev = vProps.devs[k];
          placedDevs[dev] = 0;
          paths[i*nPhysDevs + dev] = PATH_DIS;
          paths[dev*nPhysDevs + i] = PATH_DIS;
          TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, setting distance -> %d as PATH_DIS", dev, i);
        }
        i = 0;
      }
    }
  }

out:
  free(paths);
  return res;
}

struct kvDict nicPathKvList[] = {
  { "LOC",  PATH_LOC },
  { "PORT", PATH_PORT },
  { "PIX",  PATH_PIX },
  { "PXB",  PATH_PXB },
  { "PXN",  PATH_PXN },
  { "PHB",  PATH_PHB },
  { "SYS",  PATH_SYS },
  { NULL, 0 }
};

ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
  ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
  ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
  for (int i = 0; i < vProps->ndevs; i++) {
    NCCLCHECK(getProperties(vProps->devs[i], props + i));
    struct ncclXmlNode* physNetNode;
    NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name));
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i,  props[i].name);
  }

  int path = PATH_LOC;
  NCCLCHECK(ncclTopoGetPath(physNetNodes, vProps->ndevs, &path, parent));
  if (path == PATH_LOC) {
    *parent = NULL;
  } else if (parent && strcmp((*parent)->name, "pci") == 0) {
    // If the common parent is PCI, we must reparent the new NIC under a made up busId
    NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
  }
  TRACE(NCCL_GRAPH, "Selected parent %s with path %d", (*parent)->name, path);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVNics(ncclComm_t comm, struct ncclXml* xml, ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*getProperties)(int, ncclNetProperties_t*), int physicalDevs) {
  int* placedDevs = NULL;
  struct ncclXmlNode** physNetNodes = NULL;
  if (physicalDevs == 0) return ncclSuccess;

  ncclCalloc(&physNetNodes, physicalDevs);
  ncclResult_t res = ncclSuccess;

  ncclNetProperties_t* props = NULL;
  ncclCalloc(&props, physicalDevs);
  for (int i = 0; i < physicalDevs; i++) {
    NCCLCHECKGOTO(getProperties(i, props + i), res, out);
    struct ncclXmlNode* physNetNode;
    NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i,  props[i].name);
  }

  // By default, don't merge any devices
  int mergeLevel;
  mergeLevel = PATH_PORT;
  { // Avoids warnings related to jumping to "out"
    const char* mergeLevelEnv = ncclGetEnv("NCCL_NET_MERGE_LEVEL");
    if (mergeLevelEnv) kvConvertToInt(mergeLevelEnv, &mergeLevel, nicPathKvList);
    const char* forceMerge = ncclGetEnv("NCCL_NET_FORCE_MERGE");
    NCCLCHECK(ncclCalloc(&placedDevs, physicalDevs));
    memset(placedDevs, 0, sizeof(int)*physicalDevs);

    if (forceMerge) {
      NCCLCHECKGOTO(ncclTopoForceMerge(comm, xml, forceMerge, placedDevs, props, physNetNodes, physicalDevs, makeVDevice), res, out);
    }
  }
  NCCLCHECKGOTO(ncclTopoAutoMerge(comm, xml, mergeLevel, placedDevs, props, physNetNodes, physicalDevs, makeVDevice), res, out);

out:
  free(physNetNodes);
  free(props);
  if (placedDevs) free(placedDevs);
  return res;
}

static ncclResult_t ncclTopoPopulateNics(ncclComm_t comm, ncclXml* xml, int startIndex, int endIndex, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), const char* netName, int coll, int keep, int virtualNics) {
  for (int n = startIndex; n < endIndex; n++) {
    ncclNetProperties_t props;
    NCCLCHECK(getProperties(n, &props));
    struct ncclXmlNode* netNode = NULL;
    struct ncclXmlNode* parent = NULL;
    if (virtualNics) {
      struct ncclXmlNode* net = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
      // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for this vNIC
      // Only run this if the net doesn't exist locally - this may alter the XML state
      if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, getProperties, &props.vProps, &parent));
    }

    NCCLCHECK(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode, parent));

    const char* colAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));

    // If coll == 0 but the netNode is tagged as coll, don't update the keep value
    if (colAttr == NULL || coll != 0 || strcmp(colAttr,"1") != 0) NCCLCHECK(xmlSetAttrInt(netNode, "keep", keep));
    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    bool gdrSupport = (props.ptrSupport & NCCL_PTR_CUDA) || (comm->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
    INFO(NCCL_NET,"NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netName, gdrSupport ? "Enabled" : "Disabled", n, props.name);
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));
    // Only set coll if it's not 0
    if (coll) NCCLCHECK(xmlInitAttrInt(netNode, "coll", coll));

    const char* keepAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
    NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
    INFO(NCCL_GRAPH, "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s keep=%s coll=%s",
      props.name, props.pciPath, keepAttr, colAttr);
  }

  return ncclSuccess;
}

struct ncclTopoNetState {
  int nVirtualNics;
  int nPhysicalNics;
  const char* name;
};

// Calls to network plugin APIs should be protected. This function should be called inside a per-process lock.
static ncclResult_t ncclTopoProcessNet(ncclComm_t comm, ncclXml* xml, int coll, const char* dumpXmlFile, ncclTopoNetState* state, ncclResult_t (*getProperties)(int, ncclNetProperties_t*), ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*), ncclResult_t (*devices)(int*), const char* netName) {
  int usePhysicalDevices = (dumpXmlFile || makeVDevice == NULL);
  if (state->nPhysicalNics == -1) NCCLCHECK(devices(&state->nPhysicalNics));
  // Enumerate physical devices
  NCCLCHECK(ncclTopoPopulateNics(comm, xml, 0, state->nPhysicalNics, getProperties, netName, coll, 1, 0));
  if (!usePhysicalDevices) {
    if (state->nVirtualNics == -1) {
      NCCLCHECK(ncclTopoMakeVNics(comm, xml, makeVDevice, getProperties, state->nPhysicalNics));
      int nDevs;
      NCCLCHECK(devices(&nDevs));
      state->nVirtualNics = nDevs - state->nPhysicalNics;
    }
    // Remove keep=1 for physical collnets
    if (state->nVirtualNics > 0) {
      NCCLCHECK(ncclTopoPopulateNics(comm, xml, 0, state->nPhysicalNics, getProperties, netName, coll, 0, 0));
      // Populate new devices
      NCCLCHECK(ncclTopoPopulateNics(comm, xml, state->nPhysicalNics, state->nPhysicalNics+state->nVirtualNics, getProperties, netName, coll, 1, 1));
    }
  }

  return ncclSuccess;
}

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
ncclTopoNetState netStates[NCCL_NET_MAX_PLUGINS] = {};
ncclTopoNetState collNetStates[NCCL_NET_MAX_PLUGINS] = {};
ncclResult_t ncclTopoGetSharedState(ncclTopoNetState** state, const char* name, ncclTopoNetState* states) {
  INFO(NCCL_GRAPH, "Retrieving state for %s", name);
  for (int i = 0; i < NCCL_NET_MAX_PLUGINS; i++) {
    // Empty slot
    if (states[i].name == NULL) {
      states[i].nVirtualNics = -1;
      states[i].nPhysicalNics = -1;
      states[i].name = strdup(name);
      *state = states + i;
      INFO(NCCL_GRAPH, "Initialized state %d for %s", i, name);
      return ncclSuccess;
    // Found my slot
    } else if (strcmp(states[i].name, name) == 0) {
      *state = states + i;
      return ncclSuccess;
    }
  }
  WARN("NET/TOPO : Couldn't find net with name %s", name);
  return ncclInternalError;
}

ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXml* xml;
  char* mem = NULL;
  int* localRanks = NULL;
  struct ncclXml* rankXml;
  int localRank = -1, nLocalRanks = 0;
  int netLockHeld = 0;
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
  } else {
    // Try default XML topology location
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }
  if (xml->maxIndex == 0) {
    // Create top tag
    struct ncclXmlNode* top;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ret, fail);
  }

  NCCLCHECKGOTO(ncclTopoRefreshBcmP2pLinks(), ret, fail);

  // Detect only the GPU managed by this process.  We'll get any others through XML fusion.
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  NCCLCHECKGOTO(int64ToBusId(comm->peerInfo[comm->rank].busId, busId), ret, fail);
  struct ncclXmlNode* node;
  NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), ret, fail);
  if (node) {
    NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), ret, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), ret, fail);
  }

  // Auto-detect NICs if needed. net/collnet share the same xml/graph nodes,
  // so we start with collnet so that it has precedence.
  pthread_mutex_lock(&netLock);
  netLockHeld = 1;
  INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
  ncclTopoNetState* state;
  state = NULL;
  if (collNetSupport(comm)) {
    NCCLCHECKGOTO(ncclTopoGetSharedState(&state, comm->ncclCollNet->name, collNetStates), ret, fail);
    NCCLCHECKGOTO(ncclTopoProcessNet(comm, xml, 1, dumpXmlFile, state,
      comm->ncclCollNet->getProperties, comm->ncclCollNet->makeVDevice, comm->ncclCollNet->devices, comm->ncclCollNet->name), ret, fail);
  }
  NCCLCHECKGOTO(ncclTopoGetSharedState(&state, comm->ncclNet->name, netStates), ret, fail);
  NCCLCHECKGOTO(ncclTopoProcessNet(comm, xml, 0, dumpXmlFile, state,
    comm->ncclNet->getProperties, comm->ncclNet->makeVDevice, comm->ncclNet->devices, comm->ncclNet->name), ret, fail);
  pthread_mutex_unlock(&netLock);
  netLockHeld = 0;

  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), ret, fail);

  // XML topo fusion.
  if (comm->MNNVL) {
    // MNNVL clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank)
          localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  rankXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem, xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  if (comm->MNNVL) {
    // Ensure that we have enough room when fusing topos from multiple nodes.
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES), ret, fail);
  } else {
    // In the intra-node case there's no need to enlarge the topo xml.
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }

  if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
  }

  // Only update our topo tracking structure if we aren't dumping (separate steps)
  if (dumpXmlFile == NULL) NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, comm->peerInfo[comm->rank].hostHash), ret, fail);

exit:
  if (!comm->MNNVL && localRanks) free(localRanks);
  if (mem) free(mem);
  free(xml);
  return ret;
fail:
  if (netLockHeld) pthread_mutex_unlock(&netLock);
  goto exit;
}

static ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                                     int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];
  if (paths == NULL) { *localCount = 0; return ncclSuccess; }
  for (int i=0; i<system->nodes[resultType].count; i++) {
    if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;
      minType = paths[i].type;
      if (pathType) *pathType = minType;
      count = 0;
    }
    if (paths[i].bw == maxBw && paths[i].type == minType) {
      if (count == NCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in ncclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             NCCL_TOPO_MAX_NODES, resultType, index, type);
        return ncclInternalError;
      }
      locals[count++] = i;
    }
  }
  *localCount = count;
  return ncclSuccess;
}

ncclResult_t getLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int *count) {
  int localNetCount = 0, netCountByBw = 0;
  int localNets[NCCL_TOPO_MAX_NODES];
  float totalNetBw = 0, gpuBw = 0;

  for (int l=0; l<system->nodes[GPU].nodes[gpu].nlinks; l++) {
    //assuming BW to CPU reflects the GPU bandwidth via P2P or C2C
    //caveat, this could be wrong if there is a PCIe switch,
    //and a narrower link to the CPU
    if (system->nodes[GPU].nodes[gpu].links[l].remNode->type == CPU) {
       gpuBw = system->nodes[GPU].nodes[gpu].links[l].bw;
    }
  }

  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
  for (int l=0; (l < localNetCount) && (totalNetBw < gpuBw); l++, netCountByBw++) {
     totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][localNets[l]].bw;
  }
  *count = netCountByBw;

  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu));

  int localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, NET, localNets, &localNetCount, NULL));
  if (localNetCount==0) {
    WARN("Could not find any local path from gpu %d to net.", gpu);
    return ncclInternalError;
  }

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, localNets[0], GPU, localGpus, &localGpuCount, NULL));

  int net = system->nodes[GPU].nodes[gpu].gpu.dev;
  if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);
  net += channelId%(DIVUP(localNetCount,localGpuCount));
  if (id) *id = system->nodes[NET].nodes[localNets[net%localNetCount]].id;
  if (dev) *dev = system->nodes[NET].nodes[localNets[net%localNetCount]].net.dev;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex) {
  ncclResult_t ret = ncclSuccess;
  int netIndex;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &netIndex));

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL));

  int foundGpu = -1;
  for (int c=0; c<MAXCHANNELS; c++) {
    for (int lg=0; lg<localGpuCount; lg++) {
      int g = localGpus[lg];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      int64_t id;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));
      if (netId == id) {
        foundGpu = g;
        goto exit;
      }
    }
  }
exit:
  *gpuIndex = foundGpu;
  return ret;
}

/****************************/
/* External query functions */
/****************************/

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity) {
  struct ncclTopoNode* cpu = NULL, *gpu = NULL;
  int gpuIndex, cpuIndex;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex));
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex));
  gpu = system->nodes[GPU].nodes+gpuIndex;
  cpu = system->nodes[CPU].nodes+cpuIndex;

  // Query the CPU affinity set we were provided
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&mask, affinityStr));
    TRACE(NCCL_INIT, "Current affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  // Get the affinity of the CPU close to our GPU.
  cpu_set_t cpuMask = cpu->cpu.affinity;

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&cpuMask, affinityStr));
    TRACE(NCCL_INIT, "CPU GPU affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = cpuMask;
  else
    // Use a subset of the GPU affinity set
    CPU_AND(&finalMask, &mask, &cpuMask);

  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&finalMask, affinityStr));
    INFO(NCCL_INIT, "Setting affinity for GPU %d to %s", gpu->gpu.dev, affinityStr);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[GPU].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NVS].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  if (system->nodes[GPU].count == 0) return ncclInternalError;
  int min, max;
  min = max = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  for (int g=1; g<system->nodes[GPU].count; g++) {
    min = std::min(min, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
    max = std::max(max, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
  }
  if (ccMin) *ccMin = min;
  if (ccMax) *ccMax = max;
  return ncclSuccess;
}
