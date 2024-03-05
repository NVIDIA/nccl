/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <float.h>
#include "core.h"
#include "nvmlwrap.h"
#include "xml.h"
#if defined(__x86_64__)
#include <cpuid.h>
#endif

/*******************/
/* XML File Parser */
/*******************/

ncclResult_t xmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name+3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

struct xmlHandler {
  const char * name;
  xmlHandlerFunc_t func;
};

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == MAX_NODES) {
      WARN("Error : XML parser is limited to 1024 nodes");
      return ncclInternalError;
    }
    struct ncclXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) head->subs[head->nSubs++] = node;
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/**************/
/* XML Writer */
/**************/

ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i=0; i<indent; i++) fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a=0; a<node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent+2, file, node->subs[s]));
    }
    for (int i=0; i<indent; i++) fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  NCCLCHECK(ncclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return ncclSuccess;
}

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}
ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "nvlink", ncclTopoXmlLoadNvlink }, { "c2c", ncclTopoXmlLoadC2c } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlLoadNet } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic} };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 3));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "nic", ncclTopoXmlLoadNic } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading topology %s", name);
  else INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = { { "cpu", ncclTopoXmlLoadCpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn) {
  FILE* file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      WARN("Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = { { "system", ncclTopoXmlLoadSystem } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
static void memcpylower(char* dst, const char* src, const size_t size) {
  for (int i=0; i<size; i++) dst[i] = tolower(src[i]);
}
static ncclResult_t getPciPath(const char* busId, char** path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpylower(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetStrFromSys(const char* path, const char* fileName, char* strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue+offset, 1, MAX_STR_LEN-offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
  } else {
    strValue[offset-1] = '\0';
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName, const char* attrName) {
  char strValue[MAX_STR_LEN];
  NCCLCHECK(ncclTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') { NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue)); }
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char* numaId;
    NCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return ncclInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
    NCCLCHECK(ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId:4;
        unsigned modelId:4;
        unsigned familyId:4;
        unsigned processorType:2;
        unsigned resv0:2;
        unsigned extModelId:4;
        unsigned extFamilyId:8;
        unsigned resv1:4;
      };
      uint32_t val;
    } cpuid1;
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
  NCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }
  return ncclSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) { return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')); }
int checkBDFFormat(char* bdf) {
  if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.') return 0;
  if (isHex(bdf[0]) == 0 || isHex(bdf[1] == 0) || isHex(bdf[2] == 0) || isHex(bdf[3] == 0) ||
      isHex(bdf[5] == 0) || isHex(bdf[6] == 0) || isHex(bdf[8] == 0) || isHex(bdf[9] == 0) ||
      isHex(bdf[11] == 0)) return 0;
  return 1;
}

ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
  // Fill info, then parent
  const char* busId;
  NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char* path = NULL;
  ncclDebugNoWarn = NCCL_GRAPH;
  getPciPath(busId, &path);
  ncclDebugNoWarn = 0;

  if (path) {
    NCCLCHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  ncclDebugNoWarn = NCCL_GRAPH;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device");
  }
  ncclDebugNoWarn = 0;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
  if (index == -1) {
    if (path) {
      char deviceSpeedStr[MAX_STR_LEN];
      float deviceSpeed = FLT_MAX;
      NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));
      sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
      char portSpeedStr[MAX_STR_LEN];
      float portSpeed = FLT_MAX;
      NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
      sscanf(portSpeedStr, "%f GT/s", &portSpeed);
      NCCLCHECK(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
    } else {
      NCCLCHECK(xmlSetAttr(pciNode, "link_speed", ""));
    }
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_width", &index));
  if (index == -1) {
    if (path) {
      char strValue[MAX_STR_LEN];
      NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
      int deviceWidth = strtol(strValue, NULL, 0);
      NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
      int portWidth = strtol(strValue, NULL, 0);
      NCCLCHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
    } else {
      NCCLCHECK(xmlSetAttr(pciNode, "link_width", ""));
    }
  }
  struct ncclXmlNode* parent = pciNode->parent;
  if (parent == NULL) {
    if (path) {
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      NCCLCHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
      // switch, or stop if we reach a CPU root complex.
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start>0 && path[start] != '/') start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path+start+1) == 0) {
            // This a CPU root complex. Create a CPU tag and stop there.
            struct ncclXmlNode* topNode;
            NCCLCHECK(xmlFindTag(xml, "system", &topNode));
            NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
            if (parent == NULL) {
              NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
              NCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
            }
          } else if (slashCount == 2) {
            // Continue on the upper PCI switch
            for (int i = strlen(path)-1; i>0; i--) {
              if (path[i] == '/') {
                NCCLCHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));
                if (parent == NULL) {
                  NCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                  NCCLCHECK(xmlSetAttr(parent, "busid", path+i+1));
                }
                break;
              }
            }
          }
        }
        if (parent) break;
      }
    } else {
      // No information on /sys, attach GPU to unknown CPU
      NCCLCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
      if (parent == NULL) {
        struct ncclXmlNode* topNode;
        NCCLCHECK(xmlFindTag(xml, "system", &topNode));
        NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
        NCCLCHECK(xmlSetAttr(parent, "numaid", "-1"));
        NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
      }
    }
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    int subIndex = parent->nSubs;
    const char* newBusId;
    NCCLCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
    for (int s=0; s<parent->nSubs; s++) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
      if (busId != NULL && strcmp(newBusId, busId) < 0) { subIndex = s; break; }
    }
    for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s-1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
  }
  free(path);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, nvmlDevice_t nvmlDev, struct ncclXml* xml, struct ncclXmlNode** gpuNodeRet) {
  struct ncclXmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
  if (gpuNode == NULL) NCCLCHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));

  int index = -1;

  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
  if (index == -1) {
    NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev));
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", dev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
  if (dev == -1) { *gpuNodeRet = NULL; return ncclSuccess; }

  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    int cudaMajor, cudaMinor;
    if (nvmlDev == NULL) {
      cudaDeviceProp devProp;
      CUDACHECK(cudaGetDeviceProperties(&devProp, dev));
      cudaMajor = devProp.major; cudaMinor = devProp.minor;
    } else {
      NCCLCHECK(ncclNvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", cudaMajor*10+cudaMinor));
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  struct ncclXmlNode* nvlNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "nvlink", &nvlNode));
  if (nvlNode == NULL) {
    // NVML NVLink detection
    int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : (sm < 80) ? 6 : (sm < 90) ? 12 : 18;

    if (maxNvLinks > 0 && nvmlDev == NULL) {
      WARN("No NVML device handle. Skipping nvlink detection.");
      maxNvLinks = 0;
    }

    for (int l=0; l<maxNvLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      if ((ncclNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
#if CUDART_VERSION >= 11080
      if (sm >= 90) {
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
        fv.scopeId = l;
        // fv.value will contain NV_FEATURE_ENABLED or NV_FEATURE_DISABLED
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS))
          isActive = (nvmlEnableState_t) fv.value.uiVal;
      } else /* FALLTHRU to GetNvLinkState if before SM90 */
#endif
      {
        (void) ncclNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive);
      }
      if (isActive != NVML_FEATURE_ENABLED) continue;

      // Try to figure out what's on the other side of the NVLink
      nvmlPciInfo_t remoteProc;
      if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        lowerId[c] = tolower(p[c]);
        if (p[c] == 0) break;
      }

      NCCLCHECK(xmlGetSubKv(gpuNode, "nvlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, gpuNode, "nvlink", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count+1));
      }
    }
  }
#if CUDART_VERSION >= 11080
  struct ncclXmlNode* c2cNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "c2c", &c2cNode));
  if (c2cNode == NULL) {
      if (sm >= 90) {
        int c2cLinksCount = 0;
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS)) {
          c2cLinksCount = fv.value.uiVal;
          int bw = 0;
	  int count = 0;
          for (int l=0; l<c2cLinksCount; l++) {
            nvmlFieldValue_t fvs[2];
            fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
            fvs[0].scopeId = l;
            fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
            fvs[1].scopeId = l;
            if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 2, fvs) == ncclSuccess) &&
                (fvs[0].nvmlReturn == NVML_SUCCESS) &&
                (fvs[0].value.uiVal == 1) &&
                (fvs[1].nvmlReturn == NVML_SUCCESS)) {
              bw = fvs[1].value.uiVal;
	      count++;
            }
          }
          if (count > 0) {
            NCCLCHECK(xmlAddNode(xml, gpuNode, "c2c", &c2cNode));
            NCCLCHECK(xmlSetAttrInt(c2cNode, "bw", bw));
            NCCLCHECK(xmlSetAttrInt(c2cNode, "count", count));
          }
        }
      }
  }
#endif
  // Fill target classes
  for (int s=0; s<gpuNode->nSubs; s++) {
    struct ncclXmlNode* sub = gpuNode->subs[s];
    if (strcmp(sub->name, "nvlink") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(sub, "target", &busId));
      char* path;
      ncclDebugNoWarn = NCCL_GRAPH;
      getPciPath(busId, &path);
      ncclDebugNoWarn = 0;
      if (path == NULL || strcmp(busId, "fffffff:ffff:ff") == 0) {
        // Remote NVLink device is not visible inside this VM. Assume NVSwitch.
        NCCLCHECK(xmlSetAttr(sub, "tclass", "0x068000"));
      } else {
        NCCLCHECK(ncclTopoSetAttrFromSys(sub, path, "class", "tclass"));
        free(path);
      }
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
  struct ncclXmlNode* node;
  NCCLCHECK(ncclTopoGetPciNode(xml, busId, &node));
  NCCLCHECK(xmlSetAttrIfUnset(node, "class", "0x03"));
  NCCLCHECK(ncclTopoGetXmlFromSys(node, xml));
  nvmlDevice_t nvmlDev;
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
  return ncclSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
  char subSysPath[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", sysPath);
  char* path = realpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path+offset+1);
    free(path);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode) {
  NCCLCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
  if (*netNode != NULL) return ncclSuccess;

  const char* pciSysPath = pciPath;
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
    // This is not a PCI device (virtual, usb, ...).
    if (strcmp(subSystem, "pci") != 0) {
      INFO(NCCL_GRAPH, "Topology detection: network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct ncclXmlNode* parent = NULL;
  if (pciSysPath) {
    int offset;
    for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    strcpy(busId, pciSysPath+offset+1);
    NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
    NCCLCHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else {
    // Virtual NIC, no PCI device, attach to first CPU
    NCCLCHECK(xmlFindTag(xml, "cpu", &parent));
  }

  struct ncclXmlNode* nicNode = NULL;
  NCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  NCCLCHECK(xmlAddNode(xml, nicNode, "net", netNode));
  NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode* subs[MAX_SUBS];
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs*sizeof(struct ncclXmlNode*));
    for (int s=0; s<nSubs; s++) {
      NCCLCHECK(ncclTopoTrimXmlRec(subs[s]));
    }
    if (node->nSubs == 0) NCCLCHECK(xmlRemoveNode(node));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes));
  return ncclSuccess;
}

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlGraphLoadNet }, { "gpu", ncclTopoXmlGraphLoadGpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "channel", ncclTopoXmlGraphLoadChannel } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    WARN("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = { { "graph", ncclTopoXmlGraphLoadGraph } };
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    WARN("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = { { "graphs", ncclTopoXmlGraphLoadGraphs } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}
