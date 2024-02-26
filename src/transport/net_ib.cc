/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"

#define MAXNAMESIZE 64
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclIbIfAddr;

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

static int ncclNMergedIbDevs = -1;
#define NCCL_IB_MAX_DEVS_PER_NIC 2
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  int ndevs;
  int devs[NCCL_IB_MAX_DEVS_PER_NIC]; // Points to an index in ncclIbDevs
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

static int ncclNIbDevs = -1;
struct alignas(64) ncclIbDev {
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char* pciPath;
  int realPort;
  int maxQp;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
};

#define MAX_IB_DEVS 32
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_DEVS];
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclIbRelaxedOrderingEnabled = 0;

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", 0);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 18);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);

pthread_t ncclIbAsyncThread;
static void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("NET/IB : %s:%d Got async event : %s", dev->devName, dev->portNum, str);
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (strcmp(p, ncclIbDevs[d].pciPath) == 0) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000 /* NDR */ };

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

// Compare ncclIbDev[dev] to all stored mergedIbDevs
int ncclIbFindMatchingDev(int dev) {
  for (int i = 0; i < ncclNMergedIbDevs; i++) {
    if (ncclIbMergedDevs[i].ndevs < NCCL_IB_MAX_DEVS_PER_NIC) {
      int compareDev = ncclIbMergedDevs[i].devs[0];
      if (strcmp(ncclIbDevs[dev].pciPath, ncclIbDevs[compareDev].pciPath) == 0 &&
          (ncclIbDevs[dev].guid == ncclIbDevs[compareDev].guid) &&
          (ncclIbDevs[dev].link == ncclIbDevs[compareDev].link)) {
          TRACE(NCCL_NET, "NET/IB: Matched name1=%s pciPath1=%s guid1=0x%lx link1=%u name2=%s pciPath2=%s guid2=0x%lx link2=%u",
            ncclIbDevs[dev].devName, ncclIbDevs[dev].pciPath, ncclIbDevs[dev].guid, ncclIbDevs[dev].link,
            ncclIbDevs[compareDev].devName, ncclIbDevs[compareDev].pciPath, ncclIbDevs[compareDev].guid, ncclIbDevs[compareDev].link);
          return i;
      }
    }
  }

  return ncclNMergedIbDevs;
}

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      if (ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        return ncclInternalError;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      char* userIbEnv = getenv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) return ncclInternalError;

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
            WARN("NET/IB : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].portNum = port_num;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          ncclIbDevs[ncclNIbDevs].pdRefs = 0;
          ncclIbDevs[ncclNIbDevs].pd = NULL;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          NCCLCHECK(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;

          // Enable ADAPTIVE_ROUTING by default on IB networks
          // But allow it to be overloaded by an env parameter
          ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

          TRACE(NCCL_NET,"NET/IB: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name, ncclIbDevs[ncclNIbDevs].portNum,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE", ncclIbDevs[ncclNIbDevs].speed, context, ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

          pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs);
          ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
          pthread_detach(ncclIbAsyncThread); // will not be pthread_join()'d

          int mergedDev = ncclNMergedIbDevs;
          if (ncclParamIbMergeNics()) {
            mergedDev = ncclIbFindMatchingDev(ncclNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if there's only one dev inside)
          if (mergedDev == ncclNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IB device
            ncclIbMergedDevs[mergedDev].ndevs = 1;
            ncclIbMergedDevs[mergedDev].devs[0] = ncclNIbDevs;
            ncclNMergedIbDevs++;
            strncpy(ncclIbMergedDevs[mergedDev].devName, ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE);
          // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IB device
            int ndevs = ncclIbMergedDevs[mergedDev].ndevs;
            ncclIbMergedDevs[mergedDev].devs[ndevs] = ncclNIbDevs;
            ncclIbMergedDevs[mergedDev].ndevs++;
            snprintf(ncclIbMergedDevs[mergedDev].devName + strlen(ncclIbMergedDevs[mergedDev].devName), MAXNAMESIZE+1, "+%s", ncclIbDevs[ncclNIbDevs].devName);
          }

          // Aggregate speed
          ncclIbMergedDevs[mergedDev].speed += ncclIbDevs[ncclNIbDevs].speed;
          ncclNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { return ncclInternalError; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    } else {
      char line[2048];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
      for (int d = 0; d < ncclNMergedIbDevs; d++) {
        struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + d;
        if (mergedDev->ndevs > 1) {
          // Print out merged dev info
          snprintf(line+strlen(line), 2047-strlen(line), " [%d]={", d);
          for (int i = 0; i < mergedDev->ndevs; i++) {
            int ibDev = mergedDev->devs[i];
            snprintf(line+strlen(line), 2047-strlen(line), "[%d] %s:%d/%s%s", ibDev, ncclIbDevs[ibDev].devName,
              ncclIbDevs[ibDev].portNum, ncclIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
              // Insert comma to delineate
              i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line+strlen(line), 2047-strlen(line), "}");
        } else {
          int ibDev = mergedDev->devs[0];
          snprintf(line+strlen(line), 2047-strlen(line), " [%d]%s:%d/%s", ibDev, ncclIbDevs[ibDev].devName,
            ncclIbDevs[ibDev].portNum, ncclIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
           ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
ncclResult_t ncclIbGdrSupport() {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    // Check for the nv_peer_mem module being loaded
    moduleLoaded = ((access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) &&
                    // Also support the new nvidia-peermem module
                    (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) == -1)) ? 0 : 1;
  }
  if (moduleLoaded == 0) return ncclSystemError;
  return ncclSuccess;
}

// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t ncclIbDmaBufSupport(int dev) {
  static int dmaBufSupported = -1;
  if (dmaBufSupported == -1) {
    ncclResult_t res;
    struct ibv_pd* pd;
    struct ibv_context* ctx;
    struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;

    // Test each dev
    for (int i = 0; i < mergedDev->ndevs; i++) {
      int ibDev = mergedDev->devs[i];
      ctx = ncclIbDevs[ibDev].context;
      NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
      // Test kernel DMA-BUF support with a dummy call (fd=-1)
      (void) wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL/*offset*/, 0ULL/*len*/, 0ULL/*iova*/, -1/*fd*/, 0/*flags*/);
      // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
      dmaBufSupported = (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) ? 1 : 0;
      NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
    }
  }
  if (dmaBufSupported == 0) return ncclSystemError;
  return ncclSuccess;
failure:
  dmaBufSupported = 0;
  return ncclSystemError;
}

#define NCCL_NET_IB_MAX_RECVS 8

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs+dev;
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  struct ncclIbDev* ibDev = ncclIbDevs + mergedDev->devs[0];
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

#define NCCL_IB_MAX_QPS 128

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;
  int devIndex;
};

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE
  uint64_t spn;
  uint64_t iid;

  // FIFO RDMA info
  uint32_t fifoRkey;
  union ibv_gid remoteGid;
};

// Struct containing everything needed to establish connections
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  uint64_t fifoAddr;
  int ndevs;
};

enum ncclIbCommState {
  ncclIbCommStateStart = 0,
  ncclIbCommStateConnect = 1,
  ncclIbCommStateAccept = 3,
  ncclIbCommStateSend = 4,
  ncclIbCommStateRecv = 5,
  ncclIbCommStateConnecting = 6,
  ncclIbCommStateConnected = 7,
  ncclIbCommStatePendingReady = 8,
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct ncclIbHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  struct ncclIbCommStage stage; // Used by the other side when connecting
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
};

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
const char* reqTypeStr[] = { "Unused", "Send", "Recv", "Flush" };

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int* sizes;
    } recv;
  };
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[1];
  struct ncclIbGidInfo gidInfo;
};

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage stage;
};

struct ncclIbSendFifo {
  uint64_t addr;
  int      size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  char padding[24];
};

struct ncclIbQp {
  struct ibv_qp* qp;
  int devIndex;
  int remDevIdx;
};

struct ncclIbRemSizesFifo {
  int elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
  struct ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ibv_sge sge;
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* fifoMr;
};


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

struct alignas(32) ncclIbNetCommBase {
  int ndevs;
  bool isSend;
  struct ncclIbRequest reqs[MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  int devIndex;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
};

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS+1];
  struct ncclIbRemSizesFifo remSizesFifo;
  uint64_t fifoHead;
  int ar; // Use adaptive routing when all merged devices have it enabled
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure fifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  uint32_t fifoRkey;
  struct ibv_mr* fifoMr;
  struct ibv_sge fifoSge;
  struct ibv_mr* sizesFifoMr;
};

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev    devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRemFifo remFifo;
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;
};
static_assert((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbRecvComm fifo must be 32-byte aligned");

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

static void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex, struct ncclIbNetCommDevBase* base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  pthread_mutex_lock(&ibDev->lock);
  if (0 == ibDev->pdRefs++) {
    ncclResult_t res;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context), res, failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ibDev->lock);
      return res;
    }
  }
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), NULL, NULL, 0));

  return ncclSuccess;
}

ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base) {
  ncclResult_t res;
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));

  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd), res, returning);
  }
  res = ncclSuccess;
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* base, int access_flags, struct ncclIbQp* qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, uint32_t dest_qp_num, struct ncclIbDevInfo* info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = ncclParamIbGidIndex();
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = ncclParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = ncclParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}

ncclResult_t ncclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclIbListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECK(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1));
  NCCLCHECK(ncclSocketListen(&comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&comm->sock, &handle->connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  int ready;
  *sendComm = NULL;

  if (stage->state == ncclIbCommStateConnect)    goto ib_connect_check;
  if (stage->state == ncclIbCommStateSend)       goto ib_send;
  if (stage->state == ncclIbCommStateConnecting) goto ib_connect;
  if (stage->state == ncclIbCommStateConnected)  goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECK(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECK(ncclSocketConnect(&comm->base.sock));

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECK(ncclSocketReady(&comm->base.sock, &ready));
  if (!ready) return ncclSuccess;

  // IB Setup
  struct ncclIbMergedDev* mergedDev;
  mergedDev = ncclIbMergedDevs + dev;
  comm->base.ndevs = mergedDev->ndevs;
  comm->base.nqps = ncclParamIbQpsPerConn() * comm->base.ndevs; // We must have at least 1 qp per-device
  comm->base.isSend = true;

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  for (int i = 0; i < mergedDev->ndevs; i++) {
    int ibDevN = mergedDev->devs[i];
    NCCLCHECK(ncclIbInitCommDevBase(ibDevN, &comm->devs[i].base));
    comm->ar = comm->ar && ncclIbDevs[dev].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }

  struct ncclIbConnectionMetadata meta;
  meta.ndevs = comm->base.ndevs;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    ncclIbSendCommDev* commDev = comm->devs + devIndex;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    NCCLCHECK(ncclIbCreateQp(ibDev->portNum, &commDev->base, IBV_ACCESS_REMOTE_WRITE, comm->base.qps+q));
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn      = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // Query ece capabilities (enhanced connection establishment)
    NCCLCHECK(wrap_ibv_query_ece(comm->base.qps[q].qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported));
    devIndex = (devIndex + 1) % comm->base.ndevs;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    // Send my QP Info to receiver through the socket. Hope this won't block.
    // TODO - I thought I queried this in init?
    NCCLCHECK(wrap_ibv_query_port(ibDev->context, ibDev->portNum, &ibDev->portAttr));

    // Write to the metadata struct via this pointer
    ncclIbDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port       = ibDev->portNum;
    devInfo->mtu           = ibDev->portAttr.active_mtu;
    devInfo->lid           = ibDev->portAttr.lid;

    // Prepare my fifo
    NCCLCHECK(wrap_ibv_reg_mr(&commDev->fifoMr, commDev->base.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // RoCE support
    devInfo->link_layer = commDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    if (devInfo->link_layer == IBV_LINK_LAYER_ETHERNET) {
      NCCLCHECK(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, ncclParamIbGidIndex(), &commDev->base.gidInfo.localGid));
      devInfo->spn = commDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo->iid = commDev->base.gidInfo.localGid.global.interface_id;
    }

    if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d fifoRkey=0x%x fifoLkey=0x%x",
            comm->base.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
            dev, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid, devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    } else { // RoCE
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x} GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
            comm->base.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev,
            commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, meta.qpInfo[q].ece_supported, meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options, meta.qpInfo[q].ece.comp_mask, ncclParamIbGidIndex(),
            devInfo->spn, devInfo->iid, devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    }
  }
  meta.fifoAddr = (uint64_t)comm->fifo;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)));

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset));
  if (stage->offset != sizeof(meta)) return ncclSuccess;

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct ncclIbConnectionMetadata remMeta;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata), &stage->offset));
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;
  if (comm->base.nRemDevs != comm->base.ndevs) {
    mergedDev = ncclIbMergedDevs + dev;
    WARN("NET/IB : Local mergedDev=%s has a different number of devices=%d as remoteDev=%s nRemDevs=%d",
      mergedDev->devName, comm->base.ndevs, remMeta.devName, comm->base.nRemDevs);
  }

  int link_layer;
  link_layer = remMeta.devs[0].link_layer;
  for (int i = 1; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].link_layer != link_layer) {
      WARN("NET/IB : Can't merge net devices with different link_layer. i=%d remMeta.ndevs=%d link_layer=%d rem_link_layer=%d",
      i, remMeta.ndevs, link_layer, remMeta.devs[i].link_layer);
      return ncclInternalError;
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].iid;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].spn;

    // Retain remote sizes fifo info and prepare RDMA ops
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;
  }

  for (int i=0; i < comm->base.ndevs; i++) {
    NCCLCHECK(wrap_ibv_reg_mr(comm->remSizesFifo.mrs+i, comm->devs[i].base.pd, &comm->remSizesFifo.elems, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
  }
  comm->base.nRemDevs = remMeta.ndevs;

  for (int q = 0; q < comm->base.nqps; q++) {
    struct ncclIbQpInfo* remQpInfo   = remMeta.qpInfo + q;
    struct ncclIbDevInfo* remDevInfo = remMeta.devs + remQpInfo->devIndex;

    // Assign per-QP remDev
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;

    struct ibv_qp* qp = comm->base.qps[q].qp;
    if (remQpInfo->ece_supported && remQpInfo->ece_supported)
      NCCLCHECK(wrap_ibv_set_ece(qp, &remQpInfo->ece, &remQpInfo->ece_supported));

    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo->qpn, remDevInfo));
    NCCLCHECK(ncclIbRtsQp(qp));
  }

  if (link_layer == IBV_LINK_LAYER_ETHERNET ) { // RoCE
    for (int q = 0; q < comm->base.nqps; q++) {
      struct ncclIbQp* qp = comm->base.qps + q;
      int ibDevN = comm->devs[qp->devIndex].base.ibDevN;
      struct ncclIbDev* ibDev = ncclIbDevs + ibDevN;
      INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
        ibDevN, ibDev->portNum, remMeta.qpInfo[q].qpn, remMeta.qpInfo[q].ece_supported, remMeta.qpInfo[q].ece.vendor_id, remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
    }
  }

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset));
  if (stage->offset != sizeof(int)) return ncclSuccess;

  free(stage->buffer);
  stage->state = ncclIbCommStateStart;

  *sendComm = comm;
  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = &lComm->stage;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  *recvComm = NULL;

  if (stage->state == ncclIbCommStateAccept) goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECK(ncclSocketInit(&rComm->base.sock));
  NCCLCHECK(ncclSocketAccept(&rComm->base.sock, &lComm->sock));

ib_accept_check:
  NCCLCHECK(ncclSocketReady(&rComm->base.sock, &ready));
  if (!ready) return ncclSuccess;

  struct ncclIbConnectionMetadata remMeta;
  stage->state = ncclIbCommStateRecv;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_recv:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset));
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));

  // IB setup
  // Pre-declare variables because of goto
  struct ncclIbMergedDev* mergedDev;
  struct ncclIbDev* ibDev;
  int ibDevN;
  struct ncclIbRecvCommDev* rCommDev;
  struct ncclIbDevInfo* remDevInfo;
  struct ncclIbQp* qp;

  mergedDev = ncclIbMergedDevs + lComm->dev;
  rComm->base.ndevs = mergedDev->ndevs;
  rComm->base.nqps  = ncclParamIbQpsPerConn() * rComm->base.ndevs; // We must have at least 1 qp per-device
  rComm->base.isSend = false;

  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.ndevs) {
    WARN("NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
      mergedDev->devName, rComm->base.ndevs, remMeta.devName, rComm->base.nRemDevs);
  }

  // Metadata to send back to requestor (sender)
  struct ncclIbConnectionMetadata meta;
  for (int i = 0; i < rComm->base.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = mergedDev->devs[i];
    NCCLCHECK(ncclIbInitCommDevBase(ibDevN, &rCommDev->base));
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECK(wrap_ibv_query_port(ibDev->context, ibDev->portNum, &ibDev->portAttr));
    NCCLCHECK(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, ncclParamIbGidIndex(), &rCommDev->base.gidInfo.localGid));
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id  = rComm->base.remDevs[i].iid;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].spn;
  }

  // Stripe QP creation across merged devs
  // Make sure to get correct remote peer dev and QP info
  int remDevIndex;
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIndex = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIndex;
    qp = rComm->base.qps+q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIndex;

    // Local ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECK(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_REMOTE_WRITE, qp));
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.ndevs;

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].ece_supported) {
      NCCLCHECK(wrap_ibv_set_ece(qp->qp, &remMeta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported));
  
      // Query the reduced ece for this QP (matching enhancements between the requestor and the responder)
      // Store this in our own qpInfo for returning to the requestor
      if (meta.qpInfo[q].ece_supported)
        NCCLCHECK(wrap_ibv_query_ece(qp->qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported));
    }

    NCCLCHECK(ncclIbRtrQp(qp->qp, remMeta.qpInfo[q].qpn, remDevInfo));
    NCCLCHECK(ncclIbRtsQp(qp->qp));
  }

  rComm->flushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev) == ncclSuccess)
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  for (int i = 0; i < mergedDev->ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rCommDev->base.ibDevN;
    ibDev = ncclIbDevs + ibDevN;

    // Retain remote fifo info and prepare my RDMA ops
    rCommDev->fifoRkey = remMeta.devs[i].fifoRkey;
    rComm->remFifo.addr = remMeta.fifoAddr;
    NCCLCHECK(wrap_ibv_reg_mr(&rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (ncclParamIbUseInline()) rComm->remFifo.flags = IBV_SEND_INLINE;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      NCCLCHECK(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE));
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      NCCLCHECK(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rCommDev->gpuFlush.qp));
      struct ncclIbDevInfo devInfo;
      devInfo.lid         = ibDev->portAttr.lid;
      devInfo.link_layer  = ibDev->portAttr.link_layer;
      devInfo.ib_port     = ibDev->portNum;
      devInfo.spn         = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.iid         = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu         = ibDev->portAttr.active_mtu;
      NCCLCHECK(ncclIbRtrQp(rCommDev->gpuFlush.qp.qp, rCommDev->gpuFlush.qp.qp->qp_num, &devInfo));
      NCCLCHECK(ncclIbRtsQp(rCommDev->gpuFlush.qp.qp));
    }

    // Fill Handle
    meta.devs[i].lid        = ibDev->portAttr.lid;
    meta.devs[i].link_layer = rCommDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    meta.devs[i].ib_port    = ibDev->portNum;
    meta.devs[i].spn        = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].iid        = rCommDev->base.gidInfo.localGid.global.interface_id;

    // Adjust the MTU
    remMeta.devs[i].mtu    = (enum ibv_mtu) std::min(remMeta.devs[i].mtu, ibDev->portAttr.active_mtu);
    meta.devs[i].mtu      = remMeta.devs[i].mtu;

    // Prepare sizes fifo
    NCCLCHECK(wrap_ibv_reg_mr(&rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn      = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }

  meta.ndevs = rComm->base.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) free(stage->buffer);
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)));
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));

ib_send:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(struct ncclIbConnectionMetadata), &stage->offset));
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

ib_recv_ready:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->base.sock, &rComm->base.ready, sizeof(int), &stage->offset));
  if (stage->offset != sizeof(int)) return ncclSuccess;

  free(stage->buffer);
  *recvComm = rComm;

  /* reset lComm stage */
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      r->devBases[0] = NULL;
      r->devBases[1] = NULL;
      r->events[0] = r->events[1] = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* size);

ncclResult_t ncclIbRegMrDmaBufInternal(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset, int fd, ibv_mr** mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) { // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECKGOTO(ncclRealloc(&cache->slots, cache->population, cache->capacity), res, returning);
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
      if (ncclIbRelaxedOrderingEnabled) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        NCCLCHECKGOTO(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags), res, returning);
      } else {
        if (ncclIbRelaxedOrderingEnabled) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECKGOTO(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags), res, returning);
        }
        else {
          NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags), res, returning);
        }
      }
      TRACE(NCCL_INIT|NCCL_NET,"regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr, (long long)pages*pageSize, mr->rkey, mr->lkey, fd);
      if (slot != cache->population) memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      res = ncclSuccess;
      goto returning;
    } else if ((addr >= cache->slots[slot].addr) &&
        ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      res = ncclSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  assert(size > 0);
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) malloc(sizeof(struct ncclIbMrHandle));
  for (int i = 0; i < base->ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbRegMrDmaBufInternal(devComm, data, size, type, offset, fd, mhandleWrapper->mrs + i));
  }
  *mhandle = (void*) mhandleWrapper;
  return ncclSuccess;
}

ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mhandle);
}

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECKGOTO(wrap_ibv_dereg_mr(mhandle), res, returning);
      }
      res = ncclSuccess;
      goto returning;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  res = ncclInternalError;
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  for (int i = 0; i < base->ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}

NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r*8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    int* sizes = comm->remSizesFifo.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs*sizeof(int);
  }

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr = comm->remSizesFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    ncclIbQp* qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex, &comm->devs[devIndex].base);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      // Also make sure lastWr writes remote sizes using the right lkey
      comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
      lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // Select the next qpIndex
    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;
  }

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) { WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0"); return ncclInternalError; }
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %d addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    // Populate events
    int nEvents = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
    int qpIndex = comm->base.qpIndex;
    // Count down
    while (nEvents > 0) {
      ncclIbQp* qp = comm->base.qps + qpIndex;
      int devIndex = qp->devIndex;
      ncclIbAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // Track the valid lkey for this RDMA_Write
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same set of QPs inside ncclIbMultiSend()
      qpIndex = (qpIndex+1)%comm->base.nqps;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r=0; r<nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and sanity checks
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbRequest*));
    comm->fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, int* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remFifo.fifoTail%MAX_REQUESTS;
  req->recv.sizes = comm->sizesFifo[slot];
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  // Select the next devIndex (local) and QP to use for posting this CTS message
  // Since QPs are initialized by striping across devIndex, we can simply assign this to the same value
  ncclIbQp* ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.ndevs;

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];

    // Send all applicable rkeys
    for (int j = 0; j < comm->base.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail+1;
  }
  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct fifoRkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;

  // Set the correct sge properties
  comm->devs[ctsQp->devIndex].fifoSge.addr   = (uint64_t)localElem;
  comm->devs[ctsQp->devIndex].fifoSge.length = n*sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to fifo slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each fifo posting QP gets drained
  if (slot == ctsQp->devIndex) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    ncclIbAddEvent(req, ctsQp->devIndex, &comm->devs[ctsQp->devIndex].base);
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) { WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0"); return ncclInternalError; }
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;

  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  // Select either all QPs, or one qp per-device
  const int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;

  // Post recvs
  struct ibv_recv_wr* bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
    ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex+1)%comm->base.nqps;
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    ncclIbAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;
  while (1) {
    if (r->events[0] == 0 && r->events[1] == 0) {
      TRACE(NCCL_NET, "r=%p done", r);
      *done = 1;
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) sizes[i] = r->recv.sizes[i];
      }
      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
      }
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int totalWrDone = 0;
    int wrDone = 0;
    struct ibv_wc wcs[4];

    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      // If we expect any completions from this device's CQ
      if (r->events[i]) {
        NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
        totalWrDone += wrDone;
        if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
        if (wrDone == 0) continue;
        for (int w=0; w<wrDone; w++) {
          struct ibv_wc *wc = wcs+w;
          if (wc->status != IBV_WC_SUCCESS) {
            union ncclSocketAddress addr;
            ncclSocketGetAddr(r->sock, &addr);
            char localGidString[INET6_ADDRSTRLEN] = "";
            char remoteGidString[INET6_ADDRSTRLEN] = "";
            const char* localGidStr = NULL, *remoteGidStr = NULL;
            if (r->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
              localGidStr = inet_ntop(AF_INET6, &r->devBases[i]->gidInfo.localGid, localGidString, sizeof(localGidString));
              remoteGidStr = inet_ntop(AF_INET6, &r->base->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
            }

            char line[SOCKET_NAME_MAXLEN+1];
            WARN("NET/IB : Got completion from peer %s with status=%d opcode=%d len=%d vendor err %d (%s)%s%s%s%s",
                ncclSocketToString(&addr, line), wc->status, wc->opcode, wc->byte_len, wc->vendor_err, reqTypeStr[r->type],
                localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString);
            return ncclRemoteError;
          }

          union ncclSocketAddress addr;
          ncclSocketGetAddr(r->sock, &addr);
          struct ncclIbRequest* req = r->base->reqs+(wc->wr_id & 0xff);

          #ifdef ENABLE_TRACE
          char line[SOCKET_NAME_MAXLEN+1];
          TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%d wr_id=%d r=%p type=%d events={%d,%d}, i=%d",
              ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], i);
          #endif
          if (req->type == NCCL_NET_IB_REQ_SEND) {
            for (int j = 0; j < req->nreqs; j++) {
              struct ncclIbRequest* sendReq = r->base->reqs+((wc->wr_id >> (j*8)) & 0xff);
              if ((sendReq->events[i] <= 0)) {
                WARN("NET/IB: sendReq(%p)->events={%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], i, j);
                return ncclInternalError;
              }
              sendReq->events[i]--;
            }
          } else {
            if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              if (req->type != NCCL_NET_IB_REQ_RECV) {
                WARN("NET/IB: wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM and req->type=%d", req->type);
                return ncclInternalError;
              }
              if (req->nreqs == 1) {
                req->recv.sizes[0] += wc->imm_data;
              }
            }
            req->events[i]--;
          }
        }
      }
    }

    // If no CQEs found on any device, return and come back later
    if (totalWrDone == 0) return ncclSuccess;
  }
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct ncclIbSendCommDev* commDev = comm->devs + i;
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i]));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct ncclIbRecvCommDev* commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (commDev->sizesFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->sizesFifoMr));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbRegMrDmaBuf,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

