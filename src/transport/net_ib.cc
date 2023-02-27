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
  int pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

static int ncclNIbDevs = -1;
struct alignas(64) ncclIbDev {
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t port;
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
};

#define MAX_IB_PORT 15
struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];
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
  struct ibv_context* context = (struct ibv_context*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(context, &event)) { break; }
    char *str;
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("NET/IB : Got async event : %s", str);
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);

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
    p[strlen(p)-3] = '0';
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

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
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
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/IB : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          TRACE(NCCL_INIT|NCCL_NET,"NET/IB: [%d] %s:%d/%s ", d, devices[d]->name, port,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].port = port;
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

          pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
          ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
          pthread_detach(ncclIbAsyncThread); // will not be pthread_join()'d
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
      char line[1024];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
      for (int d=0; d<ncclNIbDevs; d++) {
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
            ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
      }
      line[1023] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
           ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
ncclResult_t ncclIbGdrSupport(int ibDev) {
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
    ctx = ncclIbDevs[dev].context;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
    // Test kernel DMA-BUF support with a dummy call (fd=-1)
    (void) wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL/*offset*/, 0ULL/*len*/, 0ULL/*iova*/, -1/*fd*/, 0/*flags*/);
    // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
    dmaBufSupported = (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) ? 1 : 0;
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  }
  if (dmaBufSupported == 0) return ncclSystemError;
  return ncclSuccess;
failure:
  dmaBufSupported = 0;
  return ncclSystemError;
}

#define NCCL_NET_IB_MAX_RECVS 8

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclIbDevs[dev].devName;
  props->pciPath = ncclIbDevs[dev].pciPath;
  props->guid = ncclIbDevs[dev].guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->speed = ncclIbDevs[dev].speed;
  props->latency = 0; // Not set
  props->port = ncclIbDevs[dev].port + ncclIbDevs[dev].realPort;
  props->maxComms = ncclIbDevs[dev].maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  return ncclSuccess;
}

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

#define NCCL_IB_MAX_QPS 128

struct ncclIbQpInfo {
  uint32_t lid;
  uint8_t ib_port;
  uint8_t link_layer;
  uint32_t qpn[NCCL_IB_MAX_QPS];

  // For RoCE
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;

  // FIFO RDMA info
  uint32_t fifoRkey;
  uint64_t fifoAddr;
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

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3

struct ncclIbRequest {
  struct ncclIbVerbs* verbs;
  int type;
  int events;
  struct ncclSocket* sock;
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkey;
      int offset;
    } send;
    struct {
      int sizes[NCCL_NET_IB_MAX_RECVS];
    } recv;
  };
};

struct ncclIbVerbs {
  int dev;
  struct ibv_pd* pd; // duplicate of ncclIbDevs[dev].pd
  struct ibv_cq* cq;
  uint64_t pad[1];
  struct ncclIbRequest reqs[MAX_REQUESTS];
};

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage stage;
};

struct ncclIbSendFifo {
  uint64_t addr;
  int      size;
  uint32_t rkey;
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
};

struct ncclIbSendComm {
  struct ncclIbVerbs verbs;
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoHead;
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS+1];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ncclSocket sock;

  int ready;
  struct ibv_qp* qps[NCCL_IB_MAX_QPS];
  int nqps;
  struct ibv_mr* fifoMr;
  int ar;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");

struct ncclIbGpuFlush {
  int enabled;
  int hostMem;
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ibv_qp* qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkey;
  uint32_t flags;
  struct ibv_mr* mr;
  struct ibv_sge sge;
};

struct ncclIbRecvComm {
  struct ncclIbVerbs verbs;
  struct ncclIbRemFifo remFifo;
  struct ncclSocket sock;
  int ready;
  struct ibv_qp* qps[NCCL_IB_MAX_QPS];
  int nqps;
  struct ncclIbGpuFlush gpuFlush;
};
static_assert((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

ncclResult_t ncclIbInitVerbs(int dev, struct ibv_context* ctx, struct ncclIbVerbs* verbs) {
  verbs->dev = dev;

  pthread_mutex_lock(&ncclIbDevs[dev].lock);
  if (0 == ncclIbDevs[dev].pdRefs++) {
    ncclResult_t res;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&ncclIbDevs[dev].pd, ctx), res, failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ncclIbDevs[dev].lock);
      return res;
    }
  }
  verbs->pd = ncclIbDevs[dev].pd;
  pthread_mutex_unlock(&ncclIbDevs[dev].lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  NCCLCHECK(wrap_ibv_create_cq(&verbs->cq, ctx, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), NULL, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclIbDestroyVerbs(struct ncclIbVerbs* verbs) {
  ncclResult_t res;
  NCCLCHECK(wrap_ibv_destroy_cq(verbs->cq));

  pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
  if (0 == --ncclIbDevs[verbs->dev].pdRefs) {
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(ncclIbDevs[verbs->dev].pd), res, returning);
  }
  res = ncclSuccess;
returning:
  pthread_mutex_unlock(&ncclIbDevs[verbs->dev].lock);
  return res;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbVerbs* verbs, int access_flags, struct ibv_qp** qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = verbs->cq;
  qpInitAttr.recv_cq = verbs->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(qp, verbs->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(*qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, uint32_t qpn, struct ncclIbQpInfo* info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = qpn;
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

ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
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
  NCCLCHECK(ncclSocketInit(&comm->sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECK(ncclSocketConnect(&comm->sock));

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
  if (!ready) return ncclSuccess;

  // IB Setup
  struct ibv_context* ctx;
  ctx = ncclIbDevs[dev].context;
  NCCLCHECK(ncclIbInitVerbs(dev, ctx, &comm->verbs));
  uint8_t ib_port;
  ib_port = ncclIbDevs[dev].port;
  comm->nqps = ncclParamIbQpsPerConn();
  for (int q=0; q<comm->nqps; q++) {
    NCCLCHECK(ncclIbCreateQp(ib_port, &comm->verbs, IBV_ACCESS_REMOTE_WRITE, comm->qps+q));
  }
  comm->ar = ncclIbDevs[dev].ar; // ADAPTIVE_ROUTING

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  struct ncclIbQpInfo qpInfo;
  qpInfo.ib_port = ib_port;
  for (int q=0; q<comm->nqps; q++) qpInfo.qpn[q] = comm->qps[q]->qp_num;
  qpInfo.mtu = portAttr.active_mtu;

  // Prepare my fifo
  NCCLCHECK(wrap_ibv_reg_mr(&comm->fifoMr, comm->verbs.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
  qpInfo.fifoRkey = comm->fifoMr->rkey;
  qpInfo.fifoAddr = (uint64_t)comm->fifo;

  // RoCE support
  qpInfo.lid = portAttr.lid;
  qpInfo.link_layer = portAttr.link_layer;
  if (qpInfo.link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
    for (int q=0; q<comm->nqps; q++)
      INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d LID %d", dev, ib_port, qpInfo.qpn[q], qpInfo.mtu, qpInfo.lid);
  } else { // RoCE
    union ibv_gid gid;
    NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));
    qpInfo.spn = gid.global.subnet_prefix;
    qpInfo.iid = gid.global.interface_id;
    for (int q=0; q<comm->nqps; q++)
      INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX)", dev, ib_port, qpInfo.qpn[q], qpInfo.mtu, ncclParamIbGidIndex(), qpInfo.spn, qpInfo.iid);
  }

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(qpInfo)));
  memcpy(stage->buffer, &qpInfo, sizeof(qpInfo));

ib_send:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->sock, stage->buffer, sizeof(qpInfo), &stage->offset));
  if (stage->offset != sizeof(qpInfo)) return ncclSuccess;

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(qpInfo));

ib_connect:
  struct ncclIbQpInfo remQpInfo;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, stage->buffer, sizeof(ncclIbQpInfo), &stage->offset));
  if (stage->offset != sizeof(remQpInfo)) return ncclSuccess;

  memcpy(&remQpInfo, stage->buffer, sizeof(ncclIbQpInfo));

  for (int q=0; q<comm->nqps; q++) {
    struct ibv_qp* qp = comm->qps[q];
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn[q], &remQpInfo));
    NCCLCHECK(ncclIbRtsQp(qp));
  }

  comm->ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->sock, &comm->ready, sizeof(int), &stage->offset));
  if (stage->offset != sizeof(int)) return ncclSuccess;

  free(stage->buffer);
  stage->state = ncclIbCommStateStart;

  *sendComm = comm;
  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm) {
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
  NCCLCHECK(ncclSocketInit(&rComm->sock));
  NCCLCHECK(ncclSocketAccept(&rComm->sock, &lComm->sock));

ib_accept_check:
  NCCLCHECK(ncclSocketReady(&rComm->sock, &ready));
  if (!ready) return ncclSuccess;

  struct ncclIbQpInfo remQpInfo;
  stage->state = ncclIbCommStateRecv;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remQpInfo)));

ib_recv:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->sock, stage->buffer, sizeof(remQpInfo), &stage->offset));
  if (stage->offset != sizeof(remQpInfo)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remQpInfo, stage->buffer, sizeof(struct ncclIbQpInfo));

  // IB setup
  struct ibv_context* ctx;
  uint8_t ib_port;
  ctx = ncclIbDevs[lComm->dev].context;
  ib_port = ncclIbDevs[lComm->dev].port;
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  union ibv_gid gid;
  NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));

  // QP Creation
  NCCLCHECK(ncclIbInitVerbs(lComm->dev, ctx, &rComm->verbs));
  rComm->nqps = ncclParamIbQpsPerConn();
  for (int q=0; q<rComm->nqps; q++) {
    NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_REMOTE_WRITE, rComm->qps+q));
  }

  // Adjust the MTU
  remQpInfo.mtu = (enum ibv_mtu)std::min(remQpInfo.mtu, portAttr.active_mtu);

  // Setup QP
  for (int q=0; q<rComm->nqps; q++) {
    struct ibv_qp* qp = rComm->qps[q];
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn[q], &remQpInfo));
    NCCLCHECK(ncclIbRtsQp(qp));
  }

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remFifo.rkey = remQpInfo.fifoRkey;
  rComm->remFifo.addr = remQpInfo.fifoAddr;
  NCCLCHECK(wrap_ibv_reg_mr(&rComm->remFifo.mr, rComm->verbs.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
  rComm->remFifo.sge.lkey = rComm->remFifo.mr->lkey;
  if (ncclParamIbUseInline()) rComm->remFifo.flags = IBV_SEND_INLINE;

  // Allocate Flush dummy buffer for GPU Direct RDMA
  rComm->gpuFlush.enabled = (ncclIbGdrSupport(lComm->dev) == 0) && (ncclParamIbGdrFlushDisable() == 0) ? 1 : 0;
  if (rComm->gpuFlush.enabled) {
    NCCLCHECK(wrap_ibv_reg_mr(&rComm->gpuFlush.hostMr, rComm->verbs.pd, &rComm->gpuFlush.hostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE));
    rComm->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlush.hostMem;
    rComm->gpuFlush.sge.length = 1;
    rComm->gpuFlush.sge.lkey = rComm->gpuFlush.hostMr->lkey;
    NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->gpuFlush.qp));
    struct ncclIbQpInfo localQpInfo;
    localQpInfo.lid=portAttr.lid;
    localQpInfo.link_layer=portAttr.link_layer;
    localQpInfo.ib_port=ib_port;
    localQpInfo.spn=gid.global.subnet_prefix;
    localQpInfo.iid=gid.global.interface_id;
    localQpInfo.mtu=portAttr.active_mtu;
    NCCLCHECK(ncclIbRtrQp(rComm->gpuFlush.qp, rComm->gpuFlush.qp->qp_num, &localQpInfo));
    NCCLCHECK(ncclIbRtsQp(rComm->gpuFlush.qp));
  }

  // Fill Handle
  struct ncclIbQpInfo qpInfo;
  qpInfo.lid=portAttr.lid;
  qpInfo.link_layer=portAttr.link_layer;
  qpInfo.ib_port=ib_port;
  for (int q=0; q<rComm->nqps; q++) qpInfo.qpn[q]=rComm->qps[q]->qp_num;
  qpInfo.spn=gid.global.subnet_prefix;
  qpInfo.iid=gid.global.interface_id;
  qpInfo.mtu=remQpInfo.mtu;

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) free(stage->buffer);
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbQpInfo)));
  memcpy(stage->buffer, &qpInfo, sizeof(struct ncclIbQpInfo));

ib_send:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->sock, stage->buffer, sizeof(struct ncclIbQpInfo), &stage->offset));
  if (stage->offset < sizeof(struct ncclIbQpInfo)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

ib_recv_ready:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->sock, &rComm->ready, sizeof(int), &stage->offset));
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

ncclResult_t ncclIbGetRequest(struct ncclIbVerbs* verbs, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = verbs->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->verbs = verbs;
      r->events = 1;
      r->sock = NULL;
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

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  static_assert(offsetof(struct ncclIbSendComm, verbs) == offsetof(struct ncclIbRecvComm, verbs), "Send and recv comms must have verbs at the same offset");
  assert(size > 0);

  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

  struct ncclIbVerbs* verbs = (struct ncclIbVerbs*)comm;
  struct ncclIbMrCache* cache = &ncclIbDevs[verbs->dev].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population) { // didn't find in cache
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
        NCCLCHECKGOTO(wrap_ibv_reg_dmabuf_mr(&mr, verbs->pd, offset, pages*pageSize, addr, fd, flags), res, returning);
      } else {
        if (ncclIbRelaxedOrderingEnabled) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECKGOTO(wrap_ibv_reg_mr_iova2(&mr, verbs->pd, (void*)addr, pages*pageSize, addr, flags), res, returning);
        }
        else {
          NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, verbs->pd, (void*)addr, pages*pageSize, flags), res, returning);
        }
      }
      TRACE(NCCL_INIT,"regAddr %llx size %lld rkey %x fd %d", (unsigned long long)addr, (long long)pages*pageSize, mr->rkey, fd);
      cache->population += 1;
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      *mhandle = (void*)mr;
      res = ncclSuccess;
      goto returning;
    }
    else if (cache->slots[slot].addr == addr && cache->slots[slot].pages == pages) {
      cache->slots[slot].refs += 1;
      *mhandle = (void*)cache->slots[slot].mr;
      res = ncclSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&ncclIbDevs[verbs->dev].lock);
  return res;
}

ncclResult_t ncclIbRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, (size_t)size, type, 0ULL, -1, mhandle);
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  struct ncclIbVerbs* verbs = (struct ncclIbVerbs*)comm;
  struct ncclIbMrCache* cache = &ncclIbDevs[verbs->dev].mrCache;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECKGOTO(wrap_ibv_dereg_mr((struct ibv_mr*)mhandle), res, returning);
      }
      res = ncclSuccess;
      goto returning;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  res = ncclInternalError;
returning:
  pthread_mutex_unlock(&ncclIbDevs[verbs->dev].lock);
  return res;
}

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
    sge->lkey=reqs[r]->send.lkey;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->wr.rdma.rkey = slots[r].rkey;
    wr->next = wr+1;
    wr_id += (reqs[r] - comm->verbs.reqs) << (r*8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    if (nreqs > 32) {
      WARN("Cannot store sizes of %d requests in a 32-bits field", nreqs);
      return ncclInternalError;
    }
    for (int r=0; r<nreqs; r++) {
      immData |= (reqs[r]->send.size ? 1 : 0) << r;
    }
  }

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  for (int q=0; q<comm->nqps; q++) {
    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, comm->nqps), align) * align;
      int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
    }
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->qps[q], comm->wrs, &bad_wr));

    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, comm->nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->ready == 0) { WARN("NET/IB: ncclIbIsend() called when comm->ready == 0"); return ncclInternalError; }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  struct ibv_mr* mr = (struct ibv_mr*)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->fifoHead)%MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  int idx = comm->fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    // Sanity checks to catch user collective call count/size mismatches
    if (size > slots[r].size) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s collective mismatch error, local size %d remote size %d",
        r, nreqs, tag, ncclSocketToString(&addr, line), size, slots[r].size);
      return ncclInvalidUsage;
    } // plus any potential programming errors
    else if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkey == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %d addr %lx rkey %x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkey);
      return ncclInternalError;
    }
    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->verbs, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->sock;
    req->verbs = &comm->verbs;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.lkey = mr->lkey;
    req->send.offset = 0;
    req->events = comm->nqps;
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
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ibv_mr* mr = (struct ibv_mr*)mhandles[i];
    localElem[i].rkey = mr->rkey;
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail+1;
  }

  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->remFifo.rkey;
  comm->remFifo.sge.addr = (uint64_t)localElem;
  comm->remFifo.sge.length = n*sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->remFifo.sge;
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
  if (slot == 0) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->verbs.reqs;
    req->events++;
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->qps[0], &wr, &bad_wr));
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->ready == 0) { WARN("NET/IB: ncclIbIrecv() called when comm->ready == 0"); return ncclInternalError; }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->verbs, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->sock;
  req->nreqs = n;
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->verbs.reqs;

  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  for (int q=0; q<comm->nqps; q++) {
    struct ibv_qp* qp = comm->qps[q];
    struct ibv_recv_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_recv(qp, &wr, &bad_wr));
  }
  TIME_STOP(1);
  req->events = comm->nqps;

  *request = req;

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->gpuFlush.enabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->verbs, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->sock;
  struct ibv_mr* mr = (struct ibv_mr*)mhandles[last];

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->verbs.reqs;

  wr.wr.rdma.remote_addr = (uint64_t)data[last];
  wr.wr.rdma.rkey = mr->rkey;
  wr.sg_list = &comm->gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->gpuFlush.qp, &wr, &bad_wr));
  TIME_STOP(4);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    if (r->events == 0) {
      *done = 1;
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) sizes[i] = r->recv.sizes[i];
      }
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int wrDone = 0;
    struct ibv_wc wcs[4];
    TIME_START(3);
    NCCLCHECK(wrap_ibv_poll_cq(r->verbs->cq, 4, wcs, &wrDone));
    if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
    if (wrDone == 0) return ncclSuccess;

    for (int w=0; w<wrDone; w++) {
      struct ibv_wc *wc = wcs+w;
      if (wc->status != IBV_WC_SUCCESS) {
        char line[SOCKET_NAME_MAXLEN+1];
        union ncclSocketAddress addr;
        ncclSocketGetAddr(r->sock, &addr);
        WARN("NET/IB : Got completion from peer %s with error %d, opcode %d, len %d, vendor err %d",
             ncclSocketToString(&addr, line), wc->status, wc->opcode, wc->byte_len, wc->vendor_err);
        return ncclRemoteError;
      }

      struct ncclIbRequest* req = r->verbs->reqs+(wc->wr_id & 0xff);
      if (req->type == NCCL_NET_IB_REQ_SEND) {
        for (int i=0; i<req->nreqs; i++) {
          struct ncclIbRequest* sendReq = r->verbs->reqs+((wc->wr_id >> (i*8)) & 0xff);
          if ((sendReq->events <= 0)) return ncclInternalError;
          sendReq->events--;
        }
      } else {
        if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          if (req->type != NCCL_NET_IB_REQ_RECV) return ncclInternalError;
          if (req->nreqs > 1) {
            // In the case of a multi recv, we only set sizes to 0 or 1.
            for (int i=0; i<req->nreqs; i++) {
              req->recv.sizes[i] = (wc->imm_data >> i) & 0x1;
            }
          } else {
            req->recv.sizes[0] += wc->imm_data;
          }
        }
        req->events--;
      }
    }
  }
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    for (int q=0; q<comm->nqps; q++)
      if (comm->qps[q] != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qps[q]));
    if (comm->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->fifoMr));
    NCCLCHECK(ncclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    for (int q=0; q<comm->nqps; q++)
      if (comm->qps[q] != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qps[q]));
    if (comm->gpuFlush.enabled) {
      if (comm->gpuFlush.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->gpuFlush.qp));
      if (comm->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->gpuFlush.hostMr));
    }
    if (comm->remFifo.mr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remFifo.mr));
    NCCLCHECK(ncclIbDestroyVerbs(&comm->verbs));
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
  ncclIbCloseListen
};

