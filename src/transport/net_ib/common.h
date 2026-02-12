/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char ncclIbIfName[MAX_IF_NAME_SIZE+1];
extern union ncclSocketAddress ncclIbIfAddr;

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

extern int ncclNMergedIbDevs;
#define NCCL_IB_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

extern int ncclNIbDevs;
struct alignas(64) ncclIbDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char fullPciPath[PATH_MAX];
  char* pciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
};

#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
extern int ncclIbRelaxedOrderingEnabled;

#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // The key used for remote access to the addr exchanged by the peers
  // in ncclIbConnectionMetadata::addr
  // This member is populated differently on the sender and on the receiver
  // side.
  // The sender side populates this member with the RKey obtained after it
  // registered the CTS FIFO (on the specific device).
  // The receiver side populates this member with the RKey obtained after it
  // registered the completion records structure (on the specific device).
  uint32_t rkey;

  //remote dev info
  union ibv_gid remoteGid;
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
extern const char* ncclIbReqTypeStr[];

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

// Tracks data transfers between sender and receiver. A multi-recv/send uses a
// single record.
struct ncclIbRequestCompletionRecord {
  // This array communicates data transfer sizes from the sender to the
  // receiver. The sender writes the size of each completed data transfer to
  // this array. The receiver reads these sizes before reporting completion of
  // the corresponding receive request to the user.
  int sizes[NCCL_NET_IB_MAX_RECVS];
  // The receiver fills this array to signal the completion of a data transfer.
  // The sender can then read this array to see the receiver's status. If the
  // sender detects an error or device failure, it reads this array to
  // determine if the receiver considered the transfer complete. This prevents
  // the sender from retransmitting data if the failure was only visible on the
  // sender's side. Based on the array's contents, the sender decides if, how,
  // and on which QPs/devices to replay the transfer.
  bool completions[NCCL_IB_MAX_QPS];
};

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  // Array of counters. Each element in the array is populated with the expected
  // number of completion events that the request is expecting to be generated
  // on the device corresponding to the index of the element. After the request
  // is initialized and posted, for every completion event generated by a
  // device, the corresponding counter is decremented. When the counter reaches
  // zero it means that the request was fully completed on that device.
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to the per-device base structures to make it easier to
  // poll the device's CQ when the request is tested for progress.
  // The pointers are initialized only for the devices that the request expects
  // to receive completions from.
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  uint64_t id;
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      // Tracks whether data was transmitted on a QP for this request.
      bool sentData[NCCL_IB_MAX_QPS];
    } send;
    struct {
      struct ncclIbRequestCompletionRecord* cmplsRecords;
      // Aggregates the size of a send request when sender does not write to the
      // completion records array.
      int aggSize;
    } recv;
    struct {
      int rank;
    } iput;
  };
  int connectionId;
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
};

struct ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  char padding[16];
};

struct ncclIbQp {
  struct ibv_qp* qp;
  // The index of the device on which this QP was created on.
  int devIndex;
  // The index of the device on the remote side to which this QP is connected
  // to.
  int remDevIdx;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// Structure to describe the completion records on the sender side.
struct ncclIbRemCompletionsRecords {
  // A "shadow" structure of the receiver's completion records in which the
  // sender tracks the completion records locally on its side. Sender uses this
  // memory to place the records it writes/reads to/from the receiver's
  // completion records.
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  // Address in memory of the completion records structure on the receiver side.
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the sender chooses the
  // RKey (depending on the device being used) when it accesses the receiver's
  // completion records structure.
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* ctsFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* cmplsRecordsMr;
  struct ibv_sge sge;
};


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Forward declaration
struct ncclIbResiliency;

struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  // Array of pointers to the "actual" QPs that are used for data transfers.
  // The pointers point to QPs in the ncclIbNetCommBase::qps[] array.
  struct ncclIbQp* activeQps[NCCL_IB_MAX_QPS];
  uint64_t fifoHead;
  int nqps;
  int splitDataOnQps;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  int nDataQps;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
  struct ncclIbResiliency* resiliency;
};

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex);

// qpIndex is the index relative to a device.
// For example, if a device has 2 QPs, qpIndex can be 0 or 1.
static inline ncclResult_t ncclIbCommBaseGetQpByIndex(struct ncclIbNetCommBase* commBase, int devIndex, int qpIndex, ncclIbQp** qp) {
  assert(devIndex >= 0 && devIndex < commBase->vProps.ndevs);
  *qp = commBase->activeQps[commBase->vProps.ndevs*qpIndex + devIndex];
  return ncclSuccess;
}

// The function selects the QP to be used for the request. The QP selected
// based on the request ID and also based on the provided QP index. A request
// can be posted on multiple QPs. For example, if a request is posted on 4
// QPs, this function should be called 4 times, each time with a different
// qpIndex, ranging from 0 to 3.
// The function outputs the selected QP in the outQp argument and populates the
// outQpIndex argument with the index of the selected QP. Note that the
// outQpIndex is the index of the QP in the base::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpForRequest(struct ncclIbNetCommBase* baseComm, const uint64_t id, const uint8_t qpIndex, ncclIbQp** outQp, int* outQpIndex) {
  *outQpIndex = (id + qpIndex) % baseComm->nqps;
  *outQp = baseComm->activeQps[*outQpIndex];
  assert(*outQp != NULL);
  return ncclSuccess;
}

// Get a QP object from a QP number. If not NULL, qpIndex will also return the
// index of the QP in the ncclIbNetCommBase::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpByQpNum(struct ncclIbNetCommBase* commBase, int devIndex, uint32_t qpNum, ncclIbQp** qp, int* qpIndex) {
  assert(devIndex >= 0 && devIndex < commBase->vProps.ndevs);
  assert(qp != NULL);
  TRACE(NCCL_NET, "NET/IB: %s: Looking for QP num %u on devIndex %d among %d QPs", __func__, qpNum, devIndex, commBase->nqps / commBase->vProps.ndevs);
  for (int qpIndexInDev = 0; qpIndexInDev < (commBase->nqps / commBase->vProps.ndevs); qpIndexInDev++) {
    *qp = &(commBase->qps[commBase->vProps.ndevs*qpIndexInDev + devIndex]);
    if ((*qp)->qp->qp_num == qpNum) {
      if (qpIndex != NULL) {
        *qpIndex = *qp - commBase->qps;
      }
      return ncclSuccess;
    }
  }
  *qp = NULL;
  return ncclInternalError;
}

// Each request is transfered over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transffered over
// a single QP per device or on all QPs of each device.
static inline int ncclIbCommBaseGetNqpsPerRequest(struct ncclIbNetCommBase* baseComm) {
  assert(baseComm->nDataQps != -1);
  assert(baseComm->nqps != -1);
  return (baseComm->splitDataOnQps == 1) ? baseComm->nqps : baseComm->nDataQps;
}

static inline ncclResult_t ncclIbPostRecvWorkRequest(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(qp, wr, &bad_wr));
  return ncclSuccess;
}

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with CTS FIFO and ibv structs as they have alignment restrictions

  // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
  // are written by the receiver (The receiver side writes into it upon
  // issuing a (multi-)receive request). Each row in the 2D array corresponds
  // to a single CTS message but can describe multiple recv-requests issued
  // on the receiver side.
  struct ncclIbSendFifo ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the send requests for faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array based on the "slot" they
  // are associated with.
  struct ncclIbRequest* sendReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // Counter per "slot" on how many send request were called for a multi-recv 
  int sendReqsCnt[NET_IB_MAX_REQUESTS];
  struct ncclIbRemCompletionsRecords remCmplsRecords;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0, "ncclIbSendComm ctsFifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct ncclIbRemCtsFifo {
  // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
  // the CTS FIFO locally on its side. Receiver uses this memory to place the
  // CTS messages and populates the RDMA message "gather address" with the
  // memory of the CTS message that is sent.
  struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the receiver chooses the
  // RKey (depending on the device being used) when it posts a CTS to the
  // sender
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  // MR that is obtained after registering the "shadow" CTS FIFO on the
  // receiver's side. The LKey of this MR allows RDMA operations on the receiver
  // side to gather CTS messages (formatted by the receiver) and write them to
  // the sender's CTS FIFO.
  struct ibv_mr* ctsFifoMr;
  // MR that is obtained after registering the completion records on the
  // receiver side. The RKey of this MR is provided to the sender side, to allow
  // the sender side to access receiver's completion records using RDMA
  // operations.
  struct ibv_mr* cmplsRecordsMr;
  // SGE to avoid allocation of SGE structures on the stack when receiver
  // posts RDMA operations. The SGE is populated by the address of the memory
  // in which the CTS message formatted on the receiver is placed.
  struct ibv_sge sge;
};

#define NCCL_IB_RECV_WR_ID_DUMMY UINT64_MAX

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the recv requests to allow faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array using a hash (modulo) on
  // their ID.
  struct ncclIbRequest* recvReqs[NET_IB_MAX_REQUESTS];
  // Structure to hold all the related structures regarding the CTS FIFO
  // structure.
  struct ncclIbRemCtsFifo remCtsFifo;
  // Structure to hold all the completion records of all the outstanding
  // receive requests on the receiver side.
  struct ncclIbRequestCompletionRecord cmplsRecords[NET_IB_MAX_REQUESTS];
  int gpuFlushHostMem;
  int flushEnabled;
  bool prepostReceiveWorkRequests;
  // To avoid allocation and memset on the data-path a single structure is used
  // and only the wr_id is updated before posting a receive work request.
  struct ibv_recv_wr ibRecvWorkRequest;
};
static_assert((offsetof(struct ncclIbRecvComm, remCtsFifo) % 32) == 0, "ncclIbRecvComm ctsFifo must be 32-byte aligned");

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend);
ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm);
ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm);

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
};

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern ncclProfilerCallback_t ncclProfilerFunction;

extern std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args);

ncclResult_t ncclIbGdrSupport();
ncclResult_t ncclIbPeerMemSupport();
ncclResult_t ncclIbDmaBufSupport(int dev);

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex);
ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex);
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r);

ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle);

// Net IB plugin entry functions.
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbDevices(int* ndev);
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/);
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/);
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle);
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request);
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclIbTest(void* request, int* done, int* sizes);
ncclResult_t ncclIbCloseSend(void* sendComm);
ncclResult_t ncclIbCloseRecv(void* recvComm);
ncclResult_t ncclIbCloseListen(void* listenComm);
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t ncclIbFinalizeDevices(void);
ncclResult_t ncclIbFinalize(void* ctx);
ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr);

#endif

