/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_ND_COMMON_H_
#define NET_ND_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "utils.h"
#include "param.h"

#include "ndwrap.h"

#include <ws2tcpip.h> // for SOCKADDR_INET
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <mutex>

// Forward declarations for NCCL_PARAM functions defined in common.cc
int64_t ncclParamNdDisable();
int64_t ncclParamNdUseInline();
int64_t ncclParamNdQpsPerConnection();
int64_t ncclParamNdResiliencyPortFailover();

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)

// GPU memory type detection
enum ncclNdMemType {
  ncclNdMemTypeHost = 0,
  ncclNdMemTypeDevice = 1
};

// Maximum devices
#define MAX_ND_DEVS 32
#define MAX_ND_VDEVS (MAX_ND_DEVS * 8)
#define NCCL_ND_MAX_CUDA_DEVS 64
#define NCCL_ND_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC
#define MAX_MERGED_DEV_NAME ((MAXNAMESIZE * NCCL_ND_MAX_DEVS_PER_NIC) + NCCL_ND_MAX_DEVS_PER_NIC)

// NetworkDirect specific constants
#define NCCL_NET_ND_MAX_RECVS 8
#define NET_ND_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * NCCL_NET_ND_MAX_RECVS)
#define NCCL_ND_MAX_QPS 128
#define NCCL_ND_MAX_CQ_POLL_BATCH 256
#define NCCL_ND_CONNECT_TIMEOUT_SECONDS 60
#define NCCL_ND_DATA_TIMEOUT_SECONDS 60
#define NCCL_ND_MAX_COMMS 128
#define NCCL_ND_MAX_MRS 4096
#define NCCL_ND_CQ_POLL_BATCH 64
#define NCCL_ND_PROVIDER_CHECK_INTERVAL_MS 100
#define NCCL_ND_LINK_CHECK_INTERVAL_MS 1000

// Queue depths derived from core constants (like IB plugin)
// Send depth is 2x because we may send RDMA Write + completion notification
#define NCCL_ND_SEND_WR_DEPTH (2 * NET_ND_MAX_REQUESTS)
#define NCCL_ND_RECV_WR_DEPTH NET_ND_MAX_REQUESTS

#define NCCL_ND_MAX_INBOUND_READS 1
#define NCCL_ND_MAX_OUTBOUND_READS 1
#define NCCL_ND_LISTENER_BACKLOG 256

// Request types
#define NCCL_NET_ND_REQ_UNUSED 0
#define NCCL_NET_ND_REQ_SEND 1
#define NCCL_NET_ND_REQ_RECV 2
#define NCCL_NET_ND_REQ_FLUSH 3

#define NCCL_ND_SEND_STAGE_DATA 0
#define NCCL_ND_SEND_STAGE_WAIT_ACK 1
#define NCCL_ND_SEND_STAGE_RELEASE 2
#define NCCL_ND_RECV_STAGE_CTS 0
#define NCCL_ND_RECV_STAGE_ACK 1
#define NCCL_ND_RECV_STAGE_WAIT_RELEASE 2

// Memory region structure
struct ncclNdMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  struct IND2MemoryRegion* mr;
  // IND2MemoryRegion keeps using the overlapped file supplied at creation.
  // Cache entries are global to the adapter, so this handle must have the
  // same lifetime as the cached provider object rather than any communicator.
  HANDLE ovFile;
};

// Memory region cache
struct ncclNdMrCache {
  struct ncclNdMr* slots;
  int capacity, population;
};

// Device statistics
struct ncclNdStats {
  std::atomic<int> fatalErrorCount;
  std::atomic<int> activeComms;
  std::atomic<int> peakComms;
  std::atomic<int> activeQps;
  std::atomic<int> peakQps;
  std::atomic<int> activeCqs;
  std::atomic<int> peakCqs;
  std::atomic<int> activeMrs;
  std::atomic<int> peakMrs;
  std::atomic<int> activeRequests;
  std::atomic<int> peakRequests;
  std::atomic<uint64_t> commBackpressure;
  std::atomic<uint64_t> requestBackpressure;
  std::atomic<uint64_t> mrAdmissionFailures;
  std::atomic<uint64_t> inlineWrites;
  std::atomic<uint64_t> connectionAttempts;
  std::atomic<uint64_t> connectionSuccesses;
  std::atomic<uint64_t> connectionFailures;
  std::atomic<uint64_t> connectionTimeouts;
  std::atomic<uint64_t> dataTimeouts;
  std::atomic<uint64_t> peerDisconnects;
  std::atomic<uint64_t> cqErrors;
  std::atomic<uint64_t> registrationFailures;
  std::atomic<uint64_t> healthChecks;
  std::atomic<uint64_t> providerNotifications;
  std::atomic<uint64_t> adapterHealthFailures;
  std::atomic<uint64_t> linkEvents;
  std::atomic<uint64_t> railFailures;
  std::atomic<uint64_t> railFailovers;
  std::atomic<uint64_t> replayedRequests;
  std::atomic<uint64_t> failoverFailures;
};

// GPU capability flags
struct ncclNdGpuCaps {
  // GDR support depends on both the current CUDA device and the ND adapter.
  // -1 means that the pair has not been probed yet.
  int gpuDirectSupported[NCCL_ND_MAX_CUDA_DEVS];
  int forceFlush[NCCL_ND_MAX_CUDA_DEVS];
};

// Per-device structure
struct alignas(64) ncclNdDev {
  std::mutex mutex;
  int device;
  UINT64 adapterId;
  UINT64 interfaceLuid;
  volatile LONG64 nextHealthCheckMs;
  volatile LONG64 linkFailureGeneration;
  volatile LONG healthState;
  struct IND2Adapter* adapter;
  union ncclSocketAddress addr;
  char devName[MAXNAMESIZE];
  char pciPath[PATH_MAX];
  int speed;
  int port;
  int maxComms;
  int maxMrs;
  float latency;
  struct ncclNdMrCache mrCache;
  struct ncclNdStats stats;
  ND2_ADAPTER_INFO adapterInfo;
  struct ncclNdGpuCaps gpuCaps;
};

static inline ULONG ncclNdConfiguredInlineBytes(const struct ncclNdDev* dev) {
  if (ncclParamNdUseInline() == 0) return 0;
  return dev->adapterInfo.MaxInlineDataSize;
}

// Merged/virtual device structure
struct alignas(64) ncclNdMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME];
};

// Global device arrays
extern int ncclNNdDevs;
extern int ncclNMergedNdDevs;
extern struct ncclNdDev ncclNdDevs[MAX_ND_DEVS];
extern struct ncclNdMergedDev ncclNdMergedDevs[MAX_ND_VDEVS];
extern union ncclSocketAddress ncclNdSocketAddr;

// Per-device connection metadata
struct ncclNdDevInfo {
  UINT32 ctsFifoToken; // Remote token for CTS FIFO
  UINT32 completionToken; // Remote token for completion records
  UINT64 ctsFifoAddr; // Address of CTS FIFO
  UINT64 completionAddr; // Address of completion records
};

#define NCCL_ND_PROTOCOL_MAGIC UINT64_C(0x31444e4c43434e) // "NCCLND1"
#define NCCL_ND_PROTOCOL_VERSION 4
#define NCCL_ND_PROTOCOL_CAP_TAGGED_WRITE UINT64_C(0x1)
#define NCCL_ND_PROTOCOL_CAP_GENERATION_COMPLETION UINT64_C(0x2)
#define NCCL_ND_PROTOCOL_CAP_ORDERLY_SHUTDOWN UINT64_C(0x4)
#define NCCL_ND_PROTOCOL_CAP_MULTI_QP UINT64_C(0x8)
#define NCCL_ND_PROTOCOL_CAP_MULTI_ADAPTER UINT64_C(0x10)
#define NCCL_ND_PROTOCOL_CAP_DELIVERY_ACK UINT64_C(0x20)
#define NCCL_ND_METADATA_OPTION_FAILOVER UINT32_C(0x1)
#define NCCL_ND_PROTOCOL_REQUIRED_CAPS \
  (NCCL_ND_PROTOCOL_CAP_TAGGED_WRITE | NCCL_ND_PROTOCOL_CAP_GENERATION_COMPLETION | \
   NCCL_ND_PROTOCOL_CAP_ORDERLY_SHUTDOWN | NCCL_ND_PROTOCOL_CAP_MULTI_QP | NCCL_ND_PROTOCOL_CAP_MULTI_ADAPTER | \
   NCCL_ND_PROTOCOL_CAP_DELIVERY_ACK)
#define NCCL_ND_CONTROL_CLOSE 'C'

// A fixed-size preamble is exchanged before the metadata body so peers can
// reject incompatible layouts before interpreting addresses or remote tokens.
struct ncclNdProtocolHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t headerSize;
  uint32_t metadataSize;
  uint32_t reserved;
  uint64_t capabilities;
};
static_assert(sizeof(struct ncclNdProtocolHeader) == 32, "ND protocol header layout changed");
static_assert(sizeof(struct ncclNdDevInfo) == 24, "ND device metadata layout changed");

// Connection metadata exchanged over TCP
struct ncclNdConnectionMetadata {
  uint32_t ndevs; // Number of devices
  uint32_t options; // Enabled wire-protocol options
  struct ncclNdDevInfo devs[NCCL_ND_MAX_DEVS_PER_NIC]; // Per-device info
  char devName[MAX_MERGED_DEV_NAME]; // Device name for logging
};

static inline uint32_t ncclNdConnectionOptions(void) {
  return ncclParamNdResiliencyPortFailover() != 0 ? NCCL_ND_METADATA_OPTION_FAILOVER : 0;
}

// The fixed NCCL handle carries only the OOB address. Once TCP is established,
// the listener sends every ND address needed by a multi-adapter communicator.
struct ncclNdConnectionSetup {
  uint64_t magic;
  uint32_t version;
  uint32_t ndevs;
  uint32_t qpsPerDev;
  uint32_t reserved;
  union ncclSocketAddress addrs[NCCL_ND_MAX_DEVS_PER_NIC];
};

// Queue pair structure
struct ncclNdQp {
  struct IND2QueuePair* qp;
  struct IND2Connector* connector;
  OVERLAPPED ov;
  int opDone;
  int devIndex;
  int remDevIdx;
  ULONG inlineDataSize;
  volatile LONG64 outstanding;
  volatile LONG64 outstandingBytes;
  volatile LONG64 postedOps;
};

// CTS (Clear-to-Send) FIFO entry
struct ncclNdSendFifo {
  uint64_t addr;
  uint64_t size;
  UINT32 tokens[NCCL_ND_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
};

// Remote completion records structure (sender side)
struct ncclNdRemCompletionRecords {
  uint64_t elems[NET_ND_MAX_REQUESTS][NCCL_NET_ND_MAX_RECVS];
  uint64_t acknowledgements[NET_ND_MAX_REQUESTS][NCCL_NET_ND_MAX_RECVS];
  uint64_t addr;
  UINT32 tokens[NCCL_ND_MAX_DEVS_PER_NIC];
};

// Per-device communicator base
struct ncclNdNetCommDevBase {
  int ndDevN;
  struct IND2CompletionQueue* cq;
  HANDLE ovFile;
  OVERLAPPED cqErrorOv;
  bool cqErrorArmed;
};

// Per-device send communicator
struct alignas(8) ncclNdSendCommDev {
  struct ncclNdNetCommDevBase base;
  struct IND2MemoryRegion* ctsFifoMr;
  struct IND2MemoryRegion* cmplsRecordsMr;
  bool ctsFifoRegistered;
  bool cmplsRecordsRegistered;
};

// MR handle wrapper for multi-device
struct ncclNdMrHandle {
  struct IND2MemoryRegion* mrs[NCCL_ND_MAX_DEVS_PER_NIC];
};

struct ncclNdRequest;

// Communicators are allocated with ncclCalloc, so use Windows SRW locks whose
// all-zero representation is a valid initial state. std::mutex would require a
// constructor that calloc does not run.
struct ncclNdScopedSrwLock {
  explicit ncclNdScopedSrwLock(SRWLOCK* lock) : lock(lock) {
    AcquireSRWLockExclusive(lock);
  }
  ~ncclNdScopedSrwLock() {
    ReleaseSRWLockExclusive(lock);
  }
  ncclNdScopedSrwLock(const ncclNdScopedSrwLock&) = delete;
  ncclNdScopedSrwLock& operator=(const ncclNdScopedSrwLock&) = delete;
  SRWLOCK* lock;
};

// Request structure
struct ncclNdRequest {
  struct ncclNdNetCommBase* base;
  int type;
  uint64_t id;
  ULONGLONG deadlineMs;
  int qpIndex;
  LONG64 postedBytes;
  int replayCount;
  int events[NCCL_ND_MAX_DEVS_PER_NIC];
  struct ncclNdNetCommDevBase* devBases[NCCL_ND_MAX_DEVS_PER_NIC];
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      UINT32 localTokens[NCCL_ND_MAX_DEVS_PER_NIC];
      UINT32 remoteTokens[NCCL_ND_MAX_DEVS_PER_NIC];
      UINT64 remoteAddr;
      int offset;
      int slot;
      int index;
      int stage;
    } send;
    struct {
      int* sizes;
      int slot;
      int stage;
    } recv;
  };
};

// Base communicator structure
struct alignas(32) ncclNdNetCommBase {
  SRWLOCK reqLock;
  SRWLOCK progressLock;
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  volatile LONG fatalError;
  struct ncclNdRequest reqs[NET_ND_MAX_REQUESTS];
  struct ncclNdQp qps[NCCL_ND_MAX_QPS];
  uint64_t fifoHead;
  volatile LONG64 nextPeerCheckMs;
  volatile LONG64 nextProviderCheckMs;
  volatile LONG64 nextLinkCheckMs;
  LONG64 linkFailureGenerations[NCCL_ND_MAX_DEVS_PER_NIC];
  uint64_t commId;
  volatile LONG connectionOutcome;
  struct ncclNdStats* stats;
  volatile LONG peerClosing;
  volatile LONG failedDevicesMask;
  uint32_t admissionMask;
  int primaryDev;
  int nqps;
  int ndevs;
  int nRemDevs; // Number of remote devices (from metadata exchange)
  int qpIndex; // Round-robin QP selection index

  // CTS FIFO structures
  struct ncclNdSendFifo* localCtsFifo;
  struct ncclNdSendFifo* remCtsFifo;
  UINT32 remCtsFifoTokens[NCCL_ND_MAX_DEVS_PER_NIC];

  // Completion records
  uint64_t completionRecords[NET_ND_MAX_REQUESTS][NCCL_NET_ND_MAX_RECVS];
  uint64_t releaseRecords[NET_ND_MAX_REQUESTS][NCCL_NET_ND_MAX_RECVS];
  struct ncclNdRemCompletionRecords remCompletionRecords;
};

// Send communicator
struct ncclNdSendComm {
  struct ncclNdNetCommBase base;
  struct ncclNdSendCommDev devs[NCCL_ND_MAX_DEVS_PER_NIC];
  struct ncclSocket sock;
  int state;
  ULONGLONG deadlineMs;
  ULONGLONG ctsDeadlineMs;
  int setupOffset;
  int connectQpIndex;
  int qpsPerDev;
  struct ncclNdConnectionSetup remSetup;
  struct ncclNdProtocolHeader localHeader;
  struct ncclNdProtocolHeader remHeader;
  int headerOffset;
  struct ncclNdConnectionMetadata localMeta;
  struct ncclNdConnectionMetadata remMeta;
  int metaOffset;
  struct ncclNdRequest* sendReqs[NET_ND_MAX_REQUESTS][NCCL_NET_ND_MAX_RECVS];
  int sendReqsCnt[NET_ND_MAX_REQUESTS];
};

// Per-device receive communicator
struct ncclNdRecvCommDev {
  struct ncclNdNetCommDevBase base;
  struct IND2MemoryRegion* ctsFifoMr;
  struct IND2MemoryRegion* cmplsRecordsMr;
  bool ctsFifoRegistered;
  bool cmplsRecordsRegistered;
};

// Receive communicator
struct ncclNdRecvComm {
  struct ncclNdNetCommBase base;
  struct ncclNdRecvCommDev devs[NCCL_ND_MAX_DEVS_PER_NIC];
  struct ncclSocket sock;
  int state;
  ULONGLONG deadlineMs;
  int acceptQpIndex;
  int qpsPerDev;
  struct ncclNdConnectionSetup localSetup;
  int setupOffset;
  struct ncclNdProtocolHeader localHeader;
  struct ncclNdProtocolHeader remHeader;
  int headerOffset;
  struct ncclNdConnectionMetadata localMeta;
  struct ncclNdConnectionMetadata remMeta;
  int metaOffset;
  struct ncclNdRequest* recvReqs[NET_ND_MAX_REQUESTS];
};

struct ncclNdHandle {
  union ncclSocketAddress oobAddr;
  struct ncclNdSendComm* sendComm;
};

static_assert(sizeof(struct ncclNdHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclNdHandle size too large");

// Listen communicator
struct ncclNdListenComm {
  int dev;
  int ndevs;
  ncclNetVDeviceProps_t vProps;
  struct ncclSocket sock;
  struct IND2Listener* listeners[NCCL_ND_MAX_DEVS_PER_NIC];
  HANDLE ovFiles[NCCL_ND_MAX_DEVS_PER_NIC];
  union ncclSocketAddress ndAddrs[NCCL_ND_MAX_DEVS_PER_NIC];
  struct ncclNdRecvComm* pendingRecvComm;
};

// Connection states
enum ncclNdConnectState {
  ncclNdConnectInit = 0,
  ncclNdConnectTcpConnect = 1,
  ncclNdConnectWaitRequest = 2,
  ncclNdConnectTcpConnected = 3,
  ncclNdConnectNdConnect = 4,
  ncclNdConnectNdCompleteConnect = 5,
  ncclNdConnectSendHeader = 6,
  ncclNdConnectSendMeta = 7,
  ncclNdConnectRecvHeader = 8,
  ncclNdConnectRecvMeta = 9,
  ncclNdConnectReady = 10,
};

enum ncclNdAcceptState {
  ncclNdAcceptInit = 0,
  ncclNdAcceptTcpAccept = 1,
  ncclNdAcceptTcpAccepted = 2,
  ncclNdAcceptSendSetup = 3,
  ncclNdAcceptNdGetRequest = 4,
  ncclNdAcceptNdAccept = 5,
  ncclNdAcceptNdAcceptWait = 6,
  ncclNdAcceptRecvHeader = 7,
  ncclNdAcceptRecvMeta = 8,
  ncclNdAcceptSendHeader = 9,
  ncclNdAcceptSendMeta = 10,
  ncclNdAcceptReady = 11,
};

// Helper to get device-specific comm base
static inline struct ncclNdNetCommDevBase* ncclNdGetNetCommDevBase(struct ncclNdNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclNdSendComm* sComm = (struct ncclNdSendComm*)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclNdRecvComm* rComm = (struct ncclNdRecvComm*)base;
    return &rComm->devs[devIndex].base;
  }
}

// Communicator errors are sticky: the first fatal result is returned by all
// subsequent data-path calls so no request can silently remain pending after
// another request observes a provider or protocol failure.
static inline ncclResult_t ncclNdGetFatalError(struct ncclNdNetCommBase* base) {
  return (ncclResult_t)InterlockedCompareExchange(&base->fatalError, ncclSuccess, ncclSuccess);
}

static inline ncclResult_t ncclNdSetFatalError(struct ncclNdNetCommBase* base, ncclResult_t error, bool* first = NULL) {
  if (error == ncclSuccess) return ncclSuccess;
  LONG previous = InterlockedCompareExchange(&base->fatalError, (LONG)error, ncclSuccess);
  bool wasFirst = previous == ncclSuccess;
  if (first != NULL) *first = wasFirst;
  if (wasFirst && base->stats != NULL) {
    base->stats->fatalErrorCount.fetch_add(1, std::memory_order_relaxed);
  }
  return previous == ncclSuccess ? error : (ncclResult_t)previous;
}

// Stats management and checking
static inline ncclResult_t ncclNdStatsInit(struct ncclNdStats* stat) {
  stat->fatalErrorCount.store(0, std::memory_order_relaxed);
  stat->activeComms.store(0, std::memory_order_relaxed);
  stat->peakComms.store(0, std::memory_order_relaxed);
  stat->activeQps.store(0, std::memory_order_relaxed);
  stat->peakQps.store(0, std::memory_order_relaxed);
  stat->activeCqs.store(0, std::memory_order_relaxed);
  stat->peakCqs.store(0, std::memory_order_relaxed);
  stat->activeMrs.store(0, std::memory_order_relaxed);
  stat->peakMrs.store(0, std::memory_order_relaxed);
  stat->activeRequests.store(0, std::memory_order_relaxed);
  stat->peakRequests.store(0, std::memory_order_relaxed);
  stat->commBackpressure.store(0, std::memory_order_relaxed);
  stat->requestBackpressure.store(0, std::memory_order_relaxed);
  stat->mrAdmissionFailures.store(0, std::memory_order_relaxed);
  stat->inlineWrites.store(0, std::memory_order_relaxed);
  stat->connectionAttempts.store(0, std::memory_order_relaxed);
  stat->connectionSuccesses.store(0, std::memory_order_relaxed);
  stat->connectionFailures.store(0, std::memory_order_relaxed);
  stat->connectionTimeouts.store(0, std::memory_order_relaxed);
  stat->dataTimeouts.store(0, std::memory_order_relaxed);
  stat->peerDisconnects.store(0, std::memory_order_relaxed);
  stat->cqErrors.store(0, std::memory_order_relaxed);
  stat->registrationFailures.store(0, std::memory_order_relaxed);
  stat->healthChecks.store(0, std::memory_order_relaxed);
  stat->providerNotifications.store(0, std::memory_order_relaxed);
  stat->adapterHealthFailures.store(0, std::memory_order_relaxed);
  stat->linkEvents.store(0, std::memory_order_relaxed);
  stat->railFailures.store(0, std::memory_order_relaxed);
  stat->railFailovers.store(0, std::memory_order_relaxed);
  stat->replayedRequests.store(0, std::memory_order_relaxed);
  stat->failoverFailures.store(0, std::memory_order_relaxed);
  return ncclSuccess;
}
void ncclNdStatsAddResource(std::atomic<int>* active, std::atomic<int>* peak);
void ncclNdStatsRemoveResource(std::atomic<int>* active);
void ncclNdStatsConnectionStart(struct ncclNdNetCommBase* base, int dev);
void ncclNdStatsConnectionComplete(struct ncclNdNetCommBase* base);
void ncclNdStatsConnectionFailed(struct ncclNdNetCommBase* base, bool timedOut);
void ncclNdLogStats(struct ncclNdDev* dev);
bool ncclNdTryAcquireComm(struct ncclNdNetCommBase* base);
void ncclNdReleaseComm(struct ncclNdNetCommBase* base);
void ncclNdNotifyPeerClosing(struct ncclNdNetCommBase* base);
ncclResult_t ncclNdHandleRailFailure(struct ncclNdNetCommBase* base, int devIndex, const char* reason);

// Event tracking for request completion
static inline void ncclNdAddEvent(struct ncclNdRequest* req, int devIndex, struct ncclNdNetCommDevBase* base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

// Request management
ncclResult_t ncclNdGetRequest(struct ncclNdNetCommBase* base, int type, struct ncclNdRequest** req);
ncclResult_t ncclNdFreeRequest(struct ncclNdRequest* r);
ncclResult_t ncclNdGetGpuDirectSupport(struct ncclNdDev* dev, int* supported, int* forceFlush);
ncclResult_t ncclNdCheckAdapterHealth(int dev, bool* healthy);
void ncclNdRecordInterfaceChange(UINT64 interfaceLuid, bool healthy);
void ncclNdRecordDeviceInterfaceChange(int dev, bool healthy);
bool ncclNdInterfaceMonitorActive();
void ncclNdInitProtocolHeader(struct ncclNdProtocolHeader* header);
ncclResult_t ncclNdValidateProtocolHeader(const struct ncclNdProtocolHeader* header);
ncclResult_t ncclNdValidateConnectionMetadata(const struct ncclNdConnectionMetadata* metadata, int expectedDevices);
ncclResult_t ncclNdValidateConnectionSetup(const struct ncclNdConnectionSetup* setup, int expectedDevices,
                                           int expectedQpsPerDev);

// Memory registration
ncclResult_t ncclNdFinalizeMrCache(struct ncclNdDev* dev);

// Plugin entry functions
ncclResult_t ncclNdInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction,
                        ncclProfilerCallback_t profFunction);
ncclResult_t ncclNdDevices(int* ndev);
ncclResult_t ncclNdGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclNdListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclNdConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm);
ncclResult_t ncclNdAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm);
ncclResult_t ncclNdRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t ncclNdRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t ncclNdDeregMr(void* comm, void* mhandle);
ncclResult_t ncclNdIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request);
ncclResult_t ncclNdIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request);
ncclResult_t ncclNdIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclNdTest(void* request, int* done, int* sizes);
ncclResult_t ncclNdCloseSend(void* sendComm);
ncclResult_t ncclNdCloseRecv(void* recvComm);
ncclResult_t ncclNdCloseListen(void* listenComm);
ncclResult_t ncclNdMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t ncclNdFinalizeDevices(void);
ncclResult_t ncclNdFinalize(void* ctx);
ncclResult_t ncclNdSetNetAttr(void* ctx, ncclNetAttr_t* netAttr);

#endif // NET_ND_COMMON_H_
