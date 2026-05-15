/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "shmutils.h"
#include "shm.h"
#include "transport.h"
#include "compiler.h"

#define SHM_PATH_MAX 128
#define SHM_HANDLE_TYPE ncclCuMemHandleType

struct shmBuffInfo {
  void *hptr;
  void *dptr;
};

struct shmConnectInfo {
  int rank;
  ncclShmIpcDesc_t desc;
  struct shmBuffInfo buf;
};

struct shmSendResources {
  struct ncclRecvMem* remHostMem;
  struct ncclRecvMem* devRemHostMem;
  ncclShmIpcDesc_t remDesc;
  struct ncclSendMem* hostMem;
  struct ncclSendMem* devHostMem;
};

struct shmRecvResources {
  struct ncclSendMem* remHostMem;
  struct ncclSendMem* devRemHostMem;
  ncclShmIpcDesc_t remDesc;
  struct ncclRecvMem* hostMem;
  struct ncclRecvMem* devHostMem;
};

struct shmProxyResources {
  ncclShmIpcDesc_t desc;
};

struct shmRequest {
  size_t size;
  bool legacy;
};

#define SHM_SEND_SIDE 1
#define SHM_RECV_SIDE 2
NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);
NCCL_PARAM(ShmLocality, "SHM_LOCALITY", SHM_RECV_SIDE); // 1 is sender-size, 2 is receiver-size
static int shmLocality = 0;
static void initShmLocality();

/* Determine two peers can communicate with SHM */
static ncclResult_t shmCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;
  initShmLocality();

  if (ncclParamShmDisable() == 1) return ncclSuccess;

  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) return ncclSuccess;

  // Same host?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 hostHash %lx peer2 hostHash %lx", info1->hostHash, info2->hostHash);
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // Common /dev/shm (between containers) ?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 shmDev %lx peer2 shmDev %lx", info1->shmDev, info2->shmDev);
  if (info1->shmDev != info2->shmDev) return ncclSuccess;

  *ret = 1;

  return ncclSuccess;
}

#define MAX_SHM_NAME_LEN 1024

/* Create and return connect structures for this peer to connect to me */
static ncclResult_t shmSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct shmSendResources* resources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  size_t shmSize = sizeof(struct ncclSendMem);
  struct shmRequest req;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");

  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  if (shmLocality == SHM_SEND_SIDE) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += comm->buffSizes[p];
  }
  req.size = shmSize;
  if (myInfo->hostHash == peerInfo->hostHash && myInfo->pidHash == peerInfo->pidHash)
    req.legacy = true;
  else
    req.legacy = false;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 1, myInfo->rank, &send->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, (void*)&req, sizeof(struct shmRequest), (void*)info, sizeof(struct shmConnectInfo)));

  info->rank = comm->rank;
  resources->hostMem = (struct ncclSendMem*)info->buf.hptr;
  resources->devHostMem = (struct ncclSendMem*)info->buf.dptr;

  INFO(NCCL_INIT|NCCL_SHM,"Channel %02d : %d[%d] -> %d[%d] via SHM/direct", channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
  return ncclSuccess;
}

static ncclResult_t shmRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct shmRecvResources* resources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  size_t shmSize = sizeof(struct ncclRecvMem);
  struct shmRequest req;

  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");

  if (shmLocality == SHM_RECV_SIDE) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += comm->buffSizes[p];
  }
  req.size = shmSize;
  if (myInfo->hostHash == peerInfo->hostHash && myInfo->pidHash == peerInfo->pidHash)
    req.legacy = true;
  else
    req.legacy = false;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 0, myInfo->rank, &recv->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, (void*)&req, sizeof(struct shmRequest), (void*)info, sizeof(struct shmConnectInfo)));

  info->rank = comm->rank;
  resources->hostMem = (struct ncclRecvMem*)info->buf.hptr;
  resources->devHostMem = (struct ncclRecvMem*)info->buf.dptr;

  return ncclSuccess;
}

/* Connect to this peer */
static ncclResult_t shmSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;
  char* buff;

  NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, &resources->remDesc));

  buff = shmLocality == SHM_SEND_SIDE ? (char*)(resources->devHostMem + 1) : (char*)(resources->devRemHostMem + 1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = buff;
    buff += comm->buffSizes[p];
  }
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.head = &resources->devHostMem->head;
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  send->proxyConn.proxyProgress = shmTransport.send.proxyProgress;

  return ncclSuccess;
}

static ncclResult_t shmRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  char* buff;

  NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, &resources->remDesc));

  buff = shmLocality == SHM_RECV_SIDE ? (char*)(resources->devHostMem + 1) : (char*)(resources->devRemHostMem + 1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = buff;
    buff += comm->buffSizes[p];
  }
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.tail = &resources->devHostMem->tail;
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  recv->proxyConn.proxyProgress = shmTransport.recv.proxyProgress;

  return ncclSuccess;
}

static ncclResult_t shmSendFree(struct ncclComm* comm, struct ncclConnector* send) {
  struct shmRecvResources* resources = (struct shmRecvResources*)send->transportResources;
  if (resources) {
    NCCLCHECK(ncclShmIpcClose(&resources->remDesc));
    free(resources);
    send->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvFree(struct ncclComm* comm, struct ncclConnector* recv) {
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  if (resources) {
    NCCLCHECK(ncclShmIpcClose(&resources->remDesc));
    free(resources);
    recv->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct shmProxyResources* resources = (struct shmProxyResources*)connection->transportResources;

  if (resources) {
    NCCLCHECK(ncclShmIpcClose(&resources->desc));
    free(connection->transportResources);
    connection->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct shmProxyResources* resources = (struct shmProxyResources*)connection->transportResources;

  if (resources) {
    NCCLCHECK(ncclShmIpcClose(&resources->desc));
    free(connection->transportResources);
    connection->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t result = ncclSuccess;
  struct shmRequest* req = (struct shmRequest*)reqBuff;
  /* check message size */
  if (reqSize != sizeof(struct shmRequest)) return ncclInternalError;
  if (respSize != sizeof(struct shmConnectInfo)) return ncclInternalError;

  struct shmConnectInfo* info = (struct shmConnectInfo*)respBuff;
  struct shmProxyResources* proxyRes;

  NCCLCHECK(ncclCalloc(&proxyRes, 1));
  NCCLCHECKGOTO(ncclShmAllocateShareableBuffer(req->size, req->legacy, &proxyRes->desc, &info->buf.hptr, &info->buf.dptr), result, fail);
  memcpy(&info->desc, &proxyRes->desc, sizeof(ncclShmIpcDesc_t));
  connection->transportResources = proxyRes;
exit:
  return result;
fail:
  free(proxyRes);
  goto exit;
}

static ncclResult_t shmRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t result = ncclSuccess;
  struct shmRequest* req = (struct shmRequest*)reqBuff;
  /* check message size */
  if (reqSize != sizeof(struct shmRequest)) return ncclInternalError;
  if (respSize != sizeof(struct shmConnectInfo)) return ncclInternalError;

  struct shmConnectInfo* info = (struct shmConnectInfo*)respBuff;
  struct shmProxyResources* proxyRes;

  NCCLCHECK(ncclCalloc(&proxyRes, 1));
  NCCLCHECKGOTO(ncclShmAllocateShareableBuffer(req->size, req->legacy, &proxyRes->desc, &info->buf.hptr, &info->buf.dptr), result, fail);
  memcpy(&info->desc, &proxyRes->desc, sizeof(ncclShmIpcDesc_t));
  connection->transportResources = proxyRes;
exit:
  return result;
fail:
  free(proxyRes);
  goto exit;
}

static void initShmLocality() {
  static int init = 0;
  if (!init) {
    shmLocality = ncclParamShmLocality();
    if (shmLocality != SHM_SEND_SIDE && shmLocality != SHM_RECV_SIDE) {
      WARN("Ignoring SHM locality, must be 1 (sender side) or 2 (receiver side, default)");
      shmLocality = SHM_RECV_SIDE;
    }
    init = 1;
  }
}

ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t *desc, void **hptr, void **dptr) {
  if (desc == NULL || hptr == NULL) {
    WARN("Invalid argument desc %p, hptr %p", desc, hptr);
    return ncclInvalidArgument;
  }
#if CUDART_VERSION >= 12020
  if (ncclCuMemEnable() && ncclCuMemHostEnable() && !legacy) {
    // cuMem API support
    CUmemAllocationHandleType type = SHM_HANDLE_TYPE;
    CUmemGenericAllocationHandle handle;

    NCCLCHECK(ncclCuMemHostAlloc(hptr, &handle, size));
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Return the native cuMem handle for later Export/Import via UDS
      memcpy(&desc->shmci.data, &handle, sizeof(handle));
    } else {
      CUCHECK(cuMemExportToShareableHandle(&desc->shmci.handle, handle, type, 0));
    }
    desc->shmci.size = size;
    desc->shmci.ptr = *hptr;
    if (dptr) *dptr = *hptr;
    desc->legacy = false;
    INFO(NCCL_SHM, "CUMEM allocated shareable buffer %p size %zi", desc->shmci.ptr, desc->shmci.size);
  } else {
    char shmPath[SHM_PATH_MAX] = { '\0' };
    desc->shmli.shmSize = size;
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
#if defined(NCCL_OS_LINUX)
    static const char shmPrefix[] = "/dev/shm/nccl-";
#elif defined(NCCL_OS_WINDOWS)
    static const char shmPrefix[] = "Local\\nccl-shm-";
#endif
    memcpy(desc->shmli.shmSuffix, shmPath + sizeof(shmPrefix) - 1, sizeof(desc->shmli.shmSuffix));
    desc->legacy = true;
    INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
  }
#else /* CUDART_VERSION >= 12020 */
  char shmPath[SHM_PATH_MAX] = { '\0' };
  desc->shmli.shmSize = size;
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
#if defined(NCCL_OS_LINUX)
  static const char shmPrefix[] = "/dev/shm/nccl-";
#elif defined(NCCL_OS_WINDOWS)
  static const char shmPrefix[] = "Local\\nccl-shm-";
#endif
  memcpy(desc->shmli.shmSuffix, shmPath + sizeof(shmPrefix) - 1, sizeof(desc->shmli.shmSuffix));
  desc->legacy = true;
  INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, size, *hptr);
#endif /* CUDART_VERSION >= 12020 */
  return ncclSuccess;
}

ncclResult_t ncclShmImportShareableBuffer(struct ncclComm *comm, int proxyRank, ncclShmIpcDesc_t *desc, void **hptr, void **dptr, ncclShmIpcDesc_t *descOut) {
  if (comm == NULL || desc == NULL || hptr == NULL || descOut == NULL) {
    WARN("Invalid argument comm %p, desc %p, hptr %p, descOut %p", comm, desc, hptr, descOut);
    return ncclInvalidArgument;
  }
#if CUDART_VERSION >= 12020
  if (ncclCuMemEnable() && ncclCuMemHostEnable() && !desc->legacy) {
    // cuMem API support
    CUdeviceptr hostptr = 0;
    CUmemAllocationHandleType type = SHM_HANDLE_TYPE;
    CUmemGenericAllocationHandle handle;
    int cudaDev;
    CUdevice currentDev;
    CUmemAccessDesc accessDesc = {};
    int cpuNumaNodeId;
    size_t granularity;
    size_t size = desc->shmci.size;
    CUmemAllocationProp prop = {};

    // Import and map the remote memory descriptor to the local GPU
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // UDS fd support
      int fd = -1;
      // Send cuMem handle to remote for conversion to an fd
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, proxyRank, &desc->shmci.data, &fd));
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
      (void) close(fd);
    } else {
      CUCHECK(cuMemImportFromShareableHandle(&handle, &desc->shmci.handle, type));
    }

    // Get cpu numa id
    CUDACHECK(cudaGetDevice(&cudaDev));
    CUCHECK(cuDeviceGet(&currentDev, cudaDev));
    CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
    if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;

    // Get granularity
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = type;
    prop.location.id = cpuNumaNodeId;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    ALIGN_SIZE(size, granularity);

    // Reserve and map address
    CUCHECK(cuMemAddressReserve(&hostptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    CUCHECK(cuMemMap(hostptr, size, /* offset */ 0, handle, /* flags */ 0));

    // Allow access by the local GPU
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(hostptr, size, &accessDesc, 1));

    // Allow access by the local numa
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    accessDesc.location.id = cpuNumaNodeId;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(hostptr, size, &accessDesc, 1));

    descOut->shmci.ptr = *hptr = (void *)hostptr;
    descOut->legacy = false;
    if (dptr) *dptr = (void *)hostptr;
    INFO(NCCL_SHM, "CUMEM imported shareable host buffer from proxyRank %d size %zi ptr %p, granularity %ld", proxyRank, desc->shmci.size, descOut->shmci.ptr, granularity);
  } else {
    char shmPath[SHM_PATH_MAX];
#if defined(NCCL_OS_LINUX)
    snprintf(shmPath, sizeof(shmPath), "/dev/shm/nccl-%s", desc->shmli.shmSuffix);
#elif defined(NCCL_OS_WINDOWS)
    snprintf(shmPath, sizeof(shmPath), "Local\\nccl-shm-%s", desc->shmli.shmSuffix);
#endif
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), desc->shmli.shmSize, hptr, dptr, -1, &descOut->shmli.handle));
    descOut->legacy = true;
    INFO(NCCL_SHM, "MMAP imported shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
  }
#else /* CUDART_VERSION >= 12020 */
  char shmPath[SHM_PATH_MAX];
#if defined(NCCL_OS_LINUX)
  snprintf(shmPath, sizeof(shmPath), "/dev/shm/nccl-%s", desc->shmli.shmSuffix);
#elif defined(NCCL_OS_WINDOWS)
  snprintf(shmPath, sizeof(shmPath), "Local\\nccl-shm-%s", desc->shmli.shmSuffix);
#endif
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), desc->shmli.shmSize, hptr, dptr, -1, &descOut->shmli.handle));
  descOut->legacy = true;
  INFO(NCCL_SHM, "MMAP imported shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
#endif
  return ncclSuccess;
}

ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t *desc) {
  if (desc) {
#if CUDART_VERSION >= 12020
    if (ncclCuMemEnable() && ncclCuMemHostEnable() && !desc->legacy) {
      NCCLCHECK(ncclCuMemHostFree(desc->shmci.ptr));
    } else {
      NCCLCHECK(ncclShmClose(desc->shmli.handle));
    }
#else
    NCCLCHECK(ncclShmClose(desc->shmli.handle));
#endif
  }

  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  "SHM",
  shmCanConnect,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL, shmSendProxySetup, NULL, shmSendProxyFree, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL, shmRecvProxySetup, NULL, shmRecvProxyFree, NULL }
};
