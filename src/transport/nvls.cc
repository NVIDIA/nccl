/*************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the NVLink SHARP (NVLS) transport

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "proxy.h"
#include "enqueue.h"

#if CUDART_VERSION >= 12010

struct graphRegData {
  uintptr_t offset;
  size_t size;
};

struct localRegData {
  /* Registration record data */
  uintptr_t recSendbuff, recRecvbuff;
  intptr_t recSendOffset, recRecvOffset;
  /* Registration request data */
  uintptr_t reqSendbuff, reqRecvbuff;
  size_t reqSendSize, reqRecvSize;
  intptr_t reqSendOffset, reqRecvOffset;
};

struct localRequestData {
  uintptr_t reqBuff;
  size_t reqSize;
  intptr_t reqOffset;
};

ncclResult_t nvlsCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  *ret = 0;
  return ncclSuccess;
}

ncclResult_t nvlsSendFree(struct ncclConnector* send) {
  return ncclSuccess;
}

ncclResult_t nvlsRecvFree(struct ncclConnector* recv) {
  return ncclSuccess;
}

struct ncclTransport nvlsTransport = {
  "NVLS",
  nvlsCanConnect,
  { NULL, NULL, nvlsSendFree, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, nvlsRecvFree, NULL, NULL, NULL, NULL, NULL }
};

ncclResult_t nvlsGetProperties(struct ncclComm *comm, struct ncclNvlsSharedRes* resources, int dev, int nranks, size_t size) {
  CUmulticastObjectProp* prop = &resources->properties;
  memset(prop, 0, sizeof(*prop));
  prop->size = size;
  prop->numDevices = nranks;
  prop->handleTypes = NVLS_CU_MEM_HANDLE_TYPE;
  prop->flags = 0;

  // Could be changed to CU_MULTICAST_GRANULARITY_MINIMUM when 3418538 resolved
  CUCHECK(cuMulticastGetGranularity(&resources->granularity, prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

  ALIGN_SIZE(size, resources->granularity);
  prop->size = resources->size = size;

  memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));
  resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  resources->accessDesc.location.id = dev;
  resources->dev = dev;

  return ncclSuccess;
}

ncclResult_t nvlsGroupCreate(struct ncclComm *comm, CUmulticastObjectProp *prop, int rank, unsigned int nranks, CUmemGenericAllocationHandle *mcHandle, char *shareableHandle) {
  size_t size = prop->size;

  // Create a Multicast group

  INFO(NCCL_NVLS, "NVLS Creating Multicast group nranks %d size %zi on rank %d", nranks, size, rank);
  CUCHECK(cuMulticastCreate(mcHandle, prop));

  if ((NVLS_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_NONE) && (NVLS_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
    // Get a handle to pass to other ranks
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, *mcHandle, NVLS_CU_MEM_HANDLE_TYPE, 0));
  }
  else {
    memcpy(shareableHandle, mcHandle, sizeof(CUmemGenericAllocationHandle));
  }

  INFO(NCCL_NVLS, "NVLS Created Multicast group %llx nranks %d size %zi on rank %d", *mcHandle, nranks, size, rank);

  return ncclSuccess;
}

ncclResult_t nvlsGroupAddDevice(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  INFO(NCCL_NVLS, "NVLS group %llx adding dev %d", resources->mcHandle, resources->dev);
  CUCHECK(cuMulticastAddDevice(resources->mcHandle, resources->dev));
  return ncclSuccess;
}

ncclResult_t nvlsGroupConnect(struct ncclComm *comm, char *shareableHandle, int rank, CUmemGenericAllocationHandle *mcHandle) {
  CUmemAllocationHandleType type = NVLS_CU_MEM_HANDLE_TYPE;

  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = -1;
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);
    struct ncclProxyConnector proxyConn;
    int tpProxyRank = comm->topParentRanks[rank];
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, tpProxyRank, &proxyConn));
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank, *(uint64_t*)shareableHandle, rank);
    NCCLCHECK(ncclProxyClientGetFdBlocking(comm, &proxyConn, shareableHandle, &fd));
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECK(cuMemImportFromShareableHandle(mcHandle, (void *)(uintptr_t)fd, type));
    (void) close(fd);
  } else {
    if (NVLS_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_NONE) {
      CUCHECK(cuMemImportFromShareableHandle(mcHandle, (void *)shareableHandle, type));
    } else {
      memcpy(mcHandle, shareableHandle, sizeof(CUmemGenericAllocationHandle));
    }
  }
  return ncclSuccess;
}

ncclResult_t nvlsGroupDisconnect(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  return ncclSuccess;
}

ncclResult_t nvlsGroupBindMem(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  size_t size = resources->size;
  size_t granularity;
  CUdeviceptr ptr = 0;
  CUmemAllocationProp prop;

  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = resources->dev;
  prop.requestedHandleTypes = NVLS_CU_MEM_HANDLE_TYPE;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  resources->ucGran = granularity;

  // Map a VA for UC memory
  CUCHECK(cuMemAddressReserve(&ptr, size, granularity, 0U, 0));

  // Alloc local physical mem for this NVLS group
  CUCHECK(cuMemCreate(&resources->ucHandle, size, &prop, 0));
  CUCHECK(cuMemMap(ptr, size, 0, resources->ucHandle, 0));
  CUCHECK(cuMemSetAccess(ptr, size, &resources->accessDesc, 1));
  CUDACHECK(cudaMemset((void*)ptr, 0, size));
  resources->ucBuff = (char*)ptr;
  INFO(NCCL_NVLS, "NVLS Mapped UC at %p size %zi", resources->ucBuff, size);

  // Bind physical memory to the Multicast group
  // NB: It will block until all ranks have been added to the Group
  INFO(NCCL_NVLS, "NVLS Bind mem %p UC handle 0x%llx MC handle 0x%llx size %zi", (void*)ptr, resources->ucHandle, resources->mcHandle, size);
  CUCHECK(cuMulticastBindMem(resources->mcHandle, 0/*mcOffset*/, resources->ucHandle, 0/*memOffset*/, size, 0/*flags*/));

  return ncclSuccess;
}

ncclResult_t nvlsGroupUnbind(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  int dev = resources->dev;
  size_t size = resources->size;
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zi dev %d", resources->mcHandle, size, dev);

  // Unbind physical memory from group for the given device
  CUCHECK(cuMulticastUnbind(resources->mcHandle, dev, 0/*mcOffset*/, size));

  // Release the MC group resources
  NCCLCHECK(nvlsGroupDisconnect(comm, resources));

  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t size) {
  CUCHECK(cuMulticastUnbind(*mcHandler, dev, 0/*mcOffset*/, size));
  CUCHECK(cuMemUnmap(ptr, size));
  CUCHECK(cuMemAddressFree(ptr, size));
  CUCHECK(cuMemRelease(*mcHandler));
  return ncclSuccess;
}

ncclResult_t nvlsGroupMapMem(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  size_t size = resources->size;
  CUdeviceptr ptr = 0;

  // Create a VA for the NVLS
  CUCHECK(cuMemAddressReserve(&ptr, size, resources->granularity, 0U, 0));
  // Map the VA locally
  CUCHECK(cuMemMap(ptr, size, 0, resources->mcHandle, 0));
  resources->mcBuff = (char*)ptr;
  INFO(NCCL_NVLS, "NVLS Mapped MC buffer at %p size %zi", resources->mcBuff, size);

  // Having completed the BindMem we can now call SetAccess
  // NB: It will block until all ranks have bound to the Group
  CUCHECK(cuMemSetAccess((CUdeviceptr)resources->mcBuff, size, &resources->accessDesc, 1));

  return ncclSuccess;
}

ncclResult_t nvlsGroupUnmapMem(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  size_t size;
  CUdeviceptr ptr;
  INFO(NCCL_NVLS, "NVLS Unmap mem UC handle 0x%llx(%p) MC handle 0x%llx(%p)",
       resources->ucHandle, resources->ucBuff, resources->mcHandle, resources->mcBuff);

  // Release the UC memory and mapping
  ptr = (CUdeviceptr)resources->ucBuff;
  size = resources->size;
  CUCHECK(cuMemUnmap(ptr, size));
  CUCHECK(cuMemAddressFree(ptr, size));
  CUCHECK(cuMemRelease(resources->ucHandle));

  // Release the MC memory and mapping
  ptr = (CUdeviceptr)resources->mcBuff;
  size = resources->size;
  CUCHECK(cuMemUnmap(ptr, size));
  CUCHECK(cuMemAddressFree(ptr, size));
  CUCHECK(cuMemRelease(resources->mcHandle));

  return ncclSuccess;
}

#include "bootstrap.h"
#include "channel.h"

#define NVLS_MEM_ALIGN_SIZE (1 << 21)

NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", 16);

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsSupport = 0;
  comm->nvlsChannels = 0;

  int gpuCount;
  NCCLCHECK(ncclTopoGetGpuCount(comm->topo, &gpuCount));
  if (!ncclParamNvlsEnable() || gpuCount <= 2) return ncclSuccess;

  CUdevice dev;
  int driverVersion;

  if (CUPFN(cuDeviceGet) == NULL) return ncclSuccess;
  CUCHECK(cuCtxGetDevice(&dev));
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  if (ncclParamNvlsEnable() == 2) {
    // NVLS Multicast support requires CUDA12.1 UMD + KMD
    if (CUPFN(cuMulticastCreate) != NULL /*&& driverVersion >= 12010 */) {
      CUCHECK(cuDeviceGetAttribute(&comm->nvlsSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    }
  } else {
    comm->nvlsSupport = 1;
  }

  INFO(NCCL_INIT, "NVLS multicast support is %savailable on dev %d", comm->nvlsSupport ? "" : "not ", dev);
  if (comm->nvlsSupport == 1) comm->nvlsChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, (int)ncclParamNvlsChannels()));
  return ncclSuccess;
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  if (comm->nvlsSupport == 0 || comm->nvlsChannels == 0) return ncclSuccess;

  int nHeads = comm->channels[0].nvls.nHeads;
  int headRank = comm->channels[0].nvls.headRank;
  char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];
  uintptr_t *nvlsShmem = NULL;
  size_t typeSize;

  CUdevice dev;
  CUCHECK(cuCtxGetDevice(&dev));

  ncclResult_t res = ncclSuccess;
  bool nvlsShare = true;
  if (parent && parent->nvlsSupport && parent->config.splitShare && parent->localRanks == comm->localRanks)
    nvlsShare = true;
  else
    nvlsShare = false;

  if (nvlsShare) {
    /* reuse NVLS resources */
    comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
    for (int c = 0; c < comm->nvlsChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, parent, true), res, cleanup);
    }

    comm->nvlsResources = parent->nvlsResources;
    ncclAtomicRefCountIncrement(&parent->nvlsResources->refCount);
  } else {
    int nChannels;
    struct ncclNvlsSharedRes* resources;

    NCCLCHECK(ncclCalloc(&resources, 1));
    comm->nvlsResources = resources;
    resources->refCount = 1;

    if (parent && parent->config.splitShare) {
      /* ranks on other nodes might share the NVLS resources, we need to cap nvlsChannels
       * to make sure nvlsChannels match for each rank. */
      comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
    }

    nChannels = resources->nChannels = comm->nvlsChannels;
    for (int c = 0; c < nChannels; c++) {
      NCCLCHECK(initNvlsChannel(comm, c, parent, false));
    }

    size_t buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];
    size_t memSize = NVLS_MEM_ALIGN_SIZE;
    size_t nvlsPerRankSize = nChannels * 2 * (buffSize + memSize);
    size_t nvlsTotalSize = nvlsPerRankSize * nHeads;

    INFO(NCCL_INIT | NCCL_NVLS, "NVLS comm %p headRank %d nHeads %d buffSize %zi memSize %zi nvlsPerRankSize %zi nvlsTotalSize %zi",
      comm, headRank, nHeads, buffSize, memSize, nvlsPerRankSize, nvlsTotalSize);

    char* shareableHandle = resources->shareableHandle;
    NCCLCHECKGOTO(nvlsGetProperties(comm, resources, dev, comm->localRanks, nvlsTotalSize), res, cleanup);
    if (comm->localRank == 0) {
      NCCLCHECKGOTO(nvlsGroupCreate(comm, &resources->properties, comm->localRank, comm->localRanks, &resources->mcHandle, shareableHandle), res, cleanup);
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
    } else {
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
      NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &resources->mcHandle), res, cleanup);
    }

    NCCLCHECKGOTO(nvlsGroupAddDevice(comm, resources), res, cleanup);
    NCCLCHECKGOTO(nvlsGroupBindMem(comm, resources), res, cleanup);
    // Local intra-node barrier to ensure everyone has bound their memory to the group
    NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);
    NCCLCHECKGOTO(nvlsGroupMapMem(comm, resources), res, cleanup);

    for (int h = 0; h < nHeads; h++) {
      int nvlsPeer = comm->nRanks + 1 + h;
      for (int c = 0; c < nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        char* mem = NULL;
        struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

        // Reduce UC -> MC
        mem = resources->ucBuff + (h * 2 * nChannels + c) * (buffSize + memSize);
        peer->send[1].transportComm = &nvlsTransport.send;
        peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
        peer->send[1].conn.head = (uint64_t*)(mem + buffSize);
        peer->send[1].conn.tail = (uint64_t*)(mem + buffSize + memSize / 2);
        mem = resources->mcBuff + (h * 2 * nChannels + c) * (buffSize + memSize);
        peer->recv[0].transportComm = &nvlsTransport.recv;
        peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
        peer->recv[0].conn.head = (uint64_t*)(mem + buffSize);
        peer->recv[0].conn.tail = (uint64_t*)(mem + buffSize + memSize / 2);
        peer->recv[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        // Broadcast MC -> UC
        mem = resources->ucBuff + ((h * 2 + 1) * nChannels + c) * (buffSize + memSize);
        peer->recv[1].transportComm = &nvlsTransport.recv;
        peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
        peer->recv[1].conn.head = (uint64_t*)(mem + buffSize);
        peer->recv[1].conn.tail = (uint64_t*)(mem + buffSize + memSize / 2);
        mem = resources->mcBuff + ((h * 2 + 1) * nChannels + c) * (buffSize + memSize);
        peer->send[0].transportComm = &nvlsTransport.send;
        peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
        peer->send[0].conn.head = (uint64_t*)(mem + buffSize);
        peer->send[0].conn.tail = (uint64_t*)(mem + buffSize + memSize / 2);
        peer->send[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, cleanup);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, cleanup);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, cleanup);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, cleanup);

        /*INFO(NCCL_INIT|NCCL_NVLS, "Peer %d Channel %d MC buff %p/%p UC Buff %p/%p",
            nvlsPeer, c,
            resources->mcBuff + (h*2*nChannels+c)*(buffSize+memSize),
            resources->mcBuff + ((h*2+1)*nChannels+c)*(buffSize+memSize),
            resources->ucBuff + (h*2*nChannels+c)*(buffSize+memSize),
            resources->ucBuff + ((h*2+1)*nChannels+c)*(buffSize+memSize));*/
      }
    }
  }

  /* create shared memory for fast NVLS buffer registration */
  typeSize = sizeof(struct localRegData);
  if (comm->localRank == 0) {
    shmPath[0] = '\0';
    NCCLCHECKGOTO(ncclShmOpen(shmPath, (sizeof(size_t) + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, comm->localRanks - 1, &comm->nvlsShmemHandle), res, cleanup);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, cleanup);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, cleanup);
    NCCLCHECKGOTO(ncclShmOpen(shmPath, (sizeof(size_t) + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, -1, &comm->nvlsShmemHandle), res, cleanup);
  }
  /* need 2 pools and a shared counter for shmem-based collectives */
  comm->nvlsShmem.cnt[0] = (size_t*)nvlsShmem;
  comm->nvlsShmem.ptr[0] = (void*)((char*)comm->nvlsShmem.cnt[0] + sizeof(size_t));
  comm->nvlsShmem.cnt[1] = (size_t*)((char*)comm->nvlsShmem.ptr[0] + typeSize * comm->localRanks);
  comm->nvlsShmem.ptr[1] = (void*)((char*)comm->nvlsShmem.cnt[1] + sizeof(size_t));
  comm->nvlsShmem.round = 0;

  return res;

cleanup:
  comm->nvlsSupport = 0;
  return res;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = (struct ncclNvlsSharedRes*)comm->nvlsResources;
  if (resources == NULL) return ncclSuccess;

  if (ncclAtomicRefCountDecrement(&resources->refCount) == 0) {
    NCCLCHECK(nvlsGroupUnbind(comm, resources));
    NCCLCHECK(nvlsGroupUnmapMem(comm, resources));
    free(resources);
    comm->nvlsResources = NULL;
  }
  return ncclSuccess;
}

ncclResult_t tryRegisterBuffer(struct ncclComm *comm, struct localRequestData *reqData, uintptr_t userBuff, size_t buffSize, CUdeviceptr *regAddr, bool *regUsed) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRegRecord *regRecord = NULL;
  struct localRequestData *myReqData = &reqData[comm->localRank];
  CUdeviceptr regPtr = 0;
  CUmulticastObjectProp prop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle mcHandle;
  size_t granularity;
  size_t minSize;
  bool localRegBufUsed = false;

  /* get minimal size of nvls buffers */
  minSize = reqData[0].reqSize;
  for (int i = 1; i < comm->localRanks; ++i) {
    if (minSize > reqData[i].reqSize)
      minSize = reqData[i].reqSize;
  }

  /* start registration */
  memcpy(&prop, &comm->nvlsResources->properties, sizeof(CUmulticastObjectProp));
  prop.size = minSize;
  CUCHECKGOTO(cuMulticastGetGranularity(&granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  if (comm->localRank == 0) {
    NCCLCHECKGOTO(nvlsGroupCreate(comm, &prop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle), ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
    NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);
  }

  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->nvlsResources->dev), ret, fail);
  CUCHECKGOTO(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)myReqData->reqBuff, minSize, 0), ret, fail);

  // Create a VA for the NVLS
  CUCHECKGOTO(cuMemAddressReserve(&regPtr, minSize, granularity, 0U, 0), ret, fail);
  // Map the VA locally
  CUCHECKGOTO(cuMemMap(regPtr, minSize, 0, mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess(regPtr, minSize, &comm->nvlsResources->accessDesc, 1), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&regRecord, 1), ret, fail);
  regRecord->buff = myReqData->reqBuff;
  regRecord->size = myReqData->reqSize;
  regRecord->regAddr = regPtr;
  regRecord->regSize = minSize;
  regRecord->dev = comm->nvlsResources->dev;
  regRecord->mcHandle = mcHandle;
  /* get all buffer addresses */
  NCCLCHECKGOTO(ncclCalloc(&regRecord->addrs, comm->localRanks), ret, fail);
  regRecord->addrs[comm->localRank] = regRecord->buff;
  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsShmem, regRecord->addrs + comm->localRank, regRecord->addrs, sizeof(uintptr_t)), ret, fail);
  /* enqueue record */
  ncclIntruQueueEnqueue(&comm->regRecordQueue, regRecord);

  localRegBufUsed = true;

exit:
  if (localRegBufUsed)
    *regAddr = (uintptr_t)regPtr + userBuff - myReqData->reqBuff;
  *regUsed = localRegBufUsed;
  return ret;
fail:
  localRegBufUsed = false;
  goto exit;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;
  bool localRegBufUsed = false;
  struct localRegData *regData = NULL;
  struct localRequestData *reqData = NULL;
  struct ncclRegRecord *regRecordHead = NULL, *sendRegRecord = NULL, *recvRegRecord = NULL;
  struct ncclRegRequest *regRequestHead = NULL, *sendRegRequest = NULL, *recvRegRequest = NULL;
  bool sendNeedReg = false, recvNeedReg = false;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;

  *outRegBufUsed = false;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks), ret, fail);

  /* first check whether the buffer has been registered and matches each other globally */
  regRecordHead = ncclIntruQueueHead(&comm->regRecordQueue);
  while (regRecordHead && ((sendRegRecord == NULL && sendbuff != NULL) || (recvRegRecord == NULL && recvbuff != NULL))) {
    /* check send reg record */
    if (sendRegRecord == NULL && regRecordHead->buff <= (uintptr_t)sendbuff &&
      regRecordHead->buff + regRecordHead->size >= (uintptr_t)sendbuff + sendbuffSize) {
      regData[comm->localRank].recSendbuff = regRecordHead->buff;
      regData[comm->localRank].recSendOffset = (uintptr_t)sendbuff - regRecordHead->buff;
      sendRegRecord = regRecordHead;
    }

    /* check recv reg record */
    if (recvRegRecord == NULL && regRecordHead->buff <= (uintptr_t)recvbuff &&
      regRecordHead->buff + regRecordHead->size >= (uintptr_t)recvbuff + recvbuffSize) {
      regData[comm->localRank].recRecvbuff = regRecordHead->buff;
      regData[comm->localRank].recRecvOffset = (uintptr_t)recvbuff - regRecordHead->buff;
      recvRegRecord = regRecordHead;
    }
    regRecordHead = regRecordHead->next;
  }

  /* prepare registration request for later reference */
  regRequestHead = ncclIntruQueueHead(&comm->regRequestQueue);
  while (regRequestHead && ((sendRegRequest == NULL && sendbuff != NULL) || (recvRegRequest == NULL && recvbuff != NULL))) {
    /* check send reg request */
    if (regRequestHead->buff <= (uintptr_t)sendbuff &&
      regRequestHead->buff + regRequestHead->size >= (uintptr_t)sendbuff + sendbuffSize) {
      regData[comm->localRank].reqSendbuff = regRequestHead->buff;
      regData[comm->localRank].reqSendSize = regRequestHead->size;
      regData[comm->localRank].reqSendOffset = (uintptr_t)sendbuff - regRequestHead->buff;
      sendRegRequest = regRequestHead;
    }

    /* check recv reg request */
    if (regRequestHead->buff <= (uintptr_t)recvbuff &&
      regRequestHead->buff + regRequestHead->size >= (uintptr_t)recvbuff + recvbuffSize) {
      regData[comm->localRank].reqRecvbuff = regRequestHead->buff;
      regData[comm->localRank].reqRecvSize = regRequestHead->size;
      regData[comm->localRank].reqRecvOffset = (uintptr_t)recvbuff - regRequestHead->buff;
      recvRegRequest = regRequestHead;
    }
    regRequestHead = regRequestHead->next;
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsShmem, regData + comm->localRank, regData, sizeof(struct localRegData)), ret, fail);

  /* first check whether all local ranks find their registered buffer */
  for (int i = 0; i < comm->localRanks; ++i) {
    if (regData[i].recSendbuff == 0 || sendRegRecord->addrs[i] != regData[i].recSendbuff) {
      sendNeedReg = true;
    }

    if (regData[i].recRecvbuff == 0 || recvRegRecord->addrs[i] != regData[i].recRecvbuff) {
      recvNeedReg = true;
    }
  }

  if (sendNeedReg == false) {
    for (int i = 0; i < comm->localRanks - 1; ++i) {
      if (regData[i].recSendOffset != regData[i + 1].recSendOffset) {
        /* offset are different, we cannot apply user buffer registration */
        goto fail;
      }
    }

    /* reuse previous registered buffer if possible */
    if (!sendNeedReg)
      regSendPtr = (CUdeviceptr)((uintptr_t)sendRegRecord->regAddr + regData[comm->localRank].recSendOffset);
  }

  if (recvNeedReg == false) {
    for (int i = 0; i < comm->localRanks - 1; ++i) {
      if (regData[i].recRecvOffset != regData[i + 1].recRecvOffset) {
        goto fail;
      }
    }

    if (!recvNeedReg)
      regRecvPtr = (CUdeviceptr)((uintptr_t)recvRegRecord->regAddr + regData[comm->localRank].recRecvOffset);
  }

  if ((!sendNeedReg || sendbuff == NULL) && (!recvNeedReg || recvbuff == NULL)) {
    localRegBufUsed = true;
    INFO(NCCL_NVLS, "rank %d reuse local-registered sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);
    goto exit;
  }

  /* Start Registration. Not found registered buffers, then check whether both send and recv buffer locate
   * in register request cache. */
  NCCLCHECKGOTO(ncclCalloc(&reqData, comm->localRanks), ret, fail);
  if (sendNeedReg && sendbuff != NULL) {
    /* copy request data got from previous shmem AG */
    intptr_t offset = regData[0].reqSendOffset;
    for (int i = 0; i < comm->localRanks; ++i) {
      if (regData[i].reqSendbuff == 0 || offset != regData[i].reqSendOffset) goto fail;
      reqData[i].reqBuff = regData[i].reqSendbuff;
      reqData[i].reqSize = regData[i].reqSendSize;
      reqData[i].reqOffset = regData[i].reqSendOffset;
    }
    tryRegisterBuffer(comm, reqData, (uintptr_t)sendbuff, sendbuffSize, &regSendPtr, &localRegBufUsed);
    if (localRegBufUsed == false) goto fail;
  }

  if (recvNeedReg && recvbuff != NULL) {
    intptr_t offset = regData[0].reqRecvOffset;
    for (int i = 0; i < comm->localRanks; ++i) {
      if (regData[i].reqRecvbuff == 0 || offset != regData[i].reqRecvOffset) goto fail;
      reqData[i].reqBuff = regData[i].reqRecvbuff;
      reqData[i].reqSize = regData[i].reqRecvSize;
      reqData[i].reqOffset = regData[i].reqRecvOffset;
    }
    tryRegisterBuffer(comm, reqData, (uintptr_t)recvbuff, recvbuffSize, &regRecvPtr, &localRegBufUsed);
    if (localRegBufUsed == false) goto fail;
  }

  INFO(NCCL_NVLS, "rank %d successfully local-registered sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);

exit:
  *outRegBufSend = (void*)regSendPtr;
  *outRegBufRecv = (void*)regRecvPtr;
  *outRegBufUsed = localRegBufUsed;
  free(regData);
  free(reqData);
  return ncclSuccess;
fail:
  localRegBufUsed = false;
  goto exit;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm *comm, struct ncclKernelPlan *plan, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;
  bool localRegBufUsed = false;
  struct ncclNvlsMcHandleList* sendRecord = NULL;
  struct ncclNvlsMcHandleList* recvRecord = NULL;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;
  CUmulticastObjectProp prop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle sendMcHandle, recvMcHandle;
  size_t sendGran, recvGran;
  bool *regBufFlags = NULL;
  struct graphRegData *rdata = NULL;
  const void *baseSend = NULL;
  const void *baseRecv = NULL;
  size_t baseSendSize = 1;
  size_t baseRecvSize = 1;

  *outRegBufUsed = false;
  NCCLCHECKGOTO(ncclCalloc(&regBufFlags, comm->localRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&rdata, comm->localRanks), ret, fail);

  if (sendbuffSize > 0 || recvbuffSize > 0) {
    /* retrieve base pointer and size */
    if (CUPFN(cuMemGetAddressRange) == nullptr) goto fail;
    if (sendbuff != NULL)
      CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&baseSend, &baseSendSize, (CUdeviceptr)sendbuff), ret, fail);
    if (recvbuff != NULL)
      CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr *)&baseRecv, &baseRecvSize, (CUdeviceptr)recvbuff), ret, fail);

    memcpy(&prop, &comm->nvlsResources->properties, sizeof(CUmulticastObjectProp));
    prop.size = baseSendSize;
    CUCHECKGOTO(cuMulticastGetGranularity(&sendGran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
    prop.size = baseRecvSize;
    CUCHECKGOTO(cuMulticastGetGranularity(&recvGran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);

    localRegBufUsed = ((uint64_t)baseSend % sendGran != 0 || (uint64_t)baseRecv % recvGran != 0) ? false : true;
    regBufFlags[comm->localRank] = localRegBufUsed;
    NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, regBufFlags, sizeof(bool)), ret, fail);
    for (int i = 0; i < comm->localRanks; ++i)
      if (regBufFlags[i] == false) goto fail;

    if (sendbuff != NULL) {
      /* check send buffer offset and size */
      rdata[comm->localRank].offset = (uintptr_t)sendbuff - (uintptr_t)baseSend;
      rdata[comm->localRank].size = baseSendSize;
      NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, rdata, sizeof(struct graphRegData)), ret, fail);
      baseSendSize = rdata[0].size;
      for (int i = 1; i < comm->localRanks; ++i) {
        if (rdata[0].offset != rdata[i].offset) goto fail;
        if (baseSendSize > rdata[i].size) baseSendSize = rdata[i].size;
      }
      if (baseSendSize % sendGran != 0) goto fail;

      prop.size = baseSendSize;

      /* register sendbuff */
      if (comm->localRank == 0) {
        NCCLCHECKGOTO(nvlsGroupCreate(comm, &prop, comm->localRank, comm->localRanks, &sendMcHandle, shareableHandle), ret, fail);
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
      } else {
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
        NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &sendMcHandle), ret, fail);
      }

      CUCHECKGOTO(cuMulticastAddDevice(sendMcHandle, comm->nvlsResources->dev), ret, fail);
      CUCHECKGOTO(cuMulticastBindAddr(sendMcHandle, 0, (CUdeviceptr)baseSend, baseSendSize, 0), ret, fail);

      // Create a VA for the NVLS
      CUCHECKGOTO(cuMemAddressReserve(&regSendPtr, baseSendSize, sendGran, 0U, 0), ret, fail);
      // Map the VA locally
      CUCHECKGOTO(cuMemMap(regSendPtr, baseSendSize, 0, sendMcHandle, 0), ret, fail);
      CUCHECKGOTO(cuMemSetAccess(regSendPtr, baseSendSize, &comm->nvlsResources->accessDesc, 1), ret, fail);

      sendRecord = ncclMemoryPoolAlloc<struct ncclNvlsMcHandleList>(&comm->memPool_ncclNvlsHandleList, &comm->memPermanent);
      sendRecord->mcHandle = sendMcHandle;
      sendRecord->ptr = regSendPtr;
      sendRecord->dev = comm->nvlsResources->dev;
      sendRecord->size = baseSendSize;
    }

    if (recvbuff != NULL) {
      rdata[comm->localRank].offset = (uintptr_t)recvbuff - (uintptr_t)baseRecv;
      rdata[comm->localRank].size = baseRecvSize;
      NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, rdata, sizeof(struct graphRegData)), ret, fail);
      baseRecvSize = rdata[0].size;
      for (int i = 1; i < comm->localRanks; ++i) {
        if (rdata[0].offset != rdata[i].offset) goto fail;
        if (baseRecvSize > rdata[i].size) baseRecvSize = rdata[i].size;
      }
      if (baseRecvSize % recvGran != 0) goto fail;

      prop.size = baseRecvSize;
      if (comm->localRank == 0) {
        NCCLCHECKGOTO(nvlsGroupCreate(comm, &prop, comm->localRank, comm->localRanks, &recvMcHandle, shareableHandle), ret, fail);
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
      } else {
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
        NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &recvMcHandle), ret, fail);
      }

      CUCHECKGOTO(cuMulticastAddDevice(recvMcHandle, comm->nvlsResources->dev), ret, fail);
      CUCHECKGOTO(cuMulticastBindAddr(recvMcHandle, 0, (CUdeviceptr)baseRecv, baseRecvSize, 0), ret, fail);

      // Create a VA for the NVLS
      CUCHECKGOTO(cuMemAddressReserve(&regRecvPtr, baseRecvSize, recvGran, 0U, 0), ret, fail);
      // Map the VA locally
      CUCHECKGOTO(cuMemMap(regRecvPtr, baseRecvSize, 0, recvMcHandle, 0), ret, fail);
      CUCHECKGOTO(cuMemSetAccess(regRecvPtr, baseRecvSize, &comm->nvlsResources->accessDesc, 1), ret, fail);

      recvRecord = ncclMemoryPoolAlloc<struct ncclNvlsMcHandleList>(&comm->memPool_ncclNvlsHandleList, &comm->memPermanent);
      recvRecord->mcHandle = recvMcHandle;
      recvRecord->ptr = regRecvPtr;
      recvRecord->dev = comm->nvlsResources->dev;
      recvRecord->size = baseRecvSize;
    }

    localRegBufUsed = true;
  }

exit:
  if (localRegBufUsed == false) {
    if (sendRecord) {
      ncclNvlsDeregBuffer(&sendRecord->mcHandle, sendRecord->ptr, sendRecord->dev, sendRecord->size);
      ncclMemoryPoolFree(&comm->memPool_ncclNvlsHandleList, sendRecord);
    }

    if (recvRecord) {
      ncclNvlsDeregBuffer(&recvRecord->mcHandle, recvRecord->ptr, recvRecord->dev, recvRecord->size);
      ncclMemoryPoolFree(&comm->memPool_ncclNvlsHandleList, recvRecord);
    }
  } else {
    if (sendRecord) {
      *outRegBufSend = (void*)((uintptr_t)regSendPtr + (uintptr_t)sendbuff - (uintptr_t)baseSend);
      ncclIntruQueueEnqueue(&plan->nvlsMcHandleQueue, sendRecord);
    }

    if (recvRecord) {
      *outRegBufRecv = (void*)((uintptr_t)regRecvPtr + (uintptr_t)recvbuff - (uintptr_t)baseRecv);
      ncclIntruQueueEnqueue(&plan->nvlsMcHandleQueue, recvRecord);
    }

    INFO(NCCL_NVLS, "rank %d successfully graph-registered sendbuff %p, recvbuff %p, sendbuff size %ld (register size %ld, sendGran %ld), recvbuff size %ld (register size %ld, recvGran %ld), reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, baseSendSize, sendGran, recvbuffSize, baseRecvSize, recvGran, (void*)regSendPtr, (void*)regRecvPtr);
  }

  *outRegBufUsed = localRegBufUsed;
  free(regBufFlags);
  free(rdata);
  /* always return success. */
  return ncclSuccess;
fail:
  localRegBufUsed = false;
  goto exit;
}

#else

/*
 * Pre CUDA 12.1 stubs
 */

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsChannels = 0;
  return ncclSuccess;
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm *comm, struct ncclKernelPlan *plan, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t size) {
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
