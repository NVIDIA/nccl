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
#include "register.h"
#include "transport.h"

#if CUDART_VERSION >= 12010

struct graphRegData {
  uintptr_t offset;
  size_t size;
};

struct localRegData {
  struct ncclReg reg;
  intptr_t offset;
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

ncclResult_t nvlsGroupCreate(struct ncclComm *comm, CUmulticastObjectProp *prop, int rank, unsigned int nranks, CUmemGenericAllocationHandle *mcHandle, char *shareableHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  size_t size = prop->size;

  // Create a Multicast group

  INFO(NCCL_NVLS, "NVLS Creating Multicast group nranks %d size %zu on rank %d", nranks, size, rank);
  CUCHECK(cuMulticastCreate(mcHandle, prop));

  if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
    // Get a handle to pass to other ranks
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, *mcHandle, ncclCuMemHandleType, 0));
  }
  else {
    memcpy(shareableHandle, mcHandle, sizeof(CUmemGenericAllocationHandle));
  }

  INFO(NCCL_NVLS, "NVLS Created Multicast group %llx nranks %d size %zu on rank %d", *mcHandle, nranks, size, rank);

  return ncclSuccess;
}

ncclResult_t nvlsGroupConnect(struct ncclComm *comm, char *shareableHandle, int rank, CUmemGenericAllocationHandle *mcHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;

  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = -1;
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);
    int tpProxyRank = comm->topParentRanks[rank];
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank, *(uint64_t*)shareableHandle, rank);
    NCCLCHECK(ncclProxyClientGetFdBlocking(comm, tpProxyRank, shareableHandle, &fd));
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECK(cuMemImportFromShareableHandle(mcHandle, (void *)(uintptr_t)fd, type));
    (void) close(fd);
  } else {
    if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
      CUCHECK(cuMemImportFromShareableHandle(mcHandle, (void *)shareableHandle, type));
    } else {
      memcpy(mcHandle, shareableHandle, sizeof(CUmemGenericAllocationHandle));
    }
  }
  return ncclSuccess;
}

ncclResult_t nvlsGroupUnbind(struct ncclComm *comm, size_t size, CUmemGenericAllocationHandle* mcHandle) {
  int dev = comm->cudaDev;
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zu dev %d", *mcHandle, size, dev);

  // Unbind physical memory from group for the given device
  CUCHECK(cuMulticastUnbind(*mcHandle, dev, 0/*mcOffset*/, size));

  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t size) {
  CUCHECK(cuMulticastUnbind(*mcHandler, dev, 0/*mcOffset*/, size));
  CUCHECK(cuMemUnmap(ptr, size));
  CUCHECK(cuMemAddressFree(ptr, size));
  CUCHECK(cuMemRelease(*mcHandler));
  return ncclSuccess;
}

ncclResult_t nvlsGroupUnmapMem(struct ncclComm *comm, size_t size, void* ucptr, CUmemGenericAllocationHandle* ucHandle, void* mcptr, CUmemGenericAllocationHandle* mcHandle) {
  INFO(NCCL_NVLS, "NVLS Unmap mem UC handle 0x%llx(%p) MC handle 0x%llx(%p)", *ucHandle, ucptr, *mcHandle, mcptr);

  // Release the UC memory and mapping
  CUCHECK(cuMemUnmap((CUdeviceptr)ucptr, size));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ucptr, size));
  CUCHECK(cuMemRelease(*ucHandle));

  // Release the MC memory and mapping
  CUCHECK(cuMemUnmap((CUdeviceptr)mcptr, size));
  CUCHECK(cuMemAddressFree((CUdeviceptr)mcptr, size));
  CUCHECK(cuMemRelease(*mcHandle));

  return ncclSuccess;
}

#include "bootstrap.h"
#include "channel.h"

#define NVLS_MEM_ALIGN_SIZE (1 << 21)

NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", 16);
NCCL_PARAM(NvlsChunkSize, "NVLS_CHUNKSIZE", 128*1024);

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsSupport = 0;
  comm->nvlsChannels = 0;

  int gpuCount;
  NCCLCHECK(ncclTopoGetGpuCount(comm->topo, &gpuCount));
  if (!ncclParamNvlsEnable() || ((!comm->MNNVL && gpuCount <= 2) || (comm->MNNVL && comm->clique.size <= 2))) return ncclSuccess;

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

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nvlsSupport && comm->nNodes > 1) {
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 1, &channel->nvls.treeUp, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->nvls.treeUp, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_NVLS], 0), ret, fail);
    INFO(NCCL_INIT, "Connected NVLS tree");
  }
exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t nvlsAllocateMem(struct ncclComm* comm, CUmulticastGranularity_flags mcOption, const CUmemAccessDesc* desc, size_t* sizePtr, CUmemGenericAllocationHandle* ucHandle, CUmemGenericAllocationHandle* mcHandle, void** ucptr, void** mcptr) {
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  ncclResult_t ret = ncclSuccess;
  size_t size = *sizePtr;
  size_t originSize = size;
  size_t ucgran, mcgran;

  memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
  mcprop.numDevices = comm->localRanks;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  mcprop.size = size;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, mcOption), ret, fail);
  ALIGN_SIZE(size, mcgran);
  *sizePtr = mcprop.size = size;

  if (comm->localRank == 0) {
    NCCLCHECKGOTO(nvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, mcHandle, shareableHandle), ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
    NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], mcHandle), ret, fail);
  }

  CUCHECKGOTO(cuMulticastAddDevice(*mcHandle, comm->cudaDev), ret, fail);

  memset(&ucprop, 0, sizeof(CUmemAllocationProp));
  ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ucprop.location.id = comm->cudaDev;
  ucprop.requestedHandleTypes = ncclCuMemHandleType;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);
  // Map a VA for UC memory
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)ucptr, size, ucgran, 0U, 0), ret, fail);

  // Alloc local physical mem for this NVLS group
  CUCHECKGOTO(cuMemCreate(ucHandle, size, &ucprop, 0), ret, fail);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ucptr, size, 0, *ucHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ucptr, size, desc, 1), ret, fail);
  CUDACHECKGOTO(cudaMemset(*ucptr, 0, size), ret, fail);

  // Bind physical memory to the Multicast group
  // NB: It will block until all ranks have been added to the Group
  CUCHECKGOTO(cuMulticastBindMem(*mcHandle, 0/*mcOffset*/, *ucHandle, 0/*memOffset*/, size, 0/*flags*/), ret, fail);

  // Map mc virtual address
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)mcptr, size, mcgran, 0U, 0), ret, fail);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*mcptr, size, 0, *mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*mcptr, size, desc, 1), ret, fail);
  INFO(NCCL_NVLS, "NVLS rank %d (dev %d) alloc done, ucptr %p ucgran %ld mcptr %p mcgran %ld size %ld (%ld)", comm->rank, comm->cudaDev, *ucptr, ucgran, *mcptr, mcgran, size, originSize);

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) {
  int nHeads = -1;
  int headRank = -1;
  ncclResult_t res = ncclSuccess;
  int nvlsStepSize = -1;
  size_t buffSize = 0;
  size_t nvlsPerRankSize = 0;
  size_t nvlsTotalSize = 0;
  struct ncclNvlsSharedRes* resources = NULL;
  int nChannels = -1;

  if (comm->nvlsSupport == 0 || comm->nvlsResources->inited) return ncclSuccess;
  // initialize after checking comm->nvlsSupport
  nHeads = comm->channels[0].nvls.nHeads;
  headRank = comm->channels[0].nvls.headRank;
  resources = comm->nvlsResources;
  nChannels = comm->nvlsResources->nChannels;
  nvlsStepSize = comm->nvlsChunkSize;
  buffSize = nvlsStepSize * NCCL_STEPS;
  nvlsPerRankSize = nChannels * 2 * buffSize;
  nvlsTotalSize = nvlsPerRankSize * nHeads;

  INFO(NCCL_INIT | NCCL_NVLS, "NVLS comm %p headRank %d nHeads %d buffSize %zu nvlsPerRankSize %zu nvlsTotalSize %zu",
       comm, headRank, nHeads, buffSize, nvlsPerRankSize, nvlsTotalSize);

  NCCLCHECKGOTO(nvlsAllocateMem(comm, CU_MULTICAST_GRANULARITY_RECOMMENDED, &resources->accessDesc, &nvlsTotalSize, &resources->ucBuffHandle, &resources->mcBuffHandle, (void**)&resources->ucBuff, (void**)&resources->mcBuff), res, fail);
  resources->buffSize = nvlsTotalSize;

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), res, fail);
  for (int h = 0; h < nHeads; h++) {
    int nvlsPeer = comm->nRanks + 1 + h;
    for (int c = 0; c < nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

      // Reduce UC -> MC
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->ucBuff + (h * 2 * nChannels + c) * buffSize;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->mcBuff + (h * 2 * nChannels + c) * buffSize;

      // Broadcast MC -> UC
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->ucBuff + ((h * 2 + 1) * nChannels + c) * buffSize;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->mcBuff + ((h * 2 + 1) * nChannels + c) * buffSize;

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
    }
  }

  NCCLCHECKGOTO(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream), res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream), res, fail);
  // For now, the barrier is a must that guarantees all buffers are mc-mapped before accessing peer's buffer
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, fail);
  comm->nvlsResources->inited = true;

exit:
  return res;
fail:
  comm->nvlsResources->inited = false;
  goto exit;
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t res = ncclSuccess;
  size_t typeSize;
  char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];
  uintptr_t *nvlsShmem = NULL;
  bool nvlsShare = parent && parent->nvlsSupport && parent->config.splitShare;
  int nHeads = comm->channels[0].nvls.nHeads;

  if (comm->nvlsSupport == 0 || comm->nvlsChannels == 0) return ncclSuccess;

  if (nvlsShare && parent->channels[0].nvls.nHeads == nHeads) {
    for (int ch = 0; ch < nHeads; ++ch) {
      bool find = false;
      for (int h = 0; h < parent->channels[0].nvls.nHeads; ++h) {
        if (comm->nvlsHeads[ch] == parent->nvlsHeads[h]) {
          // find the head
          find = true;
          break;
        }
      }
      if (find == false) {
        nvlsShare = false;
        goto setup;
      }
    }
    nvlsShare = true;
  } else {
    nvlsShare = false;
  }

setup:
  comm->nvlsChunkSize = ncclParamNvlsChunkSize();
  if (nvlsShare) {
    /* reuse NVLS resources */
    comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, parent, true), res, fail);
    }

    comm->nvlsResources = parent->nvlsResources;
    ncclAtomicRefCountIncrement(&parent->nvlsResources->refCount);
  } else {
    struct ncclNvlsSharedRes* resources = NULL;
    int nHeads = comm->channels[0].nvls.nHeads;
    int nChannels = comm->nChannels;
    size_t memSize = 16;
    size_t creditSize = nChannels * 2 * memSize * nHeads;
    int nvlsStepSize = comm->nvlsChunkSize;
  
    NCCLCHECKGOTO(ncclCalloc(&comm->nvlsResources, 1), res, fail);
    comm->nvlsResources->inited = false;
    comm->nvlsResources->refCount = 1;
    comm->nvlsResources->nChannels = comm->nvlsChannels;
    resources = comm->nvlsResources;

    if (parent && parent->nvlsSupport && parent->config.splitShare) {
      /* ranks on other nodes might share the NVLS resources, we need to cap nvlsChannels
       * to make sure nvlsChannels match for each rank. */
      comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
    }
    comm->nvlsResources->nChannels = comm->nvlsChannels;

    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, NULL, false), res, fail);
    }

    memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));
    resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    resources->accessDesc.location.id = comm->cudaDev;
    resources->dev = comm->cudaDev;

    NCCLCHECKGOTO(nvlsAllocateMem(comm, CU_MULTICAST_GRANULARITY_MINIMUM, &resources->accessDesc, &creditSize, &resources->ucCreditHandle, &resources->mcCreditHandle, (void**)&resources->ucCredit, (void**)&resources->mcCredit), res, fail);
    resources->creditSize = creditSize;

    // Set up head and tail only for now
    NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), res, fail);
    for (int h = 0; h < nHeads; h++) {
      int nvlsPeer = comm->nRanks + 1 + h;
      for (int c = 0; c < nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        char* mem = NULL;
        struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

        // Reduce UC -> MC
        mem = resources->ucCredit + (h * 2 * nChannels + c) * memSize;
        peer->send[1].transportComm = &nvlsTransport.send;
        peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[1].conn.head = (uint64_t*)mem;
        peer->send[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[1].conn.stepSize = nvlsStepSize;
        mem = resources->mcCredit + (h * 2 * nChannels + c) * memSize;
        peer->recv[0].transportComm = &nvlsTransport.recv;
        peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[0].conn.head = (uint64_t*)mem;
        peer->recv[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[0].conn.stepSize = nvlsStepSize;
        peer->recv[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        // Broadcast MC -> UC
        mem = resources->ucCredit + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->recv[1].transportComm = &nvlsTransport.recv;
        peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[1].conn.head = (uint64_t*)mem;
        peer->recv[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[1].conn.stepSize = nvlsStepSize;
        mem = resources->mcCredit + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->send[0].transportComm = &nvlsTransport.send;
        peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[0].conn.head = (uint64_t*)mem;
        peer->send[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[0].conn.stepSize = nvlsStepSize;
        peer->send[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), res, fail);
      }
    }
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream), res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream), res, fail);
  }

  // MNNVL does not support NVLS buffer registration
  if (!comm->MNNVL && comm->nvlsResources->nvlsShmemHandle == NULL) {
    /* create shared memory for fast NVLS buffer registration */
    typeSize = sizeof(struct localRegData) << 1;

    if (comm->localRank == 0) {
      shmPath[0] = '\0';
      NCCLCHECKGOTO(ncclShmOpen(shmPath, (sizeof(size_t) + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, comm->localRanks - 1, &comm->nvlsResources->nvlsShmemHandle), res, fail);
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, fail);
    } else {
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, fail);
      NCCLCHECKGOTO(ncclShmOpen(shmPath, (sizeof(size_t) + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, -1, &comm->nvlsResources->nvlsShmemHandle), res, fail);
    }
    /* need 2 pools and a shared counter for shmem-based collectives */
    comm->nvlsResources->nvlsShmem.cnt[0] = (size_t*)nvlsShmem;
    comm->nvlsResources->nvlsShmem.ptr[0] = (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[0] + sizeof(size_t));
    comm->nvlsResources->nvlsShmem.cnt[1] = (size_t*)((char*)comm->nvlsResources->nvlsShmem.ptr[0] + typeSize * comm->localRanks);
    comm->nvlsResources->nvlsShmem.ptr[1] = (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[1] + sizeof(size_t));
    comm->nvlsResources->nvlsShmem.round = 0;
    comm->nvlsResources->nvlsShmem.maxTypeSize = typeSize;
  }

exit:
  return res;
fail:
  comm->nvlsSupport = 0;
  goto exit;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = (struct ncclNvlsSharedRes*)comm->nvlsResources;
  if (resources == NULL) return ncclSuccess;

  if (ncclAtomicRefCountDecrement(&resources->refCount) == 0) {
    if (!comm->MNNVL && resources->nvlsShmemHandle)
      NCCLCHECK(ncclShmClose(resources->nvlsShmemHandle));

    if (resources->ucCredit && resources->mcCredit) {
      NCCLCHECK(nvlsGroupUnbind(comm, resources->creditSize, &resources->mcCreditHandle));
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->creditSize, resources->ucCredit, &resources->ucCreditHandle, resources->mcCredit, &resources->mcCreditHandle));
    }

    if (comm->nvlsResources->inited) {
      NCCLCHECK(nvlsGroupUnbind(comm, resources->buffSize, &resources->mcBuffHandle));
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->buffSize, resources->ucBuff, &resources->ucBuffHandle, resources->mcBuff, &resources->mcBuffHandle));
    }
    free(resources);
    comm->nvlsResources = NULL;
  }
  return ncclSuccess;
}

ncclResult_t tryRegisterBuffer(struct ncclComm *comm, uintptr_t userBuff, size_t buffSize, CUdeviceptr *regAddr, bool *regUsed) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg *regRecord = NULL;
  CUdeviceptr regPtr = 0;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle mcHandle;
  size_t minSize = SIZE_MAX;
  bool localRegBufUsed = false;
  struct localRegData* regData = NULL;
  cudaPointerAttributes attr;
  size_t ucgran, mcgran;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks), ret, fail);

  if (userBuff) {
    NCCLCHECKGOTO(ncclRegFind(comm, (void*)userBuff, buffSize, &regRecord), ret, fail);
    if (regRecord) {
      CUDACHECK(cudaPointerGetAttributes(&attr, (void*)regRecord->addr));
      if (attr.type == cudaMemoryTypeDevice) {
        size_t regSize = regRecord->pages * comm->regCache.pageSize;
        memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
        mcprop.numDevices = comm->localRanks;
        mcprop.handleTypes = ncclCuMemHandleType;
        mcprop.flags = 0;
        mcprop.size = regSize;
        CUCHECK(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

        memset(&ucprop, 0, sizeof(CUmemAllocationProp));
        ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ucprop.location.id = comm->cudaDev;
        ucprop.requestedHandleTypes = ncclCuMemHandleType;
        CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);

        CUCHECK(cuMemGetAddressRange((CUdeviceptr*)&regRecord->baseAddr, &regRecord->baseSize, (CUdeviceptr)regRecord->addr));
        if (regSize % mcgran == 0) {
          regRecord->regSize = regSize;
        } else {
          regRecord->regSize = regRecord->baseSize - (regRecord->addr - regRecord->baseAddr);
        }

        if (regRecord->addr % ucgran == 0 && regRecord->regSize % mcgran == 0) {
          regRecord->state |= NVLS_REG_POSSIBLE;
          memcpy(&regData[comm->localRank].reg, regRecord, sizeof(struct ncclReg));
          regData[comm->localRank].offset = userBuff - regRecord->addr;
        }
      }

      if ((regRecord->state & NVLS_REG_POSSIBLE) == 0) {
        regRecord->state |= NVLS_REG_NO_SUPPORT;
      }
    }
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank, regData, sizeof(struct localRegData)), ret, fail);

  for (int i = 0; i < comm->localRanks; ++i) {
    if ((regData[i].reg.state & NVLS_REG_POSSIBLE) == 0) {
      goto fail;
    }
    /* get minimal reg size of nvls buffers */
    if (minSize > regData[i].reg.regSize)
      minSize = regData[i].reg.regSize;
  }

  /* start registration */
  mcprop.size = minSize;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  if (comm->localRank == 0) {
    NCCLCHECKGOTO(nvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle), ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
    NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);
  }

  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->nvlsResources->dev), ret, fail);
  CUCHECKGOTO(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)regRecord->addr, minSize, 0), ret, fail);

  // Create a VA for the NVLS
  CUCHECKGOTO(cuMemAddressReserve(&regPtr, minSize, mcgran, 0U, 0), ret, fail);
  // Map the VA locally
  CUCHECKGOTO(cuMemMap(regPtr, minSize, 0, mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess(regPtr, minSize, &comm->nvlsResources->accessDesc, 1), ret, fail);

  regRecord->regAddr = regPtr;
  regRecord->regSize = minSize;
  regRecord->dev = comm->nvlsResources->dev;
  regRecord->mcHandle = mcHandle;
  regRecord->state |= NVLS_REG_COMPLETE;
  /* get all buffer addresses */
  regRecord->caddrs[comm->localRank] = regRecord->addr;
  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regRecord->caddrs + comm->localRank, regRecord->caddrs, sizeof(uintptr_t)), ret, fail);

  /* Although registration is done, we still need to check whether the offsets are same among ranks. */
  for (int i = 0; i < comm->localRanks - 1; ++i) {
    if (regData[i].offset != regData[i + 1].offset) {
      goto fail;
    }
  }

  localRegBufUsed = true;

exit:
  if (localRegBufUsed) *regAddr = (uintptr_t)regPtr + regData[comm->localRank].offset;
  *regUsed = localRegBufUsed;
  free(regData);
  return ret;
fail:
  localRegBufUsed = false;
  goto exit;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;
  bool localRegBufUsed = false;
  struct localRegData *regData = NULL;
  bool sendNeedReg = false, recvNeedReg = false;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;
  struct ncclReg *sendRegRecord = NULL;
  struct ncclReg *recvRegRecord = NULL;

  *outRegBufUsed = false;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks * 2), ret, fail);

  if (sendbuff) {
    NCCLCHECKGOTO(ncclRegFind(comm, sendbuff, sendbuffSize, &sendRegRecord), ret, fail);
    if (sendRegRecord) {
      memcpy(&regData[comm->localRank * 2].reg, sendRegRecord, sizeof(struct ncclReg));
      regData[comm->localRank * 2].offset = (uintptr_t)sendbuff - sendRegRecord->addr;
    }
  }

  if (recvbuff) {
    NCCLCHECKGOTO(ncclRegFind(comm, recvbuff, recvbuffSize, &recvRegRecord), ret, fail);
    if (recvRegRecord) {
      memcpy(&regData[comm->localRank * 2 + 1].reg, recvRegRecord, sizeof(struct ncclReg));
      regData[comm->localRank * 2 + 1].offset = (uintptr_t)recvbuff - recvRegRecord->addr;
    }
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank * 2, regData, sizeof(struct localRegData) * 2), ret, fail);

  /* first check whether all local ranks find their registered buffer */
  for (int i = 0; i < comm->localRanks; ++i) {
    if ((regData[i * 2].reg.state & NVLS_REG_COMPLETE) == 0 || regData[comm->localRank * 2].reg.caddrs[i] != regData[i * 2].reg.addr) {
      sendNeedReg = true;
    }

    if ((regData[i * 2 + 1].reg.state & NVLS_REG_COMPLETE) == 0 || regData[comm->localRank * 2 + 1].reg.caddrs[i] != regData[i * 2 + 1].reg.addr) {
      recvNeedReg = true;
    }

    if ((regData[i * 2].reg.state & NVLS_REG_NO_SUPPORT) || (regData[i * 2 + 1].reg.state & NVLS_REG_NO_SUPPORT)) {
      goto fail;
    }
  }

  if (sendNeedReg == false) {
    for (int i = 0; i < comm->localRanks - 1; ++i) {
      if (regData[i * 2].offset != regData[(i + 1) * 2].offset) {
        /* offset are different, we cannot apply user buffer registration */
        goto fail;
      }
    }

    /* reuse previous registered buffer if possible */
    if (!sendNeedReg)
      regSendPtr = (CUdeviceptr)((uintptr_t)sendRegRecord->regAddr + regData[comm->localRank * 2].offset);
  }

  if (recvNeedReg == false) {
    for (int i = 0; i < comm->localRanks - 1; ++i) {
      if (regData[i * 2 + 1].offset != regData[(i + 1) * 2 + 1].offset) {
        goto fail;
      }
    }

    if (!recvNeedReg)
      regRecvPtr = (CUdeviceptr)((uintptr_t)recvRegRecord->regAddr + regData[comm->localRank * 2 + 1].offset);
  }

  if ((!sendNeedReg || sendbuff == NULL) && (!recvNeedReg || recvbuff == NULL)) {
    localRegBufUsed = true;
    INFO(NCCL_NVLS, "rank %d reuse local-registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);
    goto exit;
  }

  /* Start Registration. Not found registered buffers, then check whether both send and recv buffer locate
   * in register request cache. */
  if (sendNeedReg && sendbuff) {
    tryRegisterBuffer(comm, (uintptr_t)sendbuff, sendbuffSize, &regSendPtr, &localRegBufUsed);
    if (localRegBufUsed == false) goto fail;
  }

  if (recvNeedReg && recvbuff) {
    tryRegisterBuffer(comm, (uintptr_t)recvbuff, recvbuffSize, &regRecvPtr, &localRegBufUsed);
    if (localRegBufUsed == false) goto fail;
  }

  INFO(NCCL_NVLS, "rank %d successfully local-registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);

exit:
  *outRegBufSend = (void*)regSendPtr;
  *outRegBufRecv = (void*)regRecvPtr;
  *outRegBufUsed = localRegBufUsed;
  free(regData);
  return ncclSuccess;
fail:
  localRegBufUsed = false;
  goto exit;
}

struct ncclNvlsCleanupCallback {
  struct ncclCommCallback base;
  CUmemGenericAllocationHandle mcHandle;
  CUdeviceptr ptr;
  int dev;
  size_t size;
};

static ncclResult_t cleanupNvls(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNvlsCleanupCallback* obj = (struct ncclNvlsCleanupCallback*)cb;
  NCCLCHECK(ncclNvlsDeregBuffer(&obj->mcHandle, obj->ptr, obj->dev, obj->size));
  INFO(NCCL_NVLS, "rank %d - deregistered buffer %p on device %d, size %ld", comm->rank, (void*)obj->ptr, obj->dev, obj->size);
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
    struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize,
    bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded
  ) {
  ncclResult_t ret = ncclSuccess;
  bool localRegBufUsed = false;
  struct ncclNvlsCleanupCallback* sendRecord = NULL;
  struct ncclNvlsCleanupCallback* recvRecord = NULL;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle sendMcHandle, recvMcHandle;
  size_t sendGran = 0, recvGran = 0;
  bool *regBufFlags = NULL;
  struct graphRegData *rdata = NULL;
  const void *baseSend = NULL;
  const void *baseRecv = NULL;
  size_t baseSendSize = 1;
  size_t baseRecvSize = 1;
  size_t ucgran;

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

    memset(&ucprop, 0, sizeof(CUmemAllocationProp));
    ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ucprop.location.id = comm->cudaDev;
    ucprop.requestedHandleTypes = ncclCuMemHandleType;
    CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);

    localRegBufUsed = ((uint64_t)baseSend % ucgran != 0 || (uint64_t)baseRecv % ucgran != 0) ? false : true;
    regBufFlags[comm->localRank] = localRegBufUsed;
    NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, regBufFlags, sizeof(bool)), ret, fail);
    for (int i = 0; i < comm->localRanks; ++i)
      if (regBufFlags[i] == false) goto fail;

    memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
    mcprop.numDevices = comm->localRanks;
    mcprop.handleTypes = ncclCuMemHandleType;
    mcprop.flags = 0;

    if (sendbuff != NULL) {
      mcprop.size = baseSendSize;
      CUCHECKGOTO(cuMulticastGetGranularity(&sendGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);

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

      mcprop.size = baseSendSize;

      /* register sendbuff */
      if (comm->localRank == 0) {
        NCCLCHECKGOTO(nvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &sendMcHandle, shareableHandle), ret, fail);
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

      sendRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));
      sendRecord->base.fn = cleanupNvls;
      sendRecord->mcHandle = sendMcHandle;
      sendRecord->ptr = regSendPtr;
      sendRecord->dev = comm->nvlsResources->dev;
      sendRecord->size = baseSendSize;
    }

    if (recvbuff != NULL) {
      mcprop.size = baseRecvSize;
      CUCHECKGOTO(cuMulticastGetGranularity(&recvGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);

      rdata[comm->localRank].offset = (uintptr_t)recvbuff - (uintptr_t)baseRecv;
      rdata[comm->localRank].size = baseRecvSize;
      NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, rdata, sizeof(struct graphRegData)), ret, fail);
      baseRecvSize = rdata[0].size;
      for (int i = 1; i < comm->localRanks; ++i) {
        if (rdata[0].offset != rdata[i].offset) goto fail;
        if (baseRecvSize > rdata[i].size) baseRecvSize = rdata[i].size;
      }
      if (baseRecvSize % recvGran != 0) goto fail;

      mcprop.size = baseRecvSize;
      if (comm->localRank == 0) {
        NCCLCHECKGOTO(nvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &recvMcHandle, shareableHandle), ret, fail);
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

      recvRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));
      recvRecord->base.fn = cleanupNvls;
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
      free(sendRecord);
    }

    if (recvRecord) {
      ncclNvlsDeregBuffer(&recvRecord->mcHandle, recvRecord->ptr, recvRecord->dev, recvRecord->size);
      free(recvRecord);
    }
  } else {
    if (sendRecord) {
      *outRegBufSend = (void*)((uintptr_t)regSendPtr + (uintptr_t)sendbuff - (uintptr_t)baseSend);
      ncclIntruQueueEnqueue(cleanupQueue, &sendRecord->base);
      *nCleanupQueueEltsAdded += 1;
    }

    if (recvRecord) {
      *outRegBufRecv = (void*)((uintptr_t)regRecvPtr + (uintptr_t)recvbuff - (uintptr_t)baseRecv);
      ncclIntruQueueEnqueue(cleanupQueue, &recvRecord->base);
      *nCleanupQueueEltsAdded += 1;
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

ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
    struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize,
    bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded
  ) {
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
