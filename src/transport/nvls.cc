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

ncclResult_t nvlsCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
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
  int fd = -1;
  ncclResult_t ret = ncclSuccess;
  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank, *(uint64_t*)shareableHandle, rank);
    NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, rank, shareableHandle, &fd), ret, fail);
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void *)(uintptr_t)fd, type), ret, fail);
    SYSCHECK(close(fd), "close");
  } else {
    if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
      CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void *)shareableHandle, type), ret, fail);
    } else {
      memcpy(mcHandle, shareableHandle, sizeof(CUmemGenericAllocationHandle));
    }
  }
exit:
  return ret;
fail:
  if (fd != -1) close(fd);
  goto exit;
}

ncclResult_t nvlsGroupUnbind(struct ncclComm *comm, size_t size, CUmemGenericAllocationHandle* mcHandle) {
  int dev = comm->cudaDev;
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zu dev %d", *mcHandle, size, dev);

  // Unbind physical memory from group for the given device
  if (size) CUCHECK(cuMulticastUnbind(*mcHandle, dev, 0/*mcOffset*/, size));

  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t ucsize, size_t mcsize) {
  CUCHECK(cuMulticastUnbind(*mcHandler, dev, 0/*mcOffset*/, ucsize));
  CUCHECK(cuMemUnmap(ptr, mcsize));
  CUCHECK(cuMemAddressFree(ptr, mcsize));
  CUCHECK(cuMemRelease(*mcHandler));
  INFO(NCCL_NVLS, "rank %d - NVLS deregistered buffer %p on device %d ucsize %ld mcsize %ld", comm->rank, (void*)ptr, dev, ucsize, mcsize);
  return ncclSuccess;
}

ncclResult_t nvlsGroupUnmapMem(struct ncclComm *comm, size_t ucsize, void* ucptr, CUmemGenericAllocationHandle* ucHandle, size_t mcsize, void* mcptr, CUmemGenericAllocationHandle* mcHandle) {
  INFO(NCCL_NVLS, "NVLS Unmap mem UC handle 0x%llx(%p) ucsize %zu MC handle 0x%llx(%p) mcsize %zd", *ucHandle, ucptr, ucsize, *mcHandle, mcptr, mcsize);

  // Release the UC memory and mapping
  if (ucptr) {
    CUCHECK(cuMemUnmap((CUdeviceptr)ucptr, ucsize));
    CUCHECK(cuMemAddressFree((CUdeviceptr)ucptr, ucsize));
    CUCHECK(cuMemRelease(*ucHandle));
  }

  // Release the MC memory and mapping
  if (mcptr) {
    CUCHECK(cuMemUnmap((CUdeviceptr)mcptr, mcsize));
    CUCHECK(cuMemAddressFree((CUdeviceptr)mcptr, mcsize));
    CUCHECK(cuMemRelease(*mcHandle));
  }

  return ncclSuccess;
}

#include "bootstrap.h"
#include "channel.h"

#define NVLS_MEM_ALIGN_SIZE (1 << 21)
#define NVLS_NCHANNELS_SM90 16
#define NVLS_NCHANNELS_SM100 32

NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", -2);
NCCL_PARAM(NvlsChunkSize, "NVLS_CHUNKSIZE", 128*1024);

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
  if (comm->nvlsSupport) {
    int channels = (comm->compCap >= 100) ? NVLS_NCHANNELS_SM100 : NVLS_NCHANNELS_SM90;
    if (ncclParamNvlsChannels() >= 0) channels = ncclParamNvlsChannels();
    comm->nvlsChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, channels));
  }
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

static ncclResult_t nvlsAllocateMem(struct ncclComm* comm, const CUmemAccessDesc* desc, size_t size, CUmemGenericAllocationHandle* ucHandle, CUmemGenericAllocationHandle* mcHandle, void** ucptr, void** mcptr, size_t* ucsizePtr, size_t* mcsizePtr) {
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  ncclResult_t ret = ncclSuccess;
  size_t mcsize;
  size_t ucsize;
  size_t ucgran, mcgran;
  int allocMcHandle = 0;

  mcsize = ucsize = size;
  *ucptr = *mcptr = NULL;
  memset(shareableHandle, '\0', sizeof(shareableHandle));
  memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
  mcprop.numDevices = comm->localRanks;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  mcprop.size = size;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(mcsize, mcgran);
  mcprop.size = mcsize;

  if (comm->localRank == 0) {
    NCCLCHECKGOTO(nvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, mcHandle, shareableHandle), ret, fail);
    allocMcHandle = 1;
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
    NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], mcHandle), ret, fail);
    allocMcHandle = 1;
  }

  CUCHECKGOTO(cuMulticastAddDevice(*mcHandle, comm->cudaDev), ret, fail);

  memset(&ucprop, 0, sizeof(CUmemAllocationProp));
  ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ucprop.location.id = comm->cudaDev;
  ucprop.requestedHandleTypes = ncclCuMemHandleType;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(ucsize, ucgran);
  // Map a VA for UC memory with MC alignment and size
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)ucptr, ucsize, ucgran, 0U, 0), ret, fail);

  // Alloc local physical mem for this NVLS group
  CUCHECKGOTO(cuMemCreate(ucHandle, ucsize, &ucprop, 0), ret, fail);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ucptr, ucsize, 0, *ucHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ucptr, ucsize, desc, 1), ret, fail);
  CUDACHECKGOTO(cudaMemset(*ucptr, 0, ucsize), ret, fail);

  // intra-node barrier to mitigate the possible hang in cuMulticastBindMem during abort
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);
  // Bind physical memory to the Multicast group
  // NB: It will block until all ranks have been added to the Group
  CUCHECKGOTO(cuMulticastBindMem(*mcHandle, 0/*mcOffset*/, *ucHandle, 0/*memOffset*/, ucsize, 0/*flags*/), ret, fail);

  // Map mc virtual address
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)mcptr, mcsize, mcgran, 0U, 0), ret, fail);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*mcptr, mcsize, 0, *mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*mcptr, mcsize, desc, 1), ret, fail);
  *ucsizePtr = ucsize;
  *mcsizePtr = mcsize;
  INFO(NCCL_NVLS, "NVLS rank %d (dev %d) alloc done, ucptr %p ucgran %ld mcptr %p mcgran %ld ucsize %ld mcsize %ld (inputsize %ld)", comm->rank, comm->cudaDev, *ucptr, ucgran, *mcptr, mcgran, ucsize, mcsize, size);

exit:
  return ret;
fail:
  if (allocMcHandle && *mcptr == NULL && *ucptr == NULL) CUCHECK(cuMemRelease(*mcHandle));
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
  cudaStream_t deviceStream, hostStream;

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

  NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, nvlsTotalSize, &resources->ucBuffHandle, &resources->mcBuffHandle, (void**)&resources->ucBuff, (void**)&resources->mcBuff, &resources->buffUCSize, &resources->buffMCSize), res, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), res, fail);
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), res, fail);
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

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
    }
  }

  NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), res, fail);
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
    size_t memSize = 64;
    size_t creditSize = nChannels * 2 * memSize * nHeads;
    int nvlsStepSize = comm->nvlsChunkSize;
    cudaStream_t hostStream, deviceStream;

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

    NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, creditSize, &resources->ucCreditHandle, &resources->mcCreditHandle, (void**)&resources->ucCredit, (void**)&resources->mcCredit, &resources->creditUCSize, &resources->creditMCSize), res, fail);

    // Set up head and tail only for now
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), res, fail);
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), res, fail);
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

        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), res, fail);
      }
    }
    NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), res, fail);
  }

  // MNNVL does not support NVLS buffer registration
  if (!comm->MNNVL && comm->nvlsResources->nvlsShmemHandle == NULL) {
    /* create shared memory for fast NVLS buffer registration */
    typeSize = sizeof(struct localRegData) << 1;

    if (comm->localRank == 0) {
      shmPath[0] = '\0';
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath), (sizeof(size_t) + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, comm->localRanks - 1, &comm->nvlsResources->nvlsShmemHandle), res, fail);
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, fail);
    } else {
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shmPath, sizeof(shmPath)), res, fail);
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath), (sizeof(size_t) + typeSize * comm->localRanks) * 2, (void**)&nvlsShmem, NULL, -1, &comm->nvlsResources->nvlsShmemHandle), res, fail);
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

    if (resources->ucCredit || resources->mcCredit) {
      NCCLCHECK(nvlsGroupUnbind(comm, resources->creditUCSize, &resources->mcCreditHandle));
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->creditUCSize, resources->ucCredit, &resources->ucCreditHandle, resources->creditMCSize, resources->mcCredit, &resources->mcCreditHandle));
    }

    if (comm->nvlsResources->inited) {
      NCCLCHECK(nvlsGroupUnbind(comm, resources->buffUCSize, &resources->mcBuffHandle));
      NCCLCHECK(nvlsGroupUnmapMem(comm, resources->buffUCSize, resources->ucBuff, &resources->ucBuffHandle, resources->buffMCSize, resources->mcBuff, &resources->mcBuffHandle));
    }
    free(resources);
    comm->nvlsResources = NULL;
  }
  return ncclSuccess;
}

ncclResult_t tryRegisterBuffer(struct ncclComm *comm, uintptr_t userBuff, size_t buffSize, CUdeviceptr *regAddr, int *regUsed) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg *regRecord = NULL;
  CUdeviceptr regPtr = 0;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle mcHandle;
  size_t minSize = SIZE_MAX;
  struct localRegData* regData = NULL;
  cudaPointerAttributes attr;
  size_t ucgran, mcgran, ucsize, mcsize;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks), ret, fail);

  if (userBuff) {
    NCCLCHECKGOTO(ncclRegFind(comm, (void*)userBuff, buffSize, &regRecord), ret, fail);
    if (regRecord) {
      CUDACHECKGOTO(cudaPointerGetAttributes(&attr, (void*)regRecord->addr), ret, fail);
      if (attr.type == cudaMemoryTypeDevice) {
        size_t regSize = regRecord->pages * comm->regCache.pageSize;
        memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
        mcprop.numDevices = comm->localRanks;
        mcprop.handleTypes = ncclCuMemHandleType;
        mcprop.flags = 0;
        mcprop.size = regSize;
        CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);

        memset(&ucprop, 0, sizeof(CUmemAllocationProp));
        ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ucprop.location.id = comm->cudaDev;
        ucprop.requestedHandleTypes = ncclCuMemHandleType;
        CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);

        CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&regRecord->baseAddr, &regRecord->baseSize, (CUdeviceptr)regRecord->addr), ret, fail);
        if (regRecord->addr % ucgran == 0) {
          if (regSize % ucgran != 0) {
            regRecord->regUCSize = ALIGN_SIZE(regSize, ucgran);
          } else {
            regRecord->regUCSize = regSize;
          }
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
    if (minSize > regData[i].reg.regUCSize)
      minSize = regData[i].reg.regUCSize;
  }

  /* start registration */
  mcsize = ucsize = minSize;
  mcprop.size = minSize;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(mcsize, mcgran);
  mcprop.size = mcsize;

  if (comm->localRank == 0) {
    NCCLCHECKGOTO(nvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle), ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
    NCCLCHECKGOTO(nvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);
  }

  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->nvlsResources->dev), ret, fail);
  // Coverity complains that regRecord could be NULL.  That won't in practice be the case because we've already checked
  // (regData[i].reg.state & NVLS_REG_POSSIBLE) of all local ranks, which would catch it and bail out.
  // coverity[var_deref_op]
  CUCHECKGOTO(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)regRecord->addr, ucsize, 0), ret, fail);

  // Create a VA for the NVLS
  CUCHECKGOTO(cuMemAddressReserve(&regPtr, mcsize, mcgran, 0U, 0), ret, fail);
  // Map the VA locally
  CUCHECKGOTO(cuMemMap(regPtr, mcsize, 0, mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess(regPtr, mcsize, &comm->nvlsResources->accessDesc, 1), ret, fail);

  regRecord->regAddr = regPtr;
  regRecord->regUCSize = ucsize;
  regRecord->regMCSize = mcsize;
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

  *regAddr = (uintptr_t)regPtr + regData[comm->localRank].offset;
  *regUsed = 1;
exit:
  free(regData);
  return ret;
fail:
  *regUsed = 0;
  goto exit;
}

static ncclResult_t nvlsRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, struct ncclReg *sendRegRecord, struct ncclReg *recvRegRecord, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;
  int regBufUsed = 0;
  struct localRegData *regData = NULL;
  bool sendNeedReg = false, recvNeedReg = false;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks * 2), ret, fail);

  if (sendRegRecord) {
    memcpy(&regData[comm->localRank * 2].reg, sendRegRecord, sizeof(struct ncclReg));
    regData[comm->localRank * 2].offset = (uintptr_t)sendbuff - sendRegRecord->addr;
  }

  if (recvRegRecord) {
    memcpy(&regData[comm->localRank * 2 + 1].reg, recvRegRecord, sizeof(struct ncclReg));
    regData[comm->localRank * 2 + 1].offset = (uintptr_t)recvbuff - recvRegRecord->addr;
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
    regBufUsed = 1;
    INFO(NCCL_REG, "rank %d reuse registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);
    goto exit;
  }

  /* Start Registration. Not found registered buffers, then check whether both send and recv buffer locate
   * in register request cache. */
  if (sendNeedReg && sendbuff && sendbuffSize > 0) {
    tryRegisterBuffer(comm, (uintptr_t)sendbuff, sendbuffSize, &regSendPtr, &regBufUsed);
    if (regBufUsed == 0) goto fail;
  }

  if (recvNeedReg && recvbuff && recvbuffSize > 0) {
    tryRegisterBuffer(comm, (uintptr_t)recvbuff, recvbuffSize, &regRecvPtr, &regBufUsed);
    if (regBufUsed == 0) goto fail;
  }

  INFO(NCCL_REG, "rank %d successfully registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff %p, reg recvbuff %p", comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);

exit:
  *outRegBufSend = (void*)regSendPtr;
  *outRegBufRecv = (void*)regRecvPtr;
  *outRegBufUsed = regBufUsed;
  free(regData);
  return ncclSuccess;
fail:
  regBufUsed = 0;
  INFO(NCCL_REG, "rank %d failed to NVLS register sendbuff %p sendbuffSize %ld recvbuff %p recvbuffSize %ld", comm->rank, sendbuff, sendbuffSize, recvbuff, recvbuffSize);
  goto exit;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  struct ncclReg *sendRegRecord = NULL;
  struct ncclReg *recvRegRecord = NULL;
  bool sendIsValid = false;
  bool recvIsValid = false;

  *outRegBufUsed = 0;
  if (sendbuff) {
    NCCLCHECK(ncclRegFind(comm, sendbuff, sendbuffSize, &sendRegRecord));
    NCCLCHECK(ncclRegLocalIsValid(sendRegRecord, &sendIsValid));
  } else {
    sendIsValid = true;
  }
  if (recvbuff) {
    NCCLCHECK(ncclRegFind(comm, recvbuff, recvbuffSize, &recvRegRecord));
    NCCLCHECK(ncclRegLocalIsValid(recvRegRecord, &recvIsValid));
  } else {
    recvIsValid = true;
  }

  if (sendIsValid && recvIsValid)
    NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord, outRegBufUsed, outRegBufSend, outRegBufRecv));

  return ncclSuccess;
}

struct ncclNvlsCleanupCallback {
  struct ncclCommCallback base;
  struct ncclReg *reg;
  struct ncclComm *comm;
};

static ncclResult_t cleanupNvls(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNvlsCleanupCallback* obj = (struct ncclNvlsCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
    struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize,
    int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded
  ) {
  struct ncclNvlsCleanupCallback* sendRecord = NULL;
  struct ncclNvlsCleanupCallback* recvRecord = NULL;
  void *baseSend = NULL;
  void *baseRecv = NULL;
  size_t baseSendSize = 0;
  size_t baseRecvSize = 0;
  struct ncclReg *sendRegRecord = NULL;
  struct ncclReg *recvRegRecord = NULL;

  *outRegBufUsed = 0;
  if (sendbuff) {
    CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseSend, &baseSendSize, (CUdeviceptr)sendbuff));
    NCCLCHECK(ncclCommGraphRegister(comm, baseSend, baseSendSize, (void**)&sendRegRecord));
  }

  if (recvbuff) {
    CUCHECK(cuMemGetAddressRange((CUdeviceptr *)&baseRecv, &baseRecvSize, (CUdeviceptr)recvbuff));
    NCCLCHECK(ncclCommGraphRegister(comm, baseRecv, baseRecvSize, (void**)&recvRegRecord));
  }

  NCCLCHECK(nvlsRegisterBuffer(comm, baseSend, baseRecv, baseSendSize, baseRecvSize, sendRegRecord, recvRegRecord, outRegBufUsed, outRegBufSend, outRegBufRecv));

  if (*outRegBufUsed) {
    if (sendRegRecord) {
      sendRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));
      sendRecord->base.fn = cleanupNvls;
      sendRecord->reg = sendRegRecord;
      sendRecord->comm = comm;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)sendRecord);
      *nCleanupQueueEltsAdded += 1;
    }

    if (recvRegRecord) {
      recvRecord = (struct ncclNvlsCleanupCallback*)malloc(sizeof(struct ncclNvlsCleanupCallback));
      recvRecord->base.fn = cleanupNvls;
      recvRecord->reg = recvRegRecord;
      recvRecord->comm = comm;
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)recvRecord);
      *nCleanupQueueEltsAdded += 1;
    }
  } else {
    if (sendbuff) NCCLCHECK(ncclCommGraphDeregister(comm, sendRegRecord));
    if (recvbuff) NCCLCHECK(ncclCommGraphDeregister(comm, recvRegRecord));
  }

  return ncclSuccess;
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
    int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv,
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded
  ) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t ucsize, size_t mcsize) {
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
