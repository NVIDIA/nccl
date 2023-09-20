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

#if CUDART_VERSION >= 12010

// Currently we only support POSIX_FILE_DESCRIPTOR handle exchange
#define USE_POSIX_FD 1

#if USE_POSIX_FD
#define NVLS_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#else
#define NVLS_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_NONE
#endif

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

ncclResult_t nvlsGroupCreate(struct ncclComm *comm, struct ncclNvlsSharedRes* resources, int rank, unsigned int nranks, char* shareableHandle) {
  size_t size = resources->size;

  // Create a Multicast group
  CUmulticastObjectProp* prop = &resources->properties;

  INFO(NCCL_NVLS, "NVLS Creating Multicast group nranks %d size %zi on rank %d", nranks, size, rank);
  CUCHECK(cuMulticastCreate(&resources->mcHandle, prop));

  if (NVLS_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_NONE) {
    // Get a handle to pass to other ranks
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, resources->mcHandle, NVLS_CU_MEM_HANDLE_TYPE, 0));
  }
  else {
    memcpy(shareableHandle, &resources->mcHandle, sizeof(resources->mcHandle));
  }

  INFO(NCCL_NVLS, "NVLS Created Multicast group %llx nranks %d size %zi on rank %d", resources->mcHandle, nranks, size, rank);

  return ncclSuccess;
}

ncclResult_t nvlsGroupAddDevice(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  INFO(NCCL_NVLS, "NVLS group %llx adding dev %d", resources->mcHandle, resources->dev);
  CUCHECK(cuMulticastAddDevice(resources->mcHandle, resources->dev));
  return ncclSuccess;
}

ncclResult_t nvlsGroupConnect(struct ncclComm *comm, struct ncclNvlsSharedRes* resources, int rank, char* shareableHandle) {
  CUmemAllocationHandleType type = NVLS_CU_MEM_HANDLE_TYPE;

  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = *(int *)shareableHandle;
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle from rank %d fd %d", comm->localRank, rank, fd);
    struct ncclProxyConnector proxyConn;
    int tpProxyRank = comm->topParentRanks[rank];
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, tpProxyRank, &proxyConn));
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of fd %d from rank %d", comm->localRank, fd, rank);
    NCCLCHECK(ncclProxyClientConvertFdBlocking(comm, &proxyConn, fd, (int *)shareableHandle));
    fd = *(int *)shareableHandle;
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECK(cuMemImportFromShareableHandle(&resources->mcHandle, (void *)(uintptr_t)fd, type));
  } else {
    if (NVLS_CU_MEM_HANDLE_TYPE != CU_MEM_HANDLE_TYPE_NONE) {
      CUCHECK(cuMemImportFromShareableHandle(&resources->mcHandle, (void *)shareableHandle, type));
    } else {
      memcpy(&resources->mcHandle, shareableHandle, sizeof(resources->mcHandle));
    }
  }
  return ncclSuccess;
}

ncclResult_t nvlsGroupDisconnect(struct ncclComm *comm, struct ncclNvlsSharedRes* resources) {
  CUmemAllocationHandleType type = NVLS_CU_MEM_HANDLE_TYPE;

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = *(int *)resources->shareableHandle;
    (void) close(fd);
  }

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
      NCCLCHECKGOTO(nvlsGroupCreate(comm, resources, comm->localRank, comm->localRanks, shareableHandle), res, cleanup);
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
    } else {
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
      NCCLCHECKGOTO(nvlsGroupConnect(comm, resources, comm->localRankToRank[0], shareableHandle), res, cleanup);
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

#endif /* CUDA_VERSION >= 12010 */
