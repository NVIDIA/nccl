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

#define NVLS_HANDLE_SIZE 64

struct nvlsResources {
  CUmulticastObjectProp properties;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t size;
  size_t granularity;
  CUmemGenericAllocationHandle mcHandle; // Multicast handle for NVLS buffer
  char* mcBuff; // Multicast NVLS buffer address
  CUmemGenericAllocationHandle ucHandle; // Unicast Handle for NVLS buffer
  char* ucBuff; // Unicast NVLS buffer address
};


ncclResult_t nvlsGetProperties(struct ncclComm *comm, struct nvlsResources* resources, int dev, int nranks, size_t size) {
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

ncclResult_t nvlsGroupCreate(struct ncclComm *comm, struct nvlsResources* resources, int rank, unsigned int nranks, char* shareableHandle) {
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

ncclResult_t nvlsGroupAddDevice(struct ncclComm *comm, struct nvlsResources* resources) {
  INFO(NCCL_NVLS, "NVLS group %llx adding dev %d", resources->mcHandle, resources->dev);
  CUCHECK(cuMulticastAddDevice(resources->mcHandle, resources->dev));
  return ncclSuccess;
}

ncclResult_t nvlsGroupUnbind(struct ncclComm *comm, struct nvlsResources* resources) {
  int dev = resources->dev;
  size_t size = resources->size;
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zi dev %d", resources->mcHandle, size, dev);

  // Unbind physical memory from group for the given device
  CUCHECK(cuMulticastUnbind(resources->mcHandle, dev, 0/*mcOffset*/, size));

  return ncclSuccess;
}

ncclResult_t nvlsGroupConnect(struct ncclComm *comm, struct nvlsResources* resources, int rank, char* shareableHandle) {
  CUmemAllocationHandleType type = NVLS_CU_MEM_HANDLE_TYPE;

  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    int fd = *(int *)shareableHandle;
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle from rank %d fd %d", comm->localRank, rank, fd);
    struct ncclProxyConnector proxyConn;
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, rank, &proxyConn));
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of fd %d from rank %d", comm->localRank, fd, rank);
    NCCLCHECK(ncclProxyCallBlocking(&proxyConn, ncclProxyMsgConvertFd, shareableHandle, sizeof(int), &fd, sizeof(int)));
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

ncclResult_t nvlsGroupBindMem(struct ncclComm *comm, struct nvlsResources* resources) {
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

ncclResult_t nvlsGroupMapMem(struct ncclComm *comm, struct nvlsResources* resources) {
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

ncclResult_t nvlsGroupUnmapMem(struct ncclComm *comm, struct nvlsResources* resources) {
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

NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", 16);

NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 1);

ncclResult_t ncclNvlsSetup(struct ncclComm* comm) {
  if (!ncclParamNvlsEnable() || comm->localRanks <= 1 || comm->nNodes>1) return ncclSuccess;
  CUdevice dev;
  int driverVersion;
  if (CUPFN(cuDeviceGet) == NULL) return ncclSuccess;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  comm->nvlsSupport = 0;
  // NVLS Multicast support requires CUDA12.1 UMD + KMD
  if (CUPFN(cuMulticastCreate) != NULL && driverVersion >= 12010) {
    CUCHECK(cuDeviceGetAttribute(&comm->nvlsSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
  }
  INFO(NCCL_INIT, "NVLS multicast support is %savailable on dev %d", comm->nvlsSupport ? "" : "not ", dev);
  if (comm->nvlsSupport == 0) return ncclSuccess;

  int nChannels = comm->nvlsChannels = std::max(comm->minCTAs, std::min(comm->maxCTAs, (int)ncclParamNvlsChannels()));
  int rank = comm->localRank, nranks = comm->localRanks;

  for (int c=0; c<nChannels; c++) {
    NCCLCHECK(initChannel(comm, c));
  }
  ncclResult_t res = ncclSuccess;
  struct nvlsResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  comm->nvlsResources = resources;

  size_t buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];
  size_t memSize = NVLS_MEM_ALIGN_SIZE;
  size_t nvlsPerRankSize = nChannels*2*(buffSize+memSize);
  size_t nvlsTotalSize = nvlsPerRankSize*nranks;

  INFO(NCCL_INIT|NCCL_NVLS, "NVLS comm %p rank %d nranks %d buffSize %zi memSize %zi nvlsPerRankSize %zi nvlsTotalSize %zi",
       comm, rank, nranks, buffSize, memSize, nvlsPerRankSize, nvlsTotalSize);

  char* nvlsShareableHandle = NULL;
  NCCLCHECKGOTO(ncclCalloc(&nvlsShareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
  NCCLCHECKGOTO(nvlsGetProperties(comm, resources, dev, nranks, nvlsTotalSize), res, cleanup);
  if (rank == 0) {
    NCCLCHECKGOTO(nvlsGroupCreate(comm, resources, rank, nranks, nvlsShareableHandle), res, cleanup);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, rank, nranks, 0, nvlsShareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, rank, nranks, 0, nvlsShareableHandle, NVLS_HANDLE_SIZE), res, cleanup);
    NCCLCHECKGOTO(nvlsGroupConnect(comm, resources, 0, nvlsShareableHandle), res, cleanup);
  }

  NCCLCHECKGOTO(nvlsGroupAddDevice(comm, resources), res, cleanup);
  NCCLCHECKGOTO(nvlsGroupBindMem(comm, resources), res, cleanup);
  // Local intra-node barrier to ensure everyone has bound their memory to the group
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, cleanup);
  NCCLCHECKGOTO(nvlsGroupMapMem(comm, resources), res, cleanup);

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.nHeads = nranks;
    for (int i=0; i<NCCL_MAX_NVLS_ARITY; i++) channel->nvls.up[i] = -1;
    channel->nvls.down = comm->nRanks+1+comm->localRank;
    channel->nvls.out = -1;       // Network not yet implemented.
    channel->nvls.headRank = comm->localRank;  // Network not yet implemented.
  }

  for (int r=0; r<nranks; r++) {
    int nvlsPeer = comm->nRanks+1+r;
    for (int c=0; c<nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      channel->nvls.up[r] = nvlsPeer;

      char* mem = NULL;
      struct ncclChannelPeer* peer = channel->peers+nvlsPeer;

      // Reduce UC -> MC
      mem = resources->ucBuff + (r*2*nChannels+c)*(buffSize+memSize);
      peer->send[0].transportComm = &nvlsTransport.send;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[0].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      mem = resources->mcBuff + (r*2*nChannels+c)*(buffSize+memSize);
      peer->recv[1].transportComm = &nvlsTransport.recv;
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[1].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      peer->recv[1].conn.flags |= NCCL_NVLS_MIN_POLL;

      // Broadcast MC -> UC
      mem = resources->ucBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize);
      peer->recv[0].transportComm = &nvlsTransport.recv;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->recv[0].conn.head = (uint64_t*)(mem+buffSize);
      peer->recv[0].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      mem = resources->mcBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize);
      peer->send[1].transportComm = &nvlsTransport.send;
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = mem;
      peer->send[1].conn.head = (uint64_t*)(mem+buffSize);
      peer->send[1].conn.tail = (uint64_t*)(mem+buffSize+memSize/2);
      peer->send[1].conn.flags |= NCCL_NVLS_MIN_POLL;

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[nvlsPeer].send[0], &peer->send[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[nvlsPeer].recv[0], &peer->recv[0].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[nvlsPeer].send[1], &peer->send[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeers[nvlsPeer].recv[1], &peer->recv[1].conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->hostStream.cudaStream), res, cleanup);

      /*INFO(NCCL_INIT|NCCL_NVLS, "Peer %d Channel %d MC buff %p/%p UC Buff %p/%p",
          nvlsPeer, c,
          resources->mcBuff + (r*2*nChannels+c)*(buffSize+memSize),
          resources->mcBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize),
          resources->ucBuff + (r*2*nChannels+c)*(buffSize+memSize),
          resources->ucBuff + ((r*2+1)*nChannels+c)*(buffSize+memSize));*/
    }
  }

  free(nvlsShareableHandle);
  return res;

cleanup:
  comm->nvlsSupport = 0;
  free(nvlsShareableHandle);
  return res;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct nvlsResources* resources = (struct nvlsResources*)comm->nvlsResources;
  if (resources == NULL) return ncclSuccess;
  NCCLCHECK(nvlsGroupUnbind(comm, resources));
  NCCLCHECK(nvlsGroupUnmapMem(comm, resources));
  free(resources);
  comm->nvlsResources = NULL;
  return ncclSuccess;
}

#else

/*
 * Pre CUDA 12.1 stubs
 */

ncclResult_t ncclNvlsSetup(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
