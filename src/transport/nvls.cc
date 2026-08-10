/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Implementation of the NVLink SHARP (NVLS) transport

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "proxy.h"
#include "enqueue.h"
#include "register.h"
#include "transport.h"
#include "register_inline.h"

#if CUDART_VERSION >= 12010

struct graphRegData {
  uintptr_t offset;
  size_t size;
};

struct localRegData {
  struct ncclReg reg;
  intptr_t offset;
  int handleTypes;
};

ncclResult_t nvlsCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1,
                            struct ncclPeerInfo* info2) {
  // This transport cannot be used for p2p
  *ret = 0;
  return ncclSuccess;
}

ncclResult_t nvlsSendFree(struct ncclComm* comm, struct ncclConnector* send) {
  return ncclSuccess;
}

ncclResult_t nvlsRecvFree(struct ncclComm* comm, struct ncclConnector* recv) {
  return ncclSuccess;
}

struct ncclTransport nvlsTransport = {"NVLS",
                                      nvlsCanConnect,
                                      {NULL, NULL, nvlsSendFree, NULL, NULL, NULL, NULL, NULL},
                                      {NULL, NULL, nvlsRecvFree, NULL, NULL, NULL, NULL, NULL}};

ncclResult_t ncclNvlsGroupCreate(struct ncclComm* comm, CUmulticastObjectProp* prop, int rank, unsigned int nranks,
                                 CUmemGenericAllocationHandle* mcHandle, char* shareableHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  size_t size = prop->size;

  // Create a Multicast group

  INFO(NCCL_NVLS, "NVLS Creating Multicast group nranks %d size %zu on rank %d", nranks, size, rank);
  CUCHECK(cuMulticastCreate(mcHandle, prop));

  if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
    // Get a handle to pass to other ranks
    CUCHECK(cuMemExportToShareableHandle(shareableHandle, *mcHandle, ncclCuMemHandleType, 0));
  } else {
    memcpy(shareableHandle, mcHandle, sizeof(CUmemGenericAllocationHandle));
  }

  INFO(NCCL_NVLS, "NVLS Created Multicast group %llx nranks %d size %zu on rank %d", *mcHandle, nranks, size, rank);

  return ncclSuccess;
}

ncclResult_t ncclNvlsGroupConnect(struct ncclComm* comm, char* shareableHandle, int rank,
                                  CUmemGenericAllocationHandle* mcHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  int fd = -1;
  ncclResult_t ret = ncclSuccess;
  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank,
          *(uint64_t*)shareableHandle, rank);
    NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, rank, shareableHandle, &fd), ret, fail);
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %d from rank %d", comm->localRank, fd, rank);
    CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void*)(uintptr_t)fd, type), ret, fail);
    SYSCHECK(close(fd), "close");
  } else {
    if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
      CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void*)shareableHandle, type), ret, fail);
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

ncclResult_t nvlsGroupUnbind(struct ncclComm* comm, size_t size, CUmemGenericAllocationHandle* mcHandle) {
  int dev = comm->cudaDev;
  INFO(NCCL_NVLS, "NVLS Unbind MC handle %llx size %zu dev %d", *mcHandle, size, dev);

  // Unbind physical memory from group for the given device
  if (size) CUCHECK(cuMulticastUnbind(*mcHandle, dev, 0 /*mcOffset*/, size));

  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle* mcHandler, CUdeviceptr ptr,
                                 int dev, size_t ucsize, size_t mcsize) {
  // unbind can trigger RM error if buffer is freed already by users
  // however, it is safe to ignore the error, and unbind will succeed anyway
  CUCALL(cuMulticastUnbind(*mcHandler, dev, 0 /*mcOffset*/, ucsize));
  CUCHECK(cuMemUnmap(ptr, mcsize));
  CUCHECK(cuMemAddressFree(ptr, mcsize));
  CUCHECK(cuMemRelease(*mcHandler));
  INFO(NCCL_NVLS, "rank %d - NVLS deregistered buffer %p on device %d ucsize %ld mcsize %ld", comm->rank, (void*)ptr,
       dev, ucsize, mcsize);
  return ncclSuccess;
}

ncclResult_t nvlsGroupUnmapMem(struct ncclComm* comm, size_t ucsize, void* ucptr,
                               CUmemGenericAllocationHandle* ucHandle, size_t mcsize, void* mcptr,
                               CUmemGenericAllocationHandle* mcHandle) {
  INFO(NCCL_NVLS, "NVLS Unmap mem UC handle 0x%llx(%p) ucsize %zu MC handle 0x%llx(%p) mcsize %zd", *ucHandle, ucptr,
       ucsize, *mcHandle, mcptr, mcsize);

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
#define NVLS_NCHANNELS_SM100_NVL 24

NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2);
NCCL_PARAM(NvlsChunkSize, "NVLS_CHUNKSIZE", 128 * 1024);
NCCL_PARAM(NvlsTreeMaxChunkSize, "NVLSTREE_MAX_CHUNKSIZE", -2);

// Returns optimal NVLSTree tuning parameters for SM100 multi-node configurations.
static ncclResult_t ncclNvlsTreeSm100Tuning(struct ncclComm* comm, int* nChannels, int* chunkSize,
                                            int* treeMaxChunkSize) {
  int nNodes = comm->nNodes;
  int ppn = comm->minLocalRanks;
  float nicBw = comm->minNetBw;
  int gpuToNicPathType = comm->graphs[NCCL_ALGO_NVLS].typeInter;

  *chunkSize = 128 * 1024;

  if (nNodes == 2 && nicBw >= 48.0f) {
    *nChannels = 32;
    *treeMaxChunkSize = 128 * 1024;
    if (ppn <= 4) {
      *chunkSize = 256 * 1024;
      *treeMaxChunkSize = 256 * 1024;
    } else if (ppn >= 16 || (ppn <= 8 && gpuToNicPathType <= PATH_PXB && nicBw < 96.0f)) {
      *treeMaxChunkSize = 64 * 1024;
    }
  } else if (nicBw >= 96.0f) {
    if (ppn <= 8) {
      *nChannels = 24;
      *chunkSize = 256 * 1024;
      *treeMaxChunkSize = 256 * 1024;
    } else {
      *nChannels = 32;
      *treeMaxChunkSize = (ppn < 32) ? 128 * 1024 : 64 * 1024;
    }
  } else if (nicBw >= 48.0f) {
    *nChannels = 24;
    *treeMaxChunkSize = 128 * 1024;
    if (gpuToNicPathType <= PATH_PXB) {
      *treeMaxChunkSize = 64 * 1024;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclNvlsTuning(struct ncclComm* comm) {
  int nChannels;
  int chunkSize = 0;
  int treeMaxChunkSize = 0;
  const char* chunkSizeEnv = ncclGetEnv("NCCL_NVLS_CHUNKSIZE");
  bool userSetChunkSize = (chunkSizeEnv != NULL && strlen(chunkSizeEnv) > 0);

  // Set default nChannels based on SM architecture
  if (comm->compCap >= 100) {
    nChannels = (comm->nNodes > 1) ? NVLS_NCHANNELS_SM100 : NVLS_NCHANNELS_SM100_NVL;
  } else {
    nChannels = NVLS_NCHANNELS_SM90;
  }

  // SM100 multi-node NVLSTree tuning (may adjust all three values)
  if (comm->minCompCap >= 100 && comm->nNodes > 1) {
    NCCLCHECK(ncclNvlsTreeSm100Tuning(comm, &nChannels, &chunkSize, &treeMaxChunkSize));
  }

  // User overrides take priority over tuning
  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT) nChannels = comm->config.nvlsCTAs;
  // If user has set chunk size or chunkSize is not set, use the chunk size as determined by ncclParamNvlsChunkSize()
  if (userSetChunkSize || chunkSize == 0) chunkSize = ncclParamNvlsChunkSize();

  // Determine final treeMaxChunkSize: env var > tuning > fallback
  int envTreeMaxChunkSize = (int)ncclParamNvlsTreeMaxChunkSize();
  if (envTreeMaxChunkSize == -2 && treeMaxChunkSize == 0) {
    treeMaxChunkSize = (comm->nNodes >= 4) ? 65536 : chunkSize;
  } else if (envTreeMaxChunkSize != -2) {
    treeMaxChunkSize = envTreeMaxChunkSize;
  }

  // Clamp nvlsChannels to [minCTAs, maxCTAs]
  nChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, nChannels));

  // Apply final values
  comm->nvlsChannels = nChannels;
  comm->nvlsChunkSize = chunkSize;
  comm->nvlsTreeMaxChunkSize = treeMaxChunkSize;

  INFO(NCCL_INIT, "NVLS tuning: nChannels %d chunkSize %d treeMaxChunkSize %d", comm->nvlsChannels, comm->nvlsChunkSize,
       comm->nvlsTreeMaxChunkSize);

  return ncclSuccess;
}

ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  comm->nvlsSupport = 0;
  comm->nvlsChannels = 0;

  if (comm->hasMultiRankNvml) {
    if (ncclParamNvlsEnable() == 1) {
      WARN("NCCL_NVLS_ENABLE has been set to \"1\" and communicator has multiple ranks using the same NVML device. "
           "This is not compatible with NCCL_NVLS_ENABLE=1.");
      return ncclInvalidUsage;
    }
    return ncclSuccess;
  }
  int gpuCount;
  NCCLCHECK(ncclTopoGetGpuCount(comm->topo, &gpuCount));
  if (!ncclParamNvlsEnable() || gpuCount < 2) return ncclSuccess;

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
  return ncclSuccess;
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nvlsSupport && comm->nNodes > 1) {
    for (int c = 0; c < comm->nvlsChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 1,
                                            &channel->nvls.treeUp, 0),
                    ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->nvls.treeUp, NCCL_MAX_NVLS_TREE_ARITY,
                                            channel->nvls.treeDown, 0),
                    ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_NVLS], 0), ret, fail);
    INFO(NCCL_INIT, "Connected NVLS tree");
  }
exit:
  return ret;
fail:
  goto exit;
}

// Create (local rank 0) or import (other local ranks) the multicast group
// shared by the local ranks of the comm, and add this device to it. On
// failure, any handle already obtained is released before returning.
static ncclResult_t nvlsGroupRendezvous(struct ncclComm* comm, CUmulticastObjectProp* mcprop,
                                        CUmemGenericAllocationHandle* mcHandle) {
  ncclResult_t ret = ncclSuccess;
  char shareableHandle[NVLS_HANDLE_SIZE];
  bool hasHandle = false;

  memset(shareableHandle, '\0', sizeof(shareableHandle));
  if (comm->localRank == 0) {
    NCCLCHECK(ncclNvlsGroupCreate(comm, mcprop, comm->localRank, comm->localRanks, mcHandle, shareableHandle));
    hasHandle = true;
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
    NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], mcHandle), ret, fail);
    hasHandle = true;
  }
  CUCHECKGOTO(cuMulticastAddDevice(*mcHandle, comm->cudaDev), ret, fail);

exit:
  return ret;
fail:
  if (hasHandle) {
    CUCHECKIGNORE(cuMemRelease(*mcHandle));
    *mcHandle = 0;
  }
  goto exit;
}

static ncclResult_t nvlsAllocateMem(struct ncclComm* comm, const CUmemAccessDesc* desc, size_t size,
                                    struct ncclNvlsMcMem* mem) {
  CUmemGenericAllocationHandle* ucHandle = &mem->ucHandle;
  CUmemGenericAllocationHandle* mcHandle = &mem->mcHandle;
  void** ucptr = (void**)&mem->ucPtr;
  void** mcptr = (void**)&mem->mcPtr;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  CUresult err;
  ncclResult_t ret = ncclSuccess;
  size_t mcsize;
  size_t ucsize;
  size_t ucgran, mcgran;
  int allocMcHandle = 0;

  mcsize = ucsize = size;
  *ucptr = *mcptr = NULL;
  memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
  mcprop.numDevices = comm->localRanks;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  mcprop.size = size;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(mcsize, mcgran);
  mcprop.size = mcsize;

  NCCLCHECKGOTO(nvlsGroupRendezvous(comm, &mcprop, mcHandle), ret, fail);
  allocMcHandle = 1;

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
  CUCHECKGOTO(cuMemCreate(ucHandle, ucsize, &ucprop, 0), ret, fail1);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*ucptr, ucsize, 0, *ucHandle, 0), ret, fail2);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*ucptr, ucsize, desc, 1), ret, fail3);
  CUDACHECKGOTO(cudaMemset(*ucptr, 0, ucsize), ret, fail3);
  // Track NVLS buffer as persistent memory
  NCCLCHECKGOTO(ncclMemTrack(comm->memManager, *ucptr, ucsize, *ucHandle, ncclCuMemHandleType, ncclMemPersist), ret,
                fail3);

  // intra-node barrier to mitigate the possible hang in cuMulticastBindMem during abort
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail3);
  // Bind physical memory to the Multicast group
  // NB: It will block until all ranks have been added to the Group
  // This is where we normally see issues if the system NVLS/Multicast support is broken
  err = CUPFN(cuMulticastBindMem(*mcHandle, 0 /*mcOffset*/, *ucHandle, 0 /*memOffset*/, ucsize, 0 /*flags*/));
  if (err != CUDA_SUCCESS) {
    const char* errStr;
    (void)pfn_cuGetErrorString(err, &errStr);    // Fail the job as NVLS support is not functional
    WARN("Failed to bind NVLink SHARP (NVLS) Multicast memory of size %ld : CUDA error %d '%s'.\nThis is usually "
         "caused by a system or configuration error in the Fabric Manager or NVSwitches.\nDisable NVLS "
         "(NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.",
         ucsize, err, errStr);
    ret = ncclUnhandledCudaError;
    goto fail3;
  }

  // Map mc virtual address
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)mcptr, mcsize, mcgran, 0U, 0), ret, fail);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)*mcptr, mcsize, 0, *mcHandle, 0), ret, fail);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)*mcptr, mcsize, desc, 1), ret, fail);
  mem->ucSize = ucsize;
  mem->mcSize = mcsize;

  INFO(
    NCCL_NVLS,
    "NVLS rank %d (dev %d) alloc done, ucptr %p ucgran %ld mcptr %p mcgran %ld ucsize %ld mcsize %ld (inputsize %ld)",
    comm->rank, comm->cudaDev, *ucptr, ucgran, *mcptr, mcgran, ucsize, mcsize, size);

exit:
  return ret;
fail3:
  CUCHECK(cuMemUnmap((CUdeviceptr)*ucptr, ucsize));
fail2:
  CUCHECK(cuMemRelease(*ucHandle));
fail1:
  CUCHECK(cuMemAddressFree((CUdeviceptr)*ucptr, ucsize));
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
  nChannels = comm->nvlsChannels;
  nvlsStepSize = comm->nvlsChunkSize;
  buffSize = nvlsStepSize * NCCL_STEPS;
  nvlsPerRankSize = nChannels * 2 * buffSize;
  nvlsTotalSize = nvlsPerRankSize * nHeads;

  INFO(NCCL_INIT | NCCL_NVLS,
       "NVLS comm %p headRank %d nHeads %d nvlsRanks %d buffSize %zu nvlsPerRankSize %zu nvlsTotalSize %zu", comm,
       headRank, nHeads, comm->localRanks, buffSize, nvlsPerRankSize, nvlsTotalSize);

  NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, nvlsTotalSize, &resources->buff), res, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                        /*concurrent=*/false, &hostStream),
                res, fail);
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                        /*concurrent=*/false, &deviceStream),
                res, fail);
  for (int h = 0; h < nHeads; h++) {
    int nvlsPeer = comm->nRanks + 1 + h;
    for (int c = 0; c < nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

      // Reduce UC -> MC
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->buff.ucPtr + (h * 2 * nChannels + c) * buffSize;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->buff.mcPtr + (h * 2 * nChannels + c) * buffSize;

      // Broadcast MC -> UC
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = resources->buff.ucPtr + ((h * 2 + 1) * nChannels + c) * buffSize;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = resources->buff.mcPtr + ((h * 2 + 1) * nChannels + c) * buffSize;

      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
      CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn,
                                    sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                    res, fail);
    }
  }

  NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                        /*concurrent=*/false),
                res, fail);
  NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                        /*concurrent=*/false),
                res, fail);
  // For now, the barrier is a must that guarantees all buffers are mc-mapped before accessing peer's buffer
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                res, fail);
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
  uintptr_t* nvlsShmem = NULL;
  bool nvlsShare = parent && parent->nvlsSupport && parent->shareResources && parent->localRanks == comm->localRanks;

  if (comm->nvlsSupport == 0 || comm->nvlsChannels == 0) return ncclSuccess;

  if (nvlsShare) {
    /* reuse NVLS resources */
    comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
    /* Inherit chunk sizes from the shared resource since we're reusing the parent's
     * NVLS buffers, which were allocated and laid out based on these values. */
    comm->nvlsChunkSize = parent->nvlsResources->chunkSize;
    comm->nvlsTreeMaxChunkSize = parent->nvlsResources->treeMaxChunkSize;
    for (int c = 0; c < comm->nvlsChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, parent, true), res, fail);
    }

    comm->nvlsResources = parent->nvlsResources;
    ncclAtomicRefCountIncrement(&parent->nvlsResources->refCount);
  } else {
    struct ncclNvlsSharedRes* resources = NULL;
    int nHeads = comm->channels[0].nvls.nHeads;
    size_t memSize = 64;
    cudaStream_t hostStream, deviceStream;

    if (parent != nullptr && parent->nvlsSupport && parent->shareResources) {
      /* ranks on other nodes might share the NVLS resources, we need to cap nvlsChannels
       * and match NVLS chunk sizes to make sure they agree for each rank. */
      comm->nvlsChannels = std::min(comm->nvlsChannels, parent->nvlsResources->nChannels);
      comm->nvlsChunkSize = parent->nvlsResources->chunkSize;
      comm->nvlsTreeMaxChunkSize = parent->nvlsResources->treeMaxChunkSize;
    }

    int nChannels = comm->nvlsChannels;
    size_t creditSize = nChannels * 2 * memSize * nHeads;
    int nvlsStepSize = comm->nvlsChunkSize;

    NCCLCHECKGOTO(ncclCalloc(&comm->nvlsResources, 1), res, fail);
    comm->nvlsResources->inited = false;
    comm->nvlsResources->refCount = 1;
    comm->nvlsResources->nChannels = nChannels;
    comm->nvlsResources->nHeads = nHeads;
    comm->nvlsResources->chunkSize = comm->nvlsChunkSize;
    comm->nvlsResources->treeMaxChunkSize = comm->nvlsTreeMaxChunkSize;
    resources = comm->nvlsResources;

    for (int c = 0; c < nChannels; c++) {
      NCCLCHECKGOTO(initNvlsChannel(comm, c, NULL, false), res, fail);
    }

    memset(&resources->accessDesc, 0, sizeof(resources->accessDesc));
    resources->accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    resources->accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    resources->accessDesc.location.id = comm->cudaDev;
    resources->dev = comm->cudaDev;

    NCCLCHECKGOTO(nvlsAllocateMem(comm, &resources->accessDesc, creditSize, &resources->credit), res, fail);

    // Set up head and tail only for now
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                          /*concurrent=*/false, &hostStream),
                  res, fail);
    NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode),
                                          &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream),
                  res, fail);
    for (int h = 0; h < nHeads; h++) {
      int nvlsPeer = comm->nRanks + 1 + h;
      for (int c = 0; c < nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        char* mem = NULL;
        struct ncclChannelPeer* peer = channel->peers[nvlsPeer];

        // Reduce UC -> MC
        mem = resources->credit.ucPtr + (h * 2 * nChannels + c) * memSize;
        peer->send[1].transportComm = &nvlsTransport.send;
        peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[1].conn.head = (uint64_t*)mem;
        peer->send[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[1].conn.stepSize = nvlsStepSize;
        mem = resources->credit.mcPtr + (h * 2 * nChannels + c) * memSize;
        peer->recv[0].transportComm = &nvlsTransport.recv;
        peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[0].conn.head = (uint64_t*)mem;
        peer->recv[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[0].conn.stepSize = nvlsStepSize;
        peer->recv[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        // Broadcast MC -> UC
        mem = resources->credit.ucPtr + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->recv[1].transportComm = &nvlsTransport.recv;
        peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[1].conn.head = (uint64_t*)mem;
        peer->recv[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[1].conn.stepSize = nvlsStepSize;
        mem = resources->credit.mcPtr + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->send[0].transportComm = &nvlsTransport.send;
        peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[0].conn.head = (uint64_t*)mem;
        peer->send[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[0].conn.stepSize = nvlsStepSize;
        peer->send[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[0], &peer->send[0].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[0], &peer->recv[0].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->send[1], &peer->send[1].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[nvlsPeer]->recv[1], &peer->recv[1].conn,
                                      sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream),
                      res, fail);
      }
    }
    NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                          /*concurrent=*/false),
                  res, fail);
    NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode),
                                          &comm->sharedRes->deviceStream, /*concurrent=*/false),
                  res, fail);
  }

  // MNNVL does not support NVLS buffer registration
  if (!comm->MNNVL && comm->nvlsResources->nvlsShmemHandle == NULL) {
    /* create shared memory for fast NVLS buffer registration */
    typeSize = DIVUP(sizeof(struct localRegData) << 1, CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

    if (comm->localRank == 0) {
      shmPath[0] = '\0';
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath),
                                (CACHE_LINE_SIZE * comm->localRanks + typeSize * comm->localRanks) * 2,
                                (void**)&nvlsShmem, NULL, comm->localRanks - 1, &comm->nvlsResources->nvlsShmemHandle),
                    res, fail);
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank,
                                                comm->localRanks, 0, shmPath, sizeof(shmPath)),
                    res, fail);
    } else {
      NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank,
                                                comm->localRanks, 0, shmPath, sizeof(shmPath)),
                    res, fail);
      NCCLCHECKGOTO(ncclShmOpen(shmPath, sizeof(shmPath),
                                (CACHE_LINE_SIZE * comm->localRanks + typeSize * comm->localRanks) * 2,
                                (void**)&nvlsShmem, NULL, -1, &comm->nvlsResources->nvlsShmemHandle),
                    res, fail);
    }
    /* need 2 pools and a shared counter for shmem-based collectives */
    comm->nvlsResources->nvlsShmem.cnt[0] = (size_t*)nvlsShmem;
    comm->nvlsResources->nvlsShmem.ptr[0] =
      (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[0] + CACHE_LINE_SIZE * comm->localRanks);
    comm->nvlsResources->nvlsShmem.cnt[1] =
      (size_t*)((char*)comm->nvlsResources->nvlsShmem.ptr[0] + typeSize * comm->localRanks);
    comm->nvlsResources->nvlsShmem.ptr[1] =
      (void*)((char*)comm->nvlsResources->nvlsShmem.cnt[1] + CACHE_LINE_SIZE * comm->localRanks);
    comm->nvlsResources->nvlsShmem.round = 0;
    comm->nvlsResources->nvlsShmem.maxTypeSize = typeSize;
  }

exit:
  return res;
fail:
  comm->nvlsSupport = 0;
  goto exit;
}

// Free what remains of a suspended NVLS allocation (see ncclNvlsSuspend
// below): the group and physical memory are already released, so only the VA
// reservations and the optional CPU backup are left.
static ncclResult_t nvlsFreeSuspended(struct ncclNvlsMcMem* mem) {
  if (mem->cpuBackup) {
    NCCLCHECK(ncclCudaHostFree(mem->cpuBackup));
    mem->cpuBackup = NULL;
  }
  if (mem->ucPtr) CUCHECKIGNORE(cuMemAddressFree((CUdeviceptr)mem->ucPtr, mem->ucSize));
  if (mem->mcPtr) CUCHECKIGNORE(cuMemAddressFree((CUdeviceptr)mem->mcPtr, mem->mcSize));
  return ncclSuccess;
}

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = (struct ncclNvlsSharedRes*)comm->nvlsResources;
  if (resources == NULL) return ncclSuccess;

  if (ncclAtomicRefCountDecrement(&resources->refCount) == 0) {
    if (!comm->MNNVL && resources->nvlsShmemHandle) NCCLCHECK(ncclShmClose(resources->nvlsShmemHandle));

    if (resources->mcSuspended) {
      NCCLCHECK(nvlsFreeSuspended(&resources->credit));
      NCCLCHECK(nvlsFreeSuspended(&resources->buff));
    } else {
      if (resources->credit.ucPtr || resources->credit.mcPtr) {
        NCCLCHECK(nvlsGroupUnbind(comm, resources->credit.ucSize, &resources->credit.mcHandle));
        NCCLCHECK(nvlsGroupUnmapMem(comm, resources->credit.ucSize, resources->credit.ucPtr,
                                    &resources->credit.ucHandle, resources->credit.mcSize, resources->credit.mcPtr,
                                    &resources->credit.mcHandle));
      }

      if (comm->nvlsResources->inited) {
        NCCLCHECK(nvlsGroupUnbind(comm, resources->buff.ucSize, &resources->buff.mcHandle));
        NCCLCHECK(nvlsGroupUnmapMem(comm, resources->buff.ucSize, resources->buff.ucPtr, &resources->buff.ucHandle,
                                    resources->buff.mcSize, resources->buff.mcPtr, &resources->buff.mcHandle));
      }
    }
    free(resources);
    comm->nvlsResources = NULL;
  }
  return ncclSuccess;
}

/*
 * NVLS multicast suspend/resume (ncclCommSuspend/ncclCommResume extension).
 *
 * cuda-checkpoint cannot checkpoint a process holding live multicast (00FD)
 * objects: `cuCheckpointProcessCheckpoint` hangs even on an unbound multicast
 * group. ncclNvlsSuspend therefore tears the NVLS multicast layer down
 * through the CUDA API (keeping libcuda's bookkeeping consistent, which is
 * what makes the process checkpointable):
 *
 *   - the credit buffer's UC contents are copied to a CPU backup: the FIFO
 *     head/tail counters referenced by persistent conn structs are live state
 *     that must survive. The data buffer is NOT backed up: after the
 *     quiescence sync + barrier in ncclCommMemSuspend its data slots are
 *     drained, so it is simply zeroed again on resume,
 *   - MC VAs are unmapped (VA reservations retained),
 *   - UC memory is unbound from the groups and released (VA reservations
 *     retained),
 *   - the multicast group handles are released.
 *
 * ncclNvlsResume re-creates the multicast groups with the same rendezvous
 * used at setup (rank 0 creates + exports, peers import via the proxy),
 * re-creates UC memory at the IDENTICAL VAs, restores contents, re-binds, and
 * re-maps the MC VAs at their identical addresses. Because every VA is
 * unchanged, kernels, captured CUDA graphs and conn structs remain valid; the
 * CUmem handles are new, which only teardown paths observe. If resume fails
 * part-way, nvlsResumeOne unwinds back to the suspended state (handles 0, VA
 * reservations and CPU backup intact), so resume may be retried and the comm
 * may still be destroyed cleanly.
 *
 * Suspendability must be validated with ncclNvlsSuspendCheck BEFORE any
 * destructive suspend work (ncclCommMemSuspend does this up front).
 *
 * Like setup, resume is collective across the local ranks of the comm and
 * must run for all ranks concurrently (grouped ncclCommResume or one thread
 * per comm): cuMulticastBindMem blocks until every device has been added to
 * the group.
 */

static ncclResult_t nvlsSuspendOne(struct ncclComm* comm, const char* what, struct ncclNvlsMcMem* mem,
                                   bool preserveContents) {
  if (preserveContents) {
    NCCLCHECK(ncclCudaHostCalloc((char**)&mem->cpuBackup, mem->ucSize));
    cudaError_t err = cudaMemcpy(mem->cpuBackup, mem->ucPtr, mem->ucSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      WARN("NVLS Suspend %s: failed to back up UC contents: %s", what, cudaGetErrorString(err));
      NCCLCHECK(ncclCudaHostFree(mem->cpuBackup));
      mem->cpuBackup = NULL;
      return ncclUnhandledCudaError;
    }
  }

  // Unmap the MC VA (the reservation at mcPtr is retained).
  CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)mem->mcPtr, mem->mcSize));
  // Unbind this device's memory from the group and drop the group handle.
  CUCHECKIGNORE(cuMulticastUnbind(mem->mcHandle, comm->cudaDev, 0 /*mcOffset*/, mem->ucSize));
  CUCHECKIGNORE(cuMemRelease(mem->mcHandle));
  mem->mcHandle = 0;
  // Unmap and release the UC physical memory (reservation at ucPtr retained).
  CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)mem->ucPtr, mem->ucSize));
  CUCHECKIGNORE(cuMemRelease(mem->ucHandle));
  mem->ucHandle = 0;

  INFO(NCCL_NVLS, "NVLS Suspend rank %d dev %d %s: ucptr %p ucsize %zu mcptr %p mcsize %zu released", comm->rank,
       comm->cudaDev, what, mem->ucPtr, mem->ucSize, mem->mcPtr, mem->mcSize);
  return ncclSuccess;
}

static ncclResult_t nvlsResumeOne(struct ncclComm* comm, const char* what, struct ncclNvlsMcMem* mem) {
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  ncclResult_t ret = ncclSuccess;

  // Re-create the multicast group with the same rendezvous as setup. The
  // stored sizes are already granularity-aligned.
  memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
  mcprop.numDevices = comm->localRanks;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  mcprop.size = mem->mcSize;
  NCCLCHECK(nvlsGroupRendezvous(comm, &mcprop, &mem->mcHandle));

  // Re-create UC physical memory and map it at the identical VA.
  memset(&ucprop, 0, sizeof(CUmemAllocationProp));
  ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ucprop.location.id = comm->cudaDev;
  ucprop.requestedHandleTypes = ncclCuMemHandleType;
  CUCHECKGOTO(cuMemCreate(&mem->ucHandle, mem->ucSize, &ucprop, 0), ret, failGroup);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)mem->ucPtr, mem->ucSize, 0, mem->ucHandle, 0), ret, failUcHandle);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)mem->ucPtr, mem->ucSize, &resources->accessDesc, 1), ret, failUcMapped);
  // Restore preserved contents, or zero-fill (see block comment above).
  if (mem->cpuBackup) {
    CUDACHECKGOTO(cudaMemcpy(mem->ucPtr, mem->cpuBackup, mem->ucSize, cudaMemcpyHostToDevice), ret, failUcMapped);
  } else {
    CUDACHECKGOTO(cudaMemset(mem->ucPtr, 0, mem->ucSize), ret, failUcMapped);
  }

  // As at setup: barrier before the bind (cuMulticastBindMem blocks until all
  // devices have joined the group).
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, failUcMapped);
  CUCHECKGOTO(cuMulticastBindMem(mem->mcHandle, 0 /*mcOffset*/, mem->ucHandle, 0 /*memOffset*/, mem->ucSize,
                                 0 /*flags*/),
              ret, failUcMapped);

  // Re-map the MC VA at its identical address.
  CUCHECKGOTO(cuMemMap((CUdeviceptr)mem->mcPtr, mem->mcSize, 0, mem->mcHandle, 0), ret, failBound);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)mem->mcPtr, mem->mcSize, &resources->accessDesc, 1), ret, failMcMapped);

  // Success: only now is the backup no longer needed.
  if (mem->cpuBackup) {
    NCCLCHECK(ncclCudaHostFree(mem->cpuBackup));
    mem->cpuBackup = NULL;
  }
  INFO(NCCL_NVLS, "NVLS Resume rank %d dev %d %s: ucptr %p ucsize %zu mcptr %p mcsize %zu restored", comm->rank,
       comm->cudaDev, what, mem->ucPtr, mem->ucSize, mem->mcPtr, mem->mcSize);
exit:
  return ret;

  // Unwind back to the suspended state so resume can be retried (or the comm
  // destroyed) safely.
failMcMapped:
  CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)mem->mcPtr, mem->mcSize));
failBound:
  CUCHECKIGNORE(cuMulticastUnbind(mem->mcHandle, comm->cudaDev, 0 /*mcOffset*/, mem->ucSize));
failUcMapped:
  CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)mem->ucPtr, mem->ucSize));
failUcHandle:
  CUCHECKIGNORE(cuMemRelease(mem->ucHandle));
  mem->ucHandle = 0;
failGroup:
  CUCHECKIGNORE(cuMemRelease(mem->mcHandle));
  mem->mcHandle = 0;
  goto exit;
}

// Validate that the NVLS state of this comm can be suspended. Must be called
// before any destructive suspend work so a rejection leaves the comm intact.
ncclResult_t ncclNvlsSuspendCheck(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  if (comm->nvlsSupport == 0 || resources == NULL || resources->mcSuspended) return ncclSuccess;
  if (resources->refCount > 1) {
    WARN("NVLS suspend not supported with shared NVLS resources (refCount=%d)", resources->refCount);
    return ncclInvalidUsage;
  }
  // Suspend releases only the comm's own NVLS buffers. Any other live
  // multicast object would still make the process un-checkpointable, so fail
  // loudly rather than produce a checkpoint attempt that hangs.
  for (int slot = 0; slot < comm->regCache.population; slot++) {
    struct ncclReg* reg = comm->regCache.slots[slot];
    if (reg->state & NVLS_REG_COMPLETE) {
      WARN("NVLS suspend not supported with NVLS-registered user buffers (buffer %p)", (void*)reg->begAddr);
      return ncclInvalidUsage;
    }
  }
  if (ncclDevrHasMulticastTeam(comm)) {
    WARN("NVLS suspend not supported with symmetric memory multicast teams");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t ncclNvlsSuspend(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  if (comm->nvlsSupport == 0 || resources == NULL) return ncclSuccess;
  if (resources->mcSuspended) return ncclSuccess;
  NCCLCHECK(ncclNvlsSuspendCheck(comm));

  // The ncclMemUntrack/ncclMemTrack calls below are stats-only (the NVLS
  // buffers are tracked as ncclMemPersist, which has no list entry); they
  // keep ncclCommMemStats accurate while suspended.
  if (resources->credit.ucPtr) {
    NCCLCHECK(nvlsSuspendOne(comm, "credit", &resources->credit, /*preserveContents=*/true));
    NCCLCHECK(ncclMemUntrack(comm->memManager, resources->credit.ucPtr, resources->credit.ucSize));
  }
  if (resources->inited && resources->buff.ucPtr) {
    NCCLCHECK(nvlsSuspendOne(comm, "buff", &resources->buff, /*preserveContents=*/false));
    NCCLCHECK(ncclMemUntrack(comm->memManager, resources->buff.ucPtr, resources->buff.ucSize));
  }
  resources->mcSuspended = true;
  return ncclSuccess;
}

ncclResult_t ncclNvlsResume(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = comm->nvlsResources;
  if (comm->nvlsSupport == 0 || resources == NULL) return ncclSuccess;
  if (!resources->mcSuspended) return ncclSuccess;

  if (resources->credit.ucPtr) {
    NCCLCHECK(nvlsResumeOne(comm, "credit", &resources->credit));
    NCCLCHECK(ncclMemTrack(comm->memManager, resources->credit.ucPtr, resources->credit.ucSize,
                           resources->credit.ucHandle, ncclCuMemHandleType, ncclMemPersist));
  }
  if (resources->inited && resources->buff.ucPtr) {
    NCCLCHECK(nvlsResumeOne(comm, "buff", &resources->buff));
    NCCLCHECK(ncclMemTrack(comm->memManager, resources->buff.ucPtr, resources->buff.ucSize, resources->buff.ucHandle,
                           ncclCuMemHandleType, ncclMemPersist));
  }
  resources->mcSuspended = false;
  return ncclSuccess;
}

ncclResult_t tryRegisterBuffer(struct ncclComm* comm, uintptr_t userBuff, size_t buffSize, CUdeviceptr* regAddr,
                               int* regUsed) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg* regRecord = NULL;
  CUdeviceptr regPtr = 0;
  CUmulticastObjectProp mcprop;
  CUmemAllocationProp ucprop;
  char shareableHandle[NVLS_HANDLE_SIZE];
  CUmemGenericAllocationHandle mcHandle = 0;
  size_t minSize = SIZE_MAX;
  struct localRegData* regData = NULL;
  cudaPointerAttributes attr;
  size_t ucgran, mcgran, ucsize = 0, mcsize = 0;
  bool bindComplete = false, mapComplete = false;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks), ret, fail);

  if (userBuff) {
    NCCLCHECKGOTO(ncclRegFind(comm, (void*)userBuff, buffSize, &regRecord), ret, fail);
    if (regRecord) {
      CUDACHECKGOTO(cudaPointerGetAttributes(&attr, (void*)regRecord->begAddr), ret, fail);
      if (attr.type == cudaMemoryTypeDevice) {
        size_t regSize = regRecord->endAddr - regRecord->begAddr;
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

        if (regRecord->begAddr % ucgran == 0) {
          if (regSize % ucgran != 0) {
            regRecord->regUCSize = ALIGN_SIZE(regSize, ucgran);
          } else {
            regRecord->regUCSize = regSize;
          }
          regRecord->state |= NVLS_REG_POSSIBLE;
          memcpy(&regData[comm->localRank].reg, regRecord, sizeof(struct ncclReg));
          regData[comm->localRank].offset = userBuff - regRecord->begAddr;
        }
      }

      if ((regRecord->state & NVLS_REG_POSSIBLE) == 0) {
        regRecord->state |= NVLS_REG_NO_SUPPORT;
      }
    }
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank, regData,
                                   sizeof(struct localRegData)),
                ret, fail);

  for (int i = 0; i < comm->localRanks; ++i) {
    if ((regData[i].reg.state & NVLS_REG_POSSIBLE) == 0) {
      goto fail;
    }
    // We need to check whether the offsets are the same among ranks.
    if (i > 0 && regData[i].offset != regData[i - 1].offset) {
      goto fail;
    }
    /* get minimal reg size of nvls buffers */
    if (minSize > regData[i].reg.regUCSize) minSize = regData[i].reg.regUCSize;
  }

  /* start registration */
  mcsize = ucsize = minSize;
  mcprop.size = minSize;
  CUCHECKGOTO(cuMulticastGetGranularity(&mcgran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(mcsize, mcgran);
  mcprop.size = mcsize;

  if (comm->localRank == 0) {
    NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcprop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle),
                  ret, fail);
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
    NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);
  }

  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->nvlsResources->dev), ret, fail);
  // intra-node barrier to mitigate the possible hang in cuMulticastBindAddr during abort
  // It also ensures that if cuMulticastBindAddr fails, the cleanup code won't race with the UDS proxy
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail);
  // Coverity complains that regRecord could be NULL.  That won't in practice be the case because we've already checked
  // (regData[i].reg.state & NVLS_REG_POSSIBLE) of all local ranks, which would catch it and bail out.
  // coverity[var_deref_op]
  CUresult err;
  err = CUPFN(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)regRecord->begAddr, ucsize, 0));
  if (err != CUDA_SUCCESS) {
    // Don't print an error in case of buffers that are incompatible with MC.
    if (err != CUDA_ERROR_INVALID_VALUE) {
      const char* errStr;
      CUCALL(cuGetErrorString(err, &errStr));
      INFO(NCCL_REG, "Failed to multicast-bind user buffer: CUDA error %d '%s'", err, errStr);
    }
    goto fail;
  }
  bindComplete = true;

  // Create a VA for the NVLS
  CUCHECKGOTO(cuMemAddressReserve(&regPtr, mcsize, mcgran, 0U, 0), ret, fail);
  // Map the VA locally
  CUCHECKGOTO(cuMemMap(regPtr, mcsize, 0, mcHandle, 0), ret, fail);
  mapComplete = true;
  CUCHECKGOTO(cuMemSetAccess(regPtr, mcsize, &comm->nvlsResources->accessDesc, 1), ret, fail);

  /* get all buffer addresses */
  regRecord->caddrs[comm->localRank] = regRecord->begAddr;
  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regRecord->caddrs + comm->localRank,
                                   regRecord->caddrs, sizeof(uintptr_t)),
                ret, fail);

  regRecord->regAddr = regPtr;
  regRecord->regUCSize = ucsize;
  regRecord->regMCSize = mcsize;
  regRecord->dev = comm->nvlsResources->dev;
  regRecord->mcHandle = mcHandle;
  regRecord->state |= NVLS_REG_COMPLETE;

  *regAddr = (uintptr_t)regPtr + regData[comm->localRank].offset;
  *regUsed = 1;
exit:
  free(regData);
  return ret;
fail:
  if (regPtr) {
    if (mapComplete) CUCALL(cuMemUnmap(regPtr, mcsize));
    CUCALL(cuMemAddressFree(regPtr, mcsize));
  }
  if (mcHandle) {
    if (bindComplete) CUCALL(cuMulticastUnbind(mcHandle, comm->nvlsResources->dev, 0 /*mcOffset*/, ucsize));
    CUCALL(cuMemRelease(mcHandle));
  }
  *regUsed = 0;
  goto exit;
}

static ncclResult_t nvlsRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize,
                                       size_t recvbuffSize, struct ncclReg* sendRegRecord,
                                       struct ncclReg* recvRegRecord, int* outRegBufUsed, void** outRegBufSend,
                                       void** outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;
  int regBufUsed = 0;
  struct localRegData* regData = NULL;
  bool sendNeedReg = false, recvNeedReg = false;
  CUdeviceptr regSendPtr = 0;
  CUdeviceptr regRecvPtr = 0;

  NCCLCHECKGOTO(ncclCalloc(&regData, comm->localRanks * 2), ret, fail);

  if (sendRegRecord) {
    memcpy(&regData[comm->localRank * 2].reg, sendRegRecord, sizeof(struct ncclReg));
    regData[comm->localRank * 2].offset = (uintptr_t)sendbuff - sendRegRecord->begAddr;
  }
  if (sendbuff) {
    CUCHECKGOTO(cuPointerGetAttribute((void*)&regData[comm->localRank * 2].handleTypes,
                                      CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, (CUdeviceptr)sendbuff),
                ret, fail);
  }

  if (recvRegRecord) {
    memcpy(&regData[comm->localRank * 2 + 1].reg, recvRegRecord, sizeof(struct ncclReg));
    regData[comm->localRank * 2 + 1].offset = (uintptr_t)recvbuff - recvRegRecord->begAddr;
  }
  if (recvbuff) {
    CUCHECKGOTO(cuPointerGetAttribute((void*)&regData[comm->localRank * 2 + 1].handleTypes,
                                      CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, (CUdeviceptr)recvbuff),
                ret, fail);
  }

  NCCLCHECKGOTO(ncclShmemAllgather(comm, &comm->nvlsResources->nvlsShmem, regData + comm->localRank * 2, regData,
                                   sizeof(struct localRegData) * 2),
                ret, fail);

  /* first check whether all local ranks find their registered buffer */
  for (int i = 0; i < comm->localRanks; ++i) {
    if ((regData[i * 2].reg.state & NVLS_REG_COMPLETE) == 0 ||
        regData[comm->localRank * 2].reg.caddrs[i] != regData[i * 2].reg.begAddr) {
      sendNeedReg = true;
    }

    if ((regData[i * 2 + 1].reg.state & NVLS_REG_COMPLETE) == 0 ||
        regData[comm->localRank * 2 + 1].reg.caddrs[i] != regData[i * 2 + 1].reg.begAddr) {
      recvNeedReg = true;
    }

    if ((regData[i * 2].reg.state & NVLS_REG_NO_SUPPORT) || (regData[i * 2 + 1].reg.state & NVLS_REG_NO_SUPPORT)) {
      goto fail;
    }

    if ((sendbuff && (regData[i * 2].handleTypes & ncclCuMemHandleType) == 0) ||
        (recvbuff && (regData[i * 2 + 1].handleTypes & ncclCuMemHandleType) == 0)) {
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
    INFO(NCCL_REG,
         "rank %d reuse registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg sendbuff "
         "%p, reg recvbuff %p",
         comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);
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

  INFO(NCCL_REG,
       "rank %d successfully registered NVLS sendbuff %p, recvbuff %p, sendbuff size %ld, recvbuff size %ld, reg "
       "sendbuff %p, reg recvbuff %p",
       comm->rank, sendbuff, recvbuff, sendbuffSize, recvbuffSize, (void*)regSendPtr, (void*)regRecvPtr);

exit:
  *outRegBufSend = (void*)regSendPtr;
  *outRegBufRecv = (void*)regRecvPtr;
  *outRegBufUsed = regBufUsed;
  free(regData);
  return ncclSuccess;
fail:
  regBufUsed = 0;
  INFO(NCCL_REG, "rank %d failed to NVLS register sendbuff %p sendbuffSize %ld recvbuff %p recvbuffSize %ld",
       comm->rank, sendbuff, sendbuffSize, recvbuff, recvbuffSize);
  goto exit;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  struct ncclReg* sendRegRecord = NULL;
  struct ncclReg* recvRegRecord = NULL;
  bool sendIsValid = false;
  bool recvIsValid = false;
  void* baseSend = NULL;
  void* baseRecv = NULL;
  size_t baseSendSize = 0;
  size_t baseRecvSize = 0;

  *outRegBufUsed = 0;
  if (sendbuff) {
    NCCLCHECK(ncclRegFind(comm, sendbuff, sendbuffSize, &sendRegRecord));
    NCCLCHECK(ncclRegLocalIsValid(sendRegRecord, &sendIsValid));
    if (sendIsValid) {
      int numSegments = 0;
      NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)sendbuff, sendbuffSize, (CUdeviceptr*)&baseSend, &baseSendSize,
                                         &numSegments));
      if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    }
  } else {
    sendIsValid = true;
  }

  if (recvbuff) {
    NCCLCHECK(ncclRegFind(comm, recvbuff, recvbuffSize, &recvRegRecord));
    NCCLCHECK(ncclRegLocalIsValid(recvRegRecord, &recvIsValid));
    if (recvIsValid) {
      int numSegments = 0;
      NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)recvbuff, recvbuffSize, (CUdeviceptr*)&baseRecv, &baseRecvSize,
                                         &numSegments));
      if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    }
  } else {
    recvIsValid = true;
  }

  if (sendIsValid && recvIsValid)
    NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord,
                                 outRegBufUsed, outRegBufSend, outRegBufRecv));

exit:
  return ncclSuccess;
}

struct ncclNvlsCleanupCallback {
  struct ncclCommCallback base;
  struct ncclReg* reg;
  struct ncclComm* comm;
};

static ncclResult_t cleanupNvls(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNvlsCleanupCallback* obj = (struct ncclNvlsCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded) {
  struct ncclNvlsCleanupCallback* sendRecord = NULL;
  struct ncclNvlsCleanupCallback* recvRecord = NULL;
  void* baseSend = NULL;
  void* baseRecv = NULL;
  size_t baseSendSize = 0;
  size_t baseRecvSize = 0;
  struct ncclReg* sendRegRecord = NULL;
  struct ncclReg* recvRegRecord = NULL;

  *outRegBufUsed = 0;
  if (sendbuff) {
    int numSegments = 0;
    NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)sendbuff, sendbuffSize, (CUdeviceptr*)&baseSend, &baseSendSize,
                                       &numSegments));
    if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    NCCLCHECK(ncclCommGraphRegister(comm, baseSend, baseSendSize, (void**)&sendRegRecord));
  }

  if (recvbuff) {
    int numSegments = 0;
    NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)recvbuff, recvbuffSize, (CUdeviceptr*)&baseRecv, &baseRecvSize,
                                       &numSegments));
    if (numSegments > 1 && !ncclParamMultiSegmentRegister()) goto exit;
    NCCLCHECK(ncclCommGraphRegister(comm, baseRecv, baseRecvSize, (void**)&recvRegRecord));
  }

  NCCLCHECK(nvlsRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord,
                               outRegBufUsed, outRegBufSend, outRegBufRecv));

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

exit:
  return ncclSuccess;
}

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  int factor;
  ncclResult_t ret = ncclSuccess;
  if (comm->nNodes == 1) {
    if (info->func == ncclFuncReduceScatter) {
      factor = (comm->compCap >= 100 ? 6 : 5) * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllGather) {
      factor = 4 * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllReduce) {
      if (comm->compCap >= 100) {
        factor = 8 * 8;
      } else {
        factor = 4 * 8;
      }
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else {
      goto fail;
    }
  } else {
    // Further tweaks for Blackwell with NVLS registered buffers
    if (info->func == ncclFuncReduceScatter) {
      factor = (comm->bandwidths[ncclFuncReduceScatter][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] > 400 ? 7 : 6) * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllGather) {
      factor = 6 * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (info->func == ncclFuncAllReduce) {
      if (comm->compCap >= 100 && comm->minNetBw >= 96.0f) {
        factor = 10 * 8;
      } else if (comm->compCap >= 100) {
        factor = 7 * 8;
      } else {
        factor = 6 * 8;
      }
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else {
      goto fail;
    }
  }

exit:
  return ret;
fail:
  ret = ncclInvalidArgument;
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

ncclResult_t ncclNvlsSuspendCheck(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSuspend(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsResume(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  *outRegBufUsed = false;
  return ncclSuccess;
}

ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle* mcHandler, CUdeviceptr ptr,
                                 int dev, size_t ucsize, size_t mcsize) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricInit(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricMap(struct ncclComm* comm, size_t offset, size_t ucsize, void* ucaddr) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricFree(struct ncclComm* comm, size_t ucsize, void* ucaddr) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsSymmetricFinalize(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  *recChannels = 0;
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
