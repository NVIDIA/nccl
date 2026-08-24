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

// Release a consumer's UC physical memory and its mapping (inverse of nvlsAllocBindUc's alloc).
static ncclResult_t nvlsFreeUc(struct ncclComm* comm, struct ncclNvlsUcSegment* uc) {
  if (uc->ptr) {
    INFO(NCCL_NVLS, "NVLS release UC handle %llx ptr %p size %zu", uc->handle, uc->ptr, uc->size);
    // Unmaps/frees/releases and untracks the persistent ncclMemTrack entry from nvlsAllocBindUc.
    NCCLCHECK(ncclCuMemFree(uc->ptr, comm->memManager));
    memset(uc, 0, sizeof(*uc));
  }
  return ncclSuccess;
}

#include "bootstrap.h"
#include "channel.h"

#define NVLS_MEM_ALIGN_SIZE (1 << 21)

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

static ncclResult_t ncclNvlsChannels(struct ncclComm* comm, int* nChannels, int* chunkSize, int* treeMaxChunkSize) {
  int channels = 0;

  if (comm->minCompCap >= 100 && comm->nNodes > 1) {
    NCCLCHECK(ncclNvlsTreeSm100Tuning(comm, &channels, chunkSize, treeMaxChunkSize));
  }

  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT) {
    channels = comm->config.nvlsCTAs;
  } else if (channels == 0 && comm->compCap >= 100) {
    // Use a reduced number of channels for single node/MNNVL domain on Blackwell and above.
    // comm->nNodes is not yet initialized at this point so we need to use local information.
    bool multiNode = false;
    if (comm->MNNVL) {
      multiNode = (comm->clique.size < comm->nRanks);
    } else {
      int i;
      for (i = 1; i < comm->nRanks; i++) {
        if (comm->peerInfo[i].hostHash != comm->peerInfo[0].hostHash) break;
      }
      multiNode = (i < comm->nRanks);
    }
    if (multiNode) {
      channels = RUBIN_AND_LATER(comm->compCap) ? /*RUBIN=*/64 : /*SM100=*/32;
    } else {
      channels = RUBIN_AND_LATER(comm->compCap) ? /*RUBIN=*/48 : /*SM100=*/24;
    }
  } else if (channels == 0) {
    channels = /*SM90=*/16;
  }

  *nChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, channels));
  return ncclSuccess;
}

ncclResult_t ncclNvlsTuning(struct ncclComm* comm) {
  int chunkSize = 0;
  int treeMaxChunkSize = 0;
  const char* chunkSizeEnv = ncclGetEnv("NCCL_NVLS_CHUNKSIZE");
  bool userSetChunkSize = (chunkSizeEnv != NULL && strlen(chunkSizeEnv) > 0);
  int nChannels;
  NCCLCHECK(ncclNvlsChannels(comm, &nChannels, &chunkSize, &treeMaxChunkSize));

  // User overrides take priority over tuning
  // If user has set chunk size or chunkSize is not set, use the chunk size as determined by ncclParamNvlsChunkSize()
  if (userSetChunkSize || chunkSize == 0) chunkSize = ncclParamNvlsChunkSize();

  // Determine final treeMaxChunkSize: env var > tuning > fallback
  int envTreeMaxChunkSize = (int)ncclParamNvlsTreeMaxChunkSize();
  if (envTreeMaxChunkSize == -2 && treeMaxChunkSize == 0) {
    treeMaxChunkSize = (comm->nNodes >= 4) ? 65536 : chunkSize;
  } else if (envTreeMaxChunkSize != -2) {
    treeMaxChunkSize = envTreeMaxChunkSize;
  }

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

  INFO(NCCL_INIT, "NVLS multicast support is %savailable on dev %d, transport %s, symmetric multimem %s",
       comm->nvlsSupport ? "" : "not ", dev, ncclNvlsTransportEnabled(comm) ? "enabled" : "disabled",
       ncclNvlsSymmetricMultimemEnabled(comm) ? "enabled" : "disabled");
  return ncclSuccess;
}

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && ncclNvlsTransportEnabled(comm) && comm->nNodes > 1) {
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

// Allocate UC backing for a consumer and bind it into its resolved MC-group slice.
static ncclResult_t nvlsAllocBindUc(struct ncclComm* comm, const struct ncclMcPartition* partition, size_t size,
                                    struct ncclNvlsUcSegment* outUc) {
  CUmemAllocationProp ucprop;
  ncclResult_t ret = ncclSuccess;
  size_t ucsize = size;
  size_t ucgran;
  void* ucptr = NULL;
  CUmemGenericAllocationHandle ucHandle = 0;

  memset(&ucprop, 0, sizeof(CUmemAllocationProp));
  ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ucprop.location.id = comm->cudaDev;
  ucprop.requestedHandleTypes = ncclCuMemHandleType;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail);
  ALIGN_SIZE(ucsize, ucgran);
  // Map a VA for UC memory with MC alignment and size
  CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)&ucptr, ucsize, ucgran, 0U, 0), ret, fail);

  // Alloc local physical mem for this NVLS group
  CUCHECKGOTO(cuMemCreate(&ucHandle, ucsize, &ucprop, 0), ret, fail1);
  CUCHECKGOTO(cuMemMap((CUdeviceptr)ucptr, ucsize, 0, ucHandle, 0), ret, fail2);
  CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)ucptr, ucsize, &comm->nvlsResources->accessDesc, 1), ret, fail3);
  CUDACHECKGOTO(cudaMemset(ucptr, 0, ucsize), ret, fail3);
  // Track NVLS buffer as persistent memory
  NCCLCHECKGOTO(ncclMemTrack(comm->memManager, ucptr, ucsize, ucHandle, ncclCuMemHandleType, ncclMemPersist), ret,
                fail3);

  // intra-node barrier to mitigate the possible hang in cuMulticastBindMem during abort
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail3);
  NCCLCHECKGOTO(ncclMcPartitionBindMem(partition, 0 /*offsetInPartition*/, ucHandle, 0 /*memOffset*/, ucsize), ret,
                fail3);

  INFO(NCCL_NVLS, "NVLS rank %d (dev %d) bound UC ptr %p ucsize %zu into MC slice offset %zu (inputsize %zu)",
       comm->rank, comm->cudaDev, ucptr, ucsize, partition->offset, size);

  // Publish the UC owner only on success (callers test outUc->ptr).
  outUc->handle = ucHandle;
  outUc->ptr = ucptr;
  outUc->size = ucsize;
exit:
  return ret;
fail3:
  CUCHECK(cuMemUnmap((CUdeviceptr)ucptr, ucsize));
fail2:
  CUCHECK(cuMemRelease(ucHandle));
fail1:
  CUCHECK(cuMemAddressFree((CUdeviceptr)ucptr, ucsize));
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
  cudaStream_t deviceStream, hostStream;

  if (!ncclNvlsTransportEnabled(comm) || comm->nvlsResources->inited) return ncclSuccess;
  // initialize after checking ncclNvlsTransportEnabled(comm)
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

  // A prior setup attempt may have bound the data slice and then failed (inited
  // still false); the slice offset is fixed, so reuse that binding.
  if (resources->dataUc.ptr == NULL) {
    NCCLCHECKGOTO(nvlsAllocBindUc(comm, &resources->dataPartition, nvlsTotalSize, &resources->dataUc), res, fail);
  }

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
      peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = (char*)resources->dataUc.ptr + (h * 2 * nChannels + c) * buffSize;
      peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] =
        (char*)resources->dataPartition.ptr + (h * 2 * nChannels + c) * buffSize;

      // Broadcast MC -> UC
      peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] =
        (char*)resources->dataUc.ptr + ((h * 2 + 1) * nChannels + c) * buffSize;
      peer->send[0].conn.buffs[NCCL_PROTO_SIMPLE] =
        (char*)resources->dataPartition.ptr + ((h * 2 + 1) * nChannels + c) * buffSize;

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
  bool nvlsShare =
    parent && ncclNvlsTransportEnabled(parent) && parent->shareResources && parent->localRanks == comm->localRanks;

  if (!ncclNvlsTransportEnabled(comm) || comm->nvlsChannels == 0) return ncclSuccess;

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

    if (parent != nullptr && ncclNvlsTransportEnabled(parent) && parent->shareResources) {
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

    // Build the single shared MC group for this NVLS domain. The data slice is
    // reserved here but bound later by ncclNvlsBufferSetup.
    {
      size_t buffSize = nvlsStepSize * NCCL_STEPS;
      size_t dataSize = nChannels * 2 * buffSize * nHeads;
      size_t ubSize = ncclNvlsUbSize(comm);
      struct ncclMcRequest requests[3] = {{creditSize, 0}, {dataSize, 0}, {ubSize, 0}};
      struct ncclMcPartition partitions[3];
      NCCLCHECKGOTO(ncclMcGroupBuildPartitions(comm, requests, 3, &resources->mcGroup, partitions), res, fail);
      resources->creditPartition = partitions[0];
      resources->dataPartition = partitions[1];
      if (ubSize) {
        resources->ubPartition = partitions[2];
        NCCLCHECKGOTO(ncclMcArenaInit(comm, &resources->ubArena, &resources->ubPartition), res, fail);
        resources->ubEnabled = true;
      }
      NCCLCHECKGOTO(nvlsAllocBindUc(comm, &resources->creditPartition, creditSize, &resources->creditUc), res, fail);
    }

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
        mem = (char*)resources->creditUc.ptr + (h * 2 * nChannels + c) * memSize;
        peer->send[1].transportComm = &nvlsTransport.send;
        peer->send[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->send[1].conn.head = (uint64_t*)mem;
        peer->send[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->send[1].conn.stepSize = nvlsStepSize;
        mem = (char*)resources->creditPartition.ptr + (h * 2 * nChannels + c) * memSize;
        peer->recv[0].transportComm = &nvlsTransport.recv;
        peer->recv[0].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[0].conn.head = (uint64_t*)mem;
        peer->recv[0].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[0].conn.stepSize = nvlsStepSize;
        peer->recv[0].conn.flags |= NCCL_NVLS_MIN_POLL;

        // Broadcast MC -> UC
        mem = (char*)resources->creditUc.ptr + ((h * 2 + 1) * nChannels + c) * memSize;
        peer->recv[1].transportComm = &nvlsTransport.recv;
        peer->recv[1].conn.buffs[NCCL_PROTO_SIMPLE] = NULL;
        peer->recv[1].conn.head = (uint64_t*)mem;
        peer->recv[1].conn.tail = (uint64_t*)(mem + memSize / 2);
        peer->recv[1].conn.stepSize = nvlsStepSize;
        mem = (char*)resources->creditPartition.ptr + ((h * 2 + 1) * nChannels + c) * memSize;
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
    typeSize = DIVUP(ncclNvlsUbScratchTypeSize(), CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

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

ncclResult_t ncclNvlsFree(struct ncclComm* comm) {
  struct ncclNvlsSharedRes* resources = (struct ncclNvlsSharedRes*)comm->nvlsResources;
  ncclResult_t ret = ncclSuccess;
  if (resources == NULL) return ncclSuccess;

  if (ncclAtomicRefCountDecrement(&resources->refCount) == 0) {
    if (!comm->MNNVL && resources->nvlsShmemHandle) NCCLCHECKIGNORE(ncclShmClose(resources->nvlsShmemHandle), ret);

    if (resources->creditUc.ptr) {
      NCCLCHECKIGNORE(
        ncclMcPartitionUnbind(&resources->creditPartition, 0 /*offsetInPartition*/, resources->creditUc.size), ret);
      NCCLCHECKIGNORE(nvlsFreeUc(comm, &resources->creditUc), ret);
    }
    if (resources->dataUc.ptr) {
      NCCLCHECKIGNORE(ncclMcPartitionUnbind(&resources->dataPartition, 0 /*offsetInPartition*/, resources->dataUc.size),
                      ret);
      NCCLCHECKIGNORE(nvlsFreeUc(comm, &resources->dataUc), ret);
    }
    if (resources->ubEnabled) ncclMcArenaDestroy(&resources->ubArena);
    NCCLCHECKIGNORE(ncclMcGroupDestroy(&resources->mcGroup), ret);

    free(resources);
    comm->nvlsResources = NULL;
  }
  return ret;
}

// Find the local registration record for one buffer, or leave *outReg NULL for a buffer
// that cannot take this path; a NULL record travels through the gathers as an ineligible
// payload. Even on error the caller must still call ncclNvlsUbRegister, since peers
// cannot see this rank's failure.
static ncclResult_t nvlsFindLocalReg(struct ncclComm* comm, const void* buff, size_t size, struct ncclReg** outReg) {
  struct ncclReg* reg = NULL;
  bool isValid = false;
  void* base = NULL;
  size_t baseSize = 0;
  int numSegments = 0;

  NCCLCHECK(ncclRegFind(comm, buff, size, &reg));
  if (reg == NULL) return ncclSuccess;
  NCCLCHECK(ncclRegLocalIsValid(reg, &isValid));
  if (!isValid || !ncclNvlsUbEligible(comm, buff, reg)) return ncclSuccess;
  NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)reg->begAddr, reg->endAddr - reg->begAddr, (CUdeviceptr*)&base,
                                     &baseSize, &numSegments));
  if (numSegments > 1 && !ncclParamMultiSegmentRegister()) return ncclSuccess;
  *outReg = reg;
  return ncclSuccess;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  ncclResult_t ret = ncclSuccess;

  *outRegBufUsed = 0;
  *outRegBufSend = NULL;
  *outRegBufRecv = NULL;

  // Uniform across local ranks, so no risk of deadlock.
  if (!ncclNvlsUbEnabled(comm)) return ret;

  struct ncclReg* sendRegRecord = NULL;
  struct ncclReg* recvRegRecord = NULL;

  if (sendbuff) {
    NCCLCHECKIGNORE(nvlsFindLocalReg(comm, sendbuff, sendbuffSize, &sendRegRecord), ret);
  }

  if (recvbuff) {
    NCCLCHECKIGNORE(nvlsFindLocalReg(comm, recvbuff, recvbuffSize, &recvRegRecord), ret);
  }

  NCCLCHECK(ncclNvlsUbRegister(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord,
                               outRegBufUsed, outRegBufSend, outRegBufRecv));
  return ret;
}

struct ncclNvlsCleanupCallback {
  struct ncclCommCallback base;
  struct ncclReg* reg;
  struct ncclComm* comm;
};

static ncclResult_t cleanupNvls(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclNvlsCleanupCallback* obj = (struct ncclNvlsCleanupCallback*)cb;
  ncclResult_t ret = ncclCommGraphDeregister(obj->comm, obj->reg);
  free(obj);
  return ret;
}

// Take a graph registration on one buffer, with the same failure contract as
// nvlsFindLocalReg. The reference is taken only on success, so the caller releases
// exactly the records it was handed.
static ncclResult_t nvlsGraphReg(struct ncclComm* comm, const void* buff, size_t size, struct ncclReg** outReg,
                                 struct ncclNvlsCleanupCallback** outRecord) {
  ncclResult_t ret = ncclSuccess;

  struct ncclReg* reg = NULL;
  struct ncclNvlsCleanupCallback* record = NULL;

  void* base = NULL;
  size_t baseSize = 0;
  int numSegments = 0;
  NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)buff, size, (CUdeviceptr*)&base, &baseSize, &numSegments));
  if (numSegments > 1 && !ncclParamMultiSegmentRegister()) return ret;

  NCCLCHECKGOTO(ncclCommGraphRegister(comm, base, baseSize, (void**)&reg), ret, done);
  if (reg == NULL) goto done;
  if (!ncclNvlsUbEligible(comm, buff, reg)) goto fail;

  NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);
  record->base.fn = cleanupNvls;
  record->reg = reg;
  record->comm = comm;

  *outReg = reg;
  *outRecord = record;

done:
  return ret;

fail:
  NCCLCHECKIGNORE(ncclCommGraphDeregister(comm, reg), ret);
  goto done;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded) {
  ncclResult_t ret = ncclSuccess;
  *outRegBufUsed = 0;
  *outRegBufSend = NULL;
  *outRegBufRecv = NULL;

  // Uniform across local ranks, so no risk of deadlock.
  if (!ncclNvlsUbEnabled(comm)) return ret;

  struct ncclNvlsCleanupCallback* sendRecord = NULL;
  struct ncclNvlsCleanupCallback* recvRecord = NULL;
  struct ncclReg* sendRegRecord = NULL;
  struct ncclReg* recvRegRecord = NULL;

  if (sendbuff) {
    NCCLCHECKIGNORE(nvlsGraphReg(comm, sendbuff, sendbuffSize, &sendRegRecord, &sendRecord), ret);
  }

  if (recvbuff) {
    NCCLCHECKIGNORE(nvlsGraphReg(comm, recvbuff, recvbuffSize, &recvRegRecord, &recvRecord), ret);
  }

  NCCLCHECKIGNORE(ncclNvlsUbRegister(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, sendRegRecord, recvRegRecord,
                                     outRegBufUsed, outRegBufSend, outRegBufRecv),
                  ret);

  if (ret == ncclSuccess && *outRegBufUsed) {
    if (sendRegRecord) {
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)sendRecord);
      *nCleanupQueueEltsAdded += 1;
    }

    if (recvRegRecord) {
      ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)recvRecord);
      *nCleanupQueueEltsAdded += 1;
    }
  } else {
    if (sendbuff) NCCLCHECKIGNORE(ncclCommGraphDeregister(comm, sendRegRecord), ret);
    if (recvbuff) NCCLCHECKIGNORE(ncclCommGraphDeregister(comm, recvRegRecord), ret);
    free(sendRecord);
    free(recvRecord);
  }

  return ret;
}

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, ncclFunc_t func, int* recChannels) {
  int factor;
  ncclResult_t ret = ncclSuccess;
  if (comm->nNodes == 1) {
    if (func == ncclFuncReduceScatter) {
      factor = (comm->compCap >= 100 ? 6 : 5) * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (func == ncclFuncAllGather) {
      factor = 4 * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (func == ncclFuncAllReduce) {
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
    if (func == ncclFuncReduceScatter) {
      factor =
        (comm->tuningContext.generalBandwidths[ncclFuncReduceScatter][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] > 400 ? 7 :
                                                                                                                 6) *
        8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (func == ncclFuncAllGather) {
      factor = 6 * 8;
      *recChannels =
        std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, DIVUP(factor, comm->nvlsResources->nHeads)));
    } else if (func == ncclFuncAllReduce) {
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

ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) {
  return ncclSuccess;
}

ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueEltsAdded) {
  *outRegBufUsed = 0;
  *outRegBufSend = NULL;
  *outRegBufRecv = NULL;
  return ncclSuccess;
}

ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  *outRegBufUsed = 0;
  *outRegBufSend = NULL;
  *outRegBufRecv = NULL;
  return ncclSuccess;
}

// The real implementation lives in nvls_ub.cc, which is empty before CUDA 12.1.
ncclResult_t ncclNvlsUbDeregister(struct ncclComm* comm, struct ncclReg* reg) {
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

ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, ncclFunc_t func, int* recChannels) {
  *recChannels = 0;
  return ncclSuccess;
}

#endif /* CUDA_VERSION >= 12010 */
