/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime_internal.h"
#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "param.h"
#include <stdlib.h>
#include <string.h>

extern int64_t ncclParamElasticBufferRegister();

ncclResult_t ncclDevrPopulateSegmentSizes(struct ncclDevrMemory* mem, int numSegments) {
  ncclResult_t ret = ncclSuccess;

  // If our caller does not have a VA (for instance, in ncclDevrCommCreateInternal),
  // there's only one segment with size = mem->size.
  if (mem->primaryAddr == nullptr) {
    assert(numSegments == 1);
    mem->segmentSizes[0] = mem->size;
    return ret;
  }

  size_t offset = 0;
  for (int segment = 0; segment < numSegments; segment++) {
    size_t baseSendSize = 0;
    CUdeviceptr segmentStart = reinterpret_cast<CUdeviceptr>(reinterpret_cast<char*>(mem->primaryAddr) + offset);
    CUCHECKGOTO(cuMemGetAddressRange(NULL, &baseSendSize, segmentStart), ret, exit);
    mem->segmentSizes[segment] = baseSendSize;
    offset += baseSendSize;
  }
exit:
  return ret;
}

ncclResult_t ncclDevrCheckRegistrationSupport(void* userPtr, size_t userSize, struct ncclComm* comm,
                                              bool hasSysmemSegment) {
  ncclResult_t ret = ncclSuccess;
  if (hasSysmemSegment) {
    if (!ncclParamElasticBufferRegister()) {
      WARN("VA represented by {userPtr = %p, size = %zu} contains CPU-backed physical segments, but "
           "NCCL_ELASTIC_BUFFER_REGISTER is set to 0. Please set NCCL_ELASTIC_BUFFER_REGISTER=1 and retry window "
           "registration",
           userPtr, userSize);
      ret = ncclInvalidArgument;
      goto exit;
    }
#if CUDART_VERSION >= 12080
    else if (comm->MNNVL) {
      int multiNodeLsaSupported = 0;
      CUCHECKGOTO(cuDeviceGetAttribute(&multiNodeLsaSupported, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED,
                                       comm->cudaDev),
                  ret, exit);
      if (!multiNodeLsaSupported) {
        WARN("VA represented by {userPtr = %p, size = %zu} contains CPU-backed physical segments, but the LSA team "
             "does not support multi-node IPC on CPU-backed buffers. Please retry by setting NCCL_MNNVL_ENABLE=0",
             userPtr, userSize);
        ret = ncclInvalidArgument;
        goto exit;
      }
    }
#endif
  }
exit:
  return ret;
}

ncclResult_t ncclDevrValidateHandleLocationType(CUmemGenericAllocationHandle memHandle, int segment) {
  ncclResult_t ret = ncclSuccess;
  CUmemAllocationProp prop;
  CUCHECKGOTO(cuMemGetAllocationPropertiesFromHandle(&prop, memHandle), ret, exit);
  if (prop.location.type != CU_MEM_LOCATION_TYPE_HOST_NUMA && prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    WARN("Segment %d has unsupported location type %d. Symmetric memory currently only supports "
         "CU_MEM_LOCATION_TYPE_HOST_NUMA and CU_MEM_LOCATION_TYPE_DEVICE.",
         segment, (int)prop.location.type);
    ret = ncclInvalidArgument;
    goto exit;
  }
exit:
  return ret;
}

ncclResult_t ncclDevrVerifySegmentLayouts(struct ncclDevrMemory* mem, struct ncclComm* comm) {
  if (!mem->globalHasSysmemSegment) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  size_t* globalPaddedSegmentSizes = nullptr;

  if (mem->maxGlobalNumSegments != mem->numSegments) {
    WARN("Elastic GIN: rank %d has %d segments but another rank has %d segments; all ranks must have identical segment "
         "configurations",
         comm->rank, mem->numSegments, mem->maxGlobalNumSegments);
    ret = ncclInvalidUsage;
    goto exit;
  }

  // Gather segment sizes from all ranks to verify they are identical.
  NCCLCHECKGOTO(ncclCalloc(&globalPaddedSegmentSizes, (size_t)mem->maxGlobalNumSegments * comm->nRanks), ret, exit);
  memcpy(&globalPaddedSegmentSizes[(size_t)comm->rank * mem->maxGlobalNumSegments], mem->segmentSizes,
         sizeof(size_t) * mem->numSegments);

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, globalPaddedSegmentSizes,
                                   sizeof(size_t) * mem->maxGlobalNumSegments),
                ret, exit);

  for (int rank = 0; rank < comm->nRanks; rank++) {
    for (int seg = 0; seg < mem->numSegments; seg++) {
      if (globalPaddedSegmentSizes[(size_t)rank * mem->maxGlobalNumSegments + seg] != mem->segmentSizes[seg]) {
        WARN("Elastic GIN: rank %d segment %d size %zu differs from rank %d segment %d size %zu; all ranks must have "
             "identical segment configurations",
             rank, seg, globalPaddedSegmentSizes[(size_t)rank * mem->maxGlobalNumSegments + seg], comm->rank, seg,
             mem->segmentSizes[seg]);
        ret = ncclInvalidUsage;
        goto exit;
      }
    }
  }
exit:
  free(globalPaddedSegmentSizes);
  return ret;
}

ncclResult_t ncclDevrBuildGinSegmentInfos(struct ncclDevrMemory* mem) {
  ncclResult_t ret = ncclSuccess;
  bool deviceOnlySegments = !mem->globalHasSysmemSegment;

  mem->numGinSegments = deviceOnlySegments ? 1 : mem->numSegments;
  NCCLCHECKGOTO(ncclCalloc(&mem->ginSegmentInfos, mem->numGinSegments), ret, exit);

  if (deviceOnlySegments) {
    mem->ginSegmentInfos[0].segmentSize = mem->size;
    mem->ginSegmentInfos[0].memType = CU_MEM_LOCATION_TYPE_DEVICE;
  } else {
    for (int segment = 0; segment < mem->numGinSegments; segment++) {
      CUmemAllocationProp prop;
      mem->ginSegmentInfos[segment].segmentSize = mem->segmentSizes[segment];
      CUCHECKGOTO(cuMemGetAllocationPropertiesFromHandle(&prop, mem->memHandles[segment]), ret, exit);
      mem->ginSegmentInfos[segment].memType = prop.location.type;
    }
  }

exit:
  return ret;
}

// Segment windows need their own shadow-pool allocation because they're variable in size.
ncclResult_t ncclDevrAllocAndPopulateSegmentWindows(struct ncclDevrState* devr, struct ncclDevrMemory* mem,
                                                    cudaStream_t stream,
                                                    struct ncclSegmentWindow** outSegmentWindowsDev) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSegmentWindow* segmentWindowsDev = nullptr;
  struct ncclSegmentWindow* segmentWindowsHost = nullptr;

  NCCLCHECKGOTO(ncclShadowPoolAlloc(&devr->shadows, sizeof(struct ncclSegmentWindow) * mem->numGinSegments,
                                    (void**)&segmentWindowsDev, (void**)&segmentWindowsHost, stream),
                ret, fail);

  if (devr->ginEnabled) {
    for (int segment = 0; segment < mem->numGinSegments; segment++) {
      segmentWindowsHost[segment].memType = mem->ginSegmentInfos[segment].memType;
      segmentWindowsHost[segment].segmentSize = mem->ginSegmentInfos[segment].segmentSize;
      for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++) {
        segmentWindowsHost[segment].ginWins[i] = mem->ginSegmentInfos[segment].ginDevWins[i];
      }
    }
    CUDACHECKGOTO(cudaMemcpyAsync(segmentWindowsDev, segmentWindowsHost,
                                  sizeof(struct ncclSegmentWindow) * mem->numGinSegments, cudaMemcpyHostToDevice,
                                  stream),
                  ret, fail);
  }

  *outSegmentWindowsDev = segmentWindowsDev;

exit:
  return ret;
fail:
  if (segmentWindowsDev != nullptr) ncclShadowPoolFree(&devr->shadows, segmentWindowsDev, stream);
  goto exit;
}

ncclResult_t ncclDevrReplaceSegmentWindowsIfNeeded(struct ncclDevrState* devr, struct ncclDevrMemory* mem,
                                                   struct ncclWindow_vidmem* winHost, cudaStream_t stream) {
  struct ncclSegmentWindow* segmentWindowsDev = nullptr;
  // When a window is created, numGinSegments is always set to `1`.  As we now
  // know that there are multiple segments, we need to reallocate ginMultiSegmentWins.
  if (mem->numGinSegments > 1) {
    NCCLCHECK(ncclShadowPoolFree(&devr->shadows, winHost->ginMultiSegmentWins, stream));
    NCCLCHECK(ncclDevrAllocAndPopulateSegmentWindows(devr, mem, stream, &segmentWindowsDev));
    winHost->ginMultiSegmentWins = segmentWindowsDev;
  }
  return ncclSuccess;
}
