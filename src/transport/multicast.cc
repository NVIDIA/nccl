/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Shared NVSwitch multicast (MC) group: one MC object per NVLS domain whose full
// VA is mapped once and bump-allocated to consumers (credit, data, ...) as
// immutable slices, replacing one MC object (and slot) per consumer.

#include "multicast.h"
#include "comm.h"
#include "transport.h"
#include "alloc.h"
#include "bitops.h"

#if CUDART_VERSION >= 12010

ncclResult_t ncclMcCreate(struct ncclComm* comm, CUmulticastObjectProp* prop, int rank, unsigned int nranks,
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

ncclResult_t ncclMcImport(struct ncclComm* comm, char* shareableHandle, int rank,
                          CUmemGenericAllocationHandle* mcHandle) {
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  ncclIpcFd fd = NCCL_INVALID_IPC_FD;
  ncclResult_t ret = ncclSuccess;
  INFO(NCCL_NVLS, "NVLS importing shareableHandle %p from rank %d", shareableHandle, rank);

  // Import and map the remote memory descriptor to the local GPU
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    // cuMem UDS support
    TRACE(NCCL_NVLS, "NVLS rank %d Importing shareable handle %p from rank %d", comm->localRank, shareableHandle, rank);
    TRACE(NCCL_NVLS, "NVLS rank %d request conversion of handle 0x%lx from rank %d", comm->localRank,
          *(uint64_t*)shareableHandle, rank);
    NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, rank, shareableHandle, &fd), ret, fail);
    TRACE(NCCL_NVLS, "NVLS rank %d received converted fd %lld from rank %d", comm->localRank, (long long)fd, rank);
    CUCHECKGOTO(cuMemImportFromShareableHandle(mcHandle, (void*)(uintptr_t)fd, type), ret, fail);
    SYSCHECK(ncclIpcFdClose(fd), "close");
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
  if (fd != NCCL_INVALID_IPC_FD) ncclIpcFdClose(fd);
  goto exit;
}

struct ncclMcGroup {
  CUmemGenericAllocationHandle handle;  // the MC object
  char* base;                          // mapped MC VA base
  size_t capacity;                      // total mapped VA size
  int dev;                           // local device, for unbind
};

ncclResult_t ncclMcGroupBuildPartitions(struct ncclComm* comm, const struct ncclMcRequest* requests, int nRequests,
                                        struct ncclMcGroup** outGroup, struct ncclMcPartition* outPartitions) {
  ncclResult_t ret = ncclSuccess;
  CUmulticastObjectProp mcprop = {};
  CUmemAccessDesc desc = {};
  char shareableHandle[NVLS_HANDLE_SIZE] =
    {}; // zero-init: non-FABRIC create fills only a prefix, but Build broadcasts all bytes
  CUmemGenericAllocationHandle mcHandle = 0;
  CUdeviceptr base = 0;
  struct ncclMcGroup* group = NULL;
  size_t recGran, minGran, capacity = 0;
  int mcCreated = 0, mapped = 0;

  *outGroup = NULL;

  mcprop.numDevices = comm->localRanks;
  mcprop.handleTypes = ncclCuMemHandleType;
  mcprop.flags = 0;
  mcprop.size = 0;
  for (int i = 0; i < nRequests; i++) mcprop.size += requests[i].size;
  CUCHECKGOTO(cuMulticastGetGranularity(&recGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED), ret, fail);
  CUCHECKGOTO(cuMulticastGetGranularity(&minGran, &mcprop, CU_MULTICAST_GRANULARITY_MINIMUM), ret, fail);

  // Bump-allocate an immutable slice per request. Offsets and sizes are rounded
  // to the recommended granularity (a multiple of the MC minimum) so every slice
  // boundary is a valid bind offset.
  for (int i = 0; i < nRequests; i++) {
    outPartitions[i] = {};
    if (requests[i].size == 0) continue;
    size_t align = requests[i].alignment > recGran ? requests[i].alignment : recGran;
    ALIGN_SIZE(capacity, align);
    size_t slice = requests[i].size;
    ALIGN_SIZE(slice, recGran);
    outPartitions[i].offset = capacity;
    outPartitions[i].size = slice;
    capacity += slice;
  }
  ALIGN_SIZE(capacity, recGran);
  if (capacity == 0) {
    WARN("NVLS MC group build requires at least one non-zero request");
    return ncclInternalError;
  }
  mcprop.size = capacity;

  // Own the MC handle the instant it exists so every later failure path releases
  // the scarce MC slot.
  if (comm->localRank == 0) {
    NCCLCHECKGOTO(ncclMcCreate(comm, &mcprop, comm->localRank, comm->localRanks, &mcHandle, shareableHandle), ret,
                  fail);
    mcCreated = 1;
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
  } else {
    NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                              0, shareableHandle, NVLS_HANDLE_SIZE),
                  ret, fail);
    NCCLCHECKGOTO(ncclMcImport(comm, shareableHandle, comm->localRankToRank[0], &mcHandle), ret, fail);
    mcCreated = 1;
  }
  CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->cudaDev), ret, fail);

  // cuMemMap of an MC object blocks until every device has been added. This
  // abort-aware barrier makes a peer failing before cuMulticastAddDevice trip the
  // abort flag here instead of stranding survivors in the blocking cuMemMap.
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail);

  // Reserve and map the whole MC VA once; each consumer slice is a view into it.
  CUCHECKGOTO(cuMemAddressReserve(&base, capacity, recGran, 0U, 0), ret, fail);
  CUCHECKGOTO(cuMemMap(base, capacity, 0, mcHandle, 0), ret, fail);
  mapped = 1;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = comm->cudaDev;
  CUCHECKGOTO(cuMemSetAccess(base, capacity, &desc, 1), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&group, 1), ret, fail);
  group->handle = mcHandle;
  group->base = (char*)base;
  group->capacity = capacity;
  group->dev = comm->cudaDev;
  // A partition is self-sufficient for binds: it carries the group's handle, device and
  // bind granularity alongside its own extent.
  for (int i = 0; i < nRequests; i++) {
    if (outPartitions[i].size == 0) continue;
    outPartitions[i].ptr = group->base + outPartitions[i].offset;
    outPartitions[i].mcHandle = mcHandle;
    outPartitions[i].minGranularity = minGran;
    outPartitions[i].dev = comm->cudaDev;
  }

  INFO(NCCL_NVLS, "NVLS rank %d MC group %llx capacity %zu over %d consumers", comm->localRank, mcHandle, capacity,
       nRequests);
  for (int i = 0; i < nRequests; i++)
    TRACE(NCCL_NVLS, "NVLS MC group %llx slice %d offset %zu size %zu ptr %p", mcHandle, i, outPartitions[i].offset,
          outPartitions[i].size, outPartitions[i].ptr);
  *outGroup = group;
  return ncclSuccess;
fail:
  // Best-effort (CUCALL) so a failing cleanup op cannot skip releasing the MC handle.
  if (mapped) CUCALL(cuMemUnmap(base, capacity));
  if (base) CUCALL(cuMemAddressFree(base, capacity));
  if (mcCreated) CUCALL(cuMemRelease(mcHandle));
  return ret;
}

ncclResult_t ncclMcGroupDestroy(struct ncclMcGroup** groupPtr) {
  struct ncclMcGroup* group = *groupPtr;
  if (group == NULL) return ncclSuccess;
  INFO(NCCL_NVLS, "NVLS destroy MC group %llx capacity %zu dev %d", group->handle, group->capacity, group->dev);
  // Releasing the MC handle drops any bindings still present; UC memory is freed by the caller.
  CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)group->base, group->capacity));
  CUCHECKIGNORE(cuMemAddressFree((CUdeviceptr)group->base, group->capacity));
  CUCHECKIGNORE(cuMemRelease(group->handle));
  free(group);
  *groupPtr = NULL;
  return ncclSuccess;
}

ncclResult_t ncclMcPartitionBindMem(const struct ncclMcPartition* partition, size_t offsetInPartition,
                                    CUmemGenericAllocationHandle mem, size_t memOffset, size_t bindSize) {
  // A bind overrunning its partition would corrupt the next consumer's partition; fail
  // cleanly instead (possible when UC rounding exceeds the MC-rounded partition).
  if (offsetInPartition + bindSize > partition->size) {
    WARN("NVLS MC bind of size %zu at slice offset %zu exceeds slice size %zu (UC/MC granularity mismatch)", bindSize,
         offsetInPartition, partition->size);
    return ncclInternalError;
  }
  size_t mcOffset = partition->offset + offsetInPartition;
  TRACE(NCCL_NVLS, "NVLS bind MC group %llx offset %zu memory %llx memOffset %zu bindSize %zu dev %d",
        partition->mcHandle, mcOffset, mem, memOffset, bindSize, partition->dev);
  // NB: This blocks until all ranks have been added to the group, and is where we
  // normally see issues if the system NVLS/Multicast support is broken.
  CUresult err = CUPFN(cuMulticastBindMem(partition->mcHandle, mcOffset, mem, memOffset, bindSize, 0 /*flags*/));
  if (err != CUDA_SUCCESS) {
    const char* errStr;
    (void)pfn_cuGetErrorString(err, &errStr);
    WARN("Failed to bind NVLink SHARP (NVLS) Multicast memory of size %zu at MC group %llx offset %zu : CUDA error %d "
         "'%s'.\nThis is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.\n"
         "Disable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.",
         bindSize, partition->mcHandle, mcOffset, err, errStr);
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}

ncclResult_t ncclMcPartitionUnbind(const struct ncclMcPartition* partition, size_t offsetInPartition, size_t bindSize) {
  if (bindSize > 0) {
    size_t mcOffset = partition->offset + offsetInPartition;
    INFO(NCCL_NVLS, "NVLS unbind MC group %llx offset %zu size %zu dev %d", partition->mcHandle, mcOffset, bindSize,
         partition->dev);
    CUCHECK(cuMulticastUnbind(partition->mcHandle, partition->dev, mcOffset, bindSize));
  }
  return ncclSuccess;
}

#endif // CUDART_VERSION >= 12010
