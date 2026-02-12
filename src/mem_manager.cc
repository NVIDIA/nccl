/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "alloc.h"
#include "checks.h"
#include "argcheck.h"
#include "cudawrap.h"
#include "debug.h"
#include "bootstrap.h"
#include "proxy.h"
#include "transport.h"
#include "nvtx.h"
#include "param.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <mutex>

// Internal parameter to disable memory manager for testing
NCCL_PARAM(MemManagerDisable, "DISABLE_MEM_MANAGER", 0);

// Initialize memory manager
ncclResult_t ncclMemManagerInit(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;

  ncclMemManager* mgr;
  NCCLCHECK(ncclCalloc(&mgr, 1));
  // Explicitly construct std::mutex using placement new
  new (&mgr->lock) std::mutex();

  mgr->entries = nullptr;
  mgr->numEntries = 0;
  mgr->released = 0;
  mgr->refCount = 1;
  mgr->totalPersist = 0;
  mgr->totalPersistImported = 0;
  mgr->totalScratch = 0;
  mgr->totalScratchImported = 0;
  mgr->totalOffload = 0;
  mgr->totalOffloadImported = 0;
  mgr->cpuBackupUsage = 0;
  mgr->commCudaDev = comm->cudaDev;

  __atomic_store_n(&mgr->initialized, 1, __ATOMIC_RELEASE);

  comm->memManager = mgr;

  INFO(NCCL_ALLOC, "MemManager: Initialized for device %d", comm->cudaDev);
  return ncclSuccess;
}

// Destroy memory manager and free all resources
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclSuccess;

  ncclMemManager* mgr = comm->memManager;

  if (!__atomic_load_n(&mgr->initialized, __ATOMIC_ACQUIRE)) {
    comm->memManager = nullptr;
    return ncclSuccess;
  }

  // Decrement reference count
  int refCount = ncclAtomicRefCountDecrement(&mgr->refCount);

  if (refCount > 0) {
    // Other comms still using this manager
    INFO(NCCL_ALLOC, "MemManager: Decremented refCount to %d", refCount);
    comm->memManager = nullptr;  // Clear this comm's pointer
    return ncclSuccess;
  }

  // refCount == 0, at this point proxy threads should be joined
  INFO(NCCL_ALLOC, "MemManager: Destroying (refCount=0)");
  __atomic_store_n(&mgr->initialized, 0, __ATOMIC_RELEASE);

  ncclDynMemEntry* entry = mgr->entries;
  while (entry != nullptr) {
    ncclDynMemEntry* next = entry->next;

    // Free CPU backup if exists
    if (entry->cpuBackup != nullptr) {
      ncclCudaHostFree(entry->cpuBackup);
    }

    // Close shareable FD if valid (defensive cleanup for POSIX FD handle type)
    if (!entry->isImportedFromPeer &&
        entry->desc.local.shareableHandleValid &&
        entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }

    // Only local entries have exportedPeerRanks (imported entries use desc.imported union member)
    if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
      free(entry->desc.local.exportedPeerRanks);
    }

    // Free the entry itself
    free(entry);
    entry = next;
  }

  mgr->entries = nullptr;
  mgr->numEntries = 0;

  // Explicitly call destructor for std::mutex
  mgr->lock.~mutex();
  // Free the manager struct
  free(mgr);
  comm->memManager = nullptr;

  INFO(NCCL_ALLOC, "MemManager: Destroyed");
  return ncclSuccess;
}

// Internal helper to create and track memory entry
static ncclResult_t ncclMemTrackInternal(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType,
  bool isImportedFromPeer,
  int ownerRank,
  int ownerDev,
  void* ownerPtr
) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: Cannot track allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // Persistent memory: atomic update only
  if (memType == ncclMemPersist) {
    if (isImportedFromPeer) {
      __atomic_fetch_add(&manager->totalPersistImported, size, __ATOMIC_RELAXED);
      TRACE(NCCL_ALLOC, "MemManager: Track Persistent Import ptr=%p size=%zu from rank=%d",
            ptr, size, ownerRank);
    } else {
      __atomic_fetch_add(&manager->totalPersist, size, __ATOMIC_RELAXED);
      TRACE(NCCL_ALLOC, "MemManager: Track Persistent ptr=%p size=%zu dev=%d",
            ptr, size, manager->commCudaDev);
    }
    return ncclSuccess;
  }

  // Scratch/Offload: create linked list entry
  ncclDynMemEntry* entry = (ncclDynMemEntry*)malloc(sizeof(ncclDynMemEntry));
  if (entry == nullptr) {
    WARN("MemManager: Failed to allocate memory entry");
    return ncclSystemError;
  }

  // Initialize common fields
  memset(entry, 0, sizeof(ncclDynMemEntry));
  entry->ptr = ptr;
  entry->size = size;
  entry->handle = handle;
  entry->handleType = handleType;
  entry->memType = memType;
  entry->state = ncclDynMemStateActive;
  entry->cudaDev = manager->commCudaDev;
  entry->cpuBackup = nullptr;
  entry->isImportedFromPeer = isImportedFromPeer;

  // Initialize ownership-specific fields
  if (isImportedFromPeer) {
    entry->desc.imported.ownerRank = ownerRank;
    entry->desc.imported.ownerDev = ownerDev;
    entry->desc.imported.ownerPtr = ownerPtr;
  } else {
    if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      entry->desc.local.shareableHandle.fd = -1;  // avoid using 0 which is stdin
    }
    entry->desc.local.shareableHandleValid = false;
    entry->desc.local.numExportedPeers = 0;
    entry->desc.local.exportedPeersCapacity = 0;
    entry->desc.local.exportedPeerRanks = nullptr;
  }

  { // lock the mutex to add the entry to the linked list
    std::lock_guard<std::mutex> lock(manager->lock);
    // Add to linked list (prepend)
    entry->next = manager->entries;
    manager->entries = entry;
    manager->numEntries++;
  } // lock_guard automatically releases mutex

  // Update statistics
  if (isImportedFromPeer) {
    if (memType == ncclMemScratch) {
      __atomic_fetch_add(&manager->totalScratchImported, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      __atomic_fetch_add(&manager->totalOffloadImported, size, __ATOMIC_RELAXED);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track imported ptr=%p size=%zu type=%d from rank=%d entries=%d",
          ptr, size, memType, ownerRank, manager->numEntries);
  } else {
    if (memType == ncclMemScratch) {
      __atomic_fetch_add(&manager->totalScratch, size, __ATOMIC_RELAXED);
    } else if (memType == ncclMemOffload) {
      __atomic_fetch_add(&manager->totalOffload, size, __ATOMIC_RELAXED);
    }
    TRACE(NCCL_ALLOC, "MemManager: Track ptr=%p size=%zu type=%d dev=%d entries=%d",
          ptr, size, memType, manager->commCudaDev, manager->numEntries);
  }

  return ncclSuccess;
}

// Track a new allocation
ncclResult_t ncclMemTrack(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType
) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType,
                              false, -1, -1, nullptr);
}

// Track imported allocation from peer
ncclResult_t ncclMemTrackImportFromPeer(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType,
  int ownerRank,
  int ownerDev,
  void* ownerPtr
) {
  return ncclMemTrackInternal(manager, ptr, size, handle, handleType, memType,
                              true, ownerRank, ownerDev, ownerPtr);
}

// Untrack allocation
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;

  // Atomic check to avoid locking destroyed mutex
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: Cannot untrack allocation ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }

  // Variables to save values before releasing lock
  size_t entrySize = 0;
  int numEntries __attribute__((unused)) = 0;  // May be unused if TRACE compiled out
  bool isImportedFromPeer = false;
  ncclMemType_t memType = ncclMemScratch;

  {
    std::lock_guard<std::mutex> lock(manager->lock);

    ncclDynMemEntry* prev = nullptr;
    ncclDynMemEntry* entry = manager->entries;

    while (entry != nullptr) {
      if (entry->ptr == ptr) {
        // Remove from linked list
        if (prev == nullptr) {
          manager->entries = entry->next;
        } else {
          prev->next = entry->next;
        }
        manager->numEntries--;

        // Free CPU backup if exists
        if (entry->cpuBackup != nullptr) {
          manager->cpuBackupUsage -= entry->size;
          ncclCudaHostFree(entry->cpuBackup);
        }

        // Close shareable FD if valid (defensive cleanup for POSIX FD handle type)
        if (!entry->isImportedFromPeer &&
            entry->desc.local.shareableHandleValid &&
            entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
            entry->desc.local.shareableHandle.fd >= 0) {
          close(entry->desc.local.shareableHandle.fd);
          entry->desc.local.shareableHandle.fd = -1;
          entry->desc.local.shareableHandleValid = false;
        }

        // Only local entries have exportedPeerRanks (imported entries use desc.imported union member)
        if (!entry->isImportedFromPeer && entry->desc.local.exportedPeerRanks != nullptr) {
          free(entry->desc.local.exportedPeerRanks);
        }

        // Save values before unlock for logging (may be unused if TRACE is compiled out)
        entrySize = entry->size;
        numEntries = manager->numEntries;
        isImportedFromPeer = entry->isImportedFromPeer;
        memType = entry->memType;

        // Safety check: log if tracked size doesn't match passed size
        if (entrySize != size) {
          INFO(NCCL_ALLOC, "MemManager: Untrack size mismatch ptr=%p tracked=%zu passed=%zu", ptr, entrySize, size);
        }

        free(entry);
        break;
      }
      prev = entry;
      entry = entry->next;
    }
  } // lock_guard automatically releases mutex

  // Update statistics
  if (entrySize > 0) {
    // Entry found in linked list
    if (isImportedFromPeer) {
      if (memType == ncclMemScratch) {
        __atomic_fetch_sub(&manager->totalScratchImported, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemOffload) {
        __atomic_fetch_sub(&manager->totalOffloadImported, entrySize, __ATOMIC_RELAXED);
      }
    } else {
      if (memType == ncclMemScratch) {
        __atomic_fetch_sub(&manager->totalScratch, entrySize, __ATOMIC_RELAXED);
      } else if (memType == ncclMemOffload) {
        __atomic_fetch_sub(&manager->totalOffload, entrySize, __ATOMIC_RELAXED);
      }
    }

    TRACE(NCCL_ALLOC, "MemManager: Untrack ptr=%p size=%zu entries=%d",
          ptr, entrySize, numEntries);
  } else {
    // Entry not found in linked list - must be persistent memory
    __atomic_fetch_sub(&manager->totalPersist, size, __ATOMIC_RELAXED);
    TRACE(NCCL_ALLOC, "MemManager: Untrack Persistent ptr=%p size=%zu", ptr, size);
  }

  return ncclSuccess;
}

// Track that a buffer is being shared with a peer (for suspend/resume coordination)
// NOTE: Only for dynamic memory (scratch/offload) that's in the linked list.
// Persistent memory doesn't need export tracking since it's never suspended.
// Call this after allocating dynamic memory and the peer imports it.
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank) {
  if (ncclParamMemManagerDisable()) return ncclSuccess;
  if (manager == nullptr || ptr == nullptr) return ncclInternalError;
  if (!__atomic_load_n(&manager->initialized, __ATOMIC_ACQUIRE)) {
    WARN("MemManager: Cannot mark export for ptr=%p, manager not initialized", ptr);
    return ncclInternalError;
  }
  std::lock_guard<std::mutex> lock(manager->lock);

  // Find entry in linked list (only contains scratch/offload, not persistent)
  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr && entry->ptr != ptr) {
    entry = entry->next;
  }

  if (entry == nullptr) {
    WARN("MemManager: Cannot mark export for ptr=%p - not found in tracked entries. "
         "Only dynamic memory (scratch/offload) needs export tracking for suspend/resume.", ptr);
    return ncclInternalError;
  }

  // Verify this is a local entry, not an imported one
  if (entry->isImportedFromPeer) {
    WARN("MemManager: Cannot mark export for ptr=%p - this is an imported buffer, not a local one", ptr);
    return ncclInternalError;
  }

  // Check if peer already exists
  for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
    if (entry->desc.local.exportedPeerRanks[i] == peerRank) {
      WARN("MemManager: Buffer ptr=%p already exported to peer rank %d", ptr, peerRank);
      return ncclInternalError;
    }
  }

  if (entry->desc.local.numExportedPeers >= entry->desc.local.exportedPeersCapacity) {
    int newCapacity = entry->desc.local.exportedPeersCapacity == 0 ? NCCL_MEM_EXPORT_PEERS_INIT :
                      entry->desc.local.exportedPeersCapacity * 2;
    ncclResult_t ret = ncclRealloc(&entry->desc.local.exportedPeerRanks,
                                   entry->desc.local.exportedPeersCapacity,
                                   newCapacity);
    if (ret != ncclSuccess) {
      WARN("MemManager: Failed to grow exportedPeerRanks array for ptr=%p", ptr);
      return ret;
    }
    entry->desc.local.exportedPeersCapacity = newCapacity;
  }

  // Add peer to export list
  entry->desc.local.exportedPeerRanks[entry->desc.local.numExportedPeers++] = peerRank;

  TRACE(NCCL_ALLOC, "MemManager: ExportToPeer ptr=%p peerRank=%d numExportedPeers=%d",
        ptr, peerRank, entry->desc.local.numExportedPeers);
  return ncclSuccess;
}

/*
 * Internal: Suspend all dynamic memory with P2P coordination
 *
 * Order of operations:
 * 1. First pass: Unmap all peer-imported buffers
 * 2. Second pass: Offload local buffers to CPU and suspend physical memory
 */
ncclResult_t ncclCommMemSuspend(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable())
  {
    WARN("MemManager: Suspend failed, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* manager = comm->memManager;

  if (manager->released) {
    WARN("MemManager: Already suspended");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  size_t releasedScratch = 0;
  size_t releasedOffload = 0;
  size_t releasedPeerImport = 0;
  int releasedCount = 0;
  int peerImportCount = 0;
  ncclDynMemEntry* entry = nullptr;

  CUDACHECK(cudaDeviceSynchronize());
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF), ret, fail);

  // Step 1: Unmap all peer-imported buffers first
  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer && entry->state == ncclDynMemStateActive) {
      TRACE(NCCL_ALLOC, "MemManager: Unmapping peer-imported buffer ptr=%p from rank %d",
            entry->ptr, entry->desc.imported.ownerRank);

      // Unmap our local mapping of the peer's memory
      CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)entry->ptr, entry->size));

      // Release our reference to the peer's handle
      CUCHECKIGNORE(cuMemRelease(entry->handle));
      entry->handle = 0;  // Clear invalid handle

      entry->state = ncclDynMemStateReleased;
      releasedPeerImport += entry->size;
      peerImportCount++;
    }
    entry = entry->next;
  }

  // Step 2: Offload and release local memory
  entry = manager->entries;
  while (entry != nullptr) {
    // Skip buffer imported from peer
    if (entry->isImportedFromPeer) {
      entry = entry->next;
      continue;
    }

    // Skip already released
    if (entry->state == ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    // For OFFLOAD type: copy to CPU backup first
    if (entry->memType == ncclMemOffload) {
      NCCLCHECKGOTO(ncclCudaHostCalloc((char**)&entry->cpuBackup, entry->size), ret, fail);
      if (entry->cpuBackup == nullptr) {
        WARN("MemManager: Failed to allocate CPU backup for offload");
        ret = ncclSystemError;
        goto fail;
      }

      // Copy GPU to CPU
      cudaError_t err = cudaMemcpy(entry->cpuBackup, entry->ptr, entry->size, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        ncclCudaHostFree(entry->cpuBackup);
        entry->cpuBackup = nullptr;
        WARN("MemManager: Failed to copy to CPU backup: %s", cudaGetErrorString(err));
        ret = ncclUnhandledCudaError;
        goto fail;
      }

      manager->cpuBackupUsage += entry->size;
      releasedOffload += entry->size;
    } else {
      releasedScratch += entry->size;
    }

    // Close the shareable FD if valid (for POSIX handles)
    if (entry->desc.local.shareableHandleValid &&
        entry->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
        entry->desc.local.shareableHandle.fd >= 0) {
      close(entry->desc.local.shareableHandle.fd);
      entry->desc.local.shareableHandle.fd = -1;
      entry->desc.local.shareableHandleValid = false;
    }

    // Unmap physical memory but keep virtual address reservation
    CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)entry->ptr, entry->size));

    // Release physical memory handle
    CUCHECKIGNORE(cuMemRelease(entry->handle));
    entry->handle = 0;  // Clear invalid handle

    entry->state = ncclDynMemStateReleased;
    releasedCount++;

    entry = entry->next;
  }

  manager->released = 1;

  INFO(NCCL_ALLOC, "MemManager: rank %d suspended %d local + %d peer entries (scratch=%zu, offload=%zu, peerImport=%zu, cpuBackup=%zu)",
       comm->rank, releasedCount, peerImportCount, releasedScratch, releasedOffload, releasedPeerImport, manager->cpuBackupUsage);

  return ncclSuccess;

fail:
  return ret;
}

/*
 * Internal: Resume previously suspended dynamic memory with P2P coordination
 *
 * Order of operations:
 * 1. First pass: Resume local memory (re-allocate, re-map, restore offloaded data)
 * 2. Exchange: AllGather new handle info for P2P buffers
 * 3. Second pass: Re-import peer buffers using new handles
 */
ncclResult_t ncclCommMemResume(struct ncclComm* comm) {
  if (ncclParamMemManagerDisable())
  {
    WARN("MemManager: Resume failed, memory manager is disabled");
    return ncclInvalidUsage;
  }
  if (comm == nullptr) return ncclInvalidArgument;
  if (comm->memManager == nullptr) return ncclInvalidUsage;
  ncclMemManager* manager = comm->memManager;

  if (!manager->released) {
    WARN("MemManager: Not in suspended state");
    return ncclInvalidUsage;
  }

  ncclResult_t ret = ncclSuccess;
  int restoredLocalCount = 0;
  int restoredPeerCount = 0;
  size_t restoredLocalBytes = 0;
  size_t restoredPeerBytes = 0;

  int localBroadcastCount = 0;
  int* allCounts = nullptr;
  int totalInfoCount = 0;
  ncclDynMemP2pHandleInfo* localInfos = nullptr;
  ncclDynMemP2pHandleInfo* allInfos = nullptr;

  // Step 1: Restore all local memory
  ncclDynMemEntry* entry = manager->entries;
  while (entry != nullptr) {
    // Skip peer-imported entries
    if (entry->isImportedFromPeer) {
      entry = entry->next;
      continue;
    }

    // Skip entries that weren't released
    if (entry->state != ncclDynMemStateReleased) {
      entry = entry->next;
      continue;
    }

    // Re-create physical allocation
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = entry->cudaDev;
    prop.requestedHandleTypes = entry->handleType;

    CUmemGenericAllocationHandle newHandle;
    CUCHECKGOTO(cuMemCreate(&newHandle, entry->size, &prop, 0), ret, fail);

    // Re-map to the same virtual address and set access permissions for local device
    ret = ncclCuMemMapAndSetAccess(entry->ptr, entry->size, newHandle, entry->cudaDev);
    if (ret != ncclSuccess) {
      CUCHECKIGNORE(cuMemRelease(newHandle));
      entry->handle = 0;  // Clear to avoid dangling handle
      goto fail;
    }

    // Restore peer access for all previously exported peers
    for (int i = 0; i < entry->desc.local.numExportedPeers; i++) {
      int peerRank = entry->desc.local.exportedPeerRanks[i];
      if (peerRank >= 0 && peerRank < comm->nRanks) {
        // Only set access for peers on the same node and same process
        if (comm->peerInfo[peerRank].pidHash == comm->peerInfo[comm->rank].pidHash && comm->peerInfo[peerRank].hostHash == comm->peerInfo[comm->rank].hostHash){
          int peerDev = comm->peerInfo[peerRank].cudaDev;
          CUmemAccessDesc peerAccessDesc = {};
          peerAccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          peerAccessDesc.location.id = peerDev;
          peerAccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          CUCHECKIGNORE(cuMemSetAccess((CUdeviceptr)entry->ptr, entry->size, &peerAccessDesc, 1));
          TRACE(NCCL_ALLOC, "MemManager: Restored peer access for ptr=%p to rank %d dev %d",
                entry->ptr, peerRank, peerDev);
        }
      }
    }

    // Update handle
    entry->handle = newHandle;

    // For OFFLOAD type: restore data from CPU backup
    if (entry->memType == ncclMemOffload && entry->cpuBackup != NULL) {
      cudaError_t err = cudaMemcpy(entry->ptr, entry->cpuBackup, entry->size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        WARN("MemManager: Failed to restore from CPU backup: %s (backup preserved)", cudaGetErrorString(err));
        ret = ncclUnhandledCudaError;
        goto fail;
      }
      // Free CPU backup on successful restore
      ncclCudaHostFree(entry->cpuBackup);
      entry->cpuBackup = nullptr;
      manager->cpuBackupUsage -= entry->size;
    }

    entry->desc.local.shareableHandleValid = false;
    if (entry->handleType == CU_MEM_HANDLE_TYPE_FABRIC) {
      CUresult exportRet = CUPFN(cuMemExportToShareableHandle(&entry->desc.local.shareableHandle.fabricHandle, newHandle,
                                                               CU_MEM_HANDLE_TYPE_FABRIC, 0));
      if (exportRet != CUDA_SUCCESS) {
        WARN("MemManager: cuMemExportToShareableHandle (FABRIC) failed for ptr=%p", entry->ptr);
        CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)entry->ptr, entry->size));
        CUCHECKIGNORE(cuMemRelease(newHandle));
        entry->handle = 0;
        ret = ncclUnhandledCudaError;
        goto fail;
      }
      entry->desc.local.shareableHandleValid = true;
    }

    entry->state = ncclDynMemStateActive;
    restoredLocalCount++;
    restoredLocalBytes += entry->size;

    TRACE(NCCL_ALLOC, "MemManager: Resumed local buffer ptr=%p size=%zu numExportedPeers=%d",
          entry->ptr, entry->size, entry->desc.local.numExportedPeers);

    entry = entry->next;
  }

  // Step 2: Barrier to ensure all ranks have resumed their local memory
  if (comm->bootstrap != nullptr) {
    INFO(NCCL_ALLOC, "MemManager: rank %d resumed %d local entries, waiting at barrier",
         comm->rank, restoredLocalCount);
    ret = bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF);
    if (ret != ncclSuccess) {
      WARN("MemManager: Barrier failed during resume");
      return ret;
    }
  }

  /*
   * Step 3: Exchange new handle info for P2P coordination
   *
   * Each rank broadcasts info about its local buffers that have peers.
   * We use a simple approach: each rank sends to all peers that imported its buffers.
   */

  // Count local buffers that have peers (need to broadcast new handle info)
  localBroadcastCount = 0;
  entry = manager->entries;
  while (entry != nullptr) {
    if (!entry->isImportedFromPeer && entry->desc.local.numExportedPeers > 0 && entry->state == ncclDynMemStateActive) {
      localBroadcastCount++;
    }
    entry = entry->next;
  }

  // Gather counts from all ranks using AllGather
  if (comm->bootstrap != nullptr && comm->nRanks > 1) {
    // Allocate buffer for all counts
    allCounts = (int*)malloc(comm->nRanks * sizeof(int));
    if (allCounts == nullptr) {
      WARN("MemManager: Failed to allocate allCounts");
      return ncclSystemError;
    }
    memset(allCounts, 0, comm->nRanks * sizeof(int));
    allCounts[comm->rank] = localBroadcastCount;

    // AllGather counts: each rank contributes sizeof(int) at its position
    ret = bootstrapAllGather(comm->bootstrap, allCounts, sizeof(int));
    if (ret != ncclSuccess) {
      free(allCounts);
      WARN("MemManager: AllGather counts failed");
      return ret;
    }

    // Calculate total and offsets
    int* offsets = (int*)malloc(comm->nRanks * sizeof(int));
    if (offsets == nullptr) {
      free(allCounts);
      return ncclSystemError;
    }

    totalInfoCount = 0;
    for (int r = 0; r < comm->nRanks; r++) {
      offsets[r] = totalInfoCount;
      totalInfoCount += allCounts[r];
    }

    if (totalInfoCount > 0) {
      // Prepare local info to send
      int localAllocCount = localBroadcastCount > 0 ? localBroadcastCount : 1;
      localInfos = (ncclDynMemP2pHandleInfo*)malloc(localAllocCount * sizeof(ncclDynMemP2pHandleInfo));
      if (localInfos == nullptr) {
        free(allCounts);
        free(offsets);
        return ncclSystemError;
      }

      int idx = 0;
      entry = manager->entries;
      while (entry != nullptr && idx < localBroadcastCount) {
        if (!entry->isImportedFromPeer && entry->desc.local.numExportedPeers > 0 && entry->state == ncclDynMemStateActive) {
          localInfos[idx].ptr = entry->ptr;
          localInfos[idx].ownerRank = comm->rank;
          localInfos[idx].ownerDev = entry->cudaDev;
          localInfos[idx].size = entry->size;
          localInfos[idx].handleType = entry->handleType;

          if (entry->handleType == CU_MEM_HANDLE_TYPE_FABRIC) {
            // For FABRIC: copy the exported fabric handle (can be shared directly)
            if (entry->desc.local.shareableHandleValid) {
              memcpy(&localInfos[idx].fabricHandle, &entry->desc.local.shareableHandle.fabricHandle, sizeof(CUmemFabricHandle));
            } else {
              WARN("MemManager: FABRIC handle not valid for entry ptr=%p", entry->ptr);
            }
          } else {
            // For POSIX FD: store the cuMem handle (for FD conversion via proxy)
            memcpy(&localInfos[idx].handleData, &entry->handle, sizeof(CUmemGenericAllocationHandle));
          }

          idx++;
        }
        entry = entry->next;
      }

      // Allocate buffer for all infos
      allInfos = (ncclDynMemP2pHandleInfo*)malloc(totalInfoCount * sizeof(ncclDynMemP2pHandleInfo));
      if (allInfos == nullptr) {
        free(allCounts);
        free(offsets);
        free(localInfos);
        return ncclSystemError;
      }

      // Copy local data to correct position
      if (localBroadcastCount > 0) {
        memcpy(allInfos + offsets[comm->rank], localInfos,
               localBroadcastCount * sizeof(ncclDynMemP2pHandleInfo));
      }

      // Exchange using Send/Recv (send first, then receive to avoid deadlock)
      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank && localBroadcastCount > 0) {
          ret = bootstrapSend(comm->bootstrap, r, 0xFEED,
                              localInfos,
                              localBroadcastCount * sizeof(ncclDynMemP2pHandleInfo));
          if (ret != ncclSuccess) {
            WARN("MemManager: Send to rank %d failed - handle exchange incomplete", r);
            free(offsets);
            free(allCounts);
            free(localInfos);
            free(allInfos);
            return ret;
          }
        }
      }

      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank && allCounts[r] > 0) {
          ret = bootstrapRecv(comm->bootstrap, r, 0xFEED,
                              allInfos + offsets[r],
                              allCounts[r] * sizeof(ncclDynMemP2pHandleInfo));
          if (ret != ncclSuccess) {
            WARN("MemManager: Recv from rank %d failed - handle exchange incomplete", r);
            free(offsets);
            free(allCounts);
            free(localInfos);
            free(allInfos);
            return ret;
          }
        }
      }
    }

    free(offsets);
  }

  /*
   * Step 4: Re-import peer buffers using exchanged handle info
   */

  entry = manager->entries;
  while (entry != nullptr) {
    if (entry->isImportedFromPeer && entry->state == ncclDynMemStateReleased) {
      // Find matching info from AllGather results
      ncclDynMemP2pHandleInfo* matchedInfo = nullptr;

      if (allInfos != nullptr) {
        for (int i = 0; i < totalInfoCount; i++) {
          if (allInfos[i].ownerRank == entry->desc.imported.ownerRank &&
              allInfos[i].ptr == entry->desc.imported.ownerPtr &&
              allInfos[i].size == entry->size) {
            matchedInfo = &allInfos[i];
            break;
          }
        }
      }

      if (matchedInfo == nullptr) {
        WARN("MemManager: Could not find matching handle info for ptr=%p from rank %d",
             entry->ptr, entry->desc.imported.ownerRank);
        entry = entry->next;
        continue;
      }

      TRACE(NCCL_ALLOC, "MemManager: Re-importing peer buffer ptr=%p from rank %d (owner ptr=%p)",
            entry->ptr, entry->desc.imported.ownerRank, matchedInfo->ptr);

      CUmemGenericAllocationHandle newHandle;
      CUresult curet;

      if (matchedInfo->handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        // POSIX FD handles only work within the same node - check hostHash
        if (comm->peerInfo && comm->peerInfoValid &&
            comm->peerInfo[entry->desc.imported.ownerRank].hostHash != comm->peerInfo[comm->rank].hostHash) {
          WARN("MemManager: Cannot re-import peer buffer from rank %d (different node) using POSIX FD - skipping",
               entry->desc.imported.ownerRank);
          entry = entry->next;
          continue;
        }

        // For POSIX FD: We need to get the FD from the owner
        // Use proxy to convert cuMem handle to FD
        int fd = -1;

        // The handleData contains the cuMem handle - request FD conversion
        ret = ncclProxyClientGetFdBlocking(comm, entry->desc.imported.ownerRank,
                                           &matchedInfo->handleData, &fd);
        if (ret != ncclSuccess || fd < 0) {
          WARN("MemManager: Failed to get FD from rank %d for ptr=%p",
               entry->desc.imported.ownerRank, entry->ptr);
          entry = entry->next;
          continue;
        }

        curet = CUPFN(cuMemImportFromShareableHandle(&newHandle, (void*)(uintptr_t)fd,
                                                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(fd);
      } else if (matchedInfo->handleType == CU_MEM_HANDLE_TYPE_FABRIC) {
        // For FABRIC: Import directly using the fabric handle
        curet = CUPFN(cuMemImportFromShareableHandle(&newHandle, &matchedInfo->fabricHandle,
                                                CU_MEM_HANDLE_TYPE_FABRIC));
      } else {
        WARN("MemManager: Unknown handle type %d for peer import", matchedInfo->handleType);
        entry = entry->next;
        continue;
      }

      if (curet != CUDA_SUCCESS) {
        WARN("MemManager: cuMemImportFromShareableHandle failed for ptr=%p (curet=%d)",
             entry->ptr, curet);
        entry = entry->next;
        continue;
      }

      // Re-map to the same virtual address and set access permissions
      ncclResult_t mapResult = ncclCuMemMapAndSetAccess(entry->ptr, entry->size, newHandle, comm->cudaDev);
      if (mapResult != ncclSuccess) {
        CUCHECKIGNORE(cuMemRelease(newHandle));
        entry->handle = 0;
        WARN("MemManager: ncclCuMemMapAndSetAccess failed for re-imported ptr=%p", entry->ptr);
        entry = entry->next;
        continue;
      }

      entry->handle = newHandle;
      entry->state = ncclDynMemStateActive;
      restoredPeerCount++;
      restoredPeerBytes += entry->size;

      TRACE(NCCL_ALLOC, "MemManager: Successfully re-imported peer buffer ptr=%p from rank %d",
            entry->ptr, entry->desc.imported.ownerRank);
    }
    entry = entry->next;
  }

  manager->released = 0;

  // Final barrier to ensure all ranks have completed peer import setup
  if (comm->bootstrap != nullptr) {
    INFO(NCCL_ALLOC, "MemManager: rank %d resumed %d local + %d peer entries (%zu + %zu bytes)",
         comm->rank, restoredLocalCount, restoredPeerCount, restoredLocalBytes, restoredPeerBytes);
    ret = bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xCAFE);
    if (ret != ncclSuccess) {
      // Cleanup
      if (allCounts) free(allCounts);
      if (localInfos) free(localInfos);
      if (allInfos) free(allInfos);
      WARN("MemManager: Final barrier failed during resume");
      return ret;
    }
  }

  // Cleanup
  if (allCounts) free(allCounts);
  if (localInfos) free(localInfos);
  if (allInfos) free(allInfos);

  return ncclSuccess;

fail:
  if (allCounts) free(allCounts);
  if (localInfos) free(localInfos);
  if (allInfos) free(allInfos);
  return ret;
}

/*
 * Public Communicator Suspend/Resume APIs
 */

NCCL_API(ncclResult_t, ncclCommSuspend, ncclComm_t comm, int flags);
ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "ncclCommSuspend", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (flags & NCCL_SUSPEND_MEM) {
    if (ncclParamMemManagerDisable())
    {
      WARN("MemManager: Suspend not supported, memory manager is disabled");
      return ncclInvalidUsage;
    }
    // Check if manager is shared
    if (comm->memManager && comm->memManager->refCount > 1) {
      WARN("Memory suspend not supported with split_share communicators (refCount=%d)",
           comm->memManager->refCount);
      return ncclInvalidUsage;
    }
    INFO(NCCL_INIT, "ncclCommSuspend: rank %d suspending memory", comm->rank);
    NCCLCHECK(ncclCommMemSuspend(comm));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommResume, ncclComm_t comm);
ncclResult_t ncclCommResume(ncclComm_t comm) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "ncclCommResume", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (ncclParamMemManagerDisable())
  {
    WARN("MemManager: Resume not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }
  // Check if manager is shared
  if (comm->memManager && comm->memManager->refCount > 1) {
    WARN("Memory resume not supported with split_share communicators (refCount=%d)",
         comm->memManager->refCount);
    return ncclInvalidUsage;
  }
  INFO(NCCL_INIT, "ncclCommResume: rank %d resuming all resources", comm->rank);
  NCCLCHECK(ncclCommMemResume(comm));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommMemStats, ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value);
ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "ncclCommMemStats", "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  if (value == nullptr) return ncclInvalidArgument;

  if (ncclParamMemManagerDisable()) {
    WARN("MemManager: MemStats not supported, memory manager is disabled");
    return ncclInvalidUsage;
  }

  if (comm->memManager == nullptr) {
    *value = 0;
    return ncclSuccess;
  }

  ncclMemManager* manager = comm->memManager;
  switch (stat) {
    case ncclStatGpuMemTotal:
      *value = __atomic_load_n(&manager->totalPersist, __ATOMIC_RELAXED) +
               __atomic_load_n(&manager->totalScratch, __ATOMIC_RELAXED) +
               __atomic_load_n(&manager->totalOffload, __ATOMIC_RELAXED);
      return ncclSuccess;
    case ncclStatGpuMemPersist:
      *value = __atomic_load_n(&manager->totalPersist, __ATOMIC_RELAXED);
      return ncclSuccess;
    case ncclStatGpuMemSuspend:
      *value = __atomic_load_n(&manager->totalScratch, __ATOMIC_RELAXED) +
               __atomic_load_n(&manager->totalOffload, __ATOMIC_RELAXED);
      return ncclSuccess;
    case ncclStatGpuMemSuspended:
      // Boolean: 0=active, 1=suspended
      *value = manager->released ? 1 : 0;
      return ncclSuccess;
    default:
      return ncclInvalidArgument;
  }
}
