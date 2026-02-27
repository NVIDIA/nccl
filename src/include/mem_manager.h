/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MEM_MANAGER_H_
#define NCCL_MEM_MANAGER_H_

#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

#if CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#ifndef CU_IPC_HANDLE_SIZE
#define CU_IPC_HANDLE_SIZE 64
#endif
typedef struct CUmemFabricHandle_st {
    unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

struct ncclComm;

// Initial capacity for exported peers array
#define NCCL_MEM_EXPORT_PEERS_INIT 8

// Memory Type for NCCL allocations
typedef enum {
  ncclMemPersist  = 0,  // Persistent memory - track stats only, never release/offload
  ncclMemScratch  = 1,  // Free without saving
  ncclMemOffload  = 2   // Copy to CPU before free, restore on resume
} ncclMemType_t;

// Memory entry state
typedef enum {
  ncclDynMemStateActive   = 0,  // Memory is allocated and usable
  ncclDynMemStateReleased = 1   // Memory has been released
} ncclDynMemState_t;

// Local owned memory descriptor
typedef struct ncclDynMemLocalDesc {
  // Shareable handle for P2P exports
  // TODO: Remove the 'fd' field - POSIX FD handles are converted on-demand via proxy
  // (ncclProxyClientGetFdBlocking), so we no longer export them upfront. Only FABRIC
  // handles need upfront export since they can be shared directly via messaging.
  union {
    int                          fd;            // For POSIX_FILE_DESCRIPTOR (unused)
    CUmemFabricHandle            fabricHandle;  // For FABRIC
  } shareableHandle;
  bool                           shareableHandleValid;
  // Peer tracking for P2P exports
  int                            numExportedPeers;
  int                            exportedPeersCapacity;
  int*                           exportedPeerRanks;
} ncclDynMemLocalDesc;

// Imported from peer memory descriptor
typedef struct ncclDynMemImportDesc {
  int                            ownerRank;     // Rank that owns the original buffer
  int                            ownerDev;      // CUDA device of the owner
  void*                          ownerPtr;      // Owner's virtual address
} ncclDynMemImportDesc;

// Individual tracked memory entry (only track scratch and offload allocations)
typedef struct ncclDynMemEntry {
  void*                          ptr;           // GPU virtual address
  size_t                         size;          // Allocation size
  CUmemGenericAllocationHandle   handle;        // Physical memory handle
  CUmemAllocationHandleType      handleType;
  ncclMemType_t                  memType;
  ncclDynMemState_t              state;
  int                            cudaDev;

  // CPU backup for OFFLOAD type memory
  void*                          cpuBackup;     // Host memory for offloaded data

  // Ownership type and type-specific data
  bool                           isImportedFromPeer;  // true if this is a peer-imported buffer
  union {
    ncclDynMemLocalDesc          local;
    ncclDynMemImportDesc         imported;
  } desc;

  // Linked list pointer
  struct ncclDynMemEntry*        next;
} ncclDynMemEntry;

// P2P Handle Exchange Structure
typedef struct ncclDynMemP2pHandleInfo {
  void*    ptr;
  int      ownerRank;
  int      ownerDev;
  size_t   size;
  int      handleType;
  union {
    uint64_t            handleData;
    CUmemFabricHandle   fabricHandle;
  };
} ncclDynMemP2pHandleInfo;

// Memory manager attached to ncclComm
typedef struct ncclMemManager {
  ncclDynMemEntry*  entries;  // Linked list of tracked allocations, only track scratch and offload allocations
  int               numEntries;
  std::mutex        lock;
  int               released;
  int               initialized;
  int               refCount;

  size_t            totalPersist;
  size_t            totalPersistImported;
  size_t            totalScratch;
  size_t            totalScratchImported;
  size_t            totalOffload;
  size_t            totalOffloadImported;
  size_t            cpuBackupUsage;

  int               commCudaDev;
} ncclMemManager;

// Initialize memory manager
ncclResult_t ncclMemManagerInit(struct ncclComm* comm);

// Destroy memory manager and free all resources
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm);

// Track a new allocation
ncclResult_t ncclMemTrack(
  struct ncclMemManager* manager,
  void* ptr,
  size_t size,
  CUmemGenericAllocationHandle handle,
  CUmemAllocationHandleType handleType,
  ncclMemType_t memType
);

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
);

// Untrack allocation
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size);

// Add peer info for buffers in the linked list entries (only for dynamic memory: scratch/offload)
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank);

ncclResult_t ncclCommMemSuspend(struct ncclComm* comm);
ncclResult_t ncclCommMemResume(struct ncclComm* comm);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_MEM_MANAGER_H_ */
