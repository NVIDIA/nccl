/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "dev_runtime.h"
#include "comm.h"
#include "device.h"
#include "transport.h"
#include "group.h"
#include "nccl_device.h"

NCCL_PARAM(WinStride, "WIN_STRIDE", -1);

// Complete types from src/include/dev_runtime.h
struct ncclDevrMemory {
  int refCount;
  struct ncclDevrMemory* next;
  CUmemGenericAllocationHandle memHandle;
  void* primaryAddr; // What we hope is the VA of this memory's first mapping.
  size_t size;
  size_t bigOffset; // offset in big VA space
  void* ginHostWins[NCCL_GIN_MAX_CONTEXTS];
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS];
};

struct ncclDevrWindowSorted {
  uintptr_t userAddr;
  size_t size;
  struct ncclDevrWindow* win;
};

struct ncclDevrTeam {
  struct ncclDevrTeam* next;
  struct ncclTeam team;
  CUmemGenericAllocationHandle mcHandle;
  void* mcBasePtr;
  int worldRankList[];
};

////////////////////////////////////////////////////////////////////////////////
// Helpers at the bottom:

// Find least index such that `arg < sorted[i].key` (least upper bound)
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg);

template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val);

template<typename Obj>
static void listRemove(Obj* list, int* count, int index);

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  if (devr->bigSize != 0) return ncclSuccess;

  // LSA needs to be the same size for all ranks, and it needs to represent
  // a consecutive set of ranks.
  int lsaSize = 0;
  int nodeSize = 1;
  for (int r=1; r < comm->nRanks; r++) {
    if (comm->rankToNode[r] == comm->rankToNode[r-1]) {
      nodeSize += 1;
    } else {
      lsaSize = gcd(lsaSize, nodeSize);
      nodeSize = 1;
    }
  }
  lsaSize = gcd(lsaSize, nodeSize);
  devr->lsaSize = lsaSize;
  devr->lsaSelf = comm->rank % lsaSize;
  devr->lsaRankList = (int*)malloc(devr->lsaSize*sizeof(int));
  for (int i=0; i < devr->lsaSize; i++) {
    devr->lsaRankList[i] = comm->rank + (i - devr->lsaSelf);
  }

  CUmemAllocationProp memProp = {};
  memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memProp.requestedHandleTypes = ncclCuMemHandleType;
  memProp.location.id = comm->cudaDev;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&devr->granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail_lsaRankList);

  devr->bigSize = ncclParamWinStride();
  if (-devr->bigSize <= 1) {
    devr->bigSize = 1;
    for (int r=0; r < comm->nRanks; ++r) {
      devr->bigSize = std::max<size_t>(devr->bigSize, comm->peerInfo[r].totalGlobalMem);
    }
  }
  devr->bigSize = alignUp(devr->bigSize, size_t(1)<<32);
  INFO(NCCL_INIT, "Symmetric VA size=%ldGB", (long)devr->bigSize>>30);
  
  ncclSpaceConstruct(&devr->bigSpace);
  ncclShadowPoolConstruct(&devr->shadows);
  return ncclSuccess;

fail_lsaRankList:
  free(devr->lsaRankList);
  return ret;
}

static void symTeamDestroyAll(struct ncclComm* comm); // Further down

ncclResult_t ncclDevrFinalize(struct ncclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;
  if (devr->bigSize == 0) return ncclSuccess;

  while (!ncclIntruQueueEmpty(&devr->regTaskQueue)) {
    struct ncclDevrRegTask* task = ncclIntruQueueDequeue(&devr->regTaskQueue);
    free(task);
  }
  
  symTeamDestroyAll(comm);
  { // delete windowTable
    cudaStream_t stream;
    if (cudaSuccess == cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) {
      struct ncclDevCommWindowTable* tableDev = devr->windowTable;
      while (tableDev != nullptr) {
        struct ncclDevCommWindowTable* tableHost;
        if (ncclSuccess != ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost)) break;
        struct ncclDevCommWindowTable* next = tableHost->next;
        ncclShadowPoolFree(&devr->shadows, tableDev, stream);
        tableDev = next;
      }
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
  CUdeviceptr flatAddr = reinterpret_cast<CUdeviceptr>(devr->lsaFlatBase);
  CUCHECKIGNORE(cuMemUnmap(flatAddr, devr->lsaSize*devr->bigSize));
  CUCHECKIGNORE(cuMemAddressFree(flatAddr, devr->lsaSize*devr->bigSize));
  ncclShadowPoolDestruct(&devr->shadows);
  ncclSpaceDestruct(&devr->bigSpace);
  free(devr->lsaRankList);
  free(devr->winSorted);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

static ncclResult_t symMemoryMapLsaTeam(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, size_t size, size_t bigOffset
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  CUmemAccessDesc accessDesc = {};
  union Message {
    CUmemGenericAllocationHandle memHandle;
    CUmemFabricHandle fabricHandle;
  };

  Message* messages = (Message*)calloc(devr->lsaSize, sizeof(Message));
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    messages[devr->lsaSelf].memHandle = memHandle;
  } else {
    CUCHECKGOTO(cuMemExportToShareableHandle(&messages[devr->lsaSelf].fabricHandle, memHandle, ncclCuMemHandleType, 0), ret, fail);
  }

  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, messages, sizeof(Message)), ret, fail);

  if (devr->lsaFlatBase == nullptr) { // Create on first need.
    CUdeviceptr addr;
    CUCHECKGOTO(cuMemAddressReserve(&addr, devr->lsaSize*devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail);
    devr->lsaFlatBase = reinterpret_cast<void*>(addr);
  }
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = comm->cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  for (int r = 0; r < devr->lsaSize; r++) {
    CUmemGenericAllocationHandle impHandle;
    if (r == devr->lsaSelf) {
      impHandle = memHandle;
    } else {
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int fd = -1;
        NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, devr->lsaRankList[r], &messages[r], &fd), ret, fail);
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, reinterpret_cast<void*>((uintptr_t)fd), ncclCuMemHandleType), ret, fail);
        SYSCHECKGOTO(close(fd), "close", ret, fail);
      } else {
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, (void*)&messages[r].fabricHandle, ncclCuMemHandleType), ret, fail);
      }
    }
    CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r*devr->bigSize + bigOffset);
    CUCHECKGOTO(cuMemMap(addr, size, 0, impHandle, 0), ret, fail);
    CUCHECKGOTO(cuMemSetAccess(addr, size, &accessDesc, 1), ret, fail);
    if (r != devr->lsaSelf) {
      CUCHECKGOTO(cuMemRelease(impHandle), ret, fail);
    }
  }
  // Ensure everyone has imported my mem handle.
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList, devr->lsaSelf, devr->lsaSize, 0xbeef), ret, fail);
leave:
  free(messages);
  return ret;
fail:
  goto leave;
}

static ncclResult_t symBindTeamMemory(
    struct ncclComm* comm, struct ncclDevrTeam* tm, struct ncclDevrMemory* mem
  ) {
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
  #if CUDART_VERSION >= 12010
    INFO(NCCL_NVLS, "Binding multicast memory at big=%lx to team {%d x %d}", mem->bigOffset, tm->team.nRanks, tm->team.stride);
    CUCHECK(cuMulticastBindMem(tm->mcHandle, mem->bigOffset, mem->memHandle, 0, mem->size, 0));
  #endif
  }
  return ncclSuccess;
}

static ncclResult_t symUnbindTeamMemory(
    struct ncclComm* comm, struct ncclDevrTeam* tm, struct ncclDevrMemory* mem
  ) {
  if (comm->nvlsSupport && tm->mcBasePtr != nullptr) {
  #if CUDART_VERSION >= 12010
    CUCHECK(cuMulticastUnbind(tm->mcHandle, comm->cudaDev, mem->bigOffset, mem->size));
  #endif
  }
  return ncclSuccess;
}

// Caller must barrier the team afterward.
static ncclResult_t symTeamObtain(
    struct ncclComm* comm, struct ncclTeam team, bool multimem,
    struct ncclDevrTeam** outTeam
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrTeam* t = devr->teamHead;
  bool teamIsNew = false;
  while (true) {
    if (t == nullptr) {
      teamIsNew = true;
      t = (struct ncclDevrTeam*)malloc(sizeof(struct ncclDevrTeam) + team.nRanks*sizeof(int));
      t->team = team;
      t->mcHandle = 0x0;
      t->mcBasePtr = nullptr;
      for (int i=0; i < team.nRanks; i++) {
        t->worldRankList[i] = comm->rank + (i - team.rank)*team.stride;
      }
      break;
    } else if (t->team.rank == team.rank && t->team.nRanks == team.nRanks && t->team.stride == team.stride) {
      if (!multimem || t->mcBasePtr != nullptr) {
        // Matching team is sufficient
        if (outTeam) *outTeam = t;
        return ncclSuccess;
      }
      break; // Need to enable multimem
    }
  }

  if (multimem) {
    if (!comm->nvlsSupport) {
      WARN("Multicast support requested for team but none available on system.");
      ret = ncclInvalidArgument;
      goto fail;
    } else {
    #if CUDART_VERSION >= 12010
      CUmemGenericAllocationHandle mcHandle = 0;
      CUdeviceptr mcAddr = 0;
      CUmulticastObjectProp mcProp = {};
      char shareableHandle[NVLS_HANDLE_SIZE];

      mcProp.numDevices = team.nRanks;
      mcProp.handleTypes = ncclCuMemHandleType;
      mcProp.flags = 0;
      mcProp.size = devr->bigSize;
      if (team.rank == 0) {
        NCCLCHECKGOTO(ncclNvlsGroupCreate(comm, &mcProp, team.rank, team.nRanks, &mcHandle, shareableHandle), ret, fail);
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail_mcHandle);
      } else {
        NCCLCHECKGOTO(bootstrapIntraNodeBroadcast(comm->bootstrap, t->worldRankList, team.rank, team.nRanks, 0, shareableHandle, NVLS_HANDLE_SIZE), ret, fail);
        NCCLCHECKGOTO(ncclNvlsGroupConnect(comm, shareableHandle, t->worldRankList[0], &mcHandle), ret, fail);
      }

      CUCHECKGOTO(cuMulticastAddDevice(mcHandle, comm->cudaDev), ret, fail_mcHandle);
      CUCHECKGOTO(cuMemAddressReserve(&mcAddr, devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail_mcHandle);
      CUCHECKGOTO(cuMemMap(mcAddr, devr->bigSize, 0, mcHandle, 0), ret, fail_mcHandle_mcAddr);
      { CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = comm->cudaDev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECKGOTO(cuMemSetAccess(mcAddr, devr->bigSize, &accessDesc, 1), ret, fail_mcHandle_mcAddr_unmap);
      }
      t->mcHandle = mcHandle;
      t->mcBasePtr = reinterpret_cast<void*>(mcAddr);

      // Bind new team with all existing memories.
      for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
        NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mcHandle_mcAddr_unmap_mems);
      }

      if (false) { // Error labels:
      fail_mcHandle_mcAddr_unmap_mems:
        for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
          symUnbindTeamMemory(comm, t, mem);
        }
      fail_mcHandle_mcAddr_unmap:
        CUCHECKIGNORE(cuMemUnmap(mcAddr, devr->bigSize));
        goto fail_mcHandle_mcAddr; // silence unused label warning
      fail_mcHandle_mcAddr:
        CUCHECKIGNORE(cuMemAddressFree(mcAddr, devr->bigSize));
        goto fail_mcHandle; // silence unused label warning
      fail_mcHandle:
        CUCHECKIGNORE(cuMemRelease(mcHandle));
        goto fail; // silence unused label warning
      }
    #else
      goto fail; // silence unused label warning
    #endif
    }
  }

  if (teamIsNew) {
     // Add to list
    t->next = devr->teamHead;
    devr->teamHead = t;
  }
  if (outTeam) *outTeam = t;
  return ret;

fail:
  if (teamIsNew) free(t);
  return ret;
}

static void symTeamDestroyAll(struct ncclComm* comm) {
  struct ncclDevrState* devr = &comm->devrState;
  while (devr->teamHead != nullptr) {
    struct ncclDevrTeam* t = devr->teamHead;
    devr->teamHead = t->next;
    if (t->mcBasePtr != nullptr) {
      for (struct ncclDevrMemory* m = devr->memHead; m != nullptr; m = m->next) {
        symUnbindTeamMemory(comm, t, m);
      }
      CUdeviceptr mcAddr = reinterpret_cast<CUdeviceptr>(t->mcBasePtr);
      CUCHECKIGNORE(cuMemUnmap(mcAddr, devr->bigSize));
      CUCHECKIGNORE(cuMemAddressFree(mcAddr, devr->bigSize));
      CUCHECKIGNORE(cuMemRelease(t->mcHandle));
    }
    free(t);
  }
}

static ncclResult_t symMemoryRegisterGin(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  //struct ncclDevrState* devr = &comm->devrState;
  NCCLCHECK(ncclGinConnectOnce(comm));
  NCCLCHECK(ncclGinRegister(comm, mem->primaryAddr, mem->size, mem->ginHostWins, mem->ginDevWins));
  return ncclSuccess;
}

// On success we take caller's reference on memHandle.
// Due to multicast binds for each pre-exiting team, this function requires
// caller do a world barrier before returning to user.
static ncclResult_t symMemoryObtain(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, void* memAddr, size_t size,
    struct ncclDevrMemory** outMem
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  int64_t bigOffset = 0;

  struct ncclDevrMemory* mem = devr->memHead;
  while (mem != nullptr) {
    if (mem->memHandle == memHandle) {
      CUCHECKIGNORE(cuMemRelease(memHandle));
      goto leave;
    }
    mem = mem->next;
  }

  // New memory.
  mem = (struct ncclDevrMemory*)malloc(sizeof(struct ncclDevrMemory));
  mem->refCount = 0;
  mem->memHandle = memHandle;
  mem->primaryAddr = memAddr;
  mem->size = size;
 
  // Grab offset in the big space.
  NCCLCHECKGOTO(ncclSpaceAlloc(&devr->bigSpace, devr->bigSize, size, devr->granularity, &bigOffset), ret, fail_mem);
  mem->bigOffset = bigOffset;

  // Map unicast addresses into flat VA space for lsa team.
  NCCLCHECKGOTO(symMemoryMapLsaTeam(comm, memHandle, size, bigOffset), ret, fail_mem_space);

  // If our caller doesn't have a VA then we'll use the LSA mapping.
  if (mem->primaryAddr == nullptr) {
    mem->primaryAddr = (char*)devr->lsaFlatBase + devr->lsaSelf*devr->bigSize + mem->bigOffset;
  }

  // Bind new memory with each existing team.
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mem_space_teams);
  }

  if (devr->ginEnabled) {
    NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail_mem_space_teams);
  }

  // Add to list of mems.
  mem->next = devr->memHead;
  devr->memHead = mem;

leave:
  mem->refCount += 1;
  *outMem = mem;
  return ret;

fail_mem_space_teams:
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    symUnbindTeamMemory(comm, t, mem);
  }
fail_mem_space:
  ncclSpaceFree(&devr->bigSpace, bigOffset, size);
fail_mem:
  free(mem);
//fail:
  return ret;
}

static void symMemoryDropRef(
    struct ncclComm* comm, struct ncclDevrMemory* mem
  ) {
  if (mem != nullptr && 0 == --mem->refCount) {
    struct ncclDevrState* devr = &comm->devrState;
    if (devr->ginEnabled) {
      ncclGinDeregister(comm, mem->ginHostWins);
    }
    for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
      symUnbindTeamMemory(comm, t, mem);
    }
    for (int r = 0; r < devr->lsaSize; r++) {
      CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r*devr->bigSize + mem->bigOffset);
      CUCHECKIGNORE(cuMemUnmap(addr, mem->size));
    }
    ncclSpaceFree(&devr->bigSpace, mem->bigOffset, mem->size);
    CUCHECKIGNORE(cuMemRelease(mem->memHandle));

    struct ncclDevrMemory** ptr = &devr->memHead;
    while (*ptr != mem) ptr = &(*ptr)->next;
    *ptr = mem->next; // Remove from list.

    free(mem);
  }
}

static ncclResult_t symWindowTableInitOnce(struct ncclComm* comm, cudaStream_t stream) {
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  if (tableDev == nullptr) { // Create on first need.
    NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableDev, nullptr, stream));
    devr->windowTable = tableDev;
  }
  return ncclSuccess;
}

// On success we take callers reference on `mem`.
static ncclResult_t symWindowCreate(
    struct ncclComm* comm, struct ncclDevrMemory* mem,
    size_t memOffset, void* userPtr, size_t userSize, int winFlags, void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrWindow** outWin,
    cudaStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrWindow* win;

  win = (struct ncclDevrWindow*)malloc(sizeof(struct ncclDevrWindow));
  memset(win, 0, sizeof(*win));
  win->memory = mem;
  win->size = userSize;
  win->bigOffset = mem->bigOffset + memOffset;
  win->winFlags = winFlags;
  win->localRegHandle = localReg;
  if (userPtr == nullptr) {
    // Null means caller has no VA and will use the lsa team flat VA address.
    win->userPtr = (char*)devr->lsaFlatBase + (devr->lsaSelf*devr->bigSize) + mem->bigOffset;
  } else {
    win->userPtr = userPtr;
  }

  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;
  NCCLCHECK(ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream));
  win->vidmem = winDev;
  winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset;
  winDevHost->mcOffset4K = win->bigOffset>>12;
  winDevHost->stride4G = devr->bigSize>>32;
  winDevHost->lsaRank = devr->lsaSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;
  winDevHost->ginOffset4K = memOffset>>12;
  for (int i=0; i < NCCL_GIN_MAX_CONTEXTS; i++) {
    winDevHost->ginWins[i] = mem->ginDevWins[i];
  }
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream));

  NCCLCHECK(symWindowTableInitOnce(comm, stream)); // ensure devr->windowTable exists
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  struct ncclDevCommWindowTable* tableHost;
  NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost));
  while (true) {
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userAddr + userSize;
      tableHost->entries[i].window = winDev;
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i], sizeof(tableHost->entries[i]), cudaMemcpyHostToDevice, stream));
      break;
    }
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableHost->next, nullptr, stream));
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next), cudaMemcpyHostToDevice, stream));
    }
    tableDev = tableHost->next;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableHost->next, &tableHost));
  }

  { // insert into winSorted[]
    int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
    struct ncclDevrWindowSorted winSort;
    winSort.userAddr = userAddr;
    winSort.size = userSize;
    winSort.win = win;
    listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount, i, winSort);
  }

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}

static ncclResult_t symWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclWindow_vidmem* winDevHost;
  struct ncclDevrWindow* winHost;

  NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclDevrWindow*)winDevHost->winHost;

  symMemoryDropRef(comm, winHost->memory);

  { struct ncclDevCommWindowTable* tableDev = devr->windowTable;
    struct ncclDevCommWindowTable* tableHost;
    NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost), ret, remove_winSorted);
    while (true) {
      int i = 0;
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        CUDACHECKGOTO(cudaMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream), ret, remove_winSorted);
        break;
      }
      if (tableHost->next == nullptr) break; // Error didn't find window in table
      tableDev = tableHost->next;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableHost->next, &tableHost), ret, remove_winSorted);
    }
  }
  NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDev, stream), ret, remove_winSorted);

  NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);

remove_winSorted:
  { int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1; // least upper bound is just after ours.
    listRemove(devr->winSorted, &devr->winSortedCount, i);
  }
  free(winHost);
fail:
  return ret;
}

ncclResult_t ncclDevrWindowRegisterInGroup(
    struct ncclComm* comm,
    void* userPtr, size_t userSize, int winFlags, ncclWindow_t* outWinDev
  ) {
  ncclResult_t ret = ncclSuccess;
  CUdeviceptr memAddr = 0;
  size_t memSize = 0;
  CUmemGenericAllocationHandle memHandle = 0x0;
  size_t memOffset;
  struct ncclDevrMemory* mem = nullptr;
  cudaStream_t stream = nullptr;
  void* localRegHandle = nullptr;

  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle), ret, fail);

  if (!comm->symmetricSupport) {
    // We just return the local registration handle directly in this case, as there's no reason to allocate the
    // ncclWindow_vidmem structure on the device, etc.
    *outWinDev = reinterpret_cast<struct ncclWindow_vidmem*>(localRegHandle);
    return ncclSuccess;
  }
  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    // Defer symmetric kernel init until at least one window with that flag exists.
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail);
  }

  // Get underlying cumem handle:
  CUCHECKGOTO(cuMemGetAddressRange(&memAddr, &memSize, reinterpret_cast<CUdeviceptr>(userPtr)), ret, fail_locReg);
  memOffset = reinterpret_cast<CUdeviceptr>(userPtr) - memAddr;
  if (memOffset%NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUCHECKGOTO(cuMemRetainAllocationHandle(&memHandle, reinterpret_cast<void*>(memAddr)), ret, fail_locReg);

  // Trade cumem handle for ncclDevrMemory*
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, (void*)memAddr, memSize, &mem), ret, fail_locReg_memHandle);
  memHandle = 0x0; // symMemoryObtain took our reference

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  NCCLCHECKGOTO(symWindowCreate(
      comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle, outWinDev, nullptr, stream
    ), ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference
  
  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_locReg_memHandle_mem_stream_win);

  // symWindowCreate needs barrier.
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_locReg_memHandle_mem_stream_win);

  cudaStreamDestroy(stream);
  return ret;

fail_locReg_memHandle_mem_stream_win:
  symWindowDestroy(comm, *outWinDev, stream);
  *outWinDev = nullptr;
  cudaStreamSynchronize(stream);
fail_locReg_memHandle_mem_stream:
  cudaStreamDestroy(stream);
  symMemoryDropRef(comm, mem);
fail_locReg_memHandle:
  if (memHandle != 0x0) { CUCHECKIGNORE(cuMemRelease(memHandle)); }
fail_locReg:
  ncclCommDeregister(comm, localRegHandle);
fail:
  *outWinDev = nullptr;
  return ret;
}

static ncclResult_t deepCopyDevCommRequirements(
    struct ncclDevCommRequirements const* src,
    struct ncclDevCommRequirements** dst
) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevResourceRequirements **dstRes;
  struct ncclTeamRequirements **dstTeam;

  NCCLCHECK(ncclCalloc(dst, 1));

  /* copy the entire struct now and update linked lists later */
  **dst = *src;

  dstRes = &(*dst)->resourceRequirementsList;
  for (struct ncclDevResourceRequirements* rr = src->resourceRequirementsList; rr != nullptr; rr = rr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstRes, 1), ret, fail);
    (*dstRes)->bufferSize = rr->bufferSize;
    (*dstRes)->bufferAlign = rr->bufferAlign;
    (*dstRes)->outBufferHandle = rr->outBufferHandle;
    dstRes = &(*dstRes)->next;
  }

  dstTeam = &(*dst)->teamRequirementsList;
  for (struct ncclTeamRequirements* tr = src->teamRequirementsList; tr != nullptr; tr = tr->next) {
    NCCLCHECKGOTO(ncclCalloc(dstTeam, 1), ret, fail);
    (*dstTeam)->team = tr->team;
    (*dstTeam)->multimem = tr->multimem;
    (*dstTeam)->outMultimemHandle = tr->outMultimemHandle;
    dstTeam = &(*dstTeam)->next;
  }

exit:
  return ret;
fail:
  freeDevCommRequirements(*dst);
  *dst = nullptr;
  goto exit;
}

void freeDevCommRequirements(
    struct ncclDevCommRequirements* reqs
) {
  if (reqs) {
    while (reqs->resourceRequirementsList) {
      struct ncclDevResourceRequirements* rr_next = reqs->resourceRequirementsList->next;
      free(reqs->resourceRequirementsList);
      reqs->resourceRequirementsList = rr_next;
    }

    while (reqs->teamRequirementsList) {
      struct ncclTeamRequirements* tr_next = reqs->teamRequirementsList->next;
      free(reqs->teamRequirementsList);
      reqs->teamRequirementsList = tr_next;
    }

    free(reqs);
  }
}

ncclResult_t ncclDevrCommCreateInternal(
    struct ncclComm* comm,
    struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclTeam world = ncclTeamWorld(comm);
  struct ncclTeam lsa = ncclTeamInnerFactor(world, devr->lsaSize);
  bool ginActivated = false;
  struct ncclDevrTeam* tmLsa;
  size_t bufSizeTotal;
  int nGinContexts = 0;
  int ginSignalTotal = 0, ginCounterTotal = 0;
  struct ncclDevResourceRequirements* resReqsHead;
  struct ncclDevResourceRequirements lsaBarReq;
  cudaStream_t stream = nullptr;
  struct ncclDevResourceRequirements railGinBarrierReq;
  CUmemGenericAllocationHandle memHandle = 0x0;
  struct ncclDevrMemory* mem = nullptr;
  struct ncclDevrWindow* win = nullptr;
  struct ncclWindow_vidmem* winHost = nullptr;
  size_t ginSignalShadowsOffset = 0;

  if (comm->nNodes > 1 || reqs->ginForceEnable || reqs->ginCounterCount != 0 || reqs->ginSignalCount != 0) {
    ginActivated = !devr->ginEnabled;
    devr->ginEnabled = true;
  }

  if (ginActivated) {
    NCCLCHECKGOTO(ncclGinConnectOnce(comm), ret, fail);
    nGinContexts = comm->sharedRes->ginState.ginCommCount;
    // Register all preexisting memories with GIN. Update the windows later when
    // we have a stream.
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr; mem = mem->next) {
      NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail);
    }
  }

  memset(outDevComm, 0, sizeof(*outDevComm));
  outDevComm->rank = comm->rank;
  outDevComm->nRanks = comm->nRanks;
  outDevComm->nRanks_rcp32 = idivRcp32(comm->nRanks);
  outDevComm->lsaRank = devr->lsaSelf;
  outDevComm->lsaSize = devr->lsaSize;
  outDevComm->lsaSize_rcp32 = idivRcp32(devr->lsaSize);

  NCCLCHECKGOTO(symTeamObtain(comm, lsa, /*multicast=*/reqs->lsaMultimem, &tmLsa), ret, fail);
  outDevComm->lsaMultimem.mcBasePtr = tmLsa->mcBasePtr;

  { struct ncclTeamRequirements* tr = reqs->teamRequirementsList;
    while (tr != nullptr) {
      if (tr->multimem) {
        struct ncclDevrTeam* tm;
        NCCLCHECKGOTO(symTeamObtain(comm, tr->team, tr->multimem, &tm), ret, fail);
        if (tr->outMultimemHandle != nullptr) tr->outMultimemHandle->mcBasePtr = tm->mcBasePtr;
      }
      tr = tr->next;
    }
  }

  resReqsHead = reqs->resourceRequirementsList;

  ncclLsaBarrierCreateRequirement(lsa, std::max(reqs->barrierCount, reqs->lsaBarrierCount), &outDevComm->lsaBarrier, &lsaBarReq);
  lsaBarReq.next = resReqsHead;
  resReqsHead = &lsaBarReq;

  ncclGinBarrierCreateRequirement(comm, ncclTeamRail(comm), std::max(reqs->barrierCount, reqs->railGinBarrierCount), &outDevComm->railGinBarrier, &railGinBarrierReq);
  railGinBarrierReq.next = resReqsHead;
  resReqsHead = &railGinBarrierReq;

  { struct ncclDevResourceRequirements* rr = resReqsHead;
    bufSizeTotal = 0;
    ginSignalTotal = reqs->ginSignalCount;
    ginCounterTotal = reqs->ginCounterCount;
    nGinContexts = comm->sharedRes->ginState.ginCommCount;
    while (rr != nullptr) {
      bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));
      if (rr->outBufferHandle != nullptr) *rr->outBufferHandle = bufSizeTotal/128;
      if (rr->outGinSignalStart != nullptr) *rr->outGinSignalStart = ginSignalTotal;
      if (rr->outGinCounterStart != nullptr) *rr->outGinCounterStart = ginCounterTotal;
      bufSizeTotal += rr->bufferSize;
      ginSignalTotal += rr->ginSignalCount;
      ginCounterTotal += rr->ginCounterCount;
      rr = rr->next;
    }
    bufSizeTotal= alignUp(bufSizeTotal, 128);
    ginSignalShadowsOffset = bufSizeTotal;
    bufSizeTotal += nGinContexts*ginSignalTotal*sizeof(uint64_t); // include signal shadows
    bufSizeTotal = alignUp(bufSizeTotal, devr->granularity);
  }

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  if (ginActivated) {
    // Now update the GIN handles in all existing windows. Registration of memories happened above.
    for (int i=0; i < devr->winSortedCount; i++) {
      struct ncclDevrWindow* win = devr->winSorted[i].win;
      struct ncclWindow_vidmem* winHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream);
      winHost->ginOffset4K = (win->bigOffset - win->memory->bigOffset)>>12;
      for (int i=0; i < NCCL_GIN_MAX_CONTEXTS; i++) {
        winHost->ginWins[i] = win->memory->ginDevWins[i];
      }
      CUDACHECK(cudaMemcpyAsync(win->vidmem, winHost, sizeof(struct ncclWindow_vidmem), cudaMemcpyHostToDevice, stream));
    }
  }

  NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail_stream); // ensure devr->windowTable exists
  outDevComm->windowTable = devr->windowTable;

  if (bufSizeTotal == 0) {
    outDevComm->resourceWindow = nullptr;
    outDevComm->resourceWindow_inlined = {};
  } else {
    CUmemAllocationProp memProp = {};
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memProp.requestedHandleTypes = ncclCuMemHandleType;
    // We have to assume that if GIN is possible it might be requested in the future,
    // even on single node.
    memProp.allocFlags.gpuDirectRDMACapable = comm->sharedRes->ginState.ncclGin != nullptr ? 1 : 0;
    memProp.location.id = comm->cudaDev;

    CUCHECKGOTO(cuMemCreate(&memHandle, bufSizeTotal, &memProp, 0), ret, fail);

    NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, NULL, bufSizeTotal, &mem), ret, fail);
    memHandle = 0x0; // Reference given to symMemoryObtain

    NCCLCHECKGOTO(symWindowCreate( // Requires world barrier afterward.
      comm, mem, /*memOffset=*/0, nullptr, bufSizeTotal, /*winFlags=*/0,
      /*localReg=*/nullptr, &outDevComm->resourceWindow, &win,
      stream), ret, fail);
    mem = nullptr; // Reference given to symWindowCreate
    NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost), ret, fail_stream_mem_win);
    outDevComm->resourceWindow_inlined = *winHost;
    outDevComm->ginSignalShadows = (uint64_t*)add4G((char*)winHost->lsaFlatBase + ginSignalShadowsOffset, winHost->lsaRank*winHost->stride4G);

    CUDACHECKGOTO(cudaMemsetAsync(win->userPtr, 0, bufSizeTotal, stream), ret, fail);
  }

  if (devr->ginEnabled) {
    outDevComm->ginSignalCount = ginSignalTotal;
    outDevComm->ginCounterCount = ginCounterTotal;
    NCCLCHECKGOTO(ncclGinAllocSignalsCounters(comm,
      ginSignalTotal, &outDevComm->ginSignalBase,
      ginCounterTotal, &outDevComm->ginCounterBase
    ), ret, fail_stream_mem_win);

    for (int ctx=0; ctx < nGinContexts; ctx++) {
      outDevComm->ginTypes[ctx] = (int)comm->sharedRes->ginState.ginDevHandles[ctx]->netDeviceType;
      outDevComm->ginHandles[ctx] = comm->sharedRes->ginState.ginDevHandles[ctx]->handle;
    }
  }

  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail);

  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef), ret, fail_stream_mem_win_signals);
  CUDACHECKGOTO(cudaStreamDestroy(stream), ret, fail);
  return ret;

fail_stream_mem_win_signals:
  if (devr->ginEnabled) {
    ncclGinFreeSignalsCounters(comm,
      outDevComm->ginSignalBase, outDevComm->ginSignalCount,
      outDevComm->ginCounterBase, outDevComm->ginCounterCount
    );
  }
fail_stream_mem_win:
  symWindowDestroy(comm, win->vidmem, stream);
  cudaStreamSynchronize(stream);
  symMemoryDropRef(comm, mem);
fail_stream:
  cudaStreamDestroy(stream);
fail:
  return ret;
}

////////////////////////////////////////////////////////////////////////////////

NCCL_API(ncclResult_t, ncclCommWindowRegister, ncclComm_t comm, void* ptr, size_t size, ncclWindow_t* win, int winFlags);
ncclResult_t ncclCommWindowRegister(
    struct ncclComm* comm, void* userPtr, size_t userSize,
    struct ncclWindow_vidmem** outWinDev, int winFlags
  ) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrRegTask* task;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  if (userPtr == nullptr || userSize == 0 || !(comm->symmetricSupport || ncclParamLocalRegister())) goto exit;

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = userPtr;
  task->userSize = userSize;
  task->winFlags = winFlags;
  task->outWinDev = outWinDev;
  ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclCommWindowDeregister(struct ncclComm* comm, struct ncclWindow_vidmem* winDev) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  cudaStream_t stream;

  if (winDev == nullptr) goto exit;

  if (!comm->symmetricSupport) {
    NCCLCHECKGOTO(ncclCommDeregister(comm, winDev), ret, fail);
    goto exit;
  }
  CUDACHECKGOTO(cudaGetDevice(&saveDev), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail_dev);
  NCCLCHECKGOTO(symWindowDestroy(comm, winDev, stream), ret, fail_dev_stream);
fail_dev_stream:
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
fail_dev:
  cudaSetDevice(saveDev);
fail:
exit:
  return ret;
}

ncclResult_t ncclDevrFindWindow(
    struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin
  ) {
  struct ncclDevrState* devr = &comm->devrState;
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted, devr->winSortedCount, userAddr);
  if (0 < i && (userAddr - devr->winSorted[i-1].userAddr < devr->winSorted[i-1].size)) {
    *outWin = devr->winSorted[i-1].win;
  } else {
    *outWin = nullptr;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclDevCommCreate, ncclComm_t comm, ncclDevCommRequirements_t const* reqs, ncclDevComm_t* outDevComm);
ncclResult_t ncclDevCommCreate(
    ncclComm_t comm, struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* outDevComm
  ) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrCommCreateTask* task = nullptr;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  // reqs must be deep copied to the task so background threads can safely access it
  NCCLCHECKGOTO(deepCopyDevCommRequirements(reqs, &task->reqs), ret, fail);
  task->outDevComm = outDevComm;
  ncclIntruQueueEnqueue(&comm->devrState.commCreateTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  free(task);
  goto exit;
}

NCCL_API(ncclResult_t, ncclDevCommDestroy, ncclComm_t comm, ncclDevComm_t const* devComm);
ncclResult_t ncclDevCommDestroy(
    struct ncclComm* comm, struct ncclDevComm const* devComm
  ) {
  struct ncclDevrState* devr = &comm->devrState;
  if (devr->ginEnabled) {
    ncclGinFreeSignalsCounters(comm,
      devComm->ginSignalBase, devComm->ginSignalCount,
      devComm->ginCounterBase, devComm->ginCounterCount
    );
  }
  if (devComm->resourceWindow != nullptr) {
    NCCLCHECK(ncclCommWindowDeregister(comm, devComm->resourceWindow));
  }
  return ncclSuccess;
}


// Get the corresponding pointer in another lsa rank's symmetric memory window
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank, void** outPtr) {
  if (winHost == nullptr || outPtr == nullptr) {
    return ncclInvalidArgument;
  }

  struct ncclDevrState* devr = &comm->devrState;
  
  // Validate lsaRank is within bounds
  if (lsaRank < 0 || lsaRank >= devr->lsaSize) {
    return ncclInvalidArgument;
  }

  // Validate offset is within bounds
  if (offset < 0 || offset >= winHost->size) {
    return ncclInvalidArgument;
  }

  // Calculate the address with offset for the specified lsa rank
  *outPtr = (void*)((uintptr_t)devr->lsaFlatBase + lsaRank * devr->bigSize + winHost->bigOffset + offset);
  return ncclSuccess;
}

// Get the multicast address for a given team
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, struct ncclTeam lsaTeam, void** outPtr){
  if (winHost == nullptr || outPtr == nullptr) {
    return ncclInvalidArgument;
  }

  if (!comm->nvlsSupport) {
    return ncclInvalidUsage;
  }

  bool multimem = true;
  struct ncclDevrTeam* tm;
  NCCLCHECK(symTeamObtain(comm, lsaTeam, multimem, &tm));
    
  // Return the base multicast address for this team with offset
  *outPtr = (void*)((uintptr_t)tm->mcBasePtr + winHost->bigOffset + offset);
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

// Find the least index strictly greater than arg.
template<typename Obj, typename Key>
static int listFindSortedLub(Key Obj::*key, Obj* sorted, int count, Key arg) {
  int lo = 0, hi = count;
  while (lo + 16 < hi) {
    int i = (lo + hi)/2;
    if (sorted[i].*key <= arg) lo = i+1;
    else hi = i;
  }
  int i = lo;
  while (i < hi && sorted[i].*key <= arg) i++;
  return i;
}

template<typename Obj>
static void listInsert(Obj** list, int* capacity, int* count, int index, Obj val) {
  if (*capacity < *count + 1) {
    *capacity *= 2;
    if (*capacity == 0) *capacity = 16;
    *list = (Obj*)realloc(*list, (*capacity)*sizeof(Obj));
  }
  for (int j = *count; j != index; j--) {
    (*list)[j] = (*list)[j-1];
  }
  (*list)[index] = val;
  *count += 1;
}

template<typename Obj>
static void listRemove(Obj* list, int* count, int index) {
  for (int i = index; i+1 < *count; i++) {
    list[i] = list[i+1];
  }
  *count -= 1;
}

