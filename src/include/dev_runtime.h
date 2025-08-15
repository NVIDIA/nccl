#ifndef NCCL_DEVICE_RUNTIME_H_
#define NCCL_DEVICE_RUNTIME_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "allocator.h"
#include "bitops.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////////////////
// ncclDevr[_]: runtime implements for symmetric API.

struct ncclDevrMemory;
struct ncclDevrWindow {
  struct ncclDevrMemory* memory;
  void* userPtr;
  size_t size;
  size_t bigOffset; // Offset in big VA space.
  int winFlags;
  void* localRegHandle;
  struct ncclWindow_vidmem* vidmem;
};
struct ncclDevrWindowSorted;
struct ncclDevrTeam;

struct ncclDevrRegTask {
  struct ncclDevrRegTask *next;
  void* userPtr;
  size_t userSize;
  int winFlags;
  ncclWindow_t* outWinDev;
};

struct ncclDevrState {
  // Like localRank/localRanks except "lsa" ranks must be consecutive in the world
  // and all lsa subsets have the same number of ranks. If any condition is
  // false then the lsa team is just the singleton of self.
  int lsaSelf;
  int lsaSize;
  int* lsaRankList;

  size_t granularity; // cuMemGetAllocationGranularity
  struct ncclDevrMemory* memHead;
  struct ncclDevrWindowSorted* winSorted;
  int winSortedCapacity, winSortedCount;
  struct ncclDevrTeam* teamHead;
  size_t bigSize; // size of our big logical space (128GB?)
  struct ncclSpace bigSpace; // allocates our big VA space.
  void* lsaFlatBase; // base ptr for all lsa ranks big VA's concatenated together: size = lsaRanks*bigSize
  struct ncclShadowPool shadows;
  struct ncclDevCommWindowTable* windowTable;

  struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;
};

// We assume ncclComm has a `ncclDevrState symState` member.
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm);
ncclResult_t ncclDevrFinalize(struct ncclComm* comm);

// If found *outWinHost will be populated and *outWinId >= 0, otherwise *outWinId == -1
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin);

ncclResult_t ncclDevrWindowRegisterInGroup(
  struct ncclComm* comm, void* ptr, size_t size, int winFlags, ncclWindow_t* outWinDev
);

// Get the corresponding pointer in another lsa rank's symmetric memory window
ncclResult_t ncclDevrGetLsaRankPtr(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, int lsaRank, void** outPtr);

// Get the multicast address for a given team
ncclResult_t ncclDevrGetLsaTeamPtrMC(struct ncclComm* comm, struct ncclDevrWindow* winHost, size_t offset, struct ncclTeam lsaTeam, void** outPtr);
#endif
