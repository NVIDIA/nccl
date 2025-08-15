#ifndef _NCCL_DEVICE_COMM__TYPES_H_
#define _NCCL_DEVICE_COMM__TYPES_H_
#include "../comm.h"
#include "core__types.h"
#include "mem_barrier__types.h"
#include "ll_a2a__types.h"

struct ncclDevCommWindowTable {
  struct Entry {
    uintptr_t base, size;
    ncclWindow_t window;
  } entries[32];
  struct ncclDevCommWindowTable* next;
};

struct ncclDevComm {
  int rank, nRanks;
  uint32_t nRanks_rcp32;
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  struct ncclDevCommWindowTable* windowTable;

  ncclWindow_t resourceWindow;
  ncclWindow_vidmem resourceWindow_inlined;

  ncclMultimemHandle multimem;
  ncclLsaBarrierHandle lsaBarrier;
  ncclLLA2AHandle lsaLLA2A;
};

#endif // _NCCL_DEVICE_COMM__TYPES_H_
