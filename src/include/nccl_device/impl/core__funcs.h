/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_CORE__FUNCS_H_
#define _NCCL_DEVICE_CORE__FUNCS_H_
#include "core__types.h"
#include "comm__types.h"
#include "ptr__types.h"

#if __cplusplus
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const &comm) {
  ncclTeam ans;
  ans.nRanks = comm.nRanks;
  ans.rank = comm.rank;
  ans.stride = 1;
  return ans;
}
#endif

#if __cplusplus
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const &comm) {
  ncclTeam ans;
  ans.nRanks = comm.lsaSize;
  ans.rank = comm.lsaRank;
  ans.stride = 1;
  return ans;
}
#endif

#if __cplusplus
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const& comm) {
  ncclTeam ans;
  ans.nRanks = nccl::utility::idivFast32(comm.nRanks, comm.lsaSize, comm.lsaSize_rcp32);
  ans.rank = nccl::utility::idivFast32(comm.rank, comm.lsaSize, comm.lsaSize_rcp32);
  ans.stride = comm.lsaSize;
  return ans;
}
#endif

NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return 0 <= arank && arank < a.nRanks && amod == 0;
}

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  //uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return arank;
}

#if __cplusplus
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const& comm, ncclTeam tm, int rank) {
  return comm.rank + (rank - tm.rank)*tm.stride;
}
#endif

#if __cplusplus
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const& comm, ncclTeam tm, int rank) {
  return comm.lsaRank + (rank - tm.rank)*tm.stride;
}
#endif

NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize) {
  ncclTeam_t ans;
  ans.nRanks = innerSize;
  ans.rank = parent.rank%innerSize;
  ans.stride = parent.stride;
  return ans;
}

NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize) {
  ncclTeam_t ans;
  ans.nRanks = parent.nRanks/innerSize;
  ans.rank = parent.rank/innerSize;
  ans.stride = parent.stride*innerSize;
  return ans;
}

NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index) {
  int stride = subset.stride/parent.stride;
  int below = parent.rank - subset.rank*stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset.nRanks-1)*stride;
  }
  if (index < below) {
    return index;
  } else if (index-below < (subset.nRanks-1)*(stride-1)) {
    return below + 1 + ((index-below)/(stride-1))*stride + (index-below)%(stride-1);
  } else {
    return below + 1 + (subset.nRanks-1)*stride + (index - below - (subset.nRanks-1)*(stride-1));
  }
}

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int i = nccl::utility::loadConst(&w->lsaRank);
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int i = peer;
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int worldRank = nccl::utility::loadConst(&w->worldRank);
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);
  int i = lsaRank + (peer - worldRank);
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);
  int i = lsaRank + (peer - tm.rank)*tm.stride;
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mm) {
  void* ptr = mm.mcBasePtr;
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + nccl::utility::loadConst(&w->mcOffset4K);
  return (void*)((char*)ptr + offset);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& comm) {
  return ncclGetMultimemPointer(w, offset, comm.lsaMultimem);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclWindow_t ncclFindWindow(Coop coop, ncclDevComm const& comm, void const *ptr) {
  using nccl::utility::loadConst;
  auto coalesced = ncclCoopCoalesced(coop);
  ncclDevCommWindowTable* t = comm.windowTable;
  while (true) {
    bool found = false;
    int index = coalesced.thread_rank();
    #pragma unroll 1
    while (index < 32) {
      uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
      ncclDevCommWindowTable::Entry e = loadConst(&t->entries[index]);
      if ((e.base != 0) && (e.size != 0) && (e.window != 0)) {
        if (uptr - uintptr_t(e.base) < uintptr_t(e.size)) {
          found = true;
          break;
        }
      }
      index += coalesced.size();
    }
    uint32_t mask = __ballot_sync(ncclCoopGetLaneMask(coalesced), found);
    if (mask != 0) {
      int source = __popc(mask-1);
      index = __shfl_sync(ncclCoopGetLaneMask(coalesced), index, source);
      return loadConst(&t->entries[index].window);
    }
    t = loadConst(&t->next);
  }
}
#endif

NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t h) {
  return ((size_t)h)*128;
}

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const& comm, ncclDevResourceHandle h) {
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, comm.lsaRank*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const& comm, ncclDevResourceHandle h, int peer) {
  int r = peer;
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, r*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const& comm, ncclDevResourceHandle h, ncclTeam team, int peer) {
  int r = comm.lsaRank + (peer - team.rank)*team.stride;
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, r*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(ncclDevComm const& comm, ncclDevResourceHandle h, ncclMultimemHandle mm) {
  void* ptr = mm.mcBasePtr;
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + comm.resourceWindow_inlined.mcOffset4K;
  ptr = reinterpret_cast<char(*)[128]>(ptr) + h;
  return ptr;
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const& comm, ncclDevResourceHandle h) {
  return ncclGetResourceBufferMultimemPointer(comm, h, comm.lsaMultimem);
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const& comm, ncclDevResourceHandle h) {
  return ncclSymPtr<char>(comm.resourceWindow, size_t(h)*128);
}
#endif

#endif
