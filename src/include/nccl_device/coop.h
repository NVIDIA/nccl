/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_COOP_H_
#define _NCCL_DEVICE_COOP_H_
#include "utility.h"

// ncclCoop[Foo]: NCCL's versions of CUDA's Cooperative Groups. They conform
// to just this subset of the CUDA API:
//   int Coop::thread_rank();
//   int Coop::size();
//   int Coop::num_threads(); // same as size()
//   void Coop::sync();

#if __CUDACC__
template<int nThreadsPow2>
struct ncclCoopTile { // An aligned pow2 set of threads within the warp.
  static_assert(nccl::utility::isPow2(nThreadsPow2) && nThreadsPow2 <= 32, "Condition required");

  NCCL_DEVICE_INLINE int thread_rank() const {
    return nccl::utility::lane() % nThreadsPow2;
  }
  NCCL_DEVICE_INLINE constexpr int size() const { return nThreadsPow2; }
  NCCL_DEVICE_INLINE constexpr int num_threads() const { return nThreadsPow2; }

  NCCL_DEVICE_INLINE uint32_t laneMask() const {
    return (-1u>>(32-nThreadsPow2))<<(nccl::utility::lane() & -nThreadsPow2);
  }
  NCCL_DEVICE_INLINE void sync() {
    if (nThreadsPow2 > 1) __syncwarp(laneMask());
  }
};
#endif

#if __CUDACC__
typedef ncclCoopTile<1> ncclCoopThread;
typedef ncclCoopTile<32> ncclCoopWarp;
#endif

#if __CUDACC__
struct ncclCoopLanes { // Some lanes of this warp.
  uint32_t lmask;

  NCCL_DEVICE_INLINE constexpr ncclCoopLanes(uint32_t lmask=-1u): lmask(lmask) {}

  NCCL_DEVICE_INLINE int thread_rank() const {
    return __popc(lmask & nccl::utility::lanemask_lt());
  }
  NCCL_DEVICE_INLINE int size() const {
    return __popc(lmask);
  }
  NCCL_DEVICE_INLINE int num_threads() const {
    return __popc(lmask);
  }
  NCCL_DEVICE_INLINE void sync() {
    __syncwarp(lmask);
  }
};
#endif

#if __CUDACC__
// A set of consecutive warps that the user has also supplied with a unique
// id from [0..15]. It is an error for two different warp spans with the same
// id to be in a collective concurrently.
struct ncclCoopWarpSpan {
  uint32_t warp0:8, nWarps:8, id:8;

  NCCL_DEVICE_INLINE constexpr ncclCoopWarpSpan(int warp0, int nWarps, int id):
    warp0(warp0), nWarps(nWarps), id(id) {
  }

  NCCL_DEVICE_INLINE int thread_rank() const {
    return threadIdx.x - 32*warp0;
  }
  NCCL_DEVICE_INLINE int size() const {
    return 32*nWarps;
  }
  NCCL_DEVICE_INLINE int num_threads() const {
    return 32*nWarps;
  }

  NCCL_DEVICE_INLINE void sync() {
    //asm volatile("barrier.sync %0, %1;" :: "r"(1+id), "r"(32*nWarps) : "memory");
    __barrier_sync_count(1+id, 32*nWarps);
  }
};
#endif

#if __CUDACC__
struct ncclCoopCta {
  NCCL_DEVICE_INLINE int thread_rank() const { return threadIdx.x; }
  NCCL_DEVICE_INLINE int size() const { return blockDim.x; }
  NCCL_DEVICE_INLINE int num_threads() const { return blockDim.x; }
  NCCL_DEVICE_INLINE void sync() { __syncthreads(); }
};
#endif

#if __CUDACC__
template<int nThreadsPow2>
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopTile<nThreadsPow2> coop) {
  return coop.laneMask();
}
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopLanes coop) {
  return coop.lmask;
}
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopWarpSpan coop) {
  return -1u;
}
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopCta coop) {
  return -1u;
}
#endif

#if __CUDACC__
// ncclCoopIsThread:
// At compile time do we know the given coop is a single thread only.
template<int nThreads>
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopTile<nThreads>) {
  return nThreads == 1;
}
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopLanes) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopWarpSpan) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopCta) { return false; }
#endif

#if __CUDACC__
template<int nThreads>
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopTile<nThreads>) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopLanes) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopWarpSpan) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopCta) { return false; }
#endif

#if __CUDACC__
// Pick threads of our warp that are safe to use collectively.
NCCL_DEVICE_INLINE ncclCoopLanes ncclCoopCoalesced() {
  return ncclCoopLanes{__activemask()};
}
#endif

#if __CUDACC__
// Pick threads of our warp that are safe to use collectively given that this
// is a collective on the provided cooperative group.
template<typename Coop>
NCCL_DEVICE_INLINE ncclCoopTile<32> ncclCoopCoalesced(Coop) {
  return ncclCoopTile<32>();
}
NCCL_DEVICE_INLINE ncclCoopLanes ncclCoopCoalesced(ncclCoopLanes coop) {
  return coop;
}
template<int nThreads>
NCCL_DEVICE_INLINE ncclCoopTile<nThreads> ncclCoopCoalesced(ncclCoopTile<nThreads> coop) {
  return coop;
}
#endif

#if __CUDACC__
template<int nThreads, typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopTile<nThreads>, T value, int root, bool entrySync=true) {
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(-1u, u[i], root, nThreads);
  return v;
}
template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopLanes coop, T value, int root, bool entrySync=true) {
  uint32_t m = coop.lmask;
  uint32_t r = root == 0 ? __ffs(m)-1 : __fns(m, 0, 1+root);
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(m, u[i], r);
  return v;
}

NCCL_DEVICE_INLINE ulong2* ncclCoopBcast_WarpSpan_stash() {
  __shared__ ulong2 stash[15];
  return stash;
}

template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopWarpSpan coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(ncclCoopBcast_WarpSpan_stash()[0]), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)&ncclCoopBcast_WarpSpan_stash()[coop.id] = value;
  coop.sync();
  return *(T*)&ncclCoopBcast_WarpSpan_stash()[coop.id];
}

NCCL_DEVICE_INLINE ulong2* ncclCoopBcast_Cta_stash() {
  __shared__ ulong2 stash;
  return &stash;
}

template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopCta coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(*ncclCoopBcast_Cta_stash()), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)ncclCoopBcast_Cta_stash() = value;
  coop.sync();
  return *(T*)ncclCoopBcast_Cta_stash();
}
#endif

#endif
