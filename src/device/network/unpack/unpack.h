/*************************************************************************
 * Copyright (c) 2023, Google LLC.  All rights reserved.
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef NET_DEVICE_UNPACK_H
#define NET_DEVICE_UNPACK_H

#include "unpack_defs.h"

#include "op128.h"
#include "align.h"
#include "device.h"
#include "common.h"

// #define ALIGNED_LOAD

inline __device__ void load64gpu(const uint64_t* ptr, uint64_t &v) {
  #if __CUDA_ARCH__ >= 700
      asm volatile("ld.relaxed.gpu.u64 {%0}, [%1];"
      : "=l"(v) : "l"(ptr));
  #else
      asm volatile("ld.volatile.global.u64 {%0}, [%1];"
      : "=l"(v) : "l"(ptr));
  #endif
}

#define PAGE_META_SIZE 16
#define META_LOAD_SIZE 16
#define DATA_LOAD_SIZE 16

// Map internal association of handle with group and peer index (called once at init time)
inline __device__ void ncclNetDeviceUnpackSetup(void* ohandle, const int group, const int index) {
  struct unpackNetDeviceHandle* handle = (struct unpackNetDeviceHandle*) ohandle;
  ncclShmem.groups[group].devicePlugin.unpack.g_meta[index] = handle->meta;
  ncclShmem.devicePlugin.unpack.bounce_buf = handle->bounce_buf;
  ncclShmem.groups[group].devicePlugin.unpack.head = handle->head;
}

inline __device__ void ncclNetDeviceIncrementHead(const int group) {
  ncclShmem.groups[group].devicePlugin.unpack.head++;
}

inline __device__ void ncclNetDeviceSaveHead(void* ohandle, const int group) {
  struct unpackNetDeviceHandle* handle = (struct unpackNetDeviceHandle*) ohandle;
  handle->head = ncclShmem.groups[group].devicePlugin.unpack.head;
}

template <uint8_t sz>
inline __device__ void bulkLoad(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<sz> *reg, const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off){
  bulkLoad<1>(t, len, cpy_src, cpy_dst, reg, w, g_meta, s_meta, src_off, dst_off);
}

template <>
inline __device__ void bulkLoad<1>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<1> reg[16], const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off){
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {

#ifdef ALIGNED_LOAD
    load128 ((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
#pragma unroll
    for (int i=0; i<16; i++) {
      reg[i] = ld_volatile_global<1>((uintptr_t)((uint8_t*)(cpy_src + data_s) + i));
    }
#endif

#pragma unroll
    for (int i=0; i<16; i++) {
      st_global<1>((uintptr_t)((uint8_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<2>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<2> reg[8], const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off){
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128 ((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
#pragma unroll
    for (int i=0; i<8; i++) {
      reg[i] = ld_volatile_global<2>((uintptr_t)((uint16_t*)(cpy_src + data_s) + i));
    }
#endif


#pragma unroll
    for (int i=0; i<8; i++) {
      st_global<2>((uintptr_t)((uint16_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<4>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<4> reg[4], const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off){
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128 ((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
#pragma unroll
    for (int i=0; i<4; i++) {
      reg[i] = ld_volatile_global<4>((uintptr_t)((uint32_t *)(cpy_src + data_s) + i));
    }
#endif

#pragma unroll
    for (int i=0; i<4; i++) {
      st_global<4>((uintptr_t)((uint32_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<8>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<8> reg[2], const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off){
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
#ifdef ALIGNED_LOAD
    load128 ((uint64_t*)(cpy_src + data_s), reg.u64[0], reg.u64[1]);
#else
#pragma unroll
    for (int i=0; i<2; i++) {
      reg[i] = ld_volatile_global<8>((uintptr_t)((uint64_t*)(cpy_src + data_s) + i));
    }
#endif

#pragma unroll
    for (int i=0; i<2; i++) {
      st_global<8>((uintptr_t)((uint64_t*)(cpy_dst + data_s) + i), reg[i]);
    }
  }
}

template <>
inline __device__ void bulkLoad<16>(const int t, const uint32_t len, char* cpy_src, char* cpy_dst, BytePack<16> reg[1], const int w, loadMeta* g_meta, loadMeta* s_meta, uint32_t src_off, uint64_t dst_off){
  uint64_t data_s;
  for (data_s = t * DATA_LOAD_SIZE; data_s + DATA_LOAD_SIZE - 1 < len; data_s += WARP_SIZE * DATA_LOAD_SIZE) {
    reg[0] = ld_volatile_global<16>((uintptr_t)(cpy_src + data_s));
    st_global<16>((uintptr_t)(cpy_dst + data_s), reg[0]);
  }
}

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif
inline __device__ int ppw(const int nbytes, int nw) {
  int v = DIVUP(nbytes, SLICE_PAGE_SIZE);
  v = DIVUP(v, nw);
  while (v > WARP_SHM_PAGE_CNT) {
    v = DIVUP(v, 2);
  }
  return v;
}

// This function is called by all threads
// Pack data from the internal iovec to the supplied flat buffer using all the
// threads
template <int Recv>
inline __device__ void ncclNetDeviceUnpack(
    const int tid, const int tidInBlock, const int nworkers, const int group, int mask, int Src, int workSize);

template <>
inline __device__ void ncclNetDeviceUnpack</*Recv=*/0>(
    const int tid, const int tidInBlock, const int nworkers, const int group, int mask, int Src, int workSize) {
  // send unpack empty
}

inline __device__ void ncclNetDeviceUnpackInner(
    const int tid, const int tidInBlock, const int nworkers, const int group, const int index,
    void *src, const int nbytes, const uint64_t step);

template <>
inline __device__ void ncclNetDeviceUnpack</*Recv=*/1>(
    const int tid, const int tidInBlock, const int nworkers, const int group, int mask, int Src, int workSize) {

  while (mask != 0) {
    int ix = __ffs(mask)-1; // Get the first set bit of the mask (this should correlate to a peer index)
    mask &= mask-1; // Drop the first set bit of the mask

    // Pack data from the internal iovec to the supplied flat srcs buffer using all the threads
    // + Src is necessary in the case of accessing the user buffer directly
    ncclNetDeviceUnpackInner(tid, tidInBlock, nworkers, group /* in case they need to use split warps shared memory partitioning*/,
        ix, ncclShmem.groups[group].srcs[ix + Src], workSize, ncclShmem.groups[group].devicePlugin.unpack.head);
  }
}

inline __device__ void ncclNetDeviceUnpackInner(
    const int tid, const int tidInBlock, const int nworkers, const int group, const int index,
    void *src, const int nbytes, const uint64_t step) {
  // from src/collectives/device/common_kernel.h
  const int w = tid / WARP_SIZE;        // Warp number
  const int nw = nworkers / WARP_SIZE;  // Number of warps
  const int t = tid % WARP_SIZE;        // Thread (inside the warp)

  BytePack<16> reg;
  loadMeta meta;

  uint64_t head;
  struct netUnpackMeta* g_meta_struct;
  void* bounce_buf;

  loadMeta* g_meta;
  loadMeta* s_meta;
  uint64_t meta_cnt;

  // hack head use per-warp
  head          = step;
  g_meta_struct = ncclShmem.groups[group].devicePlugin.unpack.g_meta[index];
  bounce_buf    = ncclShmem.devicePlugin.unpack.bounce_buf;

  __syncwarp();

  head %= NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH;

  g_meta = g_meta_struct->mem[head];

  // Currently, even/odd groups perform send/recv separately. We don't really need space for send side.
  // Total size is N page per warp * 16 B per page * 20 WARPS max = 320 * N bytes, N == WARP_SHM_PAGE_CNT
  static_assert(ncclShmemScratchWarpSize() >= WARP_SHM_SIZE, "Each warp must have enough scratch space");
  s_meta = (loadMeta*) ncclScratchForWarp(tidInBlock / WARP_SIZE); // (loadMeta*) (ncclShmem.devicePlugin.unpack.meta + shm_off);

  load64gpu(g_meta_struct->cnt + head, meta_cnt);

  int PPW = ppw(nbytes, nw);

  for (uint64_t meta_s = w * PPW; meta_s < meta_cnt; meta_s += nw * PPW) {

    uint64_t iter_meta_cnt = meta_cnt - meta_s;
    iter_meta_cnt = iter_meta_cnt < PPW ? iter_meta_cnt : PPW;

    // TODO: this load size needs to work if not aligned, but since the two are both 16...
    if (t < PPW * PAGE_META_SIZE / META_LOAD_SIZE && t < iter_meta_cnt) {  // avoid last iter load garbage data
      load128((const uint64_t*) (g_meta + (meta_s + t)), reg.u64[0], reg.u64[1]);

      storeShmem128(shmemCvtPtr((uint64_t *)(s_meta + (w * PPW + t))), reg.u64[0], reg.u64[1]);
    }

    __syncwarp();

    for (int x = 0; x < iter_meta_cnt; x++) {
      int meta_idx = x + w * PPW;
      
      // load page offs
      loadShmem128(shmemCvtPtr((uint64_t*) (s_meta + meta_idx)), meta.r64[0], meta.r64[1]);

      if (meta.len >= DATA_LOAD_SIZE) {
        // fast path, but need to adapt to alignment issue

        // bulk copy data
        uint8_t align_off = (meta.src_off | meta.dst_off) % DATA_LOAD_SIZE;
        align_off = align_off & -align_off;  // keep the lowest bit
        if (align_off == 0) {  // 0x16
          bulkLoad<16>(t, meta.len, (char*) bounce_buf + meta.src_off, (char*) src + meta.dst_off, &reg, w, g_meta, s_meta, meta.src_off, meta.dst_off);
        } else if (align_off & 0x8) {
          bulkLoad<8>(t, meta.len, (char*) bounce_buf + meta.src_off, (char*) src + meta.dst_off, (BytePack<8>*) &reg, w, g_meta, s_meta, meta.src_off, meta.dst_off);
        } else if (align_off & 0x4) {
          bulkLoad<4>(t, meta.len, (char*) bounce_buf + meta.src_off, (char*) src + meta.dst_off, (BytePack<4>*) &reg, w, g_meta, s_meta, meta.src_off, meta.dst_off);
        } else if (align_off & 0x2) {
          bulkLoad<2>(t, meta.len, (char*) bounce_buf + meta.src_off, (char*) src + meta.dst_off, (BytePack<2>*) &reg, w, g_meta, s_meta, meta.src_off, meta.dst_off);
        } else { // if (align_off & 0x1)
          bulkLoad<1>(t, meta.len, (char*) bounce_buf + meta.src_off, (char*) src + meta.dst_off, (BytePack<1>*) &reg, w, g_meta, s_meta, meta.src_off, meta.dst_off);
        }
      }

      // must be less than 16 bytes
      if (t < meta.len % DATA_LOAD_SIZE) {
        volatile char* cpy_src = (char*) bounce_buf + meta.src_off + (meta.len / DATA_LOAD_SIZE) * DATA_LOAD_SIZE + t;
        volatile char* cpy_dst = (char*) src        + meta.dst_off + (meta.len / DATA_LOAD_SIZE) * DATA_LOAD_SIZE + t;
        *cpy_dst = *cpy_src;
      }
    }

    __syncwarp();
  }
}

#endif  // NET_DEVICE_UNPACK_DEFS_H_
