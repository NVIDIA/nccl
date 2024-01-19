/*************************************************************************
 * Copyright (c) 2023, Google LLC.  All rights reserved.
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef NET_DEVICE_UNPACK_DEFS_H
#define NET_DEVICE_UNPACK_DEFS_H

#include <stdint.h>

#include "device.h"

#define NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH 16

union alignas(16) loadMeta {
  uint64_t r64[2];
  struct {
    uint32_t src_off;
    uint32_t len;
    uint64_t dst_off;
  };
};
static_assert(sizeof(union loadMeta) == 16, "Must be 16-byte aligned");

/****** global memory ******/

#define NET_UNPACK_MAX_QUEUE_DEPTH 16  // MAX_REQUESTS
#define NET_UNPACK_MAX_SLICE_SIZE 4194304  // 4MB per Irecv call
#define SLICE_PAGE_SIZE 4096
#define NET_UNPACK_MAX_SLICE_PAGES \
  (NET_UNPACK_MAX_SLICE_SIZE / SLICE_PAGE_SIZE * 2)  // * 2 for slack, wasteful..

struct netUnpackMeta {
  loadMeta mem[NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH][NET_UNPACK_MAX_SLICE_PAGES];
  uint64_t cnt[NCCL_NET_DEVICE_UNPACK_MAX_QUEUE_DEPTH];
};

struct unpackNetDeviceHandle {
  struct netUnpackMeta *meta;  // mapped
  void* bounce_buf;
  uint64_t head;
};

/****** shared memory ******/

#define NET_UNPACK_MAX_GROUPS 16 // Forked from NCCL_MAX_GROUPS in devcomm.h
#define NET_UNPACK_MAX_NPEERS 2  // The most you should have is 2 network peers per-group (indexed by index)
#define WARP_SHM_PAGE_CNT 4
#define WARP_SHM_SIZE (WARP_SHM_PAGE_CNT * sizeof(union loadMeta))
struct unpackShmem {
  void* bounce_buf;
};

struct unpackGroupShmem {
  int unpackNetDeviceIndexMask; // We store a single unpackNetDeviceIndex because only one peer can be network recv
  uint64_t head;
  struct netUnpackMeta* g_meta[NET_UNPACK_MAX_NPEERS]; // head of handle to index into meta for meta copy
};

#endif // NET_DEVICE_UNPACK_DEFS_H_
