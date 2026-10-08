/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_IMPL_COOP_H_
#define _NCCL_DEVICE_WRAPPER_IMPL_COOP_H_

// Internal implementation fragment; include only from nccl_device_wrapper.cu
// after its prerequisite headers.

struct ncclCoopAny {
  struct Storage {
    alignas(alignof(void*)) char space[16];
  };
  struct VTable {
    int (*thread_rank)(void const*);
    int (*size)(void const*);
    void (*sync)(void*);
  };

  template <typename Impl>
  __device__ static int thread_rank(void const* o) {
    return static_cast<Impl const*>(o)->thread_rank();
  }
  template <typename Impl>
  __device__ static int size(void const* o) {
    return static_cast<Impl const*>(o)->size();
  }
  template <typename Impl>
  __device__ static void sync(void* o) {
    static_cast<Impl*>(o)->sync();
  }

  template <typename Impl>
  __device__ static VTable const* get_vtable() {
    static_assert(sizeof(Impl) <= sizeof(Storage), "Incompatible coop type size");
    static_assert(alignof(Impl) <= alignof(Storage), "Incompatible coop type alignment");
    static constexpr VTable v = {&thread_rank<Impl>, &size<Impl>, &sync<Impl>};
    return &v;
  }

  Storage storage;
  VTable const* vtable;

  ncclCoopAny(ncclCoopAny const&) = default;
  ncclCoopAny(ncclCoopAny&&) = default;
  ncclCoopAny() = default;

  template <typename Impl>
  __device__ ncclCoopAny(Impl impl) {
    ::new (&this->storage) Impl(impl);
    this->vtable = get_vtable<Impl>();
  }

  __device__ int thread_rank() const {
    return vtable->thread_rank(&storage);
  }
  __device__ int size() const {
    return vtable->size(&storage);
  }
  __device__ int num_threads() const {
    return vtable->size(&storage);
  }
  __device__ void sync() {
    vtable->sync(&storage);
  }
};

static_assert(sizeof(ncclIrCoop) == sizeof(ncclCoopAny), "Coop storage size mismatch");
static_assert(alignof(ncclIrCoop) == alignof(ncclCoopAny), "Coop storage alignment mismatch");

NCCL_DEVICE_INLINE void ncclIrCoopInitThread(ncclIrCoop* coop) {
  ::new (coop) ncclCoopAny(ncclCoopThread());
}
NCCL_DEVICE_INLINE void ncclIrCoopInitWarp(ncclIrCoop* coop) {
  ::new (coop) ncclCoopAny(ncclCoopWarp());
}
NCCL_DEVICE_INLINE void ncclIrCoopInitLanes(ncclIrCoop* coop, uint32_t lane_mask) {
  ::new (coop) ncclCoopAny(ncclCoopLanes(lane_mask));
}
NCCL_DEVICE_INLINE void ncclIrCoopInitWarpSpan(ncclIrCoop* coop, int warp0, int nWarps, int id) {
  ::new (coop) ncclCoopAny(ncclCoopWarpSpan(warp0, nWarps, id));
}
NCCL_DEVICE_INLINE void ncclIrCoopInitCta(ncclIrCoop* coop) {
  ::new (coop) ncclCoopAny(ncclCoopCta());
}

NCCL_DEVICE_INLINE int ncclIrCoopThreadRank(ncclIrCoop const* coop) {
  return reinterpret_cast<ncclCoopAny const*>(coop)->thread_rank();
}
NCCL_DEVICE_INLINE int ncclIrCoopSize(ncclIrCoop const* coop) {
  return reinterpret_cast<ncclCoopAny const*>(coop)->size();
}
NCCL_DEVICE_INLINE int ncclIrCoopNumThreads(ncclIrCoop const* coop) {
  return reinterpret_cast<ncclCoopAny const*>(coop)->num_threads();
}
NCCL_DEVICE_INLINE void ncclIrCoopSync(ncclIrCoop* coop) {
  reinterpret_cast<ncclCoopAny*>(coop)->sync();
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_COOP_H_
