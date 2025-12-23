/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_PTR_H_
#define _NCCL_DEVICE_PTR_H_
#include "core.h"
#include <stdint.h>

#if __cplusplus
template<typename T>
struct ncclSymPtr {
  using ElementType = T;
  ncclWindow_t window;
  size_t offset;

  NCCL_HOST_DEVICE_INLINE constexpr ncclSymPtr(ncclWindow_t window=nullptr, size_t offset=0);

  template<typename U>
  NCCL_HOST_DEVICE_INLINE operator ncclSymPtr<U>() const;

  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(long long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned long long d);

  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(long long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned long long d);

  #if NCCL_CHECK_CUDACC
  NCCL_DEVICE_INLINE T* localPtr() const;
  NCCL_DEVICE_INLINE T* lsaPtr(int peer) const;
  NCCL_DEVICE_INLINE T* peerPtr(int peer) const;
  NCCL_DEVICE_INLINE T* peerPtr(ncclTeam team, int peer) const;
  NCCL_DEVICE_INLINE T* multimemPtr(ncclMultimemHandle mmHandle) const;
  NCCL_DEVICE_INLINE T* lsaMultimemPtr(ncclDevComm const&) const;
  #endif
};

template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator+(ncclSymPtr<T> p, Int d);
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator-(ncclSymPtr<T> p, Int d);
template<typename T>
NCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(ncclSymPtr<T> a, ncclSymPtr<T> b);

template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator==(ncclSymPtr<T> a, ncclSymPtr<T> b);
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator!=(ncclSymPtr<T> a, ncclSymPtr<T> b);
#endif

#endif
