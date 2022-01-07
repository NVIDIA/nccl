/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

inline __device__ int loadInt(int* ptr) {
  int v;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
      : "=r"(v) : "l"(ptr));
  return v;
}

typedef uint64_t PackType;

template<typename Fn>
struct FuncTraits /*{
  __device__ static T preOp(Fn, T);
  __device__ static T postOp(Fn, T);
}*/;

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const;
  __device__ PackType preOp(FUNC fn, PackType x) const;
  __device__ PackType postOp(FUNC fn, PackType x) const;
};

template<class FUNC>
struct MULTI<FUNC, int8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().preOp(fn, u.elt[i]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().postOp(fn, u.elt[i]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().preOp(fn, u.elt[i]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint8_t elt[8];
    } u;
    u.pack = x;
    #pragma unroll
    for (int i=0; i < 8; i++)
      u.elt[i] = FuncTraits<FUNC>().postOp(fn, u.elt[i]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, int32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().preOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().preOp(fn, u.elt[1]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().postOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().postOp(fn, u.elt[1]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().preOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().preOp(fn, u.elt[1]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint32_t elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().postOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().postOp(fn, u.elt[1]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  union Converter {
    PackType pack;
    half2 h2[2];
  };
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    Converter cx, cy, cr;
    cx.pack = x;
    cy.pack = y;
    cr.h2[0] = fn(cx.h2[0], cy.h2[0]);
    cr.h2[1] = fn(cx.h2[1], cy.h2[1]);
    return cr.pack;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().preOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().preOp(fn, c.h2[1]);
    return c.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().postOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().postOp(fn, c.h2[1]);
    return c.pack;
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<class FUNC>
struct MULTI<FUNC, __nv_bfloat16> {
  static_assert(sizeof(PackType) == 4 * sizeof(__nv_bfloat16),
      "PackType must be four times the size of __nv_bfloat16.");

  union Converter {
    PackType pack;
    __nv_bfloat162 h2[2];
  };
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    Converter cx, cy, cr;
    cx.pack = x;
    cy.pack = y;
    cr.h2[0] = fn(cx.h2[0], cy.h2[0]);
    cr.h2[1] = fn(cx.h2[1], cy.h2[1]);
    return cr.pack;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().preOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().preOp(fn, c.h2[1]);
    return c.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    Converter c;
    c.pack = x;
    c.h2[0] = FuncTraits<FUNC>().postOp(fn, c.h2[0]);
    c.h2[1] = FuncTraits<FUNC>().postOp(fn, c.h2[1]);
    return c.pack;
  }
};
#endif

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = fn(cx.a, cy.a);
    cr.b = fn(cx.b, cy.b);

    return cr.storage;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      float elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().preOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().preOp(fn, u.elt[1]);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      float elt[2];
    } u;
    u.pack = x;
    u.elt[0] = FuncTraits<FUNC>().postOp(fn, u.elt[0]);
    u.elt[1] = FuncTraits<FUNC>().postOp(fn, u.elt[1]);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    double rv = fn(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      double elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().preOp(fn, u.elt);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      double elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().postOp(fn, u.elt);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint64_t> {
  static_assert(sizeof(PackType) == sizeof(uint64_t),
      "PackType must be the same size as uint64_t.");
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    uint64_t rv = fn(x, y);
    return rv;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().preOp(fn, u.elt);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      uint64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().postOp(fn, u.elt);
    return u.pack;
  }
};

template<class FUNC>
struct MULTI<FUNC, int64_t> {
  static_assert(sizeof(PackType) == sizeof(int64_t),
      "PackType must be the same size as int64_t.");
  __device__ PackType operator()(FUNC fn, const PackType x, const PackType y) const {
    int64_t rv = fn((int64_t)x, (int64_t)y);
    return rv;
  }
  __device__ PackType preOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().preOp(fn, u.elt);
    return u.pack;
  }
  __device__ PackType postOp(FUNC fn, PackType x) const {
    union {
      PackType pack;
      int64_t elt;
    } u;
    u.pack = x;
    u.elt = FuncTraits<FUNC>().postOp(fn, u.elt);
    return u.pack;
  }
};

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#if CUDART_VERSION < 9000
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#else
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r = ((half*)ptr)[0];
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ((half*)ptr)[0] = val;
}
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> inline __device__
__nv_bfloat16 vFetch<__nv_bfloat16>(const volatile __nv_bfloat16* ptr) {
  __nv_bfloat16 r;
  r = ((__nv_bfloat16*)ptr)[0];
  return r;
}

template<> inline __device__
void vStore<__nv_bfloat16>(volatile __nv_bfloat16* ptr, const __nv_bfloat16 val) {
  ((__nv_bfloat16*)ptr)[0] = val;
}
#endif

typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(FUNC fn, Pack128& x, Pack128 const& y) const {
    x.x = MULTI<FUNC, T>()(fn, x.x, y.x);
    x.y = MULTI<FUNC, T>()(fn, x.y, y.y);
  }
  __device__ void preOp(FUNC fn, Pack128 &x) const {
    x.x = MULTI<FUNC, T>().preOp(fn, x.x);
    x.y = MULTI<FUNC, T>().preOp(fn, x.y);
  }
  __device__ void postOp(FUNC fn, Pack128 &x) const {
    x.x = MULTI<FUNC, T>().postOp(fn, x.x);
    x.y = MULTI<FUNC, T>().postOp(fn, x.y);
  }
};

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int PreOpN, typename Int>
__device__ __forceinline__ void ReduceCopyMulti(const int w, const int nw, const int t,
    uint64_t* redOpArgs, bool postOp, int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const Int Nelem
  ) {
  const Int inc = nw * UNROLL * WARP_SIZE;
  Int offset = w * UNROLL * WARP_SIZE + t;

  const T* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = s[i]+elemOffset+offset;
  T* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = d[i]+elemOffset+offset;

  while (offset < Nelem) {
    T vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) vals[u] = vFetch(srcs[0]+u*WARP_SIZE);
    if (PreOpN) {
      FUNC fn(redOpArgs[0]);
      for (int u = 0; u < UNROLL; ++u) vals[u] = FuncTraits<FUNC>().preOp(fn, vals[u]);
    }

    #pragma unroll
    for (int i=1; i<MINSRCS; i++) {
      T vals2[UNROLL];
      FUNC fn(redOpArgs[i]);
      for (int u = 0; u < UNROLL; ++u) vals2[u] = vFetch(srcs[i]+u*WARP_SIZE);
      if (i<PreOpN) {
        for (int u = 0; u < UNROLL; ++u) vals2[u] = FuncTraits<FUNC>().preOp(fn, vals2[u]);
      }
      for (int u = 0; u < UNROLL; ++u) vals[u] = fn(vals[u], vals2[u]);
    }
    #pragma unroll
    for (int i=MINSRCS; i<MAXSRCS; i++) {
      if (i<nsrcs) {
        T vals2[UNROLL];
        FUNC fn(redOpArgs[i]);
        for (int u = 0; u < UNROLL; ++u) vals2[u] = vFetch(srcs[i]+u*WARP_SIZE);
        if (i<PreOpN) {
          for (int u = 0; u < UNROLL; ++u) vals2[u] = FuncTraits<FUNC>().preOp(fn, vals2[u]);
        }
        for (int u = 0; u < UNROLL; ++u) vals[u] = fn(vals[u], vals2[u]);
      }
    }

    if (postOp) {
      FUNC fn(redOpArgs[0]);
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) vals[u] = FuncTraits<FUNC>().postOp(fn, vals[u]);
    }

    // Store
    #pragma unroll
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) vStore(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll
    for (int i=MINDSTS; i<MAXDSTS; i++) {
      if (i<ndsts) {
        for (int u = 0; u < UNROLL; ++u) vStore(dsts[i]+u*WARP_SIZE, vals[u]);
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int PreOpN, typename Int>
__device__ __forceinline__ void ReduceCopy128bMulti(const int w, const int nw, const int t,
    uint64_t* redOpArgs, bool postOp, int nsrcs, const T** s, int ndsts, T** d, const int elemOffset, const Int Npack
  ) {
  const Int inc = nw * UNROLL * WARP_SIZE;
  Int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);
    if (PreOpN) {
      FUNC fn(redOpArgs[0]);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>().preOp(fn, vals[u]);
    }

    #pragma unroll
    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      FUNC fn(redOpArgs[i]);
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      if (i<PreOpN) {
        for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>().preOp(fn, vals2[u]);
      }
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(fn, vals[u], vals2[u]);
    }
    #pragma unroll
    for (int i=MINSRCS; i<MAXSRCS; i++) {
      if (i<nsrcs) {
        Pack128 vals2[UNROLL];
        FUNC fn(redOpArgs[i]);
        for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
        if (i<PreOpN) {
          for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>().preOp(fn, vals2[u]);
        }
        for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(fn, vals[u], vals2[u]);
      }
    }

    if (postOp) {
      FUNC fn(redOpArgs[0]);
      #pragma unroll
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>().postOp(fn, vals[u]);
    }

    // Store
    #pragma unroll
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll
    for (int i=MINDSTS; i<MAXDSTS; i++) {
      if (i<ndsts) {
        for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
      }
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

#define PACKELEMS (sizeof(Pack128) / sizeof(T))

template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS, int PreOpN, typename Int>
__device__ __forceinline__ void ReduceOrCopyMulti(
    const int tid, const int nthreads, uint64_t* redOpArgs, bool postOp, int nsrcs, const T** srcs, int ndsts, T** dsts, Int N
  ) {
  Int Nrem = N;
  if (Nrem <= 0) return;

  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // Check that all is 16B aligned. If not don't use 16B load/stores.
  int align = 0;
  #pragma unroll
  for (int i=0; i<MINSRCS; i++) align |= ptrAlign128(srcs[i]);
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) align |= ptrAlign128(srcs[i]);
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) align |= ptrAlign128(dsts[i]);
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) align |= ptrAlign128(dsts[i]);

  Int offset = 0;
  if (align == 0) {
    // fast path: use 128b loads/stores to do the bulk of the work,
    // assuming the pointers we have are all 128-bit aligned.

    // main loop
    Int Npack = (Nrem / (PACKELEMS*UNROLL*WARP_SIZE)) * (UNROLL*WARP_SIZE); // round down
    Int Nelem = Npack * PACKELEMS;

    ReduceCopy128bMulti<FUNC, T, UNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, PreOpN>
      (w, nw, t, redOpArgs, postOp, nsrcs, srcs, ndsts, dsts, offset, Npack);

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;

    // slightly less optimized for section when we don't have full unrolling
    Npack = Nrem / PACKELEMS;
    Nelem = Npack * PACKELEMS;

    ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, PreOpN>
      (w, nw, t, redOpArgs, postOp, nsrcs, srcs, ndsts, dsts, offset, Npack);

    Nrem -= Nelem;
    if (Nrem == 0) return;
    offset += Nelem;
  }

  // unrolled, by-type (mostly for unaligned buffers)
  Int Nelem = (Nrem / (UNROLL*PACKELEMS/2*WARP_SIZE)) * (UNROLL*PACKELEMS/2*WARP_SIZE); // round down

  ReduceCopyMulti<FUNC, T, UNROLL*PACKELEMS/2, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, PreOpN>
    (w, nw, t, redOpArgs, postOp, nsrcs, srcs, ndsts, dsts, offset, Nelem);

  Nrem -= Nelem;
  if (Nrem == 0) return;
  offset += Nelem;

  // no unroll, by type. Should finish what's remaining.
  ReduceCopyMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS, PreOpN>
    (w, nw, t, redOpArgs, postOp, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

#endif // COMMON_KERNEL_H_
