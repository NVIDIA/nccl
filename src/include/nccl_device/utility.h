#ifndef _NCCL_DEVICE_UTILITY_H_
#define _NCCL_DEVICE_UTILITY_H_

#if __CUDACC__
  #define NCCL_DEVICE_INLINE __device__ __forceinline__
  #define NCCL_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
  #ifndef __host__
    #define __host__
  #endif
  #define NCCL_DEVICE_INLINE
  #define NCCL_HOST_DEVICE_INLINE inline __attribute__((always_inline))
#endif

#include <stdint.h>

#if __CUDACC__
#include <cuda/atomic>
#endif

namespace nccl {
namespace utility {

template<typename T>
T&& declval() noexcept {
  static_assert(sizeof(T)!=sizeof(T), "You can't evaluate declval.");
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z roundUp(X x, Y y) {
  return (x+y-1) - (x+y-1)%y;
}
template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z roundDown(X x, Y y) {
  return x - x%y;
}

// assumes second argument is a power of 2
template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z alignUp(X x, Y a) {
  return (x + a-1) & -Z(a);
}
template<typename T>
NCCL_HOST_DEVICE_INLINE T* alignUp(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(x) + a-1) & -uintptr_t(a));
}
template<typename T>
NCCL_HOST_DEVICE_INLINE void* alignUp(void const* x, size_t a) {
  return reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(x) + a-1) & -uintptr_t(a));
}

// assumes second argument is a power of 2
template<typename X, typename Y, typename Z = decltype(X()+int())>
NCCL_HOST_DEVICE_INLINE constexpr Z alignDown(X x, Y a) {
  return x & -Z(a);
}
template<typename T>
NCCL_HOST_DEVICE_INLINE T* alignDown(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(x) & -uintptr_t(a));
}
template<typename T>
NCCL_HOST_DEVICE_INLINE void* alignDown(void const* x, size_t a) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(x) & -uintptr_t(a));
}

template<typename T>
NCCL_HOST_DEVICE_INLINE T add4G(T base, int delta4G) {
  union { uint32_t u32[2]; T tmp; };
  tmp = base;
  u32[1] += delta4G;
  return tmp;
}


template<typename Int>
NCCL_HOST_DEVICE_INLINE constexpr bool isPow2(Int x) {
  return (x & (x-1)) == 0;
}

// Produce the reciprocal of x for use in idivByRcp
NCCL_HOST_DEVICE_INLINE constexpr uint32_t idivRcp32(uint32_t x) {
  return uint32_t(-1)/x + isPow2(x);
}
NCCL_HOST_DEVICE_INLINE constexpr uint64_t idivRcp64(uint64_t x) {
  return uint64_t(-1)/x + isPow2(x);
}

NCCL_HOST_DEVICE_INLINE uint32_t mul32hi(uint32_t a, uint32_t b) {
#if __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return uint64_t(a)*b >> 32;
#endif
}
NCCL_HOST_DEVICE_INLINE uint64_t mul64hi(uint64_t a, uint64_t b) {
#if __CUDA_ARCH__
  return __umul64hi(a, b);
#else
  return (uint64_t)(((unsigned __int128)a)*b >> 64);
#endif
}

// Produce the reciprocal of x*y given their respective reciprocals. This incurs
// no integer division on device.
NCCL_HOST_DEVICE_INLINE uint32_t imulRcp32(uint32_t x, uint32_t xrcp, uint32_t y, uint32_t yrcp) {
  if (xrcp == 0) return yrcp;
  if (yrcp == 0) return xrcp;
  uint32_t rcp = mul32hi(xrcp, yrcp);
  uint32_t rem = -x*y*rcp;
  if (x*y <= rem) rcp += 1;
  return rcp;
}
NCCL_HOST_DEVICE_INLINE uint64_t imulRcp64(uint64_t x, uint64_t xrcp, uint64_t y, uint64_t yrcp) {
  if (xrcp == 0) return yrcp;
  if (yrcp == 0) return xrcp;
  uint64_t rcp = mul64hi(xrcp, yrcp);
  uint64_t rem = -x*y*rcp;
  if (x*y <= rem) rcp += 1;
  return rcp;
}

// Fast unsigned integer division where divisor has precomputed reciprocal.
// idivFast(x, y, idivRcp(y)) == x/y
NCCL_HOST_DEVICE_INLINE void idivmodFast32(uint32_t *quo, uint32_t *rem, uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q = yrcp == 0 ? x : mul32hi(x, yrcp);
  uint32_t r = x - y*q;
  if (r >= y) { q += 1; r -= y; }
  *quo = q;
  *rem = r;
}
NCCL_HOST_DEVICE_INLINE void idivmodFast64(uint64_t *quo, uint64_t *rem, uint64_t x, uint64_t y, uint64_t yrcp) {
  uint32_t q = yrcp == 0 ? x : mul64hi(x, yrcp);
  uint32_t r = x - y*q;
  if (r >= y) { q += 1; r -= y; }
  *quo = q;
  *rem = r;
}

NCCL_HOST_DEVICE_INLINE uint32_t idivFast32(uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q, r;
  idivmodFast32(&q, &r, x, y, yrcp);
  return q;
}
NCCL_HOST_DEVICE_INLINE uint32_t idivFast64(uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q, r;
  idivmodFast64(&q, &r, x, y, yrcp);
  return q;
}

NCCL_HOST_DEVICE_INLINE uint32_t imodFast32(uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q, r;
  idivmodFast32(&q, &r, x, y, yrcp);
  return r;
}
NCCL_HOST_DEVICE_INLINE uint32_t imodFast64(uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q, r;
  idivmodFast64(&q, &r, x, y, yrcp);
  return r;
}

#if __CUDACC__
// Precomputed integer reciprocoals for denominator values 1..64 inclusive.
// Pass these to idivFast64() for fast division on the GPU.
NCCL_DEVICE_INLINE uint64_t idivRcp64_upto64(int x) {
  static constexpr uint64_t table[65] = {
    idivRcp64(0x01), idivRcp64(0x01), idivRcp64(0x02), idivRcp64(0x03),
    idivRcp64(0x04), idivRcp64(0x05), idivRcp64(0x06), idivRcp64(0x07),
    idivRcp64(0x08), idivRcp64(0x09), idivRcp64(0x0a), idivRcp64(0x0b),
    idivRcp64(0x0c), idivRcp64(0x0d), idivRcp64(0x0e), idivRcp64(0x0f),
    idivRcp64(0x10), idivRcp64(0x11), idivRcp64(0x12), idivRcp64(0x13),
    idivRcp64(0x14), idivRcp64(0x15), idivRcp64(0x16), idivRcp64(0x17),
    idivRcp64(0x18), idivRcp64(0x19), idivRcp64(0x1a), idivRcp64(0x1b),
    idivRcp64(0x1c), idivRcp64(0x1d), idivRcp64(0x1e), idivRcp64(0x1f),
    idivRcp64(0x20), idivRcp64(0x21), idivRcp64(0x22), idivRcp64(0x23),
    idivRcp64(0x24), idivRcp64(0x25), idivRcp64(0x26), idivRcp64(0x27),
    idivRcp64(0x28), idivRcp64(0x29), idivRcp64(0x2a), idivRcp64(0x2b),
    idivRcp64(0x2c), idivRcp64(0x2d), idivRcp64(0x2e), idivRcp64(0x2f),
    idivRcp64(0x30), idivRcp64(0x31), idivRcp64(0x32), idivRcp64(0x33),
    idivRcp64(0x34), idivRcp64(0x35), idivRcp64(0x36), idivRcp64(0x37),
    idivRcp64(0x38), idivRcp64(0x39), idivRcp64(0x3a), idivRcp64(0x3b),
    idivRcp64(0x3c), idivRcp64(0x3d), idivRcp64(0x3e), idivRcp64(0x3f),
    idivRcp64(0x40)
  };
  return table[x];
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE uint32_t idivRcp32_upto64(int x) {
  return idivRcp64_upto64(x)>>32;
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE void fenceAcquireGpu() {
  static __device__ int dummy;
  int tmp;
  asm volatile("ld.acquire.gpu.s32 %0,[%1];" : "=r"(tmp) : "l"(&dummy) : "memory");
  dummy = tmp;
}
NCCL_DEVICE_INLINE void fenceReleaseGpu() {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE cuda::memory_order acquireOrderOf(cuda::memory_order ord) {
  return ord == cuda::memory_order_release ? cuda::memory_order_relaxed :
         ord == cuda::memory_order_acq_rel ? cuda::memory_order_acquire :
         ord;
}
NCCL_DEVICE_INLINE cuda::memory_order releaseOrderOf(cuda::memory_order ord) {
  return ord == cuda::memory_order_acquire ? cuda::memory_order_relaxed :
         ord == cuda::memory_order_acq_rel ? cuda::memory_order_release :
         ord;
}
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE int lane() {
  int ret;
  asm("mov.u32 %0, %%laneid;" : "=r"(ret));
  return ret;
}
NCCL_DEVICE_INLINE unsigned int lanemask_lt() {
  unsigned int ret;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(ret));
  return ret;
}
#endif

#if __CUDACC__
// Load anything, but cache like its constant memory.
template<typename T>
NCCL_DEVICE_INLINE T loadConst(T const *p) {
  if (alignof(T) == 1) {
    union { uint8_t part[sizeof(T)]; T ret; };
    for (int i=0; i < (int)sizeof(T); i++) part[i] = __ldg((uint8_t const*)p + i);
    return ret;
  } else if (alignof(T) == 2) {
    union { uint16_t part[sizeof(T)/2]; T ret; };
    for (int i=0; i < (int)sizeof(T)/2; i++) part[i] = __ldg((uint16_t const*)p + i);
    return ret;
  } else if (alignof(T) == 4) {
    union { uint32_t part[sizeof(T)/4]; T ret; };
    for (int i=0; i < (int)sizeof(T)/4; i++) part[i] = __ldg((uint32_t const*)p + i);
    return ret;
  } else if (alignof(T) == 8) {
    union { uint64_t part[sizeof(T)/8]; T ret; };
    for (int i=0; i < (int)sizeof(T)/8; i++) part[i] = __ldg((uint64_t const*)p + i);
    return ret;
  } else { // alignof(T) >= 16
    union { ulonglong2 part[sizeof(T)/16]; T ret; };
    for (int i=0; i < (int)sizeof(T)/16; i++) part[i] = __ldg((ulonglong2 const*)p + i);
    return ret;
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Optional<T>: Holds a T that may or may not be constructed. An Optional
// constructed with a Present<Arg...> will have its T constructed via the
// T::T(Arg...) constructor. An Optional constructed with a Absent will not
// have its T constructed.

template<int ...vals>
struct IntSeq {};

template<int n, int m, int ...i>
struct IntSeqUpTo: IntSeqUpTo<n, m+1, i..., m> {};
template<int n, int ...i>
struct IntSeqUpTo<n, n, i...> { using Type = IntSeq<i...>; };

// Present<Arg...>: Packs a list of arguments together to be passed to Optional<T>.
template<typename ...Arg>
struct Present;
template<>
struct Present<> {};
template<typename H, typename ...T>
struct Present<H, T...> {
  H h;
  Present<T...> t;

  NCCL_HOST_DEVICE_INLINE H get(IntSeq<0>) {
    return static_cast<H>(h);
  }
  template<int i>
  NCCL_HOST_DEVICE_INLINE decltype(auto) get(IntSeq<i>) {
    return t.get(IntSeq<i-1>{});
  }
};

NCCL_HOST_DEVICE_INLINE Present<> present() {
  return Present<>{};
}
template<typename H, typename ...T>
NCCL_HOST_DEVICE_INLINE Present<H&&, T&&...> present(H&& h, T&& ...t) {
  return Present<H&&, T&&...>{static_cast<H&&>(h), present(static_cast<T&&>(t)...)};
}

struct Absent {};

template<typename T>
struct Optional {
  bool present; // Is `thing` constructed.
  union { T thing; };

  // Construct with absent thing:
  NCCL_HOST_DEVICE_INLINE constexpr Optional(): present(false) {}
  NCCL_HOST_DEVICE_INLINE constexpr Optional(Absent): present(false) {}

  // Helper constructor
  template<int ...i, typename ...Arg>
  NCCL_HOST_DEVICE_INLINE Optional(Present<Arg...> args, IntSeq<i...>):
    present(true),
    thing{args.get(IntSeq<i>())...} {
  }
  // Construct with present thing:
  template<typename ...Arg>
  NCCL_HOST_DEVICE_INLINE Optional(Present<Arg...> args):
    Optional(args, IntSeqUpTo<sizeof...(Arg), 0>::Type()) {
  }

  NCCL_HOST_DEVICE_INLINE ~Optional() {
    if (present) thing.~T();
  }
};

}}

#endif
