/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BITOPS_H_
#define NCCL_BITOPS_H_

#include <stdint.h>
#include <string.h>
#include "compiler.h"

#if !__NVCC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

template<typename Int>
constexpr static __host__ __device__ Int minval(Int a) { return a; }
template<typename Int, typename ...More>
constexpr static __host__ __device__ Int minval(Int a, Int b, More ...more) {
  #if __CUDA_ARCH__
    return minval(min(a, b), more...);
  #else
    return minval(a < b ? a : b, more...);
  #endif
}

template<typename Int>
constexpr static __host__ __device__ Int maxval(Int a) { return a; }
template<typename Int, typename ...More>
constexpr static __host__ __device__ Int maxval(Int a, Int b, More ...more) {
  #if __CUDA_ARCH__
    return maxval(max(a, b), more...);
  #else
    return maxval(a > b ? a : b, more...);
  #endif
}

#define BIT(x) (1UL << (x))
#define MASK(x) ((1UL << x) - 1UL)

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_POWER(x, y) \
    ((x) > (y) ? ROUNDUP(x, y) : ((y)/((y)/(x))))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<typename X, typename Y, typename Z = decltype(X()+Y())>
static __host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
static __host__ __device__ constexpr Z roundUp(X x, Y y) {
  return (x+y-1) - (x+y-1)%y;
}
template<typename X, typename Y, typename Z = decltype(X()+Y())>
static __host__ __device__ constexpr Z roundDown(X x, Y y) {
  return x - x%y;
}

// assumes second argument is a power of 2
template<typename X, typename Y, typename Z = decltype(X()+Y())>
static __host__ __device__ constexpr Z alignUp(X x, Y a) {
  return (x + a-1) & -Z(a);
}
template<typename T>
static __host__ __device__ T* alignUp(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(x) + a-1) & -uintptr_t(a));
}

// assumes second argument is a power of 2
template<typename X, typename Y, typename Z = decltype(X()+Y())>
static __host__ __device__ constexpr Z alignDown(X x, Y a) {
  return x & -Z(a);
}

template<typename T>
static __host__ __device__ T* alignDown(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(x) & -uintptr_t(a));
}

template<typename Int>
constexpr __host__ __device__ bool isPow2(Int x) {
  return (x & (x-1)) == 0;
}

template<typename T>
static __host__ __device__ T add4G(T base, int delta4G) {
  union { T tmp; uint32_t u32[2]; };
  tmp = base;
  u32[1] += delta4G;
  return tmp;
}

template<typename T>
static __host__ __device__ T incWrap4G(T ptr, uint32_t delta4G, uint32_t lo4G, uint32_t hi4G) {
  union { T tmp; uint32_t u32[2]; };
  tmp = ptr;
  u32[1] += delta4G;
  if (u32[1] >= hi4G) u32[1] -= hi4G-lo4G;
  return tmp;
}

template<typename T>
static __host__ __device__ T decWrap4G(T ptr, uint32_t delta4G, uint32_t lo4G, uint32_t hi4G) {
  union { T tmp; uint32_t u32[2]; };
  tmp = ptr;
  u32[1] -= delta4G;
  if (u32[1] < lo4G) u32[1] += hi4G-lo4G;
  return tmp;
}

// Produce the reciprocal of x for use in idivByRcp
constexpr __host__ __device__ uint32_t idivRcp32(uint32_t x) {
  return uint32_t(uint64_t(0x100000000)/x);
}
constexpr __host__ __device__ uint64_t idivRcp64(uint64_t x) {
  return uint64_t(-1)/x + isPow2(x);
}

static __host__ __device__ uint32_t mul32hi(uint32_t a, uint32_t b) {
#if __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return uint64_t(a)*b >> 32;
#endif
}
static __host__ __device__ uint64_t mul64hi(uint64_t a, uint64_t b) {
#if __CUDA_ARCH__
  return __umul64hi(a, b);
#else
  return (uint64_t)(((unsigned __int128)a)*b >> 64);
#endif
}

// Produce the reciprocal of x*y given their respective reciprocals. This incurs
// no integer division on device.
static __host__ __device__ uint32_t imulRcp32(uint32_t x, uint32_t xrcp, uint32_t y, uint32_t yrcp) {
  if (xrcp == 0) return yrcp;
  if (yrcp == 0) return xrcp;
  uint32_t rcp = mul32hi(xrcp, yrcp);
  uint32_t rem = -x*y*rcp;
  if (x*y <= rem) rcp += 1;
  return rcp;
}
static __host__ __device__ uint64_t imulRcp64(uint64_t x, uint64_t xrcp, uint64_t y, uint64_t yrcp) {
  if (xrcp == 0) return yrcp;
  if (yrcp == 0) return xrcp;
  uint64_t rcp = mul64hi(xrcp, yrcp);
  uint64_t rem = -x*y*rcp;
  if (x*y <= rem) rcp += 1;
  return rcp;
}

// Fast integer division where divisor has precomputed reciprocal.
// idivFast(x, y, idivRcp(y)) == x/y
static __host__ __device__ void idivmodFast32(uint32_t *quo, uint32_t *rem, uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q = x, r = 0;
  if (yrcp != 0) {
    q = mul32hi(x, yrcp);
    r = x - y*q;
    if (r >= y) { q += 1; r -= y; }
  }
  *quo = q;
  *rem = r;
}
static __host__ __device__ void idivmodFast64(uint64_t *quo, uint64_t *rem, uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q = x, r = 0;
  if (yrcp != 0) {
    q = mul64hi(x, yrcp);
    r = x - y*q;
    if (r >= y) { q += 1; r -= y; }
  }
  *quo = q;
  *rem = r;
}

static __host__ __device__ uint32_t idivFast32(uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q, r;
  idivmodFast32(&q, &r, x, y, yrcp);
  return q;
}
static __host__ __device__ uint32_t idivFast64(uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q, r;
  idivmodFast64(&q, &r, x, y, yrcp);
  return q;
}

static __host__ __device__ uint32_t imodFast32(uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q, r;
  idivmodFast32(&q, &r, x, y, yrcp);
  return r;
}
static __host__ __device__ uint32_t imodFast64(uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q, r;
  idivmodFast64(&q, &r, x, y, yrcp);
  return r;
}

template<typename Int>
static __host__ __device__ int countOneBits(Int x) {
#if __CUDA_ARCH__
  if (sizeof(Int) <= sizeof(unsigned int)) {
    return __popc((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    return __popcll((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
    return -1;
  }
#else
  if (sizeof(Int) <= sizeof(unsigned int)) {
    return COMPILER_POPCOUNT32((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    return COMPILER_POPCOUNT64((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    return COMPILER_POPCOUNT64((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
    return -1;
  }
#endif
}

// Returns index of first one bit or returns -1 if mask is zero.
template<typename Int>
static __host__ __device__ int firstOneBit(Int mask) {
  int i;
#if __CUDA_ARCH__
  if (sizeof(Int) <= sizeof(int)) {
    i = __ffs((int)mask);
  } else if (sizeof(Int) <= sizeof(long long)) {
    i = __ffsll((long long)mask);
  } else {
    static_assert(sizeof(Int) <= sizeof(long long), "Unsupported integer size.");
  }
#else
  if (sizeof(Int) <= sizeof(int)) {
    i = COMPILER_FFS((int)mask);
  } else if (sizeof(Int) <= sizeof(long)) {
    i = COMPILER_FFSL((long)mask);
  } else if (sizeof(Int) <= sizeof(long long)) {
    i = COMPILER_FFSLL((long long)mask);
  } else {
    static_assert(sizeof(Int) <= sizeof(long long), "Unsupported integer size.");
  }
#endif
  return i-1;
}

template<typename Int>
static __host__ __device__ int popFirstOneBit(Int* mask) {
  Int tmp = *mask;
  *mask &= *mask-1;
  return firstOneBit(tmp);
}

template<typename Int>
static __host__ __device__ int log2Down(Int x) {
  int w, n;
#if __CUDA_ARCH__
  if (sizeof(Int) <= sizeof(int)) {
    w = 8*sizeof(int);
    n = __clz((int)x);
  } else if (sizeof(Int) <= sizeof(long long)) {
    w = 8*sizeof(long long);
    n = __clzll((long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(long long), "Unsupported integer size.");
  }
#else
  if (x == 0) {
    return -1;
  } else if (sizeof(Int) <= sizeof(unsigned int)) {
    w = 8*sizeof(unsigned int);
    n = COMPILER_CLZ((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    w = 8*sizeof(unsigned long);
    n = COMPILER_CLZL((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    w = 8*sizeof(unsigned long long);
    n = COMPILER_CLZLL((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
  }
#endif
  return (w-1)-n;
}

template<typename Int>
static __host__ __device__ int log2Up(Int x) {
  int w, n;
  if (x != 0) x -= 1;
#if __CUDA_ARCH__
  if (sizeof(Int) <= sizeof(int)) {
    w = 8*sizeof(int);
    n = __clz((int)x);
  } else if (sizeof(Int) <= sizeof(long long)) {
    w = 8*sizeof(long long);
    n = __clzll((long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(long long), "Unsupported integer size.");
  }
#else
  if (x == 0) {
    return 0;
  } else if (sizeof(Int) <= sizeof(unsigned int)) {
    w = 8*sizeof(unsigned int);
    n = COMPILER_CLZ((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    w = 8*sizeof(unsigned long);
    n = COMPILER_CLZL((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    w = 8*sizeof(unsigned long long);
    n = COMPILER_CLZLL((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
  }
#endif
  return w-n;
}

template<typename Int>
static __host__ __device__ Int pow2Up(Int x) {
  return Int(1)<<log2Up(x);
}

template<typename Int>
static __host__ __device__ Int pow2Down(Int x) {
  // True, log2Down can return -1, but we don't normally pass 0 as an argument...
  // coverity[negative_shift]
  return Int(1)<<log2Down(x);
}

template<typename UInt, int nSubBits>
static __host__ UInt reverseSubBits(UInt x) {
  if (nSubBits >= 16 && 8*sizeof(UInt) == nSubBits) {
    switch (8*sizeof(UInt)) {
    case 16: x = COMPILER_BSWAP16(x); break;
    case 32: x = COMPILER_BSWAP32(x); break;
    case 64: x = COMPILER_BSWAP64(x); break;
    default: static_assert(8*sizeof(UInt) <= 64, "Unsupported integer type.");
    }
    return reverseSubBits<UInt, 8>(x);
  } else if (nSubBits <= 1) {
    return x;
  } else {
    UInt m = UInt(-1)/((UInt(1)<<(nSubBits/2))+1);
    x = (x & m)<<(nSubBits/2) | (x & ~m)>>(nSubBits/2);
    return reverseSubBits<UInt, nSubBits/2>(x);
  }
}

template<typename T> struct ncclToUnsigned;
template<> struct ncclToUnsigned<char> { using type = unsigned char; };
template<> struct ncclToUnsigned<signed char> { using type = unsigned char; };
template<> struct ncclToUnsigned<unsigned char> { using type = unsigned char; };
template<> struct ncclToUnsigned<signed short> { using type = unsigned short; };
template<> struct ncclToUnsigned<unsigned short> { using type = unsigned short; };
template<> struct ncclToUnsigned<signed int> { using type = unsigned int; };
template<> struct ncclToUnsigned<unsigned int> { using type = unsigned int; };
template<> struct ncclToUnsigned<signed long> { using type = unsigned long; };
template<> struct ncclToUnsigned<unsigned long> { using type = unsigned long; };
template<> struct ncclToUnsigned<signed long long> { using type = unsigned long long; };
template<> struct ncclToUnsigned<unsigned long long> { using type = unsigned long long; };

// Reverse the bottom nBits bits of x. The top bits will be overwritten with 0's.
template<typename Int>
static __host__ __device__ Int reverseBits(Int x, int nBits) {
  using UInt = typename ncclToUnsigned<Int>::type;
  union { UInt ux; Int sx; };
  sx = x;
  #if __CUDA_ARCH__
    if (sizeof(Int) <= sizeof(unsigned int)) {
      ux = __brev(ux);
    } else if (sizeof(Int) <= sizeof(unsigned long long)) {
      ux = __brevll(ux);
    } else {
      static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer type.");
    }
  #else
    ux = reverseSubBits<UInt, 8*sizeof(UInt)>(ux);
  #endif
  ux = nBits==0 ? 0 : ux>>(8*sizeof(UInt)-nBits);
  return sx;
}

////////////////////////////////////////////////////////////////////////////////
// Custom 8 bit floating point format for approximating 32 bit uints. This format
// has nearly the full range of uint32_t except it only keeps the top 3 bits
// beneath the leading 1 bit and thus has a max value of 0xf0000000.

static __host__ __device__ uint32_t u32fpEncode(uint32_t x, int bitsPerPow2) {
  int log2x;
  #if __CUDA_ARCH__
    log2x = 31-__clz(x|1);
  #else
    log2x = 31-COMPILER_CLZ(x|1);
  #endif
  uint32_t mantissa = x>>(log2x >= bitsPerPow2 ? log2x-bitsPerPow2 : 0) & ((1u<<bitsPerPow2)-1);
  uint32_t exponent = log2x >= bitsPerPow2 ? log2x-(bitsPerPow2-1) : 0;
  return exponent<<bitsPerPow2 | mantissa;
}

static __host__ __device__ uint32_t u32fpDecode(uint32_t x, int bitsPerPow2) {
  uint32_t exponent = x>>bitsPerPow2;
  uint32_t mantissa = (x & ((1u<<bitsPerPow2)-1)) | (exponent!=0 ? 0x8 : 0);
  if (exponent != 0) exponent -= 1;
  return mantissa<<exponent;
}

constexpr uint32_t u32fp8MaxValue() { return 0xf0000000; }

static __host__ __device__ uint8_t u32fp8Encode(uint32_t x) {
  return u32fpEncode(x, 3);
}
static __host__ __device__ uint32_t u32fp8Decode(uint8_t x) {
  return u32fpDecode(x, 3);
}

// The hash isn't just a function of the bytes but also where the bytes are split
// into different calls to eatHash().
static __host__ __device__ void eatHash(uint64_t acc[2], const void* bytes, size_t size) {
  char const* ptr = (char const*)bytes;
  acc[0] ^= size;
  while (size != 0) {
    // Mix the accumulator bits.
    acc[0] += acc[1];
    acc[1] ^= acc[0];
    acc[0] ^= acc[0] >> 31;
    acc[0] *= 0x9de62bbc8cef3ce3;
    acc[1] ^= acc[1] >> 32;
    acc[1] *= 0x485cd6311b599e79;
    // Read in a chunk of input.
    size_t chunkSize = size < sizeof(uint64_t) ? size : sizeof(uint64_t);
    uint64_t x = 0;
    memcpy(&x, ptr, chunkSize);
    ptr += chunkSize;
    size -= chunkSize;
    // Add to accumulator.
    acc[0] += x;
  }
}

template<typename T>
static __host__ __device__ void eatHash(uint64_t acc[2], const T* bytes) {
  eatHash(acc, (const void*)bytes, sizeof(T));
}

static __host__ __device__ uint64_t digestHash(uint64_t const acc[2]) {
  uint64_t h = acc[0];
  h ^= h >> 31;
  h *= 0xbac3bd562846de6b;
  h += acc[1];
  h ^= h >> 32;
  h *= 0x995a187a14e7b445;
  return h;
}

static __host__ __device__ uint64_t getHash(const void* bytes, size_t size) {
  uint64_t acc[2] = {1, 1};
  eatHash(acc, bytes, size);
  return digestHash(acc);
}
template<typename T>
static __host__ __device__ uint64_t getHash(const T* bytes) {
  return getHash((const void*)bytes, sizeof(T));
}

#endif
