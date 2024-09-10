/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BITOPS_H_
#define NCCL_BITOPS_H_

#include <stdint.h>

#if !__NVCC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_POWER(x, y) \
    ((x) > (y) ? ROUNDUP(x, y) : ((y)/((y)/(x))))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z roundUp(X x, Y y) {
  return (x+y-1) - (x+y-1)%y;
}
template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z roundDown(X x, Y y) {
  return x - x%y;
}

// assumes second argument is a power of 2
template<typename X, typename Z = decltype(X()+int())>
__host__ __device__ constexpr Z alignUp(X x, int a) {
  return (x + a-1) & Z(-a);
}
// assumes second argument is a power of 2
template<typename X, typename Z = decltype(X()+int())>
__host__ __device__ constexpr Z alignDown(X x, int a) {
  return x & Z(-a);
}

template<typename Int>
inline __host__ __device__ int countOneBits(Int x) {
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
    return __builtin_popcount((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    return __builtin_popcountl((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    return __builtin_popcountll((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
    return -1;
  }
#endif
}

// Returns index of first one bit or returns -1 if mask is zero.
template<typename Int>
inline __host__ __device__ int firstOneBit(Int mask) {
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
    i = __builtin_ffs((int)mask);
  } else if (sizeof(Int) <= sizeof(long)) {
    i = __builtin_ffsl((long)mask);
  } else if (sizeof(Int) <= sizeof(long long)) {
    i = __builtin_ffsll((long long)mask);
  } else {
    static_assert(sizeof(Int) <= sizeof(long long), "Unsupported integer size.");
  }
#endif
  return i-1;
}

template<typename Int>
inline __host__ __device__ int popFirstOneBit(Int* mask) {
  Int tmp = *mask;
  *mask &= *mask-1;
  return firstOneBit(tmp);
}

template<typename Int>
inline __host__ __device__ int log2Down(Int x) {
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
    n = __builtin_clz((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    w = 8*sizeof(unsigned long);
    n = __builtin_clzl((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    w = 8*sizeof(unsigned long long);
    n = __builtin_clzll((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
  }
#endif
  return (w-1)-n;
}

template<typename Int>
inline __host__ __device__ int log2Up(Int x) {
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
    n = __builtin_clz((unsigned int)x);
  } else if (sizeof(Int) <= sizeof(unsigned long)) {
    w = 8*sizeof(unsigned long);
    n = __builtin_clzl((unsigned long)x);
  } else if (sizeof(Int) <= sizeof(unsigned long long)) {
    w = 8*sizeof(unsigned long long);
    n = __builtin_clzll((unsigned long long)x);
  } else {
    static_assert(sizeof(Int) <= sizeof(unsigned long long), "Unsupported integer size.");
  }
#endif
  return w-n;
}

template<typename Int>
inline __host__ __device__ Int pow2Up(Int x) {
  return Int(1)<<log2Up(x);
}

template<typename Int>
inline __host__ __device__ Int pow2Down(Int x) {
  // True, log2Down can return -1, but we don't normally pass 0 as an argument...
  // coverity[negative_shift]
  return Int(1)<<log2Down(x);
}

template<typename UInt, int nSubBits>
inline __host__ UInt reverseSubBits(UInt x) {
  if (nSubBits >= 16 && 8*sizeof(UInt) == nSubBits) {
    switch (8*sizeof(UInt)) {
    case 16: x = __builtin_bswap16(x); break;
    case 32: x = __builtin_bswap32(x); break;
    case 64: x = __builtin_bswap64(x); break;
    default: static_assert(8*sizeof(UInt) <= 64, "Unsupported integer type.");
    }
    return reverseSubBits<UInt, 8>(x);
  } else if (nSubBits == 1) {
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
inline __host__ __device__ Int reverseBits(Int x, int nBits) {
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

inline __host__ __device__ uint32_t u32fpEncode(uint32_t x, int bitsPerPow2) {
  int log2x;
  #if __CUDA_ARCH__
    log2x = 31-__clz(x|1);
  #else
    log2x = 31-__builtin_clz(x|1);
  #endif
  uint32_t mantissa = x>>(log2x >= bitsPerPow2 ? log2x-bitsPerPow2 : 0) & ((1u<<bitsPerPow2)-1);
  uint32_t exponent = log2x >= bitsPerPow2 ? log2x-(bitsPerPow2-1) : 0;
  return exponent<<bitsPerPow2 | mantissa;
}

inline __host__ __device__ uint32_t u32fpDecode(uint32_t x, int bitsPerPow2) {
  uint32_t exponent = x>>bitsPerPow2;
  uint32_t mantissa = (x & ((1u<<bitsPerPow2)-1)) | (exponent!=0 ? 0x8 : 0);
  if (exponent != 0) exponent -= 1;
  return mantissa<<exponent;
}

constexpr uint32_t u32fp8MaxValue() { return 0xf0000000; }

inline __host__ __device__ uint8_t u32fp8Encode(uint32_t x) {
  return u32fpEncode(x, 3);
}
inline __host__ __device__ uint32_t u32fp8Decode(uint8_t x) {
  return u32fpDecode(x, 3);
}

inline __host__ __device__ uint64_t getHash(const char* string, int n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

#endif
