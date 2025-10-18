/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "op128.h"
#include <limits>
#include <type_traits>

template<typename T>
struct IsFloatingPoint: std::false_type {};
template<>
struct IsFloatingPoint<half>: std::true_type {};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct IsFloatingPoint<__nv_bfloat16>: std::true_type {};
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
struct IsFloatingPoint<__nv_fp8_e4m3>: std::true_type {};
template<>
struct IsFloatingPoint<__nv_fp8_e5m2>: std::true_type {};
#endif
template<>
struct IsFloatingPoint<float>: std::true_type {};
template<>
struct IsFloatingPoint<double>: std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// The reduction function classes. All classes must:
//  1. Expose the `EltType` typedef.
//  2. Have constructor taking no arguments (default constructible).
//  3. Have constructor taking `uint64_t opArg`.

template<typename T>
struct FuncCopy { using EltType = T; __device__ __forceinline__ FuncCopy(uint64_t opArg=0) {}; };
template<typename T>
struct FuncSum  { using EltType = T; __device__ __forceinline__ FuncSum(uint64_t opArg=0) {}; };
template<typename T>
struct FuncProd { using EltType = T; __device__ __forceinline__ FuncProd(uint64_t opArg=0) {}; };

template<typename T>
struct FuncMinMax {
  using EltType = T;
  BytePack<sizeof(T)> xormask; // only used by integers
  bool isMinNotMax; // only used by floats
  __device__ __forceinline__ FuncMinMax(uint64_t opArg=0) {
    xormask.native = opArg;
    isMinNotMax = (opArg&1)==0;
  }
};

template<typename T> struct FuncPreMulSum;
template<typename T> struct FuncSumPostDiv;

////////////////////////////////////////////////////////////////////////////////
// Trait class for handling the reduction argument.

template<typename Fn>
struct RedOpArg { // default case: no argument
  static constexpr bool ArgUsed = false;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) { return 0; }
};

template<typename T>
struct RedOpArg<FuncMinMax<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) {
    union { uint64_t u64; T val; };
    u64 = 0;
    val = *(T*)ptr;
    return u64;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Trait classes for reduction functions. Given a function (FuncSum, etc.)
// and a number of elements in a pack, will reduce, preOp, or postOp a pack
// of elements. These classes are intended to be specialized for specific
// combinations of reduction function and pack size.

template<typename A, typename B, int EltPerPackA>
struct Apply_Cast/*{
  static BytePack<EltPerPackA*sizeof(B)/sizeof(A)> cast(BytePack<EltPerPackA*sizeof(A)> a);
}*/;

template<typename Fn, int EltPerPack>
struct Apply_Reduce /*{
  static BytePack<EltPerPack*sizeof(T)> reduce(
    Fn fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b
  );
}*/;
template<typename Fn, int EltPerPack>
struct Apply_PreOp/*{
  static constexpr bool IsIdentity;
  static BytePack<EltPerPack*sizeof(T)> preOp(Fn fn, BytePack<EltPerPack*sizeof(T)> a);
}*/;
template<typename Fn, int EltPerPack>
struct Apply_PostOp/*{
  static constexpr bool IsIdentity;
  static BytePack<EltPerPack*sizeof(T)> postOp(Fn fn, BytePack<EltPerPack*sizeof(T)> a);
}*/;
template<typename Fn>
struct LoadMultimem_BigPackSize/*{
  // If non-zero, then this and sizeof(T) are valid pack sizes for LoadMultimem,
  // otherwise there are no valid pack sizes for LoadMultimem.
  static constexpr int BigPackSize = 0;
}*/;
template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem/*{
  static BytePack<BytePerPack> load(Fn fn, uintptr_t addr);
}*/;


// Helpers for dealing with BytePack<0>'s
template<typename A, typename B, int EltPerPack>
struct Apply_Cast_MaybeEmpty: Apply_Cast<A, B, EltPerPack> {};
template<typename A, typename B>
struct Apply_Cast_MaybeEmpty<A, B, /*EltPerPack=*/0> {
  __device__ constexpr static BytePack<0> cast(BytePack<0> a) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_Reduce_MaybeEmpty: Apply_Reduce<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_Reduce_MaybeEmpty<Fn, 0> {
  __device__ constexpr static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PreOp_MaybeEmpty: Apply_PreOp<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_PreOp_MaybeEmpty<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> preOp(Fn fn, BytePack<0> a) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PostOp_MaybeEmpty: Apply_PostOp<Fn, EltPerPack> {};
template<typename Fn>
struct Apply_PostOp_MaybeEmpty<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __device__ constexpr static BytePack<0> postOp(Fn fn, BytePack<0> a) { return {}; }
};

template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem_MaybeEmpty: Apply_LoadMultimem<Fn, BytePerPack> {};
template<typename Fn>
struct Apply_LoadMultimem_MaybeEmpty<Fn, 0> {
  __device__ constexpr static BytePack<0> load(Fn fn, uintptr_t addr) { return {}; }
};

////////////////////////////////////////////////////////////////////////////////
// Public API for calling the trait classes. These take the data elements as a
// pack of any type, which could be a BytePack<?> or any integral type (uint64_t,
// uint32_t, etc.), and will return a new pack where each element has been
// transformed appropriately.

template<typename A, typename B, typename PackA>
__device__ __forceinline__ BytePack<BytePackOf<PackA>::Size*sizeof(B)/sizeof(A)> applyCast(PackA a) {
  return Apply_Cast_MaybeEmpty<A, B, BytePackOf<PackA>::Size/sizeof(A)>::cast(toPack(a));
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyReduce(Fn fn, Pack a, Pack b) {
  return fromPack<Pack>(
    Apply_Reduce_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::reduce(fn, toPack(a), toPack(b))
  );
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPreOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PreOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::preOp(fn, toPack(a))
  );
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPostOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PostOp_MaybeEmpty<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::postOp(fn, toPack(a))
  );
}

template<typename Fn, int BytePerPack>
__device__ __forceinline__ BytePack<BytePerPack> applyLoadMultimem(Fn fn, uintptr_t addr) {
  return Apply_LoadMultimem_MaybeEmpty<Fn, BytePerPack>::load(fn, addr);
}

////////////////////////////////////////////////////////////////////////////////
// Apply_Cast

template<typename A, typename B, int EltPerPack>
struct Apply_Cast {
  __device__ __forceinline__ static BytePack<EltPerPack*sizeof(B)> cast(BytePack<EltPerPack*sizeof(A)> a) {
    BytePack<EltPerPack*sizeof(B)> b;
    b.half[0] = Apply_Cast<A, B, EltPerPack/2>::cast(a.half[0]);
    b.half[1] = Apply_Cast<A, B, EltPerPack/2>::cast(a.half[1]);
    return b;
  }
};

template<typename A, typename B>
struct Apply_Cast<A, B, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(B)> cast(BytePack<sizeof(A)> a) {
    return toPack(B(fromPack<A>(a)));
  }
};

template<>
struct Apply_Cast<__half, float, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(float)> cast(BytePack<sizeof(__half)> a) {
    return toPack(__half2float(fromPack<__half>(a)));
  }
};
template<>
struct Apply_Cast<float, __half, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(__half)> cast(BytePack<sizeof(float)> a) {
    return toPack(__float2half_rn(fromPack<float>(a)));
  }
};

template<>
struct Apply_Cast<__half, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4*2> cast(BytePack<2*2> a) {
    return toPack(__half22float2(fromPack<__half2>(a)));
  }
};
template<>
struct Apply_Cast<float, __half, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<2*2> cast(BytePack<4*2> a) {
    return toPack(__float22half2_rn(fromPack<float2>(a)));
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__) && (CUDART_RUNTIME >= 12000 || __CUDA_ARCH__ >= 800)
template<>
struct Apply_Cast<__nv_bfloat16, float, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<4*2> cast(BytePack<2*2> a) {
    return toPack(__bfloat1622float2(fromPack<__nv_bfloat162>(a)));
  }
};
template<>
struct Apply_Cast<float ,__nv_bfloat16, /*EltPerPack=*/2> {
  __device__ __forceinline__ static BytePack<2*2> cast(BytePack<4*2> a) {
    return toPack(__float22bfloat162_rn(fromPack<float2>(a)));
  }
};
#endif

#define EASY_CAST(A, B, EltPerPack, VecA, VecB) \
  template<> \
  struct Apply_Cast<A, B, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(B)*EltPerPack> cast(BytePack<sizeof(A)*EltPerPack> a) { \
      return toPack(VecB(fromPack<VecA>(a))); \
    } \
  }; \
  template<> \
  struct Apply_Cast<B, A, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(A)*EltPerPack> cast(BytePack<sizeof(B)*EltPerPack> b) { \
      return toPack(VecA(fromPack<VecB>(b))); \
    } \
  };

#if defined(__CUDA_FP8_TYPES_EXIST__)
EASY_CAST(__nv_fp8_e5m2, float, 2, __nv_fp8x2_e5m2, float2)
EASY_CAST(__nv_fp8_e5m2, float, 4, __nv_fp8x4_e5m2, float4)

EASY_CAST(__nv_fp8_e4m3, float, 2, __nv_fp8x2_e4m3, float2)
EASY_CAST(__nv_fp8_e4m3, float, 4, __nv_fp8x4_e4m3, float4)
#endif
#undef EASY_CAST

////////////////////////////////////////////////////////////////////////////////
// Apply_Reduce

// Nonsensical base case
template<typename Fn>
struct Apply_Reduce<Fn, /*EltPerPack=*/0> {
  __device__ __forceinline__ static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) {
    return  {};
  }
};

// General recursive definition (EltPerPack > 1). This is how we iterate over
// all elements in a pack of any size, by breaking it into halves. Eventually
// we'll hit a base case (a more specific template specialization which takes
// precedence).
template<typename Fn, int EltPerPack>
struct Apply_Reduce {
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> reduce(Fn fn, BytePack<Size> a, BytePack<Size> b) {
    a.half[0] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[0], b.half[0]);
    a.half[1] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[1], b.half[1]);
    return a;
  }
};

// Base case definitions (EltPerPack == 1)
template<typename T>
struct Apply_Reduce<FuncCopy<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncCopy<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return a;
  }
};
template<typename T>
struct Apply_Reduce<FuncSum<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncSum<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) + fromPack<T>(b));
  }
};
template<typename T>
struct Apply_Reduce<FuncProd<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncProd<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) * fromPack<T>(b));
  }
};
template<typename T>
struct Apply_Reduce<FuncMinMax<T>, /*EltPerPack=*/1> {
  __device__ __forceinline__ static BytePack<sizeof(T)> reduce(FuncMinMax<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return (a.native ^ fn.xormask.native) < (b.native ^ fn.xormask.native) ? a : b;
  }
};

// Optimizations for specfic types and element count combinations:
template<>
struct Apply_Reduce<FuncSum<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncSum<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t even = 0x00ff00ffu;
    uint32_t x = (a.native &  even) + (b.native &  even);
    uint32_t y = (a.native & ~even) + (b.native & ~even);
    //a.native = (x & even) | (y & ~even);
    a.native = __byte_perm(x, y, 0x7250);
    return a;
  }
};

template<>
struct Apply_Reduce<FuncMinMax<uint8_t>, /*EltPerPack=*/4> {
  __device__ static BytePack<4> reduce(FuncMinMax<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
    constexpr uint32_t ones = 0x01010101u;
    constexpr uint32_t even = 0x00ff00ffu; // even byte mask
    // Replicate xormask to all bytes
    uint32_t x = fn.xormask.native * ones;
    // Transform inputs by xormask
    uint32_t ax = a.native ^ x;
    uint32_t bx = b.native ^ x;
    // Use 9-bit arithmetic to compute d=a-b
    uint32_t d0 = (ax    & even) + (~bx      & even) + ones;
    uint32_t d1 = (ax>>8 & even) + (~(bx>>8) & even) + ones;
    // Move sign bit of each 9-bit delta into the least bit of origin byte
    //uint32_t s = (d0>>8 & ones & even) | (d1 & ones & ~even);
    uint32_t s = __byte_perm(d0, d1, 0x7351) & ones;
    // Broadcast least bit across whole byte
    s *= 0xffu;
    // Compose result by selecting bytes via: signbit(a-b)==1 ? a : b
    a.native = (a.native & s) | (b.native & ~s);
    return a;
  }
};

template<>
struct Apply_Reduce<FuncProd<uint8_t>, /*EltPerPack=*/4> {
  __device__ __forceinline__ static BytePack<4> reduce(FuncProd<uint8_t> fn, BytePack<4> apack, BytePack<4> bpack) {
    uint32_t a = apack.native;
    uint32_t b = bpack.native;
    uint32_t ab0 = (a*b) & 0xffu;
    asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(ab0) : "r"(a&0xff00u), "r"(b&0xff00u));
    uint32_t ab1;
    asm volatile("mul.hi.u32 %0, %1, %2;"     : "=r"(ab1) : "r"(a&0xff0000), "r"(b&0xff0000));
    asm volatile("mad.hi.u32 %0, %1, %2, %0;" : "+r"(ab1) : "r"(a&0xff000000u), "r"(b&0xff000000u));
    apack.native = __byte_perm(ab0, ab1, 0x6420);
    return apack;
  }
};

#define SPECIALIZE_REDUCE(Fn, T, EltPerPack, Vec, expr_of_fn_x_y) \
  template<> \
  struct Apply_Reduce<Fn<T>, EltPerPack> { \
    __device__ __forceinline__ static BytePack<sizeof(Vec)> reduce( \
        Fn<T> fn, BytePack<sizeof(Vec)> a, BytePack<sizeof(Vec)> b \
      ) { \
      Vec x = fromPack<Vec>(a); \
      Vec y = fromPack<Vec>(b); \
      return toPack<Vec>(expr_of_fn_x_y); \
    } \
  };

SPECIALIZE_REDUCE(FuncMinMax, float, 1, float, fn.isMinNotMax ? fminf(x, y) : fmaxf(x, y))
SPECIALIZE_REDUCE(FuncMinMax, double, 1, double, fn.isMinNotMax ? fmin(x, y) : fmax(x, y))

#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  SPECIALIZE_REDUCE(FuncSum, half, 1, half, __hadd(x, y))
  // Coverity recommends the use of std::move here but, given that half is a scalar,
  // a plain copy will be just as efficient.
  // coverity[copy_constructor_call]
  SPECIALIZE_REDUCE(FuncSum, half, 2, half2, __hadd2(x, y))
  SPECIALIZE_REDUCE(FuncProd, half, 1, half, __hmul(x, y))
  // coverity[copy_constructor_call]
  SPECIALIZE_REDUCE(FuncProd, half, 2, half2, __hmul2(x, y))
#else
  SPECIALIZE_REDUCE(FuncSum, half, 1, half, __float2half(__half2float(x) + __half2float(y)))
  SPECIALIZE_REDUCE(FuncProd, half, 1, half, __float2half(__half2float(x) * __half2float(y)))
#endif

#if __CUDA_ARCH__ >= 800
  SPECIALIZE_REDUCE(FuncMinMax, half, 1, half, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
  // coverity[copy_constructor_call]
  SPECIALIZE_REDUCE(FuncMinMax, half, 2, half2, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))
#else
  SPECIALIZE_REDUCE(FuncMinMax, half, 1, half, __float2half(fn.isMinNotMax ? fminf(__half2float(x), __half2float(y)) : fmaxf(__half2float(x), __half2float(y))))
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 800
  SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __hadd(x, y))
  // coverity[copy_constructor_call]
  SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 2, __nv_bfloat162, __hadd2(x, y))
  SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 1, __nv_bfloat16, __hmul(x, y))
  // coverity[copy_constructor_call]
  SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 2, __nv_bfloat162, __hmul2(x, y))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 1, __nv_bfloat16, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
  // coverity[copy_constructor_call]
  SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 2, __nv_bfloat162, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))
#else
  SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(__bfloat162float(x) + __bfloat162float(y)))
  SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(__bfloat162float(x) * __bfloat162float(y)))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(fn.isMinNotMax ? fminf(__bfloat162float(x), __bfloat162float(y)) : fmaxf(__bfloat162float(x), __bfloat162float(y))))
#endif
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
  SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e4m3, 1, __nv_fp8_e4m3, __nv_fp8_e4m3(__hadd(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e4m3, 2, __nv_fp8x2_e4m3, __nv_fp8x2_e4m3(__hadd2(__half2(x),__half2(y))))
  SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e4m3, 1, __nv_fp8_e4m3, __nv_fp8_e4m3(__hmul(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e4m3, 2, __nv_fp8x2_e4m3, __nv_fp8x2_e4m3(__hmul2(__half2(x),__half2(y))))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e4m3, 1, __nv_fp8_e4m3, __nv_fp8_e4m3(fn.isMinNotMax ? __hmin(__half(x),__half(y)) : __hmax(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e4m3, 2, __nv_fp8x2_e4m3, __nv_fp8x2_e4m3(fn.isMinNotMax ? __hmin2(__half2(x),__half2(y)) : __hmax2(__half2(x),__half2(y))))

  SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e5m2, 1, __nv_fp8_e5m2, __nv_fp8_e5m2(__hadd(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncSum, __nv_fp8_e5m2, 2, __nv_fp8x2_e5m2, __nv_fp8x2_e5m2(__hadd2(__half2(x),__half2(y))))
  SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e5m2, 1, __nv_fp8_e5m2, __nv_fp8_e5m2(__hmul(__half(x),__half(y))))
  SPECIALIZE_REDUCE(FuncProd, __nv_fp8_e5m2, 2, __nv_fp8x2_e5m2, __nv_fp8x2_e5m2(__hmul2(__half2(x),__half2(y))))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e5m2, 1, __nv_fp8_e5m2, __nv_fp8_e5m2(fn.isMinNotMax ? __hmin(__half(x), __half(y)) : __hmax(__half(x), __half(y))))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_fp8_e5m2, 2, __nv_fp8x2_e5m2, __nv_fp8x2_e5m2(fn.isMinNotMax ? __hmin2(__half2(x), __half2(y)) : __hmax2(__half2(x), __half2(y))))
#endif
#endif

#undef SPECIALIZE_REDUCE

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp

// General recursive definition (EltPerPack > 1)
template<typename Fn, int EltPerPack>
struct Apply_PreOp {
  static constexpr bool IsIdentity = Apply_PreOp<Fn, EltPerPack/2>::IsIdentity;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    #if __cpp_if_constexpr
    if constexpr(!IsIdentity) {
    #else
    if (!IsIdentity) {
    #endif
      // The `if (!IsIdentity)` condition is not strictly necessary, but it may help
      // compiler in that it won't have to tear a register apart for no reason
      // just to put it back together again.
      a.half[0] = Apply_PreOp<Fn, EltPerPack/2>::preOp(fn, a.half[0]);
      a.half[1] = Apply_PreOp<Fn, EltPerPack/2>::preOp(fn, a.half[1]);
    }
    return a;
  }
};
// Base case definition (EltPerPack == 1), by default is identity function.
template<typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = true;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// Base case definition (EltPerPack == 0), is nonsense!
template<typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ __forceinline__ static BytePack<0> preOp(Fn fn, BytePack<0> a) {
    return {};
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PostOp

// General recursive definition (EltPerPack > 1)
template<typename Fn, int EltPerPack>
struct Apply_PostOp {
  static constexpr bool IsIdentity = Apply_PostOp<Fn, EltPerPack/2>::IsIdentity;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    #if __cpp_if_constexpr
    if constexpr(!IsIdentity) {
    #else
    if (!IsIdentity) {
    #endif
      // The `if (!IsIdentity)` condition is not strictly necessary, but it may help
      // compiler in that it won't have to tear a register apart for no reason
      // just to put it back together again.
      a.half[0] = Apply_PostOp<Fn, EltPerPack/2>::postOp(fn, a.half[0]);
      a.half[1] = Apply_PostOp<Fn, EltPerPack/2>::postOp(fn, a.half[1]);
    }
    return a;
  }
};
// Base case definition (EltPerPack == 1), by default is identity function.
template<typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = true;
  template<int Size>
  __device__ __forceinline__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// Base case definition (EltPerPack == 0), is nonsense!
template<typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ __forceinline__ static BytePack<0> postOp(Fn fn, BytePack<0> a) {
    return {};
  }
};


////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum

template<typename T>
struct RedOpArg<FuncPreMulSum<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) {
    union { uint64_t u64; T val; };
    u64 = 0;
    val = *(T*)ptr;
    return u64;
  }
};

// General definition for all integral types, float, and double.
template<typename T>
struct FuncPreMulSum {
  using EltType = T;
  T scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; T val; };
    u64 = opArg;
    scalar = val;
  }
};

template<>
// Coverity recommends the users of this type to use std::move in certain cases but,
// given that half is a scalar, a plain copy will be just as efficient.
// coverity[moveable_type]
struct FuncPreMulSum<half> {
  using EltType = half;
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  __half2 scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; __half val; };
    u64 = opArg;
    scalar.x = val;
    scalar.y = val;
  }
#else
  float scalar;
  __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; __half val; };
    u64 = opArg;
    scalar = (float)val;
  }
#endif
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
  template<>
  // Coverity recommends the users of this type to use std::move in certain cases but,
  // given that __nv_bfloat16 is a scalar, a plain copy will be just as efficient.
  // coverity[moveable_type]
  struct FuncPreMulSum<__nv_bfloat16> {
    using EltType = __nv_bfloat16;
  #if __CUDA_ARCH__ >= 800
    __nv_bfloat162 scalar;
    __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
      union { uint64_t u64; __nv_bfloat16 val; };
      u64 = opArg;
      scalar.x = val;
      scalar.y = val;
    }
  #else
    float scalar;
    __device__ __forceinline__ FuncPreMulSum(uint64_t opArg=0) {
      union { uint64_t u64; __nv_bfloat16 val; };
      u64 = opArg;
      scalar = __bfloat162float(val);
    }
  #endif
  };
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
  template<>
  struct FuncPreMulSum<__nv_fp8_e4m3> {
    using EltType = __nv_fp8_e4m3;
    __half2 scalar2;
    __device__ __forceinline__ FuncPreMulSum(uint64_t opArg) {
      union { uint64_t u64; __nv_fp8_storage_t val; };
      u64 = opArg;
      scalar2.x = __half(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3));
      scalar2.y = scalar2.x;
    }
  };

  template<>
  struct FuncPreMulSum<__nv_fp8_e5m2> {
    using EltType = __nv_fp8_e5m2;
    __half2 scalar2;
    __device__ __forceinline__ FuncPreMulSum(uint64_t opArg) {
      union { uint64_t u64; __nv_fp8_storage_t val; };
      u64 = opArg;
      scalar2.x = __half(__nv_cvt_fp8_to_halfraw(val, __NV_E5M2));
      scalar2.y = scalar2.x;
    }
  };
#endif
#endif

template<typename T, int EltPerPack>
struct Apply_Reduce<FuncPreMulSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack*sizeof(T)> reduce(FuncPreMulSum<T> fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b) {
    // FuncPreMulSum reduce dispatches to FuncSum.
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

// PreOp of FuncPreMulSum for integral types, float, and double.
template<typename T>
struct Apply_PreOp<FuncPreMulSum<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> preOp(FuncPreMulSum<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fromPack<T>(a) * fn.scalar);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for float16.

template<>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(half)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half)> a) {
    #if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
      return toPack<half>(__hmul(fromPack<half>(a), fn.scalar.x));
    #else
      return toPack<half>(__float2half(__half2float(fromPack<half>(a)) * fn.scalar));
    #endif
  }
};
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  template<>
  struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/2> {
    static constexpr bool IsIdentity = false;
    __device__ __forceinline__ static BytePack<sizeof(half2)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half2)> a) {
      return toPack<half2>(__hmul2(fromPack<half2>(a), fn.scalar));
    }
  };
#endif

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for bfloat16.

#if defined(__CUDA_BF16_TYPES_EXIST__)
  template<>
  struct Apply_PreOp<FuncPreMulSum<__nv_bfloat16>, /*EltPerPack=*/1> {
    static constexpr bool IsIdentity = false;
    __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat16)> preOp(
        FuncPreMulSum<__nv_bfloat16> fn, BytePack<sizeof(__nv_bfloat16)> a
      ) {
      #if __CUDA_ARCH__ >= 800
        return toPack<__nv_bfloat16>(__hmul(fromPack<__nv_bfloat16>(a), fn.scalar.x));
      #else
        return toPack<__nv_bfloat16>(__float2bfloat16(__bfloat162float(fromPack<__nv_bfloat16>(a)) * fn.scalar));
      #endif
    }
  };
  #if __CUDA_ARCH__ >= 800
    template<>
    struct Apply_PreOp<FuncPreMulSum<__nv_bfloat16>, /*EltPerPack=*/2> {
      static constexpr bool IsIdentity = false;
      __device__ __forceinline__ static BytePack<sizeof(__nv_bfloat162)> preOp(
          FuncPreMulSum<__nv_bfloat16> fn, BytePack<sizeof(__nv_bfloat162)> a
        ) {
        return toPack<__nv_bfloat162>(__hmul2(fromPack<__nv_bfloat162>(a), fn.scalar));
      }
    };
  #endif
#endif

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for fp8.

#if defined(__CUDA_FP8_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 900
  template<>
  struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e4m3>, /*EltPerPack=*/1> {
    static constexpr bool IsIdentity = false;
    __device__ __forceinline__ static BytePack<sizeof(__nv_fp8_e4m3)> preOp(
        FuncPreMulSum<__nv_fp8_e4m3> fn, BytePack<sizeof(__nv_fp8_e4m3)> a
      ) {
      return toPack<__nv_fp8_e4m3>(__nv_fp8_e4m3(__hmul(__half(fromPack<__nv_fp8_e4m3>(a)), fn.scalar2.x)));
    }
  };
  template<>
  struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e4m3>, /*EltPerPack=*/2> {
    static constexpr bool IsIdentity = false;
    __device__ __forceinline__ static BytePack<sizeof(__nv_fp8x2_e4m3)> preOp(
        FuncPreMulSum<__nv_fp8_e4m3> fn, BytePack<sizeof(__nv_fp8x2_e4m3)> a
      ) {
      return toPack<__nv_fp8x2_e4m3>(__nv_fp8x2_e4m3(__hmul2(__half2(fromPack<__nv_fp8x2_e4m3>(a)), fn.scalar2)));
    }
  };

  template<>
  struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e5m2>, /*EltPerPack=*/1> {
    static constexpr bool IsIdentity = false;
    __device__ __forceinline__ static BytePack<sizeof(__nv_fp8_e5m2)> preOp(
        FuncPreMulSum<__nv_fp8_e5m2> fn, BytePack<sizeof(__nv_fp8_e5m2)> a
      ) {
      return toPack<__nv_fp8_e5m2>(__nv_fp8_e5m2(__hmul(__half(fromPack<__nv_fp8_e5m2>(a)), fn.scalar2.x)));
    }
  };
  template<>
  struct Apply_PreOp<FuncPreMulSum<__nv_fp8_e5m2>, /*EltPerPack=*/2> {
    static constexpr bool IsIdentity = false;
    __device__ __forceinline__ static BytePack<sizeof(__nv_fp8x2_e5m2)> preOp(
        FuncPreMulSum<__nv_fp8_e5m2> fn, BytePack<sizeof(__nv_fp8x2_e5m2)> a
      ) {
      return toPack<__nv_fp8x2_e5m2>(__nv_fp8x2_e5m2(__hmul2(__half2(fromPack<__nv_fp8x2_e5m2>(a)), fn.scalar2)));
    }
  };
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
// FuncSumPostDiv

template<typename T>
struct RedOpArg<FuncSumPostDiv<T>> {
  static constexpr bool ArgUsed = true;
  __device__ __forceinline__ static uint64_t loadArg(void *ptr) {
    return *(uint64_t*)ptr;
  }
};

template<typename T>
struct FuncSumPostDiv {
  static_assert(T(0) < T(-1), "FuncSumPostDiv is only for implementing ncclAvg on uint types.");
  using EltType = T;
  using UintType = typename std::conditional<sizeof(T)==8, uint64_t, uint32_t>::type;
  uint32_t divisor:31, isSigned:1;
  UintType recip;

  __device__ __forceinline__ FuncSumPostDiv(uint64_t opArg=0) {
    isSigned = opArg & 1;
    divisor = opArg >> 1;
    recip =  UintType(-1)/divisor;
  }
  __device__ __forceinline__ T divide(T x) {
    // x is negative iff we are in signed mode and the top bit is set
    bool xneg = isSigned && (x & ~(T(-1)>>1));
    // Compute abs(x):
    // T(-x) vs -T(x) is critical. We have to negate then truncate the bits. Consider
    // if we are doing signed 8-bit types, thus T=uint8_t. The value -1 is encoded
    // as 0xff. -T(0xff) when promoted to 32-bit (which is implicit by compiler)
    // gives 0xffffff01, but T(-0xff) is 0x1, and that is the abs value we want.
    UintType xabs = xneg ? T(-x) : x;
    // Compute quotient by multiplying by reciprical.
    UintType q = sizeof(T)==8 ? __umul64hi(xabs, recip) : __umulhi(xabs, recip);
    // Quotient may be off by one so do a fixup.
    if (xabs - q*divisor >= divisor) q += 1;
    // If original x was negative then we have to negate it back since we were
    // working with its abs val.
    return xneg ? -T(q) : T(q);
  }
};

template<typename T, int EltPerPack>
struct Apply_Reduce<FuncSumPostDiv<T>, EltPerPack>:
    Apply_Reduce<FuncSum<T>, EltPerPack> {
  __device__ __forceinline__ static BytePack<EltPerPack*sizeof(T)> reduce(FuncSumPostDiv<T> fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b) {
    // FuncSumPostDiv reduce dispatches to FuncSum.
    return Apply_Reduce<FuncSum<T>, EltPerPack>::reduce(FuncSum<T>(), a, b);
  }
};

template<typename T>
struct Apply_PostOp<FuncSumPostDiv<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ __forceinline__ static BytePack<sizeof(T)> postOp(FuncSumPostDiv<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fn.divide(fromPack<T>(a)));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_LoadMultimem

#define RegCode_for_size_1 "r"
#define RegCode_for_size_2 "h"
#define RegCode_for_size_4 "r"
#define RegCode_for_size_8 "l"

#define RegSize_for_size_1 4
#define RegSize_for_size_2 2
#define RegSize_for_size_4 4
#define RegSize_for_size_8 8

#define PtxAcc_for_u32
#define PtxAcc_for_s32
#define PtxAcc_for_s64
#define PtxAcc_for_u64
#define PtxAcc_for_f32
#define PtxAcc_for_f64
#if CUDART_VERSION >= 12020
  #define PtxAcc_for_f16 ".acc::f32"
  #define PtxAcc_for_bf16 ".acc::f32"
  #define PtxAcc_for_f16x2 ".acc::f32"
  #define PtxAcc_for_bf16x2 ".acc::f32"
#else
  #define PtxAcc_for_f16
  #define PtxAcc_for_bf16
  #define PtxAcc_for_f16x2
  #define PtxAcc_for_bf16x2
#endif
#define PtxAcc_for_e4m3 ".acc::f16"
#define PtxAcc_for_e5m2 ".acc::f16"
#define PtxAcc_for_e4m3x4 ".acc::f16"
#define PtxAcc_for_e5m2x4 ".acc::f16"

#define DEFINE_Apply_LoadMultimem_sum(T, ptx_ty, PackSize) \
  template<> \
  struct Apply_LoadMultimem<FuncSum<T>, PackSize> { \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncSum<T> fn, uintptr_t addr) { \
      BytePack<RegSize_for_size_##PackSize> reg; \
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add" PtxAcc_for_##ptx_ty "." #ptx_ty " %0, [%1];" \
        : "=" RegCode_for_size_##PackSize(reg.native) \
        : "l"(addr) : "memory"); \
      BytePack<PackSize> ans; \
      ans.native = reg.native; \
      return ans; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax(T, ptx_ty, PackSize) \
  template<> \
  struct Apply_LoadMultimem<FuncMinMax<T>, PackSize> { \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncMinMax<T> fn, uintptr_t addr) { \
      BytePack<RegSize_for_size_##PackSize> reg; \
      if (fn.isMinNotMax) { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min." #ptx_ty " %0, [%1];" \
          : "=" RegCode_for_size_##PackSize(reg.native) \
          : "l"(addr) : "memory"); \
      } else { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max." #ptx_ty " %0, [%1];" \
          : "=" RegCode_for_size_##PackSize(reg.native) \
          : "l"(addr) : "memory"); \
      } \
      BytePack<PackSize> ans; \
      ans.native = reg.native; \
      return ans; \
    } \
  };

#define DEFINE_Apply_LoadMultimem_sum_v4(T, ptx_ty, VecEltSize) \
  template<> \
  struct Apply_LoadMultimem<FuncSum<T>, 4*(VecEltSize)> { \
    static constexpr int PackSize = 4*(VecEltSize); \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncSum<T> fn, uintptr_t addr) { \
      union { BytePack<PackSize> ans; BytePack<VecEltSize> elts[4]; }; \
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add" PtxAcc_for_##ptx_ty ".v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
        : "=" RegCode_for_size_##VecEltSize(elts[0].native), \
          "=" RegCode_for_size_##VecEltSize(elts[1].native), \
          "=" RegCode_for_size_##VecEltSize(elts[2].native), \
          "=" RegCode_for_size_##VecEltSize(elts[3].native) \
        : "l"(addr) : "memory"); \
      return ans; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax_v4(T, ptx_ty, VecEltSize) \
  template<> \
  struct Apply_LoadMultimem<FuncMinMax<T>, 4*(VecEltSize)> { \
    static constexpr int PackSize = 4*(VecEltSize); \
    __device__ __forceinline__ static BytePack<PackSize> load(FuncMinMax<T> fn, uintptr_t addr) { \
      union { BytePack<PackSize> ans; BytePack<VecEltSize> elts[4]; }; \
      if (fn.isMinNotMax) { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
          : "=" RegCode_for_size_##VecEltSize(elts[0].native), \
            "=" RegCode_for_size_##VecEltSize(elts[1].native), \
            "=" RegCode_for_size_##VecEltSize(elts[2].native), \
            "=" RegCode_for_size_##VecEltSize(elts[3].native) \
          : "l"(addr) : "memory"); \
      } else { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
          : "=" RegCode_for_size_##VecEltSize(elts[0].native), \
            "=" RegCode_for_size_##VecEltSize(elts[1].native), \
            "=" RegCode_for_size_##VecEltSize(elts[2].native), \
            "=" RegCode_for_size_##VecEltSize(elts[3].native) \
          : "l"(addr) : "memory"); \
      } \
      return ans; \
    } \
  };

#define DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(T, ptx_ty, VecEltSize) \
  DEFINE_Apply_LoadMultimem_sum_v4(T, ptx_ty, VecEltSize) \
  template<> \
  struct Apply_LoadMultimem<FuncSum<T>, sizeof(T)> { \
    __device__ __forceinline__ static BytePack<sizeof(T)> load(FuncSum<T> fn, uintptr_t addr) { \
      union { BytePack<VecEltSize> tmp; BytePack<sizeof(T)> elts[(VecEltSize)/sizeof(T)]; }; \
      asm volatile("multimem.ld_reduce.relaxed.sys.global.add" PtxAcc_for_##ptx_ty "." #ptx_ty " %0, [%1];" \
        : "=" RegCode_for_size_##VecEltSize(tmp.native) \
        : "l"(addr & -uintptr_t(VecEltSize)) : "memory"); \
      return elts[(addr/sizeof(T))%((VecEltSize)/sizeof(T))]; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(T, ptx_ty, VecEltSize) \
  DEFINE_Apply_LoadMultimem_minmax_v4(T, ptx_ty, VecEltSize) \
  template<> \
  struct Apply_LoadMultimem<FuncMinMax<T>, sizeof(T)> { \
    __device__ __forceinline__ static BytePack<sizeof(T)> load(FuncMinMax<T> fn, uintptr_t addr) { \
      union { BytePack<VecEltSize> tmp; BytePack<sizeof(T)> elts[(VecEltSize)/sizeof(T)]; }; \
      if (fn.isMinNotMax) { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.min." #ptx_ty " %0, [%1];" \
          : "=" RegCode_for_size_##VecEltSize(tmp.native) \
          : "l"(addr & -uintptr_t(VecEltSize)) : "memory"); \
      } else { \
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max." #ptx_ty " %0, [%1];" \
          : "=" RegCode_for_size_##VecEltSize(tmp.native) \
          : "l"(addr & -uintptr_t(VecEltSize)) : "memory"); \
      } \
      return elts[(addr/sizeof(T))%((VecEltSize)/sizeof(T))]; \
    } \
  };

template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem {
  __device__ __forceinline__ static BytePack<BytePerPack> load(Fn fn, uintptr_t addr) {
    __trap();
    return {};
  }
};

#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
  template<typename Fn>
  struct LoadMultimem_BigPackSize {
    using T = typename Fn::EltType;
    static constexpr bool IsSum = std::is_same<Fn, FuncSum<T>>::value ||
                                  std::is_same<Fn, FuncPreMulSum<T>>::value ||
                                  std::is_same<Fn, FuncSumPostDiv<T>>::value;
    static constexpr bool IsMinMax = std::is_same<Fn, FuncMinMax<T>>::value;
    static constexpr bool IsFloat = IsFloatingPoint<T>::value;
    static constexpr int BigPackSize =
      IsFloat && IsSum && sizeof(T) < 8 ? 16 :
      IsFloat && IsSum ? sizeof(T) :
      IsFloat && IsMinMax && sizeof(T)==2 ? 16 :
      !IsFloat && (IsSum||IsMinMax) && sizeof(T)>=4 ? sizeof(T) :
      /*multimem.ld_reduce not supported:*/ 0;
  };

  DEFINE_Apply_LoadMultimem_sum(uint32_t, u32, 4)
  DEFINE_Apply_LoadMultimem_minmax(uint32_t, u32, 4)

  DEFINE_Apply_LoadMultimem_sum(int32_t, s32, 4)
  DEFINE_Apply_LoadMultimem_minmax(int32_t, s32, 4)

  DEFINE_Apply_LoadMultimem_sum(uint64_t, u64, 8)
  DEFINE_Apply_LoadMultimem_minmax(uint64_t, u64, 8)

  DEFINE_Apply_LoadMultimem_sum(int64_t, u64, 8)
  DEFINE_Apply_LoadMultimem_minmax(int64_t, s64, 8)

  DEFINE_Apply_LoadMultimem_sum(float, f32, 4)
  DEFINE_Apply_LoadMultimem_sum_v4(float, f32, 4)

  DEFINE_Apply_LoadMultimem_sum(double, f64, 8)

  DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(half, f16x2, 4)
  DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(half, f16x2, 4)

  #if defined(__CUDA_BF16_TYPES_EXIST__)
    DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(__nv_bfloat16, bf16x2, 4)
    DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(__nv_bfloat16, bf16x2, 4)
  #endif

  #if NCCL_CUDA_ARCH_SPECIFIC == 1000 || NCCL_CUDA_ARCH_SPECIFIC == 1010 || NCCL_CUDA_ARCH_FAMILY_SPECIFIC == 1000 || NCCL_CUDA_ARCH_FAMILY_SPECIFIC == 1010 || NCCL_CUDA_ARCH_SPECIFIC == 1200 || NCCL_CUDA_ARCH_SPECIFIC == 1210
    DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(__nv_fp8_e4m3, e4m3x4, 4)
    DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(__nv_fp8_e4m3, e4m3x4, 4)
    DEFINE_Apply_LoadMultimem_sum_v4_and_xparts(__nv_fp8_e5m2, e5m2x4, 4)
    DEFINE_Apply_LoadMultimem_minmax_v4_and_xparts(__nv_fp8_e5m2, e5m2x4, 4)
  #endif
#else
  template<typename Fn>
  struct LoadMultimem_BigPackSize {
    static constexpr int BigPackSize = 0;
  };
#endif

#undef DEFINE_Apply_LoadMultimem
#undef DEFINE_Apply_LoadMultimem_v4
#undef DEFINE_Apply_LoadMultimem_v4x2_and_subhalf

#undef RegCode_for_size_2
#undef RegCode_for_size_4
#undef RegCode_for_size_8

#undef RegSize_for_size_1
#undef RegSize_for_size_2
#undef RegSize_for_size_4
#undef RegSize_for_size_8

#undef PtxAcc_for_u32
#undef PtxAcc_for_s32
#undef PtxAcc_for_s64
#undef PtxAcc_for_u64
#undef PtxAcc_for_f32
#undef PtxAcc_for_f64
#undef PtxAcc_for_f16
#undef PtxAcc_for_bf16
#undef PtxAcc_for_f16x2
#undef PtxAcc_for_bf16x2
#undef PtxAcc_for_e4m3
#undef PtxAcc_for_e5m2
#undef PtxAcc_for_e4m3x4
#undef PtxAcc_for_e5m2x4

#endif // REDUCE_KERNEL_H_
