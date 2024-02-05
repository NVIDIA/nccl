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
struct FuncCopy { using EltType = T; __device__ FuncCopy(uint64_t opArg=0) {}; };
template<typename T>
struct FuncSum  { using EltType = T; __device__ FuncSum(uint64_t opArg=0) {}; };
template<typename T>
struct FuncProd { using EltType = T; __device__ FuncProd(uint64_t opArg=0) {}; };
template<typename T>
struct FuncMinMax {
  using EltType = T;
  BytePack<sizeof(T)> xormask; // only used by integers
  bool isMinNotMax; // only used by floats
  __device__ FuncMinMax(uint64_t opArg=0) {
    xormask.native = opArg;
    isMinNotMax = (opArg&1)==0;
  }
};
template<typename T> struct FuncPreMulSum;
template<typename T> struct FuncSumPostDiv;

////////////////////////////////////////////////////////////////////////////////
// Trait classes for reduction functions. Given a function (FuncSum, etc.)
// and a number of elements in a pack, will reduce, preOp, or postOp a pack
// of elements. These classes are intended to be specialized for specific
// combinations of reduction function and pack size.

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

////////////////////////////////////////////////////////////////////////////////
// Public API for calling the trait classes. These take the data elements as a
// pack of any type, which could be a BytePack<?> or any integral type (uint64_t,
// uint32_t, etc.), and will return a new pack where each element has been
// transformed appropriately.

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyReduce(Fn fn, Pack a, Pack b) {
  return fromPack<Pack>(
    Apply_Reduce<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::reduce(fn, toPack(a), toPack(b))
  );
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPreOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PreOp<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::preOp(fn, toPack(a))
  );
}

template<typename Fn, typename Pack>
__device__ __forceinline__ Pack applyPostOp(Fn fn, Pack a) {
  return fromPack<Pack>(
    Apply_PostOp<Fn, BytePackOf<Pack>::Size/sizeof(typename Fn::EltType)>
      ::postOp(fn, toPack(a))
  );
}

template<typename Fn, int BytePerPack>
__device__ __forceinline__ BytePack<BytePerPack> applyLoadMultimem(Fn fn, uintptr_t addr) {
  return Apply_LoadMultimem<Fn, BytePerPack>::load(fn, addr);
}

////////////////////////////////////////////////////////////////////////////////
// Apply_Reduce

// Nonsensical base case
template<typename Fn>
struct Apply_Reduce<Fn, /*EltPerPack=*/0> {
  __device__ static BytePack<0> reduce(Fn fn, BytePack<0> a, BytePack<0> b) {
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
  __device__ static BytePack<Size> reduce(Fn fn, BytePack<Size> a, BytePack<Size> b) {
    a.half[0] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[0], b.half[0]);
    a.half[1] = Apply_Reduce<Fn, EltPerPack/2>::reduce(fn, a.half[1], b.half[1]);
    return a;
  }
};

// Base case definitions (EltPerPack == 1)
template<typename T>
struct Apply_Reduce<FuncCopy<T>, /*EltPerPack=*/1> {
  __device__ static BytePack<sizeof(T)> reduce(FuncCopy<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return a;
  }
};
template<typename T>
struct Apply_Reduce<FuncSum<T>, /*EltPerPack=*/1> {
  __device__ static BytePack<sizeof(T)> reduce(FuncSum<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) + fromPack<T>(b));
  }
};
template<typename T>
struct Apply_Reduce<FuncProd<T>, /*EltPerPack=*/1> {
  __device__ static BytePack<sizeof(T)> reduce(FuncProd<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return toPack<T>(fromPack<T>(a) * fromPack<T>(b));
  }
};
template<typename T>
struct Apply_Reduce<FuncMinMax<T>, /*EltPerPack=*/1> {
  __device__ static BytePack<sizeof(T)> reduce(FuncMinMax<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    return (a.native ^ fn.xormask.native) < (b.native ^ fn.xormask.native) ? a : b;
  }
};

// Optimizations for specfic types and element count combinations:
template<>
struct Apply_Reduce<FuncSum<uint8_t>, /*EltPerPack=*/4> {
  __device__ static BytePack<4> reduce(FuncSum<uint8_t> fn, BytePack<4> a, BytePack<4> b) {
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
  __device__ static BytePack<4> reduce(FuncProd<uint8_t> fn, BytePack<4> apack, BytePack<4> bpack) {
    uint32_t a = apack.native;
    uint32_t b = bpack.native;
    uint32_t ab0 = (a*b) & 0xffu;
    asm("mad.lo.u32 %0, %1, %2, %0;" : "+r"(ab0) : "r"(a&0xff00u), "r"(b&0xff00u));
    uint32_t ab1;
    asm("mul.hi.u32 %0, %1, %2;"     : "=r"(ab1) : "r"(a&0xff0000), "r"(b&0xff0000));
    asm("mad.hi.u32 %0, %1, %2, %0;" : "+r"(ab1) : "r"(a&0xff000000u), "r"(b&0xff000000u));
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
  SPECIALIZE_REDUCE(FuncSum, half, 2, half2, __hadd2(x, y))
  SPECIALIZE_REDUCE(FuncProd, half, 1, half, __hmul(x, y))
  SPECIALIZE_REDUCE(FuncProd, half, 2, half2, __hmul2(x, y))
#else
  SPECIALIZE_REDUCE(FuncSum, half, 1, half, __float2half(__half2float(x) + __half2float(y)))
  SPECIALIZE_REDUCE(FuncProd, half, 1, half, __float2half(__half2float(x) * __half2float(y)))
#endif

#if __CUDA_ARCH__ >= 800
  SPECIALIZE_REDUCE(FuncMinMax, half, 1, half, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
  SPECIALIZE_REDUCE(FuncMinMax, half, 2, half2, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))
#else
  SPECIALIZE_REDUCE(FuncMinMax, half, 1, half, __float2half(fn.isMinNotMax ? fminf(__half2float(x), __half2float(y)) : fmaxf(__half2float(x), __half2float(y))))
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
#if __CUDA_ARCH__ >= 800
  SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __hadd(x, y))
  SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 2, __nv_bfloat162, __hadd2(x, y))
  SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 1, __nv_bfloat16, __hmul(x, y))
  SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 2, __nv_bfloat162, __hmul2(x, y))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 1, __nv_bfloat16, fn.isMinNotMax ? __hmin(x, y) : __hmax(x, y))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 2, __nv_bfloat162, fn.isMinNotMax ? __hmin2(x, y) : __hmax2(x, y))
#else
  SPECIALIZE_REDUCE(FuncSum, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(__bfloat162float(x) + __bfloat162float(y)))
  SPECIALIZE_REDUCE(FuncProd, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(__bfloat162float(x) * __bfloat162float(y)))
  SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, 1, __nv_bfloat16, __float2bfloat16(fn.isMinNotMax ? fminf(__bfloat162float(x), __bfloat162float(y)) : fmaxf(__bfloat162float(x), __bfloat162float(y))))
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
  __device__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
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
  __device__ static BytePack<Size> preOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// Base case definition (EltPerPack == 0), is nonsense!
template<typename Fn>
struct Apply_PreOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ static BytePack<0> preOp(Fn fn, BytePack<0> a) {
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
  __device__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
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
  __device__ static BytePack<Size> postOp(Fn fn, BytePack<Size> a) {
    return a;
  }
};
// Base case definition (EltPerPack == 0), is nonsense!
template<typename Fn>
struct Apply_PostOp<Fn, /*EltPerPack=*/0> {
  static constexpr bool IsIdentity = true;
  __device__ static BytePack<0> postOp(Fn fn, BytePack<0> a) {
    return {};
  }
};


////////////////////////////////////////////////////////////////////////////////
// FuncPreMulSum

// General definition for all integral types, float, and double.
template<typename T>
struct FuncPreMulSum {
  using EltType = T;
  T scalar;
  __device__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; T val; };
    u64 = opArg;
    scalar = val;
  }
};

template<>
struct FuncPreMulSum<half> {
  using EltType = half;
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  half2 scalar;
  __device__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; half val; };
    u64 = opArg;
    scalar.x = val;
    scalar.y = val;
  }
#else
  float scalar;
  __device__ FuncPreMulSum(uint64_t opArg=0) {
    union { uint64_t u64; half val; };
    u64 = opArg;
    scalar = __half2float(val);
  }
#endif
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
  template<>
  struct FuncPreMulSum<__nv_bfloat16> {
    using EltType = __nv_bfloat16;
  #if __CUDA_ARCH__ >= 800
    __nv_bfloat162 scalar;
    __device__ FuncPreMulSum(uint64_t opArg=0) {
      union { uint64_t u64; __nv_bfloat16 val; };
      u64 = opArg;
      scalar.x = val;
      scalar.y = val;
    }
  #else
    float scalar;
    __device__ FuncPreMulSum(uint64_t opArg=0) {
      union { uint64_t u64; __nv_bfloat16 val; };
      u64 = opArg;
      scalar = __bfloat162float(val);
    }
  #endif
  };
#endif

template<typename T>
struct Apply_Reduce<FuncPreMulSum<T>, /*EltPerPack=*/1> {
  __device__ static BytePack<sizeof(T)> reduce(FuncPreMulSum<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    // FuncPreMulSum reduce dispatches to FuncSum.
    return Apply_Reduce<FuncSum<T>, 1>::reduce(FuncSum<T>(), a, b);
  }
};

// PreOp of FuncPreMulSum for integral types, float, and double.
template<typename T>
struct Apply_PreOp<FuncPreMulSum<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ static BytePack<sizeof(T)> preOp(FuncPreMulSum<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fromPack<T>(a) * fn.scalar);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_PreOp of FuncPreMulSum for float16.

template<>
struct Apply_PreOp<FuncPreMulSum<half>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ static BytePack<sizeof(half)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half)> a) {
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
    __device__ static BytePack<sizeof(half2)> preOp(FuncPreMulSum<half> fn, BytePack<sizeof(half2)> a) {
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
    __device__ static BytePack<sizeof(__nv_bfloat16)> preOp(
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
      __device__ static BytePack<sizeof(__nv_bfloat162)> preOp(
          FuncPreMulSum<__nv_bfloat16> fn, BytePack<sizeof(__nv_bfloat162)> a
        ) {
        return toPack<__nv_bfloat162>(__hmul2(fromPack<__nv_bfloat162>(a), fn.scalar));
      }
    };
  #endif
#endif

////////////////////////////////////////////////////////////////////////////////
// FuncSumPostDiv

template<typename T, bool IsFloating=IsFloatingPoint<T>::value>
struct FuncSumPostDiv_IntOnly;

template<typename T>
struct FuncSumPostDiv: FuncSumPostDiv_IntOnly<T> {
  __device__ FuncSumPostDiv(uint64_t opArg=0):
    FuncSumPostDiv_IntOnly<T>(opArg) {
  }
};

template<typename T>
struct FuncSumPostDiv_IntOnly<T, /*IsFloating=*/false>: FuncSum<T> {
  using EltType = T;
  int divisor;
  __device__ FuncSumPostDiv_IntOnly(uint64_t opArg=0): divisor(opArg) {}
};

template<typename T>
struct FuncSumPostDiv_IntOnly<T, /*IsFloating=*/true> {
  static_assert(sizeof(T)!=sizeof(T), "FuncSumPostDiv is only for implementing ncclAvg on integral types.");
};

template<typename T>
struct Apply_Reduce<FuncSumPostDiv<T>, /*EltPerPack=*/1>:
    Apply_Reduce<FuncSum<T>, 1> {
  __device__ static BytePack<sizeof(T)> reduce(FuncSumPostDiv<T> fn, BytePack<sizeof(T)> a, BytePack<sizeof(T)> b) {
    // FuncSumPostDiv reduce dispatches to FuncSum.
    return Apply_Reduce<FuncSum<T>, 1>::reduce(FuncSum<T>(), a, b);
  }
};

template<typename T>
struct Apply_PostOp<FuncSumPostDiv<T>, /*EltPerPack=*/1> {
  static constexpr bool IsIdentity = false;
  __device__ static BytePack<sizeof(T)> postOp(FuncSumPostDiv<T> fn, BytePack<sizeof(T)> a) {
    return toPack<T>(fromPack<T>(a) / fn.divisor);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Apply_LoadMultimem

#define SIZEOF_BytePack_field_u16 2
#define PTX_REG_BytePack_field_u16 "h"

#define SIZEOF_BytePack_field_u32 4
#define PTX_REG_BytePack_field_u32 "r"

#define SIZEOF_BytePack_field_u64 8
#define PTX_REG_BytePack_field_u64 "l"

#define DEFINE_Apply_LoadMultimem_sum(T, ptx_ty, pack_field) \
  template<> \
  struct Apply_LoadMultimem<FuncSum<T>, SIZEOF_BytePack_field_##pack_field> { \
    static constexpr int PackSize = SIZEOF_BytePack_field_##pack_field; \
    __device__ static BytePack<PackSize> load(FuncSum<T> fn, uintptr_t addr) { \
      BytePack<PackSize> ans; \
      asm("multimem.ld_reduce.relaxed.sys.global.add." #ptx_ty " %0, [%1];" \
        : "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field) \
        : "l"(addr)); \
      return ans; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax(T, ptx_ty, pack_field) \
  template<> \
  struct Apply_LoadMultimem<FuncMinMax<T>, SIZEOF_BytePack_field_##pack_field> { \
    static constexpr int PackSize = SIZEOF_BytePack_field_##pack_field; \
    __device__ static BytePack<PackSize> load(FuncMinMax<T> fn, uintptr_t addr) { \
      BytePack<PackSize> ans; \
      if (fn.isMinNotMax) { \
        asm("multimem.ld_reduce.relaxed.sys.global.min." #ptx_ty " %0, [%1];" \
          : "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field) \
          : "l"(addr)); \
      } else { \
        asm("multimem.ld_reduce.relaxed.sys.global.max." #ptx_ty " %0, [%1];" \
          : "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field) \
          : "l"(addr)); \
      } \
      return ans; \
    } \
  };

#define DEFINE_Apply_LoadMultimem_sum_v4(T, ptx_ty, pack_field) \
  template<> \
  struct Apply_LoadMultimem<FuncSum<T>, 4*(SIZEOF_BytePack_field_##pack_field)> { \
    static constexpr int PackSize = 4*(SIZEOF_BytePack_field_##pack_field); \
    __device__ static BytePack<PackSize> load(FuncSum<T> fn, uintptr_t addr) { \
      BytePack<PackSize> ans; \
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
        : "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[0]), \
          "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[1]), \
          "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[2]), \
          "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[3]) \
        : "l"(addr)); \
      return ans; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax_v4(T, ptx_ty, pack_field) \
  template<> \
  struct Apply_LoadMultimem<FuncMinMax<T>, 4*(SIZEOF_BytePack_field_##pack_field)> { \
    static constexpr int PackSize = 4*(SIZEOF_BytePack_field_##pack_field); \
    __device__ static BytePack<PackSize> load(FuncMinMax<T> fn, uintptr_t addr) { \
      BytePack<PackSize> ans; \
      if (fn.isMinNotMax) { \
        asm("multimem.ld_reduce.relaxed.sys.global.min.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
          : "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[0]), \
            "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[1]), \
            "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[2]), \
            "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[3]) \
          : "l"(addr)); \
      } else { \
        asm("multimem.ld_reduce.relaxed.sys.global.max.v4." #ptx_ty " {%0,%1,%2,%3}, [%4];" \
          : "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[0]), \
            "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[1]), \
            "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[2]), \
            "=" PTX_REG_BytePack_field_##pack_field(ans.pack_field[3]) \
          : "l"(addr)); \
      } \
      return ans; \
    } \
  };

#define DEFINE_Apply_LoadMultimem_sum_v4x2_and_subhalf(T, ptx_ty, pack_field) \
  DEFINE_Apply_LoadMultimem_sum_v4(T, ptx_ty, pack_field) \
  template<> \
  struct Apply_LoadMultimem<FuncSum<T>, sizeof(T)> { \
    __device__ static BytePack<sizeof(T)> load(FuncSum<T> fn, uintptr_t addr) { \
      BytePack<2*sizeof(T)> tmp; \
      asm("multimem.ld_reduce.relaxed.sys.global.add." #ptx_ty " %0, [%1];" \
        : "=" PTX_REG_BytePack_field_##pack_field(tmp.pack_field) \
        : "l"(addr & -uintptr_t(2*sizeof(T)))); \
      return tmp.half[(addr/sizeof(T))%2]; \
    } \
  };
#define DEFINE_Apply_LoadMultimem_minmax_v4x2_and_subhalf(T, ptx_ty, pack_field) \
  DEFINE_Apply_LoadMultimem_minmax_v4(T, ptx_ty, pack_field) \
  template<> \
  struct Apply_LoadMultimem<FuncMinMax<T>, sizeof(T)> { \
    __device__ static BytePack<sizeof(T)> load(FuncMinMax<T> fn, uintptr_t addr) { \
      BytePack<2*sizeof(T)> tmp; \
      if (fn.isMinNotMax) { \
        asm("multimem.ld_reduce.relaxed.sys.global.min." #ptx_ty " %0, [%1];" \
          : "=" PTX_REG_BytePack_field_##pack_field(tmp.pack_field) \
          : "l"(addr & -uintptr_t(2*sizeof(T)))); \
      } else { \
        asm("multimem.ld_reduce.relaxed.sys.global.max." #ptx_ty " %0, [%1];" \
          : "=" PTX_REG_BytePack_field_##pack_field(tmp.pack_field) \
          : "l"(addr & -uintptr_t(2*sizeof(T)))); \
      } \
      return tmp.half[(addr/sizeof(T))%2]; \
    } \
  };

template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem {
  __device__ static BytePack<BytePerPack> load(Fn fn, uintptr_t addr) {
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
      IsFloat && IsSum ? 8 :
      IsFloat && IsMinMax && sizeof(T)==2 ? 16 :
      !IsFloat && (IsSum||IsMinMax) && sizeof(T)>=4 ? sizeof(T) :
      /*multimem.ld_reduce not supported:*/ 0;
  };

  DEFINE_Apply_LoadMultimem_sum(uint32_t, u32, u32)
  DEFINE_Apply_LoadMultimem_minmax(uint32_t, u32, u32)

  DEFINE_Apply_LoadMultimem_sum(int32_t, s32, u32)
  DEFINE_Apply_LoadMultimem_minmax(int32_t, s32, u32)

  DEFINE_Apply_LoadMultimem_sum(uint64_t, u64, u64)
  DEFINE_Apply_LoadMultimem_minmax(uint64_t, u64, u64)

  DEFINE_Apply_LoadMultimem_sum(int64_t, u64, u64)
  DEFINE_Apply_LoadMultimem_minmax(int64_t, s64, u64)

  DEFINE_Apply_LoadMultimem_sum(float, f32, u32)
  DEFINE_Apply_LoadMultimem_sum_v4(float, f32, u32)

  DEFINE_Apply_LoadMultimem_sum(double, f64, u64)

  DEFINE_Apply_LoadMultimem_sum_v4x2_and_subhalf(half, f16x2, u32)
  DEFINE_Apply_LoadMultimem_minmax_v4x2_and_subhalf(half, f16x2, u32)

  #if defined(__CUDA_BF16_TYPES_EXIST__)
    DEFINE_Apply_LoadMultimem_sum_v4x2_and_subhalf(__nv_bfloat16, bf16x2, u32)
    DEFINE_Apply_LoadMultimem_minmax_v4x2_and_subhalf(__nv_bfloat16, bf16x2, u32)
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
#undef SIZEOF_BytePack_field_u64
#undef PTX_REG_BytePack_field_u64
#undef SIZEOF_BytePack_field_u32
#undef PTX_REG_BytePack_field_u32
#undef SIZEOF_BytePack_field_u16
#undef PTX_REG_BytePack_field_u16

#endif // REDUCE_KERNEL_H_
