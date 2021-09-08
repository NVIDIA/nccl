/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef NCCL_REDUCE_KERNEL_H_
#define NCCL_REDUCE_KERNEL_H_

#include "common_kernel.h"
#include <limits>
#include <type_traits>

template<typename T>
struct FuncNull {
  __device__ FuncNull(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return 0;
  }
};

template<typename T>
struct FuncSum {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T>
struct FuncProd {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T>
struct FuncMax {
  __device__ FuncMax(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template<typename T>
struct FuncMin {
  __device__ FuncMin(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};

template<typename Fn>
struct FuncTraits { // generic implementation for FuncSum,Prod,Min,Max
  static constexpr bool IsPreOpIdentity = true;
  static constexpr bool IsPostOpIdentity = true;

  template<typename T>
  __device__ static T preOp(Fn, T x) { return x; }
  template<typename T>
  __device__ static T postOp(Fn, T x) { return x; }
};

#define MASK0 0x00ff00ff
#define MASK1 0xff00ff00
static __device__ uint32_t addChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit addition */
  const uint32_t x0 = x & MASK0;
  const uint32_t x1 = x & MASK1;
  const uint32_t y0 = y & MASK0;
  const uint32_t y1 = y & MASK1;
  const uint32_t r0 = (x0+y0);
  const uint32_t r1 = (x1+y1);
  return (r0 & MASK0) | (r1 & MASK1);
}

template<>
struct FuncSum<int8_t> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x+y;
  }
};
template<>
struct FuncSum<uint8_t> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    return addChar4(x, y);
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x+y;
  }
};

static __device__ uint32_t mulChar4(const uint32_t x, const uint32_t y) {
  /* This can be used both for signed and unsigned 8-bit multiplication */
  union converter { uint32_t storage; char4 a; };
  converter cx, cy, cr;
  cx.storage = x;
  cy.storage = y;
  cr.a.x = cx.a.x * cy.a.x;
  cr.a.y = cx.a.y * cy.a.y;
  cr.a.z = cx.a.z * cy.a.z;
  cr.a.w = cx.a.w * cy.a.w;
  return cr.storage;
}

template<>
struct FuncProd<int8_t> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return x*y;
  }
};
template<>
struct FuncProd<uint8_t> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
    return mulChar4(x, y);
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return x*y;
  }
};

template<>
struct FuncMax<int8_t> {
  __device__ FuncMax(uint64_t opArg=0) {}
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmax4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x>y) ? x : y;
  }
};
template<>
struct FuncMax<uint8_t> {
  __device__ FuncMax(uint64_t opArg=0) {}
  union converter { uint32_t storage; uchar4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x>y) ? x : y;
  }
};

template<>
struct FuncMin<int8_t> {
  __device__ FuncMin(uint64_t opArg=0) {}
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmin4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ int8_t operator()(const int8_t x, const int8_t y) const {
    return (x<y) ? x : y;
  }
};
template<>
struct FuncMin<uint8_t> {
  __device__ FuncMin(uint64_t opArg=0) {}
  union converter { uint32_t storage; uchar4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vmin4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ uint8_t operator()(const uint8_t x, const uint8_t y) const {
    return (x<y) ? x : y;
  }
};

template<>
struct FuncSum<half> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hadd(x, y);
#else
    return __float2half( __half2float(x) + __half2float(y) );
#endif
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct FuncSum<__nv_bfloat16> {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ __nv_bfloat162 operator()(const __nv_bfloat162 x, const __nv_bfloat162 y) const {
#if __CUDA_ARCH__ >= 800
    return __hadd2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#endif
   }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 x, const __nv_bfloat16 y) const {
#if __CUDA_ARCH__ >= 800
    return __hadd(x, y);
#else
    return __float2bfloat16( __bfloat162float(x) + __bfloat162float(y) );
#endif
  }
};
#endif

template<>
struct FuncProd<half> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ half2 operator()(const half2 x, const half2 y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul2(x, y);
#else
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x * fy.x;
    fr.y = fx.y * fy.y;
    return __float22half2_rn(fr);
#endif
  }
  __device__ half operator()(const half x, const half y) const {
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    return __hmul(x, y);
#else
    return __float2half( __half2float(x) * __half2float(y) );
#endif
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct FuncProd<__nv_bfloat16> {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ __nv_bfloat162 operator()(const __nv_bfloat162 x, const __nv_bfloat162 y) const {
#if __CUDA_ARCH__ >= 800
    return __hmul2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#endif
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 x, const __nv_bfloat16 y) const {
#if __CUDA_ARCH__ >= 800
    return __hmul(x, y);
#else
    return __float2bfloat16( __bfloat162float(x) * __bfloat162float(y) );
#endif
  }
};
#endif

template<>
struct FuncMax<half> {
  __device__ FuncMax(uint64_t opArg=0) {}
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fmaxf(fx.x, fy.x);
    fr.y = fmaxf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fmaxf(fx, fy);
    return __float2half(fm);
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct FuncMax<__nv_bfloat16> {
  __device__ FuncMax(uint64_t opArg=0) {}
  __device__ __nv_bfloat162 operator()(const __nv_bfloat162 x, const __nv_bfloat162 y) const {
#if __CUDA_ARCH__ >= 800
    return __hmax2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fmaxf(fxl, fyl), fmaxf(fxh, fyh));
#endif
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 x, const __nv_bfloat16 y) const {
#if __CUDA_ARCH__ >= 800
    return __hmax(x, y);
#else
    float fx, fy;
    fx = __bfloat162float(x);
    fy = __bfloat162float(y);
    return __float2bfloat16(fmaxf(fx, fy));
#endif
  }
};
#endif

template<>
struct FuncMin<half> {
  __device__ FuncMin(uint64_t opArg=0) {}
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fminf(fx.x, fy.x);
    fr.y = fminf(fx.y, fy.y);
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fminf(fx, fy);
    return __float2half(fm);
  }
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct FuncMin<__nv_bfloat16> {
  __device__ FuncMin(uint64_t opArg=0) {}
   __device__ __nv_bfloat162 operator()(const __nv_bfloat162 x, const __nv_bfloat162 y) const {
#if __CUDA_ARCH__ >= 800
    return __hmin2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fminf(fxl, fyl), fminf(fxh, fyh));
#endif
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 x, const __nv_bfloat16 y) const {
#if __CUDA_ARCH__ >= 800
    return __hmin(x, y);
#else
    float fx, fy;
    fx = __bfloat162float(x);
    fy = __bfloat162float(y);
    return __float2bfloat16(fminf(fx, fy));
#endif
  }
};
#endif

template<>
struct FuncMax<float> {
  __device__ FuncMax(uint64_t opArg=0) {}
  __device__ float operator()(float x, float y) const {
    return fmaxf(x, y);
  }
};
template<>
struct FuncMin<float> {
  __device__ FuncMin(uint64_t opArg=0) {}
  __device__ float operator()(float x, float y) const {
    return fminf(x, y);
  }
};

template<>
struct FuncMax<double> {
  __device__ FuncMax(uint64_t opArg=0) {}
  __device__ double operator()(double x, double y) const {
    return fmax(x, y);
  }
};
template<>
struct FuncMin<double> {
  __device__ FuncMin(uint64_t opArg=0) {}
  __device__ double operator()(double x, double y) const {
    return fmin(x, y);
  }
};

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

template<typename T, bool IsFloating=IsFloatingPoint<T>::value>
struct FuncSumPostDiv;

template<typename T>
struct FuncSumPostDiv<T, /*IsFloating=*/false>: FuncSum<T> {
  static constexpr bool IsPreOpIdentity = true;
  static constexpr bool IsPostOpIdentity = false;
  int n;
  __device__ FuncSumPostDiv(uint64_t opArg): n(opArg) {}
  // inherits FuncSum::operator()
  __device__ T preOp(T x) const { return x; }
  __device__ T postOp(T x) const { return T(x/n); }
};

template<typename T>
struct FuncSumPostDiv<T, /*IsFloating=*/true> {
  static_assert(sizeof(T)!=sizeof(T), "FuncSumPostDiv is only for implementing ncclAvg on integral types.");
};

template<typename T>
struct FuncPreMulSum: FuncSum<T> { // integral T since all floats are specialized below
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;
  T scale;
  __device__ FuncPreMulSum(uint64_t opArg) { scale = *(T*)&opArg; }
  // inherits FuncSum::operator()
  __device__ T preOp(T x) const { return x*scale; }
  __device__ T postOp(T x) const { return x; }
};

template<>
struct FuncPreMulSum<double>: FuncSum<double> {
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;
  double scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale = *(double*)&opArg;
  }
  // inherits FuncSum::operator()
  __device__ double preOp(double x) const {
    return IsPreOpIdentity ? x : x*scale;
  }
  __device__ double postOp(double x) const {
    return IsPostOpIdentity ? x : x*scale;
  }
};

template<>
struct FuncPreMulSum<float>: FuncSum<float> {
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;
  float scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale = *(float*)&opArg;
  }
  // inherits FuncSum::operator()
  __device__ float preOp(float x) const {
    return IsPreOpIdentity ? x : x*scale;
  }
  __device__ float postOp(float x) const {
    return IsPostOpIdentity ? x : x*scale;
  }
};

template<>
struct FuncPreMulSum<half>: FuncSum<half> {
  // Change these to switch between all prescale, all postscale, or both by sqrt(N).
  // Obviously, the only invalid combination is both true. An improvement would be
  // make this parameterized as a build time setting and passed here through
  // preprocessor definitions.
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;

#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  half2 scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale.x = *(half*)&opArg;
    scale.y = scale.x;
  }
  // inherits FuncSum::operator()
  __device__ half preOp(half x) const {
    return IsPreOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ half2 preOp(half2 x) const {
    return IsPreOpIdentity ? x : __hmul2(x, scale);
  }
  __device__ half postOp(half x) const {
    return IsPostOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ half2 postOp(half2 x) const {
    return IsPostOpIdentity ? x : __hmul2(x, scale);
  }
#else
  float scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale = __half2float(*(half*)&opArg);
  }
  // inherits FuncSum::operator()
  __device__ half preOp(half x) const {
    return IsPreOpIdentity ? x : __float2half(__half2float(x)*scale);
  }
  __device__ half2 preOp(half2 x) const {
    if (IsPreOpIdentity)
      return x;
    else {
      float2 a = __half22float2(x);
      a.x *= scale;
      a.y *= scale;
      return __float22half2_rn(a);
    }
  }
  __device__ half postOp(half x) const {
    return IsPostOpIdentity ? x : __float2half(__half2float(x)*scale);
  }
  __device__ half2 postOp(half2 x) const {
    if (IsPostOpIdentity)
      return x;
    else {
      float2 a = __half22float2(x);
      a.x *= scale;
      a.y *= scale;
      return __float22half2_rn(a);
    }
  }
#endif
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct FuncPreMulSum<__nv_bfloat16>: FuncSum<__nv_bfloat16> {
  // Change these to switch between all prescale, all postscale, or both by sqrt(N).
  // Obviously, the only invalid combination is both true. An improvement would be
  // make this parameterized as a build time setting and passed here through
  // preprocessor definitions.
  static constexpr bool IsPreOpIdentity = false;
  static constexpr bool IsPostOpIdentity = true;

#if __CUDA_ARCH__ >= 800
  __nv_bfloat162 scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale.x = *(__nv_bfloat16*)&opArg;
    scale.y = scale.x;
  }
  // inherits FuncSum::operator()
  __device__ __nv_bfloat16 preOp(__nv_bfloat16 x) const {
    return IsPreOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ __nv_bfloat162 preOp(__nv_bfloat162 x) const {
    return IsPreOpIdentity ? x : __hmul2(x, scale);
  }
  __device__ __nv_bfloat16 postOp(__nv_bfloat16 x) const {
    return IsPostOpIdentity ? x : __hmul(x, scale.x);
  }
  __device__ __nv_bfloat162 postOp(__nv_bfloat162 x) const {
    return IsPostOpIdentity ? x : __hmul2(x, scale);
  }
#else
  float scale;
  __device__ FuncPreMulSum(uint64_t opArg) {
    scale = *(__nv_bfloat16*)&opArg;
  }
  // inherits FuncSum::operator()
  __device__ __nv_bfloat16 preOp(__nv_bfloat16 x) const {
    return IsPreOpIdentity ? x : __float2bfloat16(__bfloat162float(x)*scale);
  }
  __device__ __nv_bfloat162 preOp(__nv_bfloat162 x) const {
    if (IsPreOpIdentity)
      return x;
    else {
      float fxl, fxh;
      fxl = __low2float(x);
      fxh = __high2float(x);
      return __floats2bfloat162_rn(fxl * scale, fxh * scale);
    }
  }
  __device__ __nv_bfloat16 postOp(__nv_bfloat16 x) const {
    return IsPostOpIdentity ? x : __float2bfloat16(__bfloat162float(x)*scale);
  }
  __device__ __nv_bfloat162 postOp(__nv_bfloat162 x) const {
    if (IsPostOpIdentity)
      return x;
    else {
      float fxl, fxh;
      fxl = __low2float(x);
      fxh = __high2float(x);
      return __floats2bfloat162_rn(fxl * scale, fxh * scale);
    }
  }
#endif
};
#endif

template<typename T>
struct FuncTraits<FuncPreMulSum<T>> {
  static constexpr bool IsPreOpIdentity = FuncPreMulSum<T>::IsPreOpIdentity;
  static constexpr bool IsPostOpIdentity = FuncPreMulSum<T>::IsPostOpIdentity;

  template<typename U>
  __device__ static U preOp(FuncPreMulSum<T> fn, U x) {
    return fn.preOp(x);
  }
  template<typename U>
  __device__ static U postOp(FuncPreMulSum<T> fn, U x) {
    return fn.postOp(x);
  }
};
template<typename T>
struct FuncTraits<FuncSumPostDiv<T>> {
  static constexpr bool IsPreOpIdentity = FuncSumPostDiv<T>::IsPreOpIdentity;
  static constexpr bool IsPostOpIdentity = FuncSumPostDiv<T>::IsPostOpIdentity;

  template<typename U>
  __device__ static U preOp(FuncSumPostDiv<T> fn, U x) {
    return fn.preOp(x);
  }
  template<typename U>
  __device__ static U postOp(FuncSumPostDiv<T> fn, U x) {
    return fn.postOp(x);
  }
};
#endif // REDUCE_KERNEL_H_
