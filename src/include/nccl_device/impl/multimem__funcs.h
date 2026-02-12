/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_MULTIMEM__FUNCS_H_
#define _NCCL_DEVICE_MULTIMEM__FUNCS_H_

#include "../utility.h"
#include "vector__types.h"
#include <cuda_runtime.h>
#include <cassert>
#include <type_traits>

#if NCCL_CHECK_CUDACC

namespace nccl {
namespace utility {

// Load helper that selects multimem vs LSA and validates support at compile time.
template<typename Pack, bool UseMultimem, typename RedOp, int Count = Pack::Count>
struct LoadImpl {
  NCCL_DEVICE_INLINE static Pack run(const Pack* addr) {
    using PackEltType = typename Pack::EltType;
    static_assert(!UseMultimem || std::is_same<RedOp, OpSum<PackEltType>>::value,
                  "Multimem sources only support OpSum - use LSA sources for custom RedOp");
    static_assert(!UseMultimem || (!std::is_same<PackEltType, int8_t>::value &&
                                   !std::is_same<PackEltType, uint8_t>::value),
                  "int8_t and uint8_t are not supported for multimem sources - use LSA sources");
#if defined(__CUDA_FP4_TYPES_EXIST__)
    static_assert(!UseMultimem || !std::is_same<PackEltType, __nv_fp4_e2m1>::value,
                  "__nv_fp4_e2m1 is not supported for multimem sources - use LSA sources");
#endif
    #if __CUDA_ARCH__ < 900
    if (UseMultimem) {
      assert(false && "multimem is not supported on architectures < sm_90");
      return Pack{};
    }
    #else
    static_assert(!UseMultimem,
                  "multimem load not implemented for this pack type. "
                  "A multimem specialization is not possible for this type.");
    #endif
    return *addr;
  }
};

// Empty packs (0 elements) - just return empty pack
template<typename Pack, bool UseMultimem, typename RedOp>
struct LoadImpl<Pack, UseMultimem, RedOp, 0> {
  NCCL_DEVICE_INLINE static Pack run(const Pack* addr) {
    return Pack{};
  }
};

template<typename Pack, bool UseMultimem, typename RedOp>
NCCL_DEVICE_INLINE Pack load(const Pack* addr) {
  return LoadImpl<Pack, UseMultimem, RedOp>::run(addr);
}
#if __CUDA_ARCH__ >= 900

// Double precision - single element
template<>
NCCL_DEVICE_INLINE EltPack<double, 1> load<EltPack<double, 1>, true, OpSum<double>>(const EltPack<double, 1>* addr) {
    EltPack<double, 1> result;
    double value;
    asm volatile("multimem.ld_reduce.global.add.f64 %0, [%1];"
                 : "=d"(value)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value;
    return result;
  }

// Double precision - 2 elements (2 x 64-bit = 128 bits)
// Note: No 128-bit multimem.ld_reduce for double, use 2 separate .f64 operations
template<>
NCCL_DEVICE_INLINE EltPack<double, 2> load<EltPack<double, 2>, true, OpSum<double>>(const EltPack<double, 2>* addr) {
    EltPack<double, 2> result;
    double* elems = result.elts();
    const char* base_addr = reinterpret_cast<const char*>(addr);

    // Load 2 separate f64 values (no vector instruction available)
    // Use unrolled loop calling the 1-element version
    #pragma unroll
    for (int i = 0; i < 2; i++) {
      EltPack<double, 1> loaded = load<EltPack<double, 1>, true, OpSum<double>>(
          reinterpret_cast<const EltPack<double, 1>*>(base_addr + i * sizeof(double)));
      elems[i] = loaded.elts()[0];
    }
    return result;
  }

// Float precision - single element
template<>
NCCL_DEVICE_INLINE EltPack<float, 1> load<EltPack<float, 1>, true, OpSum<float>>(const EltPack<float, 1>* addr) {
    EltPack<float, 1> result;
    float value;
    asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];"
                 : "=f"(value)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value;
    return result;
  }

// Float precision - 2 elements
template<>
NCCL_DEVICE_INLINE EltPack<float, 2> load<EltPack<float, 2>, true, OpSum<float>>(const EltPack<float, 2>* addr) {
    EltPack<float, 2> result;
    float2 value;
    asm volatile("multimem.ld_reduce.global.add.v2.f32 {%0, %1}, [%2];"
                 : "=f"(value.x), "=f"(value.y)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value.x;
    result.elts()[1] = value.y;
    return result;
  }

// Float precision - 4 elements
template<>
NCCL_DEVICE_INLINE EltPack<float, 4> load<EltPack<float, 4>, true, OpSum<float>>(const EltPack<float, 4>* addr) {
    EltPack<float, 4> result;
    float4 value;
    asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    float* elems = result.elts();
    elems[0] = value.x;
    elems[1] = value.y;
    elems[2] = value.z;
    elems[3] = value.w;
    return result;
  }

// Half precision - 2 elements (minimum: 2 halves = 32 bits)
template<>
NCCL_DEVICE_INLINE EltPack<half, 2> load<EltPack<half, 2>, true, OpSum<half>>(const EltPack<half, 2>* addr) {
    EltPack<half, 2> result;
    uint32_t raw;
    asm volatile("multimem.ld_reduce.global.add.acc::f32.f16x2 %0, [%1];"
                 : "=r"(raw)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw;
      half elts[2];
    } packed{raw};
    result.elts()[0] = packed.elts[0];
    result.elts()[1] = packed.elts[1];
    return result;
  }

// Half precision - single element (trick: load as f16x2, extract one half)
template<>
NCCL_DEVICE_INLINE EltPack<half, 1> load<EltPack<half, 1>, true, OpSum<half>>(const EltPack<half, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
    return EltPack<half, 1>{};
#else
    // Align address to 4 bytes for f16x2 load
    const char* charAddr = reinterpret_cast<const char*>(addr);
    const size_t offset = reinterpret_cast<size_t>(addr) & 3;
    const char* alignedAddr = charAddr - offset;
    // Load as EltPack<half, 2> and extract the correct half
    EltPack<half, 2> loaded = load<EltPack<half, 2>, true, OpSum<half>>(
        reinterpret_cast<const EltPack<half, 2>*>(alignedAddr));
    EltPack<half, 1> result;
    const half* loadedElts = loaded.elts();
    // Extract the correct half based on address offset
    result.elts()[0] = loadedElts[offset / sizeof(half)];
    return result;
#endif
  }

// Half precision - 8 elements (8 halves = 128 bits)
template<>
NCCL_DEVICE_INLINE EltPack<half, 8> load<EltPack<half, 8>, true, OpSum<half>>(const EltPack<half, 8>* addr) {
    EltPack<half, 8> result;
    uint32_t raw[4];
    asm volatile("multimem.ld_reduce.global.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw[4];
      half elts[8];
    } packed{{raw[0], raw[1], raw[2], raw[3]}};
    half* out = result.elts();
    #pragma unroll 8
    for (int i = 0; i < 8; i++) out[i] = packed.elts[i];
    return result;
  }

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Bfloat16 precision - 2 elements (minimum: 2 bf16s = 32 bits)
// Uses bf16x2 multimem instruction (not f16x2 - bf16 has different format than half)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 2> load<EltPack<__nv_bfloat16, 2>, true, OpSum<__nv_bfloat16>>(const EltPack<__nv_bfloat16, 2>* addr) {
    EltPack<__nv_bfloat16, 2> result;
    uint32_t raw;
    // Use bf16x2 instruction - bf16 requires its own instruction format
    asm volatile("multimem.ld_reduce.global.add.acc::f32.bf16x2 %0, [%1];"
                 : "=r"(raw)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw;
      __nv_bfloat16 elts[2];
    } packed{raw};
    result.elts()[0] = packed.elts[0];
    result.elts()[1] = packed.elts[1];
    return result;
  }

// Bfloat16 precision - single element (trick: load as bf16x2, extract one bf16)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 1> load<EltPack<__nv_bfloat16, 1>, true, OpSum<__nv_bfloat16>>(const EltPack<__nv_bfloat16, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
    return EltPack<__nv_bfloat16, 1>{};
#else
    // Align address to 4 bytes for bf16x2 load
    const char* charAddr = reinterpret_cast<const char*>(addr);
    const size_t offset = reinterpret_cast<size_t>(addr) & 3;
    const char* alignedAddr = charAddr - offset;
    // Load as EltPack<__nv_bfloat16, 2> and extract the correct bf16
    EltPack<__nv_bfloat16, 2> loaded = load<EltPack<__nv_bfloat16, 2>, true, OpSum<__nv_bfloat16>>(
        reinterpret_cast<const EltPack<__nv_bfloat16, 2>*>(alignedAddr));
    EltPack<__nv_bfloat16, 1> result;
    const __nv_bfloat16* loadedElts = loaded.elts();
    // Extract the correct bf16 based on address offset
    result.elts()[0] = loadedElts[offset / sizeof(__nv_bfloat16)];
    return result;
#endif
  }

// Bfloat16 precision - 8 elements (8 bf16s = 128 bits)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 8> load<EltPack<__nv_bfloat16, 8>, true, OpSum<__nv_bfloat16>>(const EltPack<__nv_bfloat16, 8>* addr) {
    // Use v4.bf16x2 instruction - bf16 requires its own instruction format
    EltPack<__nv_bfloat16, 8> result;
    uint32_t raw[4];
    asm volatile("multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw[4];
      __nv_bfloat16 elts[8];
    } packed{{raw[0], raw[1], raw[2], raw[3]}};
    __nv_bfloat16* out = result.elts();
    #pragma unroll 8
    for (int i = 0; i < 8; i++) out[i] = packed.elts[i];
    return result;
  }
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// FP8 E4M3 precision - 4 elements (minimum: 4 fp8s = 32 bits)
// Uses .acc::f16 accumulation (accumulates to half precision)
//   - Specific architectures: sm_100a, sm_101a/sm_110a (renamed from PTX ISA 9.0), sm_120a, sm_121a
//   - Family-specific architectures (PTX ISA 8.8+): sm_100f+ or higher, sm_101f+/sm_110f+ or higher
//   - NOT supported on sm_103 (10.3) or other unsupported variants
// Uses __CUDA_ARCH_FAMILY_SPECIFIC__ and __CUDA_ARCH_SPECIFIC__ when available (CUDA 12.9+)
// Only define FP8 multimem specializations for supported architectures - otherwise fallback to generic template
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 4> load<EltPack<__nv_fp8_e4m3, 4>, true, OpSum<__nv_fp8_e4m3>>(const EltPack<__nv_fp8_e4m3, 4>* addr) {
    #if (defined(__CUDA_ARCH_SPECIFIC__) && \
         (__CUDA_ARCH_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_SPECIFIC__ == 1010 || \
          __CUDA_ARCH_SPECIFIC__ == 1200 || \
          __CUDA_ARCH_SPECIFIC__ == 1210)) || \
        (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
         (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
    EltPack<__nv_fp8_e4m3, 4> result;
    uint32_t raw;
    // Use e4m3x4 instruction with .acc::f16 accumulation
    asm volatile("multimem.ld_reduce.global.add.acc::f16.e4m3x4 %0, [%1];"
                 : "=r"(raw)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw;
      __nv_fp8_e4m3 elts[4];
    } packed{raw};
    __nv_fp8_e4m3* out = result.elts();
    #pragma unroll
    for (int i = 0; i < 4; i++) out[i] = packed.elts[i];
    return result;
    #else
    assert(false && "FP8 multimem is not supported on this architecture.");
    return EltPack<__nv_fp8_e4m3, 4>{};
    #endif
  }

// FP8 E4M3 precision - 2 elements (trick: load as e4m3x4, extract two fp8s)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 2> load<EltPack<__nv_fp8_e4m3, 2>, true, OpSum<__nv_fp8_e4m3>>(const EltPack<__nv_fp8_e4m3, 2>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
    return EltPack<__nv_fp8_e4m3, 2>{};
#else
    // Align address to 4 bytes for e4m3x4 load
    const char* charAddr = reinterpret_cast<const char*>(addr);
    const size_t offset = reinterpret_cast<size_t>(addr) & 3;
    const char* alignedAddr = charAddr - offset;
    // Load as EltPack<__nv_fp8_e4m3, 4> and extract the correct two fp8s
    EltPack<__nv_fp8_e4m3, 4> loaded = load<EltPack<__nv_fp8_e4m3, 4>, true, OpSum<__nv_fp8_e4m3>>(
        reinterpret_cast<const EltPack<__nv_fp8_e4m3, 4>*>(alignedAddr));
    EltPack<__nv_fp8_e4m3, 2> result;
    // Extract the correct two fp8s based on address offset
    const __nv_fp8_e4m3* loadedElts = loaded.elts();
    const int startIdx = offset / sizeof(__nv_fp8_e4m3);
    __nv_fp8_e4m3* resultElts = result.elts();
    resultElts[0] = loadedElts[startIdx];
    resultElts[1] = loadedElts[startIdx + 1];
    return result;
#endif
}

// FP8 E4M3 precision - single element (trick: load as e4m3x4, extract one fp8)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 1> load<EltPack<__nv_fp8_e4m3, 1>, true, OpSum<__nv_fp8_e4m3>>(const EltPack<__nv_fp8_e4m3, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
    return EltPack<__nv_fp8_e4m3, 1>{};
#else
    // Align address to 4 bytes for e4m3x4 load
    const char* charAddr = reinterpret_cast<const char*>(addr);
    const size_t offset = reinterpret_cast<size_t>(addr) & 3;
    const char* alignedAddr = charAddr - offset;
    // Load as EltPack<__nv_fp8_e4m3, 4> and extract the correct fp8
    EltPack<__nv_fp8_e4m3, 4> loaded = load<EltPack<__nv_fp8_e4m3, 4>, true, OpSum<__nv_fp8_e4m3>>(
        reinterpret_cast<const EltPack<__nv_fp8_e4m3, 4>*>(alignedAddr));
    EltPack<__nv_fp8_e4m3, 1> result;
    // Extract the correct fp8 based on address offset
    const __nv_fp8_e4m3* loadedElts = loaded.elts();
    result.elts()[0] = loadedElts[offset / sizeof(__nv_fp8_e4m3)];
    return result;
#endif
}

// FP8 E4M3 precision - 16 elements (16 fp8s = 128 bits)
// Only define FP8 multimem specializations for supported architectures - otherwise fallback to generic template
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 16> load<EltPack<__nv_fp8_e4m3, 16>, true, OpSum<__nv_fp8_e4m3>>(const EltPack<__nv_fp8_e4m3, 16>* addr) {
    #if (defined(__CUDA_ARCH_SPECIFIC__) && \
         (__CUDA_ARCH_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_SPECIFIC__ == 1010 || \
          __CUDA_ARCH_SPECIFIC__ == 1200 || \
          __CUDA_ARCH_SPECIFIC__ == 1210)) || \
        (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
         (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
    EltPack<__nv_fp8_e4m3, 16> result;
    uint32_t raw[4];
    // Use v4.e4m3x4 instruction (4 x e4m3x4 = 16 elements)
    asm volatile("multimem.ld_reduce.global.add.acc::f16.v4.e4m3x4 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw[4];
      __nv_fp8_e4m3 elts[16];
    } packed{{raw[0], raw[1], raw[2], raw[3]}};
    __nv_fp8_e4m3* out = result.elts();
    #pragma unroll 16
    for (int i = 0; i < 16; i++) out[i] = packed.elts[i];
    return result;
    #else
    assert(false && "FP8 multimem with is not supported on this architecture.");
    return EltPack<__nv_fp8_e4m3, 16>{};
    #endif
  }

// FP8 E5M2 precision - 4 elements (minimum: 4 fp8s = 32 bits)
// Uses .acc::f16 accumulation (accumulates to half precision)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 4> load<EltPack<__nv_fp8_e5m2, 4>, true, OpSum<__nv_fp8_e5m2>>(const EltPack<__nv_fp8_e5m2, 4>* addr) {
    #if (defined(__CUDA_ARCH_SPECIFIC__) && \
         (__CUDA_ARCH_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_SPECIFIC__ == 1010 || \
          __CUDA_ARCH_SPECIFIC__ == 1200 || \
          __CUDA_ARCH_SPECIFIC__ == 1210)) || \
        (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
         (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
    EltPack<__nv_fp8_e5m2, 4> result;
    uint32_t raw;
    // Use e5m2x4 instruction with .acc::f16 accumulation
    asm volatile("multimem.ld_reduce.global.add.acc::f16.e5m2x4 %0, [%1];"
                 : "=r"(raw)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw;
      __nv_fp8_e5m2 elts[4];
    } packed{raw};
    __nv_fp8_e5m2* out = result.elts();
    #pragma unroll
    for (int i = 0; i < 4; i++) out[i] = packed.elts[i];
    return result;
    #else
    assert(false && "FP8 multimem with is not supported on this architecture.");
    return EltPack<__nv_fp8_e5m2, 4>{};
    #endif
  }

// FP8 E5M2 precision - 2 elements (trick: load as e5m2x4, extract two fp8s)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 2> load<EltPack<__nv_fp8_e5m2, 2>, true, OpSum<__nv_fp8_e5m2>>(const EltPack<__nv_fp8_e5m2, 2>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
    return EltPack<__nv_fp8_e5m2, 2>{};
#else
    // Align address to 4 bytes for e5m2x4 load
    const char* charAddr = reinterpret_cast<const char*>(addr);
    const size_t offset = reinterpret_cast<size_t>(addr) & 3;
    const char* alignedAddr = charAddr - offset;
    // Load as EltPack<__nv_fp8_e5m2, 4> and extract the correct two fp8s
    EltPack<__nv_fp8_e5m2, 4> loaded = load<EltPack<__nv_fp8_e5m2, 4>, true, OpSum<__nv_fp8_e5m2>>(
        reinterpret_cast<const EltPack<__nv_fp8_e5m2, 4>*>(alignedAddr));
    EltPack<__nv_fp8_e5m2, 2> result;
    // Extract the correct two fp8s based on address offset
    const __nv_fp8_e5m2* loadedElts = loaded.elts();
    const int startIdx = offset / sizeof(__nv_fp8_e5m2);
    __nv_fp8_e5m2* resultElts = result.elts();
    resultElts[0] = loadedElts[startIdx];
    resultElts[1] = loadedElts[startIdx + 1];
    return result;
#endif
}

// FP8 E5M2 precision - single element (trick: load as e5m2x4, extract one fp8)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 1> load<EltPack<__nv_fp8_e5m2, 1>, true, OpSum<__nv_fp8_e5m2>>(const EltPack<__nv_fp8_e5m2, 1>* addr) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands");
    return EltPack<__nv_fp8_e5m2, 1>{};
#else
    // Align address to 4 bytes for e5m2x4 load
    const char* charAddr = reinterpret_cast<const char*>(addr);
    const size_t offset = reinterpret_cast<size_t>(addr) & 3;
    const char* alignedAddr = charAddr - offset;
    // Load as EltPack<__nv_fp8_e5m2, 4> and extract the correct fp8
    EltPack<__nv_fp8_e5m2, 4> loaded = load<EltPack<__nv_fp8_e5m2, 4>, true, OpSum<__nv_fp8_e5m2>>(
        reinterpret_cast<const EltPack<__nv_fp8_e5m2, 4>*>(alignedAddr));
    EltPack<__nv_fp8_e5m2, 1> result;
    // Extract the correct fp8 based on address offset
    const __nv_fp8_e5m2* loadedElts = loaded.elts();
    result.elts()[0] = loadedElts[offset / sizeof(__nv_fp8_e5m2)];
    return result;
#endif
}

// FP8 E5M2 precision - 16 elements (16 fp8s = 128 bits)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 16> load<EltPack<__nv_fp8_e5m2, 16>, true, OpSum<__nv_fp8_e5m2>>(const EltPack<__nv_fp8_e5m2, 16>* addr) {
    #if (defined(__CUDA_ARCH_SPECIFIC__) && \
         (__CUDA_ARCH_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_SPECIFIC__ == 1010 || \
          __CUDA_ARCH_SPECIFIC__ == 1200 || \
          __CUDA_ARCH_SPECIFIC__ == 1210)) || \
        (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
         (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || \
          __CUDA_ARCH_FAMILY_SPECIFIC__ == 1010))
    EltPack<__nv_fp8_e5m2, 16> result;
    uint32_t raw[4];
    // Use v4.e5m2x4 instruction (4 x e5m2x4 = 16 elements)
    asm volatile("multimem.ld_reduce.global.add.acc::f16.v4.e5m2x4 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw[0]), "=r"(raw[1]), "=r"(raw[2]), "=r"(raw[3])
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    union {
      uint32_t raw[4];
      __nv_fp8_e5m2 elts[16];
    } packed{{raw[0], raw[1], raw[2], raw[3]}};
    __nv_fp8_e5m2* out = result.elts();
    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = packed.elts[i];
    return result;
    #else
    assert(false && "FP8 multimem with is not supported on this architecture.");
    return EltPack<__nv_fp8_e5m2, 16>{};
    #endif
  }
#endif // __CUDA_FP8_TYPES_EXIST__

// int32_t - single element
template<>
NCCL_DEVICE_INLINE EltPack<int32_t, 1> load<EltPack<int32_t, 1>, true, OpSum<int32_t>>(const EltPack<int32_t, 1>* addr) {
    EltPack<int32_t, 1> result;
    int32_t value;
    asm volatile("multimem.ld_reduce.global.add.s32 %0, [%1];"
                 : "=r"(value)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value;
    return result;
  }

// int32_t - 4 elements (4 x 32-bit = 128 bits)
// Note: No 128-bit multimem.ld_reduce for integers, use 4 separate .s32 operations
template<>
NCCL_DEVICE_INLINE EltPack<int32_t, 4> load<EltPack<int32_t, 4>, true, OpSum<int32_t>>(const EltPack<int32_t, 4>* addr) {
    EltPack<int32_t, 4> result;
    int32_t* elems = result.elts();
    const char* base_addr = reinterpret_cast<const char*>(addr);

    // Load 4 separate s32 values (no vector instruction available)
    // Use unrolled loop calling the 1-element version
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      EltPack<int32_t, 1> loaded = load<EltPack<int32_t, 1>, true, OpSum<int32_t>>(
          reinterpret_cast<const EltPack<int32_t, 1>*>(base_addr + i * sizeof(int32_t)));
      elems[i] = loaded.elts()[0];
    }
    return result;
  }

// uint32_t - single element
template<>
NCCL_DEVICE_INLINE EltPack<uint32_t, 1> load<EltPack<uint32_t, 1>, true, OpSum<uint32_t>>(const EltPack<uint32_t, 1>* addr) {
    EltPack<uint32_t, 1> result;
    uint32_t value;
    asm volatile("multimem.ld_reduce.global.add.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value;
    return result;
  }

// uint32_t - 4 elements (4 x 32-bit = 128 bits)
// Note: No 128-bit multimem.ld_reduce for integers, use 4 separate .u32 operations
template<>
NCCL_DEVICE_INLINE EltPack<uint32_t, 4> load<EltPack<uint32_t, 4>, true, OpSum<uint32_t>>(const EltPack<uint32_t, 4>* addr) {
    EltPack<uint32_t, 4> result;
    uint32_t* elems = result.elts();
    const char* base_addr = reinterpret_cast<const char*>(addr);

    // Load 4 separate u32 values (no vector instruction available)
    // Use unrolled loop calling the 1-element version
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      EltPack<uint32_t, 1> loaded = load<EltPack<uint32_t, 1>, true, OpSum<uint32_t>>(
          reinterpret_cast<const EltPack<uint32_t, 1>*>(base_addr + i * sizeof(uint32_t)));
      elems[i] = loaded.elts()[0];
    }
    return result;
  }

// int64_t - single element
// Note: Uses .u64 (add doesn't support .s64 for signed 64-bit)
template<>
NCCL_DEVICE_INLINE EltPack<int64_t, 1> load<EltPack<int64_t, 1>, true, OpSum<int64_t>>(const EltPack<int64_t, 1>* addr) {
    EltPack<int64_t, 1> result;
    int64_t value;
    asm volatile("multimem.ld_reduce.global.add.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value;
    return result;
  }

// int64_t - 2 elements (2 x 64-bit = 128 bits)
// Note: No 128-bit multimem.ld_reduce for integers, use 2 separate .u64 operations
template<>
NCCL_DEVICE_INLINE EltPack<int64_t, 2> load<EltPack<int64_t, 2>, true, OpSum<int64_t>>(const EltPack<int64_t, 2>* addr) {
    EltPack<int64_t, 2> result;
    int64_t* elems = result.elts();
    const char* base_addr = reinterpret_cast<const char*>(addr);

    // Load 2 separate u64 values (no vector instruction available)
    // Use unrolled loop calling the 1-element version
    #pragma unroll
    for (int i = 0; i < 2; i++) {
      EltPack<int64_t, 1> loaded = load<EltPack<int64_t, 1>, true, OpSum<int64_t>>(
          reinterpret_cast<const EltPack<int64_t, 1>*>(base_addr + i * sizeof(int64_t)));
      elems[i] = loaded.elts()[0];
    }
    return result;
  }

// uint64_t - single element
template<>
NCCL_DEVICE_INLINE EltPack<uint64_t, 1> load<EltPack<uint64_t, 1>, true, OpSum<uint64_t>>(const EltPack<uint64_t, 1>* addr) {
    EltPack<uint64_t, 1> result;
    uint64_t value;
    asm volatile("multimem.ld_reduce.global.add.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(__cvta_generic_to_global(addr))
                 : "memory");
    result.elts()[0] = value;
    return result;
  }

// uint64_t - 2 elements (2 x 64-bit = 128 bits)
// Note: No 128-bit multimem.ld_reduce for integers, use 2 separate .u64 operations
template<>
NCCL_DEVICE_INLINE EltPack<uint64_t, 2> load<EltPack<uint64_t, 2>, true, OpSum<uint64_t>>(const EltPack<uint64_t, 2>* addr) {
    EltPack<uint64_t, 2> result;
    uint64_t* elems = result.elts();
    const char* base_addr = reinterpret_cast<const char*>(addr);

    // Load 2 separate u64 values (no vector instruction available)
    // Use unrolled loop calling the 1-element version
    #pragma unroll
    for (int i = 0; i < 2; i++) {
      EltPack<uint64_t, 1> loaded = load<EltPack<uint64_t, 1>, true, OpSum<uint64_t>>(
          reinterpret_cast<const EltPack<uint64_t, 1>*>(base_addr + i * sizeof(uint64_t)));
      elems[i] = loaded.elts()[0];
    }
    return result;
  }

#endif // __CUDA_ARCH__ >= 900

// Store helper that selects multimem vs LSA at compile time.
template<typename Pack, bool UseMultimem>
NCCL_DEVICE_INLINE void store(Pack* addr, const Pack& val) {
  if NCCL_IF_CONSTEXPR (UseMultimem) {
    multimemStore(addr, val);
  } else {
    *addr = val;
  }
}

// Multimem Store
// Typeless: stores bytes directly based on pack byte size only

#if __CUDA_ARCH__ >= 900

// Typeless byte pack union - allows converting EltPack to typeless bytes
template<int Bytes>
union BytePack {
  char bytes[Bytes];
  uint16_t u16[(Bytes + 1) / 2];
  uint32_t u32[(Bytes + 3) / 4];
  uint64_t u64[(Bytes + 7) / 8];
  float f32[(Bytes + 3) / 4];
  double f64[(Bytes + 7) / 8];
};

// Typeless multimem store - specialized by byte size only
template<int Bytes>
NCCL_DEVICE_INLINE void multimem_st_global(uintptr_t addr, const BytePack<Bytes>& val);

template<>
NCCL_DEVICE_INLINE void multimem_st_global<0>(uintptr_t addr, const BytePack<0>& val) {
  // nop
}

template<>
NCCL_DEVICE_INLINE void multimem_st_global<1>(uintptr_t addr, const BytePack<1>& val) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
    assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#multimem-addresses.");
    return;
#else
  asm volatile("st.global.b8 [%0], %1;" :: "l"(addr), "r"((uint32_t)val.bytes[0]) : "memory");
#endif
}

template<>
NCCL_DEVICE_INLINE void multimem_st_global<2>(uintptr_t addr, const BytePack<2>& val) {
#ifndef NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE
      assert(false && "Experimental NCCL device code detected; you may accept the risk "
                    "of this not being available in the future. If you accept that risk, "
                    "set NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE during compilation or "
                    "refactor the code. More details "
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#multimem-addresses.");
    return;
#else
  asm volatile("st.global.b16 [%0], %1;" :: "l"(addr), "h"(val.u16[0]) : "memory");
#endif
}

template<>
NCCL_DEVICE_INLINE void multimem_st_global<4>(uintptr_t addr, const BytePack<4>& val) {
  asm volatile("multimem.st.global.b32 [%0], %1;" :: "l"(addr), "r"(val.u32[0]) : "memory");
}

template<>
NCCL_DEVICE_INLINE void multimem_st_global<8>(uintptr_t addr, const BytePack<8>& val) {
  asm volatile("multimem.st.global.b64 [%0], %1;" :: "l"(addr), "l"(val.u64[0]) : "memory");
}

template<>
NCCL_DEVICE_INLINE void multimem_st_global<16>(uintptr_t addr, const BytePack<16>& val) {
  // Use v4.f32 for 16 bytes (4 x 32-bit values) - multimem.st requires .f32 qualifier for vectors
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(addr), "r"(val.u32[0]), "r"(val.u32[1]), "r"(val.u32[2]), "r"(val.u32[3])
               : "memory");
}

#endif // __CUDA_ARCH__ >= 900

// Multimem store - converts EltPack to typeless BytePack
template<typename Pack>
NCCL_DEVICE_INLINE void multimemStore(void* addr, const Pack& pack) {
  // Check architecture requirement
  #if __CUDA_ARCH__ < 900
    assert(false && "multimemStore requires CUDA architecture >= 900 (sm_90 or higher)");
    return;
  #else
    const size_t multimem_addr = __cvta_generic_to_global(addr);
    // Convert EltPack to typeless BytePack via union
    union {
      Pack eltPack;
      BytePack<Pack::Bytes> bytePack;
    } converter;
    converter.eltPack = pack;
    multimem_st_global<Pack::Bytes>(multimem_addr, converter.bytePack);
  #endif
}

} // namespace utility
} // namespace nccl

#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_MULTIMEM__FUNCS_H_
