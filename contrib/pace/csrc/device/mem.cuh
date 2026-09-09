#ifndef PACE_DEVICE_MEM_CUH
#define PACE_DEVICE_MEM_CUH

// Single-rank device toolkit: vectorized loads/stores, memory-order
// primitives, warp/group reductions & scans, TMA/mbarrier, cache-policy
// helpers and generic partition math. Nothing here touches another rank.

#include <cstdint>
#include "device/configs.cuh"

struct float8 {
    float4 low, high;
};

template <typename T>
__forceinline__ __device__ void get_work_range(const T& works, const int& workers, const int& worker_id, T& start, T& end) {
    const T basic_tokens = works / workers;
    const T remain_tokens = works - basic_tokens * workers;
    start = basic_tokens * worker_id + (worker_id > remain_tokens ? remain_tokens : worker_id);
    end = start + basic_tokens + (worker_id < remain_tokens ? 1 : 0);
}

template <typename T>
__forceinline__ __device__ void get_range_index(const T& members, const T& ranges, const T& member_id, T& range_id) {
    const T basic_range_len = members / ranges;
    const T remain_members = members - basic_range_len * ranges;
    const T long_part = remain_members * (basic_range_len + 1);
    if (member_id < long_part) {
        range_id = member_id / (basic_range_len + 1);
    } else {
        range_id = remain_members + (member_id - long_part) / basic_range_len;
    }
}

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_WARP_REDUCE(UNROLL_FACTOR, LANE_ID, N, DST, SRC_FUNC, N_SRC, ACC_RATIO, LD_FUNC, ST_FUNC, RESET_FUNC, REDUCE_FUNC, REVERT_FUNC, CHECK_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    float4 unrolled_values[(UNROLL_FACTOR) * ACC_RATIO]; \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR) * ACC_RATIO; ++ __j) \
            RESET_FUNC(unrolled_values[__j]); \
        for (int __src_index = 0; __src_index < (N_SRC); ++__src_index) {\
            auto __src = SRC_FUNC(__src_index);\
            _Pragma("unroll") \
            for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
                const auto ld_value = LD_FUNC(__src + __i + __j * 32);\
                CHECK_FUNC(ld_value, __src_index);\
                REDUCE_FUNC(unrolled_values + __j * ACC_RATIO, ld_value); \
            }\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, REVERT_FUNC(unrolled_values + __j * ACC_RATIO)); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR) * ACC_RATIO; ++ __j) \
            RESET_FUNC(unrolled_values[__j]); \
        for (int __src_index = 0; __src_index < (N_SRC); ++__src_index) {\
            auto __src = SRC_FUNC(__src_index);\
            _Pragma("unroll") \
            for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
                if (__i + __j * 32 < (N)) { \
                    const auto ld_value = LD_FUNC(__src + __i + __j * 32);\
                    CHECK_FUNC(ld_value, __src_index);\
                    REDUCE_FUNC(unrolled_values + __j * ACC_RATIO, ld_value); \
                } \
            } \
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, REVERT_FUNC(unrolled_values + __j * ACC_RATIO)); \
            } \
        } \
    } \
}

#define UNROLLED_GROUP_COPY(UNROLL_FACTOR, ID, GSIZE, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    const int kLoopStride = (GSIZE) * (UNROLL_FACTOR); \
    const int kMainPart = (N) / kLoopStride * kLoopStride; \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (ID); __i < kMainPart; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * (GSIZE)); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * (GSIZE), unrolled_values[__j]); \
    } \
    { \
        int __i = kMainPart + (ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * (GSIZE)); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                ST_FUNC(__dst + __i + __j * (GSIZE), unrolled_values[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_BLOCK_COPY(UNROLL_FACTOR, N, DST, SRC, LD_FUNC, ST_FUNC) {\
    UNROLLED_GROUP_COPY(UNROLL_FACTOR, threadIdx.x, blockDim.x, N, DST, SRC, LD_FUNC, ST_FUNC);\
}

#define UNROLLED_WARP_TRANS_AND_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC, TRANS_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            TRANS_FUNC(unrolled_values[__j]); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                TRANS_FUNC(unrolled_values[__j]); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_WARP_2WAY_TRANS_AND_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, SRC_BETA, BETA_RATIO, LD_FUNC, LD_FUNC_BETA, ST_FUNC, TRANS_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)], unrolled_values_beta[(UNROLL_FACTOR) * BETA_RATIO]; \
    auto __src = (SRC); \
    auto __src_beta = (SRC_BETA);\
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            _Pragma("unroll") \
            for (int __k = 0; __k < (BETA_RATIO); ++ __k) {\
                unrolled_values_beta[__j * (BETA_RATIO) + __k] = LD_FUNC_BETA(__src_beta + (__i + __j * 32) * BETA_RATIO + __k); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            TRANS_FUNC(unrolled_values[__j], unrolled_values_beta + __j * BETA_RATIO); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                _Pragma("unroll") \
                for (int __k = 0; __k < (BETA_RATIO); ++ __k) {\
                    unrolled_values_beta[__j * (BETA_RATIO) + __k] = LD_FUNC_BETA(__src_beta + (__i + __j * 32) * BETA_RATIO + __k); \
                } \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                TRANS_FUNC(unrolled_values[__j], unrolled_values_beta + __j * BETA_RATIO); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_WARP_TRANS_AND_COPY_INT4_INT2(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC, TRANS_FUNC) \
{ \
    constexpr int kLoopStride = 32 * (UNROLL_FACTOR); \
    int4 unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            TRANS_FUNC(unrolled_values[__j]); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * 32, *reinterpret_cast<int2*>(&unrolled_values[__j])); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                TRANS_FUNC(unrolled_values[__j]); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * 32 < (N)) { \
                ST_FUNC(__dst + __i + __j * 32, *reinterpret_cast<int2*>(&unrolled_values[__j])); \
            } \
        } \
    } \
}

#define WEIGHT_MUL_BF16_INT4(imm_int4, weight) {\
    float2 imm;\
    _Pragma("unroll") \
    for (int i = 0; i < kBF162PerInt4; ++i) {\
        imm = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(&imm_int4)[i]);\
        imm.x *= weight;\
        imm.y *= weight;\
        reinterpret_cast<__nv_bfloat162*>(&imm_int4)[i] = __float22bfloat162_rn(imm);\
    }\
}

__device__ __forceinline__ void f4_add(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

#define UNROLLED_GROUP_F32_2WAY_REDUCE(UNROLL_FACTOR, ID, GSIZE, N, DST, SRC0, SRC1, LD_FUNC0, LD_FUNC1, ST_FUNC) \
{ \
    const int kLoopStride = (GSIZE) * (UNROLL_FACTOR); \
    const int kMainPart = (N) / kLoopStride * kLoopStride; \
    float4 unrolled_values[(UNROLL_FACTOR)]; \
    auto __dst = (DST); \
    auto __src0 = (SRC0);\
    auto __src1 = (SRC1);\
    for (int __i = (ID); __i < kMainPart; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            unrolled_values[__j] = {0.0f, 0.0f, 0.0f, 0.0f}; \
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            const auto ld_value = LD_FUNC0(__src0 + __i + __j * (GSIZE)); \
            f4_add(unrolled_values[__j], ld_value);\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            const auto ld_value = LD_FUNC1(__src1 + __i + __j * (GSIZE)); \
            f4_add(unrolled_values[__j], ld_value);\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * (GSIZE), unrolled_values[__j]); \
    } \
    { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            unrolled_values[__j] = {0.0f, 0.0f, 0.0f, 0.0f}; \
        }\
        int __i = kMainPart + (ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                const auto ld_value = LD_FUNC0(__src0 + __i + __j * (GSIZE)); \
                f4_add(unrolled_values[__j], ld_value);\
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                const auto ld_value = LD_FUNC1(__src1 + __i + __j * (GSIZE)); \
                f4_add(unrolled_values[__j], ld_value);\
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                ST_FUNC(__dst + __i + __j * (GSIZE), unrolled_values[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_BLOCK_F32_2WAY_REDUCE(UNROLL_FACTOR, N, DST, SRC0, SRC1, LD_FUNC0, LD_FUNC1, ST_FUNC) {\
    UNROLLED_GROUP_F32_2WAY_REDUCE(UNROLL_FACTOR, threadIdx.x, blockDim.x, N, DST, SRC0, SRC1, LD_FUNC0, LD_FUNC1, ST_FUNC);\
}

// V2: separate arrays for src0/src1, issue both batches before any add so the
// compiler exposes 2*UF inflight loads per thread. Higher reg pressure
// (~2*UF float4 alive simultaneously) than the original single-array form.
#define UNROLLED_GROUP_F32_2WAY_REDUCE_V2(UNROLL_FACTOR, ID, GSIZE, N, DST, SRC0, SRC1, LD_FUNC0, LD_FUNC1, ST_FUNC) \
{ \
    const int kLoopStride = (GSIZE) * (UNROLL_FACTOR); \
    const int kMainPart = (N) / kLoopStride * kLoopStride; \
    float4 v0[(UNROLL_FACTOR)]; \
    float4 v1[(UNROLL_FACTOR)]; \
    auto __dst = (DST); \
    auto __src0 = (SRC0);\
    auto __src1 = (SRC1);\
    for (int __i = (ID); __i < kMainPart; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            v0[__j] = LD_FUNC0(__src0 + __i + __j * (GSIZE)); \
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            v1[__j] = LD_FUNC1(__src1 + __i + __j * (GSIZE)); \
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            f4_add(v0[__j], v1[__j]);\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * (GSIZE), v0[__j]); \
    } \
    { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            v0[__j] = {0.0f, 0.0f, 0.0f, 0.0f}; \
            v1[__j] = {0.0f, 0.0f, 0.0f, 0.0f}; \
        }\
        int __i = kMainPart + (ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                v0[__j] = LD_FUNC0(__src0 + __i + __j * (GSIZE)); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                v1[__j] = LD_FUNC1(__src1 + __i + __j * (GSIZE)); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            f4_add(v0[__j], v1[__j]);\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                ST_FUNC(__dst + __i + __j * (GSIZE), v0[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_BLOCK_F32_2WAY_REDUCE_V2(UNROLL_FACTOR, N, DST, SRC0, SRC1, LD_FUNC0, LD_FUNC1, ST_FUNC) {\
    UNROLLED_GROUP_F32_2WAY_REDUCE_V2(UNROLL_FACTOR, threadIdx.x, blockDim.x, N, DST, SRC0, SRC1, LD_FUNC0, LD_FUNC1, ST_FUNC);\
}

#define UNROLLED_GROUP_F32_REDUCE(UNROLL_FACTOR, ID, GSIZE, N, DST, SRC_FUNC, N_SRC, LD_FUNC, ST_FUNC) \
{ \
    const int kLoopStride = (GSIZE) * (UNROLL_FACTOR); \
    const int kMainPart = (N) / kLoopStride * kLoopStride; \
    float4 unrolled_values[(UNROLL_FACTOR)]; \
    auto __dst = (DST); \
    for (int __i = (ID); __i < kMainPart; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            unrolled_values[__j] = {0.0f, 0.0f, 0.0f, 0.0f}; \
        }\
        for (int __src_index = 0; __src_index < (N_SRC); ++__src_index) {\
            auto __src = SRC_FUNC(__src_index);\
            _Pragma("unroll") \
            for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
                const auto ld_value = LD_FUNC(__src + __i + __j * (GSIZE)); \
                f4_add(unrolled_values[__j], ld_value);\
            }\
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * (GSIZE), unrolled_values[__j]); \
    } \
    { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) {\
            unrolled_values[__j] = {0.0f, 0.0f, 0.0f, 0.0f}; \
        }\
        int __i = kMainPart + (ID); \
        for (int __src_index = 0; __src_index < (N_SRC); ++__src_index) {\
            auto __src = SRC_FUNC(__src_index);\
            _Pragma("unroll") \
            for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
                if (__i + __j * (GSIZE) < (N)) { \
                    const auto ld_value = LD_FUNC(__src + __i + __j * (GSIZE)); \
                    f4_add(unrolled_values[__j], ld_value);\
                } \
            } \
        }\
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * (GSIZE) < (N)) { \
                ST_FUNC(__dst + __i + __j * (GSIZE), unrolled_values[__j]); \
            } \
        } \
    } \
}

#define UNROLLED_BLOCK_F32_REDUCE(UNROLL_FACTOR, N, DST, SRC_FUNC, N_SRC, LD_FUNC, ST_FUNC) {\
    UNROLLED_GROUP_F32_REDUCE(UNROLL_FACTOR, threadIdx.x, blockDim.x, N, DST, SRC_FUNC, N_SRC, LD_FUNC, ST_FUNC);\
}


#define UNROLLED_MINIWARP_COPY(UNROLL_FACTOR, MINIWARP_SIZE, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = MINIWARP_SIZE * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * MINIWARP_SIZE); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * MINIWARP_SIZE, unrolled_values[__j]); \
    } \
    { \
        int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * MINIWARP_SIZE < (N)) { \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * MINIWARP_SIZE); \
            } \
        } \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) { \
            if (__i + __j * MINIWARP_SIZE < (N)) { \
                ST_FUNC(__dst + __i + __j * MINIWARP_SIZE, unrolled_values[__j]); \
            } \
        } \
    } \
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

__device__  __forceinline__ uint64_t ld_acquire_global(const uint64_t *ptr) {
    uint64_t ret;
    asm volatile( "ld.acquire.gpu.global.u64 %0, [%1];"
            : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
    uint64_t ret;
    asm volatile( "ld.acquire.sys.global.u64 %0, [%1];"
            : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
    int ret;
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_cta(const uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.cta.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint32_t ld_acquire_cta(const uint32_t *ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.cta.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int4 ld_nc_global(const int4 *ptr) {
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
            : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int2 ld_nc_global(const int2 *ptr) {
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];"
            : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int ld_nc_global(const int *ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];"
            : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ __nv_bfloat162 ld_nc_global(const __nv_bfloat162 *ptr) {
    int ret;
    ret = ld_nc_global(reinterpret_cast<const int*>(ptr));
    return *reinterpret_cast<__nv_bfloat162*>(&ret);
}

__device__  __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
    uint16_t ret;
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];"
            : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__  __forceinline__ float4 ld_nc_global(const float4 *ptr) {
    float4 ret;
    asm volatile(LD_NC_FUNC ".v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float2 ld_nc_global(const float2 *ptr) {
    float2 ret;
    asm volatile(LD_NC_FUNC ".v2.f32 {%0, %1}, [%2];"
            : "=f"(ret.x), "=f"(ret.y) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_nc_global(const float *ptr) {
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];"
            : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int16_t ld_nc_global(const int16_t *ptr) {
    int16_t ret;
    asm volatile(LD_NC_FUNC ".s16 %0, [%1];"
            : "=h"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ __nv_bfloat16 ld_nc_global(const __nv_bfloat16 *ptr) {
    int16_t ret;
    ret = ld_nc_global(reinterpret_cast<const int16_t*>(ptr));
    return *reinterpret_cast<__nv_bfloat16*>(&ret);
}

__device__  __forceinline__ float8 ld_nc_global(const float8 *ptr) {
    float8 ret;
#if __CUDA_ARCH__ >= 1000
    // SM100+: .v8.f32 (256-bit vector load) is supported.
    asm volatile(LD_NC_FUNC ".v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=f"(ret.low.x), "=f"(ret.low.y), "=f"(ret.low.z), "=f"(ret.low.w), "=f"(ret.high.x), "=f"(ret.high.y), "=f"(ret.high.z), "=f"(ret.high.w) : "l"(ptr));
#else
    // SM90 (H100): ptxas caps vector width at 128 bits (.v4); emit two .v4.f32 loads
    // of the two contiguous float4 halves (low at +0B, high at +16B).
    const float4 *p4 = reinterpret_cast<const float4*>(ptr);
    asm volatile(LD_NC_FUNC ".v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(ret.low.x), "=f"(ret.low.y), "=f"(ret.low.z), "=f"(ret.low.w) : "l"(p4));
    asm volatile(LD_NC_FUNC ".v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(ret.high.x), "=f"(ret.high.y), "=f"(ret.high.z), "=f"(ret.high.w) : "l"(p4 + 1));
#endif
    return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

__device__  __forceinline__ void st_na_global(int4 *ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};"
            ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__  __forceinline__ void st_na_global(int2 *ptr, const int2& value) {
    asm volatile(ST_NA_FUNC ".v2.s32 [%0], {%1, %2};"
            ::"l"(ptr), "r"(value.x), "r"(value.y));
}

__device__  __forceinline__ void st_na_global(float4 *ptr, const float4& value) {
    asm volatile(ST_NA_FUNC ".v4.f32 [%0], {%1, %2, %3, %4};"
            ::"l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w));
}

// Store float4 to global, optionally pre-multiplying by `inv_num_ranks` (reduce
// average). Shared by the RS kernels.
template <bool kNeedAvg>
__device__ __forceinline__ void wrap_st_dev(float4* ptr, float4& value, const float& inv_num_ranks) {
    if constexpr (kNeedAvg) {
        value.x *= inv_num_ranks;
        value.y *= inv_num_ranks;
        value.z *= inv_num_ranks;
        value.w *= inv_num_ranks;
    }
    st_na_global(ptr, value);
}

__device__  __forceinline__ void st_na_global(float2 *ptr, const float2& value) {
    asm volatile(ST_NA_FUNC ".v2.f32 [%0], {%1, %2};"
            ::"l"(ptr), "f"(value.x), "f"(value.y));
}

__device__  __forceinline__ void st_na_global(float *ptr, const float& value) {
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;"
            ::"l"(ptr), "f"(value));
}

__device__  __forceinline__ void st_na_global(float8 *ptr, const float8& value) {
#if __CUDA_ARCH__ >= 1000
    // SM100+: .v8.f32 (256-bit vector store) is supported.
    asm volatile(ST_NA_FUNC ".v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            ::"l"(ptr), "f"(value.low.x), "f"(value.low.y), "f"(value.low.z), "f"(value.low.w), "f"(value.high.x), "f"(value.high.y), "f"(value.high.z), "f"(value.high.w));
#else
    // SM90 (H100): ptxas caps vector width at 128 bits (.v4); emit two .v4.f32 stores
    // of the two contiguous float4 halves (low at +0B, high at +16B).
    float4 *p4 = reinterpret_cast<float4*>(ptr);
    asm volatile(ST_NA_FUNC ".v4.f32 [%0], {%1, %2, %3, %4};"
            ::"l"(p4), "f"(value.low.x), "f"(value.low.y), "f"(value.low.z), "f"(value.low.w));
    asm volatile(ST_NA_FUNC ".v4.f32 [%0], {%1, %2, %3, %4};"
            ::"l"(p4 + 1), "f"(value.high.x), "f"(value.high.y), "f"(value.high.z), "f"(value.high.w));
#endif
}

// Final output store for the RS kernels: optionally fuse the reduce-average
// (inv_num_ranks) and an extra post-multiply (extra_post_mul) into a single
// scale, then either cast float4 -> bf16x2-pair into the squeezed (half-width)
// output buffer, or store float4 directly. kNeedAvg=false recovers the
// avg-less variant (avg was applied upstream). Shared by the RS kernels.
template <bool kNeedCast, bool kMul, bool kNeedAvg>
__device__ __forceinline__ void out_cast_if_needed_dev(float4* ptr, float4& val,
    const float& extra_post_mul, const float& inv_num_ranks, void* out_ptr) {
    float scale = 1.0f;
    if constexpr (kNeedAvg) scale *= inv_num_ranks;
    if constexpr (kMul) scale *= extra_post_mul;
    constexpr bool kNeedScale = kNeedAvg || kMul;
    if constexpr (kNeedCast) {
        float2 casted;
        __nv_bfloat162* cptr = reinterpret_cast<__nv_bfloat162*>(&casted);
        if constexpr (kNeedScale) {
            cptr[0] = __float22bfloat162_rn(make_float2(val.x * scale, val.y * scale));
            cptr[1] = __float22bfloat162_rn(make_float2(val.z * scale, val.w * scale));
        } else {
            cptr[0] = __float22bfloat162_rn(reinterpret_cast<const float2*>(&val)[0]);
            cptr[1] = __float22bfloat162_rn(reinterpret_cast<const float2*>(&val)[1]);
        }
        float2* squeeze_ptr = reinterpret_cast<float2*>(out_ptr) + (ptr - reinterpret_cast<float4*>(out_ptr));
        st_na_global(squeeze_ptr, casted);
    } else {
        if constexpr (kNeedScale) {
            val.x *= scale; val.y *= scale;
            val.z *= scale; val.w *= scale;
        }
        st_na_global(ptr, val);
    }
}

__device__  __forceinline__ void st_na_global(uint64_t *ptr, const uint64_t& value) {
    asm volatile(ST_NA_FUNC ".u64 [%0], %1;" ::"l"(ptr), "l"(value));
}

__device__  __forceinline__ void st_na_global(int64_t *ptr, const int64_t& value) {
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

__device__  __forceinline__ void st_na_global(int *ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

__device__  __forceinline__ void st_na_global(uint8_t *ptr, const uint8_t& value) {
    asm volatile(ST_NA_FUNC ".u8 [%0], %1;" ::"l"(ptr), "h"(static_cast<uint16_t>(value)));
}

__device__  __forceinline__ void st_release_cta(const int *ptr, int val) {
    asm volatile("st.release.cta.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__  __forceinline__ void st_release_cta(const uint32_t *ptr, uint32_t val) {
    asm volatile("st.release.cta.u32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__  __forceinline__ void st_release_cta(const uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.cta.u64 [%0], %1;"::"l"(ptr), "l"(val) : "memory");
}

__device__  __forceinline__ void st_release_global(const int *ptr, int val) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__  __forceinline__ void st_release_global(const uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;"::"l"(ptr), "l"(val) : "memory");
}

__device__  __forceinline__ void st_release_sys_global(const int *ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__  __forceinline__ void st_release_sys_global(const uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"::"l"(ptr), "l"(val) : "memory");
}

__device__ __forceinline__ uint32_t get_lane_id() {
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ uint32_t elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES
    uint32_t pred = 0;
    asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "      elect.sync %%rx|%%px, %1;\n"
      "@%%px mov.s32 %0, 1;\n"
      "}\n"
      : "+r"(pred)
      : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
}

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

// Operation functors
template <typename T> struct ReduceSum { __device__ T operator()(T a, T b) const { return a + b; } };
template <typename T> struct ReduceMax { __device__ T operator()(T a, T b) const { return a > b ? a : b; } };
template <typename T> struct ReduceMin { __device__ T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> struct ReduceAnd { __device__ T operator()(T a, T b) const { return a & b; } };
template <typename T> struct ReduceOr  { __device__ T operator()(T a, T b) const { return a | b; } };

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                     kNumLanesPerGroup ==  4 or kNumLanesPerGroup == 2  or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));
        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));
        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));
        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));
        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));
    }
    return value;
}

// Convenience aliases
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

template <typename T>
__forceinline__ __host__ __device__ void div_and_remain(const T& main, const T& div, T& multiple, T& remain) {
    multiple = main / div;
    remain = main - multiple * div;
}

enum EagerScope{
    EAGER_SCOPE_BLOCK = 0,
    EAGER_SCOPE_WARP  = 1,
};
template <enum EagerScope kScope, typename T=int>
__forceinline__ __device__ void hillis_steele_sum(T *ptr, const int nelems, const int thread_idx) {
    // Hillis-Steele Algorithm
    auto scope_sync = []() {
        if constexpr (kScope == EAGER_SCOPE_BLOCK) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    };
    const int num_threads = kScope == EAGER_SCOPE_BLOCK ? blockDim.x : 32;
    for (int stride = 1; stride < nelems; stride *= 2) {
        for (int idx = nelems - 1; idx - stride >= 0; idx -= num_threads) {
            const int my_idx = idx - num_threads + 1 + thread_idx;
            bool work = (my_idx - stride) >= 0;
            const auto load_v = work ? ptr[my_idx - stride] : 0;
            scope_sync();
            if (work) ptr[my_idx] += load_v;
            scope_sync();
        }
    }
};

template <enum EagerScope kScope>
__forceinline__ __device__ void hillis_steele_sum_dr(int *ptr, const int nelems, const int* first_read_ptr) {
    // Hillis-Steele Algorithm
    auto scope_sync = []() {
        if constexpr (kScope == EAGER_SCOPE_BLOCK) {
            __threadfence_block();
            __syncthreads();
        } else {
            __syncwarp();
        }
    };
    const int num_threads = kScope == EAGER_SCOPE_BLOCK ? blockDim.x : 32;
    const int thread_idx = kScope == EAGER_SCOPE_BLOCK ? threadIdx.x : threadIdx.x % 32;
    for (int stride = 1; stride < nelems; stride *= 2) {
        for (int idx = nelems - 1; idx - stride >= 0; idx -= num_threads) {
            const int my_idx = idx - num_threads + 1 + thread_idx;
            bool work = (my_idx - stride) >= 0;
            int load_v, place_v;
            if (stride == 1) {
                load_v = work ? first_read_ptr[my_idx - stride] : 0;
                place_v = first_read_ptr[my_idx];
            } else {
                load_v = work ? ptr[my_idx - stride] : 0;
                place_v = ptr[my_idx];
            }
            scope_sync();
            if (work) ptr[my_idx] = place_v + load_v;
            scope_sync();
        }
    }
};

// bound function: binary search in a sorted array
// kLowerBound=true: returns LAST index where element < target (or -1 if none)
// kLowerBound=false: returns LAST index where element <= target (or -1 if none)
// NOTE: Returns -1 when no element satisfies the condition (first element already fails)
template <bool kLowerBound = true, typename T>
__forceinline__ __device__ int bound(const T *ptr, const int nelems, const T& target) {
    if constexpr (kLowerBound) {
        if (*ptr >= target) return -1;
        if (ptr[nelems - 1] < target) return nelems;
        int lb = 0, rb = nelems;
        while (rb - lb != 1) {
            int middle = (lb + rb) / 2;
            if (ptr[middle] >= target) {
                rb = middle;
            } else {
                lb = middle;
            }
        }
        return lb;
    } else {
        if (*ptr > target) return -1;
        if (ptr[nelems - 1] <= target) return nelems;
        int lb = 0, rb = nelems;
        while (rb - lb != 1) {
            int middle = (lb + rb) / 2;
            if (ptr[middle] > target) {
                rb = middle;
            } else {
                lb = middle;
            }
        }
        return lb;
    }
}

__forceinline__ __device__ float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}


__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster; \n" :: );
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" :: "r"(arrive_count), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(mbar_int_ptr));
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    const auto& wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
    asm volatile("{\n\t"
                 ".reg .pred       P1; \n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
                 "@P1 bra DONE; \n\t"
                 "bra     LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}" :: "r"(mbar_int_ptr), "r"(wait), "r"(0x989680));
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" :: "r"(num_bytes), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" :: "r"(mbar_int_ptr));
}

__device__ __forceinline__ void tma_store_fence() {
    asm volatile ("fence.proxy.async.shared::cta;");
}

// Cross-proxy fence after TMA stores to gmem. `cp.async.bulk.wait_group.read`
// only guarantees completion from the issuing thread's view of the async proxy.
// To make the TMA-written gmem bytes observable to the generic proxy (regular
// loads from other threads / other CTAs) and to the sys proxy (e.g. RDMA
// engines reading the same buffer), a `fence.proxy.async.global` is required
// AFTER `tma_store_wait<0>()` and BEFORE any generic/sys-scope release that
// advertises the data. Without it remote consumers may observe stale bytes.

template <int kFenceLevel>
__device__ __forceinline__ void tma_store_gmem_fence() {
    if constexpr (kFenceLevel == 1) {
        asm volatile ("fence.proxy.async.release.global;");
    } else if constexpr (kFenceLevel == 2) {
        asm volatile ("fence.proxy.async.release.sys;");
    }
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes,
                                            bool evict_first = true) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(smem_int_ptr), "l"(gmem_ptr), "r"(num_bytes), "r"(mbar_int_ptr), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tma_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}

template <bool kCommitGroup = true>
__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes,
                                             bool evict_first = true) {
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n"
                 :: "l"(gmem_ptr), "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint) : "memory");
    if constexpr (kCommitGroup) {
        tma_commit_group();
    }
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

// Strong variant: waits until all-but-N groups are fully complete (including
// the gmem write phase). The `.read` variant only guarantees that the source
// smem is safe to overwrite; the gmem-side DMA may still be in flight.
// Use this (not the `.read` flavor) before publishing a sys-scope release so
// that remote consumers observing the signal will see the final bytes.
template <int N>
__device__ __forceinline__ void tma_store_wait_complete() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}


// Build an L2 evict_last cache-policy descriptor
// Used together with the PTX ld.global.L2::cache_hint cache policy
__device__ __forceinline__
uint64_t make_l2_evict_last_policy()
{
    uint64_t policy;
    // retention fraction 1.0, i.e. keep everything in L2 (evict_last = evicted last)
    asm volatile(
        "createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
        : "=l"(policy)
    );
    return policy;
}

/**
 * @brief Atomic add (int*, global memory, L2 evict_last cache hint)
 *
 * @param addr  target global memory address
 * @param val   addend
 * @return      the old value before the operation
 *
 * PTX:
 *   atom.relaxed.gpu.global.add.L2::cache_hint.s32 d, [a], b, cache-policy;
 */
__device__ __forceinline__
int atomic_add_global_l2_keep(int* addr, int val)
{
    int old;
    uint64_t cache_policy = make_l2_evict_last_policy();

    asm volatile(
        "atom.relaxed.gpu.global.add.L2::cache_hint.s32 %0, [%1], %2, %3;"
        : "=r"(old)                         // %0: d (output, old value)
        : "l"(addr), "r"(val), "l"(cache_policy)  // %1: [a], %2: b, %3: cache-policy
        : "memory"
    );

    return old;
}

#endif  // PACE_DEVICE_MEM_CUH
