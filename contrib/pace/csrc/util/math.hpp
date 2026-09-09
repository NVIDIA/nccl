#ifndef PACE_UTIL_MATH_HPP_
#define PACE_UTIL_MATH_HPP_

#include "cuda_runtime.h"

/**
 * Ceiling division
 *
 * @tparam dtype_t  Numeric type
 * @param a         Dividend
 * @param b         Divisor
 * @return          Ceiling of a/b
 */
template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

/**
 * Align value up to multiple of alignment
 *
 * @tparam dtype_t  Numeric type
 * @param a         Value to align
 * @param b         Alignment
 * @return          Value aligned up to multiple of b
 */
template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

#endif  // PACE_UTIL_MATH_HPP_
