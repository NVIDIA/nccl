/*
* Copyright 2024  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

/**
 * NVTX semantic headers require nvToolsExtPayload.h to be included beforehand.
 */

#ifndef NVTX_SEMANTIC_ID_COUNTERS_V1
#define NVTX_SEMANTIC_ID_COUNTERS_V1 2

/**
 * Flags to extend the semantics of counters.
 */
#define NVTX_COUNTERS_FLAGS_NONE  0

/**
 * Convert the fixed point value to a normalized floating point value.
 * Unsigned [0f : 1f] or signed [-1f : 1f] is determined by the underlying type
 * this flag is applied to.
 */
#define NVTX_COUNTERS_FLAG_NORMALIZE    (1 << 1)

/**
 *  Visual tools should apply scale and limits when graphing.
 */
#define NVTX_COUNTERS_FLAG_LIMIT_MIN    (1 << 2)
#define NVTX_COUNTERS_FLAG_LIMIT_MAX    (1 << 3)
#define NVTX_COUNTERS_FLAG_LIMITS \
    (NVTX_COUNTERS_FLAG_LIMIT_MIN | NVTX_COUNTERS_FLAG_LIMIT_MAX)

/**
 * Counter time scopes.
 */
#define NVTX_COUNTERS_FLAG_TIMESCOPE_POINT        (1 << 5)
#define NVTX_COUNTERS_FLAG_TIMESCOPE_SINCE_LAST   (2 << 5)
#define NVTX_COUNTERS_FLAG_TIMESCOPE_UNTIL_NEXT   (3 << 5)
#define NVTX_COUNTERS_FLAG_TIMESCOPE_SINCE_START  (4 << 5)

/**
 * Counter value types.
 */
#define NVTX_COUNTERS_FLAG_VALUETYPE_ABSOLUTE (1 << 10)
/** Delta to previous value of same counter type. */
#define NVTX_COUNTERS_FLAG_VALUETYPE_DELTA    (2 << 10)

/**
 * Datatypes for the `limits` union.
 */
#define NVTX_COUNTERS_LIMIT_I64 0
#define NVTX_COUNTERS_LIMIT_U64 1
#define NVTX_COUNTERS_LIMIT_F64 2

/**
 *\brief Specify counter semantics.
 */
typedef struct nvtxSemanticsCounter_v1 {
    /** Header of the semantic extensions (with identifier, version, etc.). */
    struct nvtxSemanticsHeader_v1 header;

    /** Flags to provide more context about the counter value. */
    uint64_t flags;

    /** Unit of the counter value (case-insensitive). */
    const char*  unit;

    /** Should be 1 if not used. */
    uint64_t unitScaleNumerator;

    /** Should be 1 if not used. */
    uint64_t unitScaleDenominator;

    /** Determines the used union member. Use defines `NVTX_COUNTER_LIMIT_*`. */
    int64_t limitType;

    /** Graph limits {minimum, maximum}. */
    union limits_t {
        int64_t  i64[2];
        uint64_t u64[2];
        double   d[2];
    } limits;
} nvtxSemanticsCounter_t;

#endif /* NVTX_SEMANTIC_ID_COUNTERS_V1 */