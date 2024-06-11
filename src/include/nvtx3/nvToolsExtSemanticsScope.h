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

#ifndef NVTX_SEMANTIC_ID_SCOPE_V1
#define NVTX_SEMANTIC_ID_SCOPE_V1 1

/**
 * \brief Specify the NVTX scope for a payload entry.
 *
 * This allows the scope to be set for a specific value or counter in a payload.
 * The scope must be known at schema registration time.
 */
typedef struct nvtxSemanticsScope_v1
{
    struct nvtxSemanticsHeader_v1 header;

    /** Specifies the scope of a payload entry, e.g. a counter or timestamp. */
    uint64_t scopeId;
} nvtxSemanticsScope_t;

#endif /* NVTX_SEMANTIC_ID_SCOPE_V1 */