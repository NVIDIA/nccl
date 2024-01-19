/*
* Copyright 2021  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_EXT_IMPL_PAYLOAD_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExtPayload.h (except when NVTX_NO_IMPL is defined).
#endif

#define NVTX_EXT_IMPL_GUARD
#include "nvtxExtImpl.h"
#undef NVTX_EXT_IMPL_GUARD

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID) \
    NAME##_v##VERSION##_mem##COMPATID
#define NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L2(NAME, VERSION, COMPATID) \
    NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID)
#define NVTX_EXT_PAYLOAD_VERSIONED_ID(NAME) \
    NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L2(NAME, NVTX_VERSION, NVTX_EXT_COMPATID_PAYLOAD)

/*
 * Function slots for the binary payload extension. First entry is the module
 * state, initialized to `0` (`NVTX_EXTENSION_FRESH`).
 */
NVTX_LINKONCE_DEFINE_GLOBAL intptr_t
NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots)[NVTX3EXT_CBID_PAYLOAD_FN_NUM + 1]
    = {0};

NVTX_LINKONCE_DEFINE_FUNCTION void NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadInitOnce)()
{
    intptr_t* fnSlots = NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots) + 1;
    nvtxExtModuleSegment_t segment = {
        0, // unused (only one segment)
        NVTX3EXT_CBID_PAYLOAD_FN_NUM,
        fnSlots
    };

    nvtxExtModuleInfo_t module = {
        NVTX_VERSION, sizeof(nvtxExtModuleInfo_t),
        NVTX_EXT_MODULEID_PAYLOAD, NVTX_EXT_COMPATID_PAYLOAD,
        1, &segment, // number of segments, segments
        NULL, // no export function needed
        // bake type sizes and alignment information into program binary
        &nvtxExtPayloadTypeInfo
    };

    NVTX_INFO( "%s\n", __FUNCTION__  );

    NVTX_VERSIONED_IDENTIFIER(nvtxExtInitOnce)(&module,
        NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots));
}

#define NVTX_EXT_FN_IMPL(ret_val, fn_name, signature, arg_names) \
typedef ret_val ( * fn_name##_impl_fntype )signature; \
NVTX_LINKONCE_DEFINE_FUNCTION ret_val fn_name signature { \
    intptr_t slot = NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots)[NVTX3EXT_CBID_##fn_name + 1]; \
    if (slot != NVTX_EXTENSION_DISABLED) { \
        if (slot) { \
            return (*(fn_name##_impl_fntype)slot) arg_names; \
        } else { \
            NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadInitOnce)(); \
            slot = NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots)[NVTX3EXT_CBID_##fn_name + 1]; \
            if (slot != NVTX_EXTENSION_DISABLED && slot) { \
                return (*(fn_name##_impl_fntype)slot) arg_names; \
            } \
        } \
    } \
    return ((ret_val)(intptr_t)-1); \
}

NVTX_EXT_FN_IMPL(uint64_t, nvtxPayloadSchemaRegister, (nvtxDomainHandle_t domain, const nvtxPayloadSchemaAttr_t* attr), (domain, attr))

NVTX_EXT_FN_IMPL(uint64_t, nvtxPayloadEnumRegister, (nvtxDomainHandle_t domain, const nvtxPayloadEnumAttr_t* attr), (domain, attr))

#undef NVTX_EXT_FN_IMPL

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */