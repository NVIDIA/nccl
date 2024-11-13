/*
* Copyright 2021-2023  NVIDIA Corporation.  All rights reserved.
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

#ifndef NVTX_EXT_IMPL_PAYLOAD_V1
#define NVTX_EXT_IMPL_PAYLOAD_V1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Macros to create versioned symbols. */
#define NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID) \
    NAME##_v##VERSION##_bpl##COMPATID
#define NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L2(NAME, VERSION, COMPATID) \
    NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID)
#define NVTX_EXT_PAYLOAD_VERSIONED_ID(NAME) \
    NVTX_EXT_PAYLOAD_VERSIONED_IDENTIFIER_L2(NAME, NVTX_VERSION, NVTX_EXT_PAYLOAD_COMPATID)

#ifdef NVTX_DISABLE

#include "nvtxExtHelperMacros.h"

#define NVTX_EXT_PAYLOAD_IMPL_FN_V1(ret_val, fn_name, signature, arg_names) \
ret_val fn_name signature { \
    NVTX_EXT_HELPER_UNUSED_ARGS arg_names \
    return ((ret_val)(intptr_t)-1); \
}

#else /* NVTX_DISABLE */

#include "nvtxExtPayloadTypeInfo.h"

/*
 * Function slots for the payload extension. First entry is the module state,
 * initialized to `0` (`NVTX_EXTENSION_FRESH`).
 */
#define NVTX_EXT_PAYLOAD_SLOT_COUNT 63
NVTX_LINKONCE_DEFINE_GLOBAL intptr_t
NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots)[NVTX_EXT_PAYLOAD_SLOT_COUNT + 1]
    = {0};

/* Avoid warnings about missing prototype. */
NVTX_LINKONCE_FWDDECL_FUNCTION void NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadInitOnce)(void);
NVTX_LINKONCE_DEFINE_FUNCTION void NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadInitOnce)()
{
    intptr_t* fnSlots = NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots) + 1;
    nvtxExtModuleSegment_t segment = {
        0, /* unused (only one segment) */
        NVTX_EXT_PAYLOAD_SLOT_COUNT,
        fnSlots
    };

    nvtxExtModuleInfo_t module = {
        NVTX_VERSION, sizeof(nvtxExtModuleInfo_t),
        NVTX_EXT_PAYLOAD_MODULEID, NVTX_EXT_PAYLOAD_COMPATID,
        1, &segment, /* number of segments, segments */
        NULL, /* no export function needed */
        /* bake type sizes and alignment information into program binary */
        &(NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadTypeInfo))
    };

    NVTX_INFO( "%s\n", __FUNCTION__  );

    NVTX_VERSIONED_IDENTIFIER(nvtxExtInitOnce)(&module,
        NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots));
}

#define NVTX_EXT_PAYLOAD_IMPL_FN_V1(ret_type, fn_name, signature, arg_names) \
typedef ret_type (*fn_name##_impl_fntype)signature; \
    NVTX_DECLSPEC ret_type NVTX_API fn_name signature { \
    intptr_t slot = NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots)[NVTX3EXT_CBID_##fn_name + 1]; \
    if (slot != NVTX_EXTENSION_DISABLED) { \
        if (slot != NVTX_EXTENSION_FRESH) { \
            return (*(fn_name##_impl_fntype)slot) arg_names; \
        } else { \
            NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadInitOnce)(); \
            /* Re-read function slot after extension initialization. */ \
            slot = NVTX_EXT_PAYLOAD_VERSIONED_ID(nvtxExtPayloadSlots)[NVTX3EXT_CBID_##fn_name + 1]; \
            if (slot != NVTX_EXTENSION_DISABLED && slot != NVTX_EXTENSION_FRESH) { \
                return (*(fn_name##_impl_fntype)slot) arg_names; \
            } \
        } \
    } \
    NVTX_EXT_FN_RETURN_INVALID(ret_type) \
}

#endif /*NVTX_DISABLE*/

/* Non-void functions. */
#define NVTX_EXT_FN_RETURN_INVALID(rtype) return ((rtype)(intptr_t)-1);

NVTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, nvtxPayloadSchemaRegister,
    (nvtxDomainHandle_t domain, const nvtxPayloadSchemaAttr_t* attr),
    (domain, attr))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, nvtxPayloadEnumRegister,
    (nvtxDomainHandle_t domain, const nvtxPayloadEnumAttr_t* attr),
    (domain, attr))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(int, nvtxRangePushPayload,
    (nvtxDomainHandle_t domain, const nvtxPayloadData_t* payloadData, size_t count),
    (domain, payloadData, count))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(int, nvtxRangePopPayload,
    (nvtxDomainHandle_t domain, const nvtxPayloadData_t* payloadData, size_t count),
    (domain, payloadData, count))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(nvtxRangeId_t, nvtxRangeStartPayload,
    (nvtxDomainHandle_t domain, const nvtxPayloadData_t* payloadData, size_t count),
    (domain, payloadData, count))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(uint8_t, nvtxDomainIsEnabled, (nvtxDomainHandle_t domain), (domain))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(uint64_t, nvtxScopeRegister, (nvtxDomainHandle_t domain,
    const nvtxScopeAttr_t* attr), (domain, attr))

#undef NVTX_EXT_FN_RETURN_INVALID
/* END: Non-void functions. */

/* void functions. */
#define NVTX_EXT_FN_RETURN_INVALID(rtype)
#define return

NVTX_EXT_PAYLOAD_IMPL_FN_V1(void, nvtxMarkPayload, (nvtxDomainHandle_t domain,
    const nvtxPayloadData_t* payloadData, size_t count), (domain, payloadData, count))

NVTX_EXT_PAYLOAD_IMPL_FN_V1(void, nvtxRangeEndPayload, (nvtxDomainHandle_t domain,
    nvtxRangeId_t id, const nvtxPayloadData_t* payloadData, size_t count),
    (domain, id, payloadData, count))

#undef return
#undef NVTX_EXT_FN_RETURN_INVALID
/* END: void functions. */

/* Keep NVTX_EXT_PAYLOAD_IMPL_FN_V1 defined for a future version of this extension. */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* NVTX_EXT_IMPL_PAYLOAD_V1 */

