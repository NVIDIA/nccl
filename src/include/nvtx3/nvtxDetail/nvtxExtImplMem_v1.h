/*
* Copyright 2009-2020,2023  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_EXT_IMPL_MEM_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExtMem.h (except when NVTX_NO_IMPL is defined).
#endif

#define NVTX_EXT_IMPL_GUARD
#include "nvtxExtImpl.h"
#undef NVTX_EXT_IMPL_GUARD

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define NVTXMEM_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID) NAME##_v##VERSION##_mem##COMPATID
#define NVTXMEM_VERSIONED_IDENTIFIER_L2(NAME, VERSION, COMPATID) NVTXMEM_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID)
#define NVTX_EXT_MEM_VERSIONED_ID(NAME) NVTXMEM_VERSIONED_IDENTIFIER_L2(NAME, NVTX_VERSION, NVTX_EXT_COMPATID_MEM)

#ifdef NVTX_DISABLE

#include "nvtxExtHelperMacros.h"

#define NVTX_EXT_FN_IMPL(ret_val, fn_name, signature, arg_names) \
ret_val fn_name signature { \
    NVTX_EXT_HELPER_UNUSED_ARGS arg_names \
    return ((ret_val)(intptr_t)-1); \
}

#else  /* NVTX_DISABLE */

/*
 * Function slots for the memory extension. First entry is the module
 * state, initialized to `0` (`NVTX_EXTENSION_FRESH`).
 */
NVTX_LINKONCE_DEFINE_GLOBAL intptr_t
NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemSlots)[NVTX3EXT_CBID_MEM_FN_NUM + 2]
    = {0};

NVTX_LINKONCE_DEFINE_FUNCTION void NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemInitOnce)()
{
    intptr_t* fnSlots = NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemSlots) + 1;
    nvtxExtModuleSegment_t segment = {
        0, /* unused (only one segment) */
        NVTX3EXT_CBID_MEM_FN_NUM,
        fnSlots
    };

    nvtxExtModuleInfo_t module = {
        NVTX_VERSION, sizeof(nvtxExtModuleInfo_t),
        NVTX_EXT_MODULEID_MEM, NVTX_EXT_COMPATID_MEM,
        1, &segment,
        NULL, /* no export function needed */
        NULL
    };

    NVTX_INFO( "%s\n", __FUNCTION__  );

    NVTX_VERSIONED_IDENTIFIER(nvtxExtInitOnce)(&module,
        NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemSlots));
}

#define NVTX_EXT_FN_IMPL(ret_type, fn_name, signature, arg_names) \
typedef ret_type ( * fn_name##_impl_fntype )signature; \
    NVTX_DECLSPEC ret_type NVTX_API fn_name signature { \
    intptr_t slot = NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemSlots)[NVTX3EXT_CBID_##fn_name + 1]; \
    if (slot != NVTX_EXTENSION_DISABLED) { \
        if (slot != NVTX_EXTENSION_FRESH) { \
            return (*(fn_name##_impl_fntype)slot) arg_names; \
        } else { \
            NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemInitOnce)(); \
            /* Re-read function slot after extension initialization. */ \
            slot = NVTX_EXT_MEM_VERSIONED_ID(nvtxExtMemSlots)[NVTX3EXT_CBID_##fn_name + 1]; \
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

NVTX_EXT_FN_IMPL(nvtxMemHeapHandle_t, nvtxMemHeapRegister, (nvtxDomainHandle_t domain, nvtxMemHeapDesc_t const* desc), (domain, desc))

NVTX_EXT_FN_IMPL(nvtxMemPermissionsHandle_t, nvtxMemPermissionsCreate, (nvtxDomainHandle_t domain, int32_t creationflags), (domain, creationflags))

#undef NVTX_EXT_FN_RETURN_INVALID
/* END: Non-void functions. */

/* void functions. */
#define NVTX_EXT_FN_RETURN_INVALID(rtype)
#define return

NVTX_EXT_FN_IMPL(void, nvtxMemHeapUnregister, (nvtxDomainHandle_t domain, nvtxMemHeapHandle_t heap), (domain, heap))

NVTX_EXT_FN_IMPL(void, nvtxMemHeapReset, (nvtxDomainHandle_t domain, nvtxMemHeapHandle_t heap), (domain, heap))

NVTX_EXT_FN_IMPL(void, nvtxMemRegionsRegister, (nvtxDomainHandle_t domain, nvtxMemRegionsRegisterBatch_t const* desc), (domain, desc))

NVTX_EXT_FN_IMPL(void, nvtxMemRegionsResize, (nvtxDomainHandle_t domain,nvtxMemRegionsResizeBatch_t const* desc), (domain, desc))

NVTX_EXT_FN_IMPL(void, nvtxMemRegionsUnregister, (nvtxDomainHandle_t domain,nvtxMemRegionsUnregisterBatch_t const* desc), (domain, desc))

NVTX_EXT_FN_IMPL(void, nvtxMemRegionsName, (nvtxDomainHandle_t domain,nvtxMemRegionsNameBatch_t const* desc), (domain, desc))

NVTX_EXT_FN_IMPL(void, nvtxMemPermissionsAssign, (nvtxDomainHandle_t domain,nvtxMemPermissionsAssignBatch_t const* desc), (domain, desc))

NVTX_EXT_FN_IMPL(void, nvtxMemPermissionsDestroy, (nvtxDomainHandle_t domain, nvtxMemPermissionsHandle_t permissions), (domain, permissions))

NVTX_EXT_FN_IMPL(void, nvtxMemPermissionsReset, (nvtxDomainHandle_t domain, nvtxMemPermissionsHandle_t permissions), (domain, permissions))

NVTX_EXT_FN_IMPL(void, nvtxMemPermissionsBind, (nvtxDomainHandle_t domain, nvtxMemPermissionsHandle_t permissions, uint32_t bindScope, uint32_t bindFlags), (domain, permissions, bindScope, bindFlags))

NVTX_EXT_FN_IMPL(void, nvtxMemPermissionsUnbind, (nvtxDomainHandle_t domain, uint32_t bindScope), (domain, bindScope))

#undef return
#undef NVTX_EXT_FN_RETURN_INVALID
/* END: void functions. */

#undef NVTX_EXT_FN_IMPL

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
