/*
* Copyright 2009-2020  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_EXT_IMPL_MEM_CUDART_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExtMemCudaRt.h (except when NVTX_NO_IMPL is defined).
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef NVTX_DISABLE

#include "nvtxExtHelperMacros.h"

#define NVTX_EXT_FN_IMPL(ret_val, fn_name, signature, arg_names) \
ret_val fn_name signature { \
    NVTX_EXT_HELPER_UNUSED_ARGS arg_names \
    return ((ret_val)(intptr_t)-1); \
}

#else  /* NVTX_DISABLE */

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

NVTX_EXT_FN_IMPL(nvtxMemPermissionsHandle_t, nvtxMemCudaGetProcessWidePermissions, (nvtxDomainHandle_t domain), (domain))

NVTX_EXT_FN_IMPL(nvtxMemPermissionsHandle_t, nvtxMemCudaGetDeviceWidePermissions, (nvtxDomainHandle_t domain, int device), (domain, device))

#undef NVTX_EXT_FN_RETURN_INVALID
/* END: Non-void functions. */

/* void functions. */
#define NVTX_EXT_FN_RETURN_INVALID(rtype)
#define return

NVTX_EXT_FN_IMPL(void, nvtxMemCudaSetPeerAccess, (nvtxDomainHandle_t domain, nvtxMemPermissionsHandle_t permissions, int devicePeer, uint32_t flags), (domain, permissions, devicePeer, flags))

#undef return
#undef NVTX_EXT_FN_RETURN_INVALID
/* END: void functions. */

#undef NVTX_EXT_FN_IMPL

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
