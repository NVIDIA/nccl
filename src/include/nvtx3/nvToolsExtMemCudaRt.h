/*
* Copyright 2009-2020  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/
#ifndef NVTOOLSEXTV3_MEM_CUDART_V1
#define NVTOOLSEXTV3_MEM_CUDART_V1

#include "nvToolsExtMem.h"

#include "cuda.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/** \brief The memory is from a CUDA runtime array.
 *
 * Relevant functions: cudaMallocArray,  cudaMalloc3DArray
 * Also cudaArray_t from other types such as cudaMipmappedArray_t
 *
 * NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE is not supported
 *
 * nvtxMemHeapRegister receives a heapDesc of type cudaArray_t because the description can be retrieved by tools through cudaArrayGetInfo()
 * nvtxMemRegionRegisterEx receives a regionDesc of type nvtxMemCudaArrayRangeDesc_t
 */
#define NVTX_MEM_TYPE_CUDA_ARRAY 0x11

/** \brief structure to describe memory in a CUDA array object
 */
typedef struct nvtxMemCudaArrayRangeDesc_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */
    uint32_t reserved0;
    cudaArray_t  src;
    size_t offset[3];
    size_t extent[3];
} nvtxMemCudaArrayRangeDesc_v1;
typedef nvtxMemCudaArrayRangeDesc_v1 nvtxMemCudaArrayRangeDesc_t;


/** \brief The memory is from a CUDA device array.
 *
 * Relevant functions: cuArrayCreate,  cuArray3DCreate
 * Also CUarray from other types such as CUmipmappedArray
 *
 * NVTX_MEM_HEAP_HANDLE_PROCESS_WIDE is not supported
 *
 * nvtxMemHeapRegister receives a heapDesc of type cudaArray_t because the description can be retrieved by tools through cudaArrayGetInfo()
 * nvtxMemRegionRegisterEx receives a regionDesc of type nvtxMemCuArrayRangeDesc_t
 */
#define NVTX_MEM_TYPE_CU_ARRAY 0x12

/** \brief structure to describe memory in a CUDA array object
 */
typedef struct nvtxMemCuArrayRangeDesc_v1
{
    uint16_t extCompatID; /* Set to NVTX_EXT_COMPATID_MEM */
    uint16_t structSize; /* Size of the structure. */
    uint32_t reserved0;
    CUarray  src;
    size_t offset[3];
    size_t extent[3];
} nvtxMemCuArrayRangeDesc_v1;
typedef nvtxMemCuArrayRangeDesc_v1 nvtxMemCuArrayRangeDesc_t;

/* Reserving 0x2-0xF for more common types */

#define NVTX_MEM_CUDA_PEER_ALL_DEVICES -1

/** \brief Get the permission object that represent the CUDA runtime device
 * or cuda driver context
 *
 * This object will allow developers to adjust permissions applied to work executed
 * on the GPU.  It may be inherited or overridden by permissions object bound
 * with NVTX_MEM_PERMISSIONS_BIND_SCOPE_CUDA_STREAM, depending on the binding flags.
 *
 * Ex. change the peer to peer access permissions between devices in entirety
 * or punch through special holes
 *
 * By default, all memory is accessible that naturally would be to a CUDA kernel until
 * modified otherwise by nvtxMemCudaSetPeerAccess or changing regions.
 *
 * This object should also represent the CUDA driver API level context.
*/
NVTX_DECLSPEC nvtxMemPermissionsHandle_t NVTX_API nvtxMemCudaGetProcessWidePermissions(
    nvtxDomainHandle_t domain);

/** \brief Get the permission object that represent the CUDA runtime device
 * or cuda driver context
 *
 * This object will allow developers to adjust permissions applied to work executed
 * on the GPU.  It may be inherited or overridden by permissions object bound
 * with NVTX_MEM_PERMISSIONS_BIND_SCOPE_CUDA_STREAM, depending on the binding flags.
 *
 * Ex. change the peer to peer access permissions between devices in entirety
 * or punch through special holes
 *
 * By default, all memory is accessible that naturally would be to a CUDA kernel until
 * modified otherwise by nvtxMemCudaSetPeerAccess or changing regions.
 *
 * This object should also represent the CUDA driver API level context.
*/
NVTX_DECLSPEC nvtxMemPermissionsHandle_t NVTX_API nvtxMemCudaGetDeviceWidePermissions(
    nvtxDomainHandle_t domain,
    int device);

/** \brief Change the default behavior for all memory mapped in from a particular device.
 *
 * While typically all memory defaults to readable and writable, users may desire to limit
 * access to reduced default permissions such as read-only and a per-device basis.
 *
 * Regions can used to further override smaller windows of memory.
 *
 * devicePeer can be NVTX_MEM_CUDA_PEER_ALL_DEVICES
 *
*/
NVTX_DECLSPEC void NVTX_API nvtxMemCudaSetPeerAccess(
    nvtxDomainHandle_t domain,
    nvtxMemPermissionsHandle_t permissions,
    int devicePeer, /* device number such as from cudaGetDevice() or NVTX_MEM_CUDA_PEER_ALL_DEVICES */
    uint32_t flags); /* NVTX_MEM_PERMISSIONS_REGION_FLAGS_* */

/** @} */ /*END defgroup*/

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

#ifndef NVTX_NO_IMPL
#define NVTX_EXT_IMPL_MEM_CUDART_GUARD /* Ensure other headers cannot be included directly */
#include "nvtxDetail/nvtxExtImplMemCudaRt_v1.h"
#undef NVTX_EXT_IMPL_MEM_CUDART_GUARD
#endif /*NVTX_NO_IMPL*/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NVTOOLSEXTV3_MEM_CUDART_V1 */
