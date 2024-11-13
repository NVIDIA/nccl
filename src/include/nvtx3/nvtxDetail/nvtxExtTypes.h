/*
* Copyright 2021  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

/* This header defines types which are used by the internal implementation
*  of NVTX and callback subscribers.  API clients do not use these types,
*  so they are defined here instead of in nvToolsExt.h to clarify they are
*  not part of the NVTX client API. */

#ifndef NVTXEXTTYPES_H
#define NVTXEXTTYPES_H

#ifndef NVTX_EXT_TYPES_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExt[EXTENSION].h.
#endif

typedef intptr_t (NVTX_API * NvtxExtGetExportFunction_t)(uint32_t exportFunctionId);

typedef struct nvtxExtModuleSegment_t
{
    size_t segmentId;
    size_t slotCount;
    intptr_t* functionSlots;
} nvtxExtModuleSegment_t;

typedef struct nvtxExtModuleInfo_t
{
    uint16_t nvtxVer;
    uint16_t structSize;
    uint16_t moduleId;
    uint16_t compatId;
    size_t segmentsCount;
    nvtxExtModuleSegment_t* segments;
    NvtxExtGetExportFunction_t getExportFunction;
    const void* extInfo;
} nvtxExtModuleInfo_t;

typedef int (NVTX_API * NvtxExtInitializeInjectionFunc_t)(nvtxExtModuleInfo_t* moduleInfo);

#endif /* NVTXEXTTYPES_H */