/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _GIN_HOST_GDAKI_H_
#define _GIN_HOST_GDAKI_H_

#ifndef DOCA_VERBS_USE_CUDA_WRAPPER
#define DOCA_VERBS_USE_CUDA_WRAPPER
#endif

#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "nccl.h"
#include "gin/gin_host.h"

ncclResult_t ncclGinGdakiCreateContext(void *collComm, int nSignals, int nCounters,
                                       void **outGinCtx, ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinGdakiDestroyContext(void *ginCtx);
ncclResult_t ncclGinGdakiRegMrSym(void *collComm, void *data, size_t size, int type, void **mhandle,
                                  void **ginHandle);
ncclResult_t ncclGinGdakiDeregMrSym(void *collComm, void *mhandle);
ncclResult_t ncclGinGdakiProgress(void *ginCtx);
ncclResult_t ncclGinGdakiQueryLastError(void *ginCtx, bool *hasError);

#endif
