/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_HOST_PROXY_H_
#define GIN_HOST_PROXY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_net.h"

ncclResult_t ncclGinProxyCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                       int nSignals, int nCounters, void **outGinCtx,
                                       ncclNetDeviceHandle_v11_t **outDevHandle);
ncclResult_t ncclGinProxyRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                  int type, int mr_flags, void **mhandle, void **ginHandle);
ncclResult_t ncclGinProxyDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle);
ncclResult_t ncclGinProxyDestroyContext(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinProxyProgress(ncclGin_t *ginComm, void *ginCtx);
ncclResult_t ncclGinProxyQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError);

#endif
