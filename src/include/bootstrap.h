/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"

ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, int rank, int nranks, void** commState);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapClose(void* commState);
#endif
