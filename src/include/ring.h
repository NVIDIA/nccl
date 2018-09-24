/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RING_H_
#define NCCL_RING_H_
#include "core.h"

ncclResult_t initRing(struct ncclComm* comm, int ringid);
ncclResult_t freeRing(struct ncclRing* ring);

#endif
