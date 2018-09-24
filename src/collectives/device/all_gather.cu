/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "all_gather.h"
#include "collectives.h"

#define UNROLL 4

#if NCCL_OP == 0
IMPL_COLL3(ncclAllGather, copy, FuncSum, i8, int8_t, ncclCollAllGather, ncclSum, ncclInt8);
#endif
