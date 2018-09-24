/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "reduce_scatter.h"
#include "collectives.h"

#define UNROLL 4

#if NCCL_OP == 0
IMPL_COLL2(ncclReduceScatter, sum,  FuncSum,  ncclCollReduceScatter, ncclSum);
#elif NCCL_OP == 1
IMPL_COLL2(ncclReduceScatter, prod, FuncProd, ncclCollReduceScatter, ncclProd);
#elif NCCL_OP == 2
IMPL_COLL2(ncclReduceScatter, min,  FuncMin,  ncclCollReduceScatter, ncclMin);
#elif NCCL_OP == 3
IMPL_COLL2(ncclReduceScatter, max,  FuncMax,  ncclCollReduceScatter, ncclMax);
#endif
