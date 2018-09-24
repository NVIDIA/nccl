/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "reduce.h"
#include "collectives.h"

#define UNROLL 4

#if NCCL_OP == 0
IMPL_COLL2(ncclReduce, sum,  FuncSum,  ncclCollReduce, ncclSum);
#elif NCCL_OP == 1
IMPL_COLL2(ncclReduce, prod, FuncProd, ncclCollReduce, ncclProd);
#elif NCCL_OP == 2
IMPL_COLL2(ncclReduce, min,  FuncMin,  ncclCollReduce, ncclMin);
#elif NCCL_OP == 3
IMPL_COLL2(ncclReduce, max,  FuncMax,  ncclCollReduce, ncclMax);
#endif
