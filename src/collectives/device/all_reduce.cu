/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "all_reduce.h"
#include "collectives.h"

#define UNROLL 4

#if NCCL_OP == 0
IMPL_COLL2(ncclAllReduce, sum,  FuncSum,  ncclCollAllReduce, ncclSum);
#elif NCCL_OP == 1
IMPL_COLL2(ncclAllReduce, prod, FuncProd, ncclCollAllReduce, ncclProd);
#elif NCCL_OP == 2
IMPL_COLL2(ncclAllReduce, min,  FuncMin,  ncclCollAllReduce, ncclMin);
#elif NCCL_OP == 3
IMPL_COLL2(ncclAllReduce, max,  FuncMax,  ncclCollAllReduce, ncclMax);
#endif
