/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_reduce.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_R(ncclAllReduce, ncclCollAllReduce);
