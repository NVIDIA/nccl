/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_gather.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_C(ncclAllGather, ncclCollAllGather);
