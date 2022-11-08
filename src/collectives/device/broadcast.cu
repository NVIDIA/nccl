/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NCCL_HOLLOW_THIS_TU NCCL_HOLLOW_BROADCAST

#include "broadcast.h"
#include "common.h"
#include "collectives.h"

IMPL_COLL_C(Broadcast);
