/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_MNNVL_H_
#define NCCL_MNNVL_H_

#include "nccl.h"
#include "comm.h"

ncclResult_t ncclMnnvlCheck(struct ncclComm* comm);

#endif
