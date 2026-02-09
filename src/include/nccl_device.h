/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_HOSTLIB_ONLY
#include "nccl_device/coop.h"
#include "nccl_device/impl/barrier__funcs.h"
#include "nccl_device/impl/comm__funcs.h"
#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/ll_a2a__funcs.h"
#include "nccl_device/impl/lsa_barrier__funcs.h"
#include "nccl_device/impl/gin__funcs.h"
#include "nccl_device/impl/gin_barrier__funcs.h"
#include "nccl_device/impl/ptr__funcs.h"
#else
// Include the types and declaration if NCCL_HOSTLIB_ONLY is defined
#include "nccl_device/coop.h"
#include "nccl_device/core.h"
#include "nccl_device/ll_a2a.h"
#include "nccl_device/barrier.h"
#include "nccl_device/ptr.h"
#include "nccl_device/impl/comm__types.h"
#include "nccl_device/impl/core__types.h"
#include "nccl_device/impl/ll_a2a__types.h"
#include "nccl_device/impl/barrier__types.h"
#include "nccl_device/impl/ptr__types.h"
#endif
