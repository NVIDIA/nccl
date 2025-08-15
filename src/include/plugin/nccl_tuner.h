/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include "nccl.h"
#include "nccl_common.h"

#include "tuner/tuner_v5.h"
#include "tuner/tuner_v4.h"
#include "tuner/tuner_v3.h"
#include "tuner/tuner_v2.h"

typedef ncclTuner_v5_t ncclTuner_t;
typedef ncclTunerConstants_v5_t ncclTunerConstants_t;
typedef ncclNvlDomainInfo_v5_t ncclNvlDomainInfo_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v5"

#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5
#define NCCL_ALGO_PAT 6
#define NCCL_NUM_ALGORITHMS NCCL_NUM_ALGORITHMS_V5 // Tree/Ring/CollNet*/PAT

#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
#define NCCL_NUM_PROTOCOLS NCCL_NUM_PROTOCOLS_V5 // Simple/LL/LL128

#define NCCL_ALGO_PROTO_IGNORE -1.0

#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
#define NCCL_NUM_HW_LINKS NCCL_NUM_HW_LINKS_V5

#define NCCL_VOLTA_COMPCAP_IDX 0
#define NCCL_AMPERE_COMPCAP_IDX 1
#define NCCL_HOPPER_COMPCAP_IDX 2
#define NCCL_BLACKWELL_COMPCAP_IDX 3
#define NCCL_NUM_COMPCAPS NCCL_NUM_COMPCAPS_V5

#define NCCL_TUNING_SCALE_1NODE 0
#define NCCL_TUNING_SCALE_2NODES 1
#define NCCL_TUNING_SCALE_4NODES 2
#define NCCL_NUM_TUNING_SCALES NCCL_NUM_TUNING_SCALES_V5

#endif
