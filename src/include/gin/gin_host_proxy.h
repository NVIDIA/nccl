/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_HOST_PROXY_H_
#define GIN_HOST_PROXY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_gin.h"

ncclResult_t ncclGinProxyInit(ncclGin_t** proxyGin);

#endif
