/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PRINT_EVENT_H_
#define PRINT_EVENT_H_

#include "nccl/common.h"
extern ncclDebugLogger_t logFn;

void debugEvent(void* eHandle, const char* tag);
void printEvent(FILE* fh, void* handle);

#endif
