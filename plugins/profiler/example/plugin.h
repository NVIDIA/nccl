/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PLUGIN_H_
#define PLUGIN_H_

__attribute__((visibility("default"))) int exampleProfilerStart(int eActivationMask, const char* name);
__attribute__((visibility("default"))) int exampleProfilerStop(void);


#endif
