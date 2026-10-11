/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_LOG_H_
#define NCCL_LOG_H_

#include "nccl.h"

// The sink interface itself is declared in nccl.h, because it is usable in two ways: an application that
// links libnccl registers a sink directly with ncclSetDebugLogSink(), and a plugin exports one under
// NCCL_LOG_PLUGIN_SYMBOL. Both use the same struct, so a sink can move between the two without being
// rewritten, and there is only one definition to keep correct.
typedef ncclLogSink_v1_t ncclLog_t;

#define NCCL_LOG_PLUGIN_SYMBOL ncclLogPlugin_v1

// The name and the string NCCL dlsym()s for, kept together so they cannot drift apart.
#define NCCL_LOG_PLUGIN_SYMBOL_STR_(x) #x
#define NCCL_LOG_PLUGIN_SYMBOL_STR(x) NCCL_LOG_PLUGIN_SYMBOL_STR_(x)
#define NCCL_LOG_PLUGIN_SYMBOL_NAME NCCL_LOG_PLUGIN_SYMBOL_STR(NCCL_LOG_PLUGIN_SYMBOL)

#endif // end include guard
