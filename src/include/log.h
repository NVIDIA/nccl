/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_LOG_H_
#define NCCL_INT_LOG_H_

#include "nccl_log.h"

// Load the log plugin named by NCCL_LOG_PLUGIN, if any, and install it as the log sink.
// A load failure leaves NCCL's default output in place and is not reported as an error.
//
// There is no matching finalize. A log sink sits on a path every other thread uses, so tearing it down
// at exit while those threads may still be logging is worse than letting the process exit with it
// installed; a plugin sink therefore lives for the life of the process.
ncclResult_t ncclLogPluginInit(void);

#endif
