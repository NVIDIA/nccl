/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DIAGNOSTICS_LOG_H_
#define NCCL_DIAGNOSTICS_LOG_H_

#include <stdio.h>
#include <unistd.h>

static char diagLogHost[64] = "";
static int diagLogPid = 0;
static void diagLogInit() {
  if (diagLogPid != 0) return;
  if (gethostname(diagLogHost, sizeof(diagLogHost) - 1) != 0) {
    snprintf(diagLogHost, sizeof(diagLogHost), "?");
  }
  diagLogHost[sizeof(diagLogHost) - 1] = '\0';
  diagLogPid = (int)getpid();
}

// clang-format off
#define DIAG_PRINT(fmt, ...) \
  do { \
    diagLogInit(); \
    fprintf(stdout, "%s:%d " fmt "\n", diagLogHost, diagLogPid, ##__VA_ARGS__); \
  } while (0)
// clang-format on

#endif // NCCL_DIAGNOSTICS_LOG_H_
