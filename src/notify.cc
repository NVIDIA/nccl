/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Poolside Inc & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "checks.h"
#include "comm.h"
#include "profiler.h"

NCCL_API(ncclResult_t, ncclNotifyTag, const char* tag, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclNotifyTag(const char* tag, ncclComm_t comm, cudaStream_t stream) {
  if (comm == NULL) return ncclInvalidArgument;
  if (tag == NULL) return ncclInvalidArgument;
  NCCLCHECK(ncclProfilerUserTagEvent(comm, tag));
  return ncclSuccess;
}
