/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_DIAGNOSTICS_CHECKS_H_
#define NCCL_RAS_DIAGNOSTICS_CHECKS_H_

#include "diagnostics.h"

ncclResult_t rasDiagnosticsGpuModelCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsGpuModelSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsCudaDriverVersionCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                         struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsCudaDriverVersionSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsEccCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsEccSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsNvLinkCollectLocal(const struct rasDiagnosticsContext* ctx,
                                              struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsNvLinkSummarize(const struct rasDiagnosticsContext* ctx,
                                           const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsNcclEnvCollectLocal(const struct rasDiagnosticsContext* ctx,
                                               struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsNcclEnvSummarize(const struct rasDiagnosticsContext* ctx,
                                            const struct rasDiagnosticsReporter* reporter, const char* data, int nData);

#endif // NCCL_RAS_DIAGNOSTICS_CHECKS_H_
