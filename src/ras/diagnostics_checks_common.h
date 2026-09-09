/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_DIAGNOSTICS_CHECKS_COMMON_H_
#define NCCL_RAS_DIAGNOSTICS_CHECKS_COMMON_H_

#include <stddef.h>

#include "diagnostics.h"

typedef ncclResult_t (*rasDiagnosticsFillLocalDataFn)(const struct rasDiagnosticsCommSnapshot* comm, void* checkData);

// Severity tags, space-padded so message text aligns across severities.
#define RAS_DIAG_TAG_OK "[OK]   "
#define RAS_DIAG_TAG_INFO "[INFO] "

// Number of individual ranks shown in a mismatch set before it is truncated with a total count.
#define RAS_DIAG_RANK_SET_MAX 8

void rasDiagnosticsCommSnapshotInit(struct rasDiagnosticsCommSnapshot* snapshot, const struct ncclComm* comm);
int rasDiagnosticsCommIdCompare(const struct rasCommId* id1, const struct rasCommId* id2);
bool rasDiagnosticsCommMatchesContext(const struct rasDiagnosticsContext* ctx, const struct ncclComm* comm);
size_t rasDiagnosticsLocalRecordStride(size_t checkDataSize);
ncclResult_t rasDiagnosticsCollectLocalRecords(const struct rasDiagnosticsContext* ctx, size_t checkDataSize,
                                               rasDiagnosticsFillLocalDataFn fillCheckData,
                                               struct rasDiagnosticsLocalData* data);
const struct rasDiagnosticsRankHeader* rasDiagnosticsRankHeaderFromRecord(const char* record);
int rasDiagnosticsRankHeaderCompare(const void* p1, const void* p2);
void rasDiagnosticsFormatRankSet(char* buf, size_t bufLen, const int* ranks, int nStored, int nTotal);
ncclResult_t __attribute__((format(printf, 3, 4))) rasDiagnosticsReport(const struct rasDiagnosticsReporter* reporter,
                                                                        const char* tag, const char* fmt, ...);
ncclResult_t rasDiagnosticsReportIncomplete(const struct rasDiagnosticsReporter* reporter, const char* checkName,
                                            const struct rasDiagnosticsRankHeader* rank, int gatheredRanks);

#endif // NCCL_RAS_DIAGNOSTICS_CHECKS_COMMON_H_
