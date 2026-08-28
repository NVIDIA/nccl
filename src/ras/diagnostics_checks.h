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
ncclResult_t rasDiagnosticsNvidiaDriverVersionCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                           struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsNvidiaDriverVersionSummarize(
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
ncclResult_t rasDiagnosticsRdmaTopoCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsRdmaTopoSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsIommuCollectLocal(const struct rasDiagnosticsContext* ctx,
                                             struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsIommuSummarize(const struct rasDiagnosticsContext* ctx,
                                          const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsAtsCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsAtsSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
void rasDiagnosticsInit();
ncclResult_t rasDiagnosticsXidCollectLocal(const struct rasDiagnosticsContext* ctx,
                                           struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsXidSummarize(const struct rasDiagnosticsContext* ctx,
                                        const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsPathsCollectLocal(const struct rasDiagnosticsContext* ctx,
                                             struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsPathsSummarize(const struct rasDiagnosticsContext* ctx,
                                          const struct rasDiagnosticsReporter* reporter, const char* data, int nData);

ncclResult_t rasDiagnosticsNetDeviceCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                 struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsNetDeviceSummarize(
  const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter* reporter, const char* data, int nData);

// The NET-device check's wire record and the helpers producing it are declared here, rather than kept
// private to diagnostics_net.cc, so that the unit test can build payloads and drive them directly.
#define RAS_DIAG_NET_DEVICE_NAME_LEN 64

// Link layer of a NET device's port. The plugin interface does not report it, so it is read from
// sysfs and stored as a small value of our own rather than the provider's encoding.
#define RAS_DIAG_NET_LINK_UNKNOWN 0
#define RAS_DIAG_NET_LINK_INFINIBAND 1
#define RAS_DIAG_NET_LINK_ETHERNET 2

// Wire record of the NET-device diagnostic, finalized at snapshot capture and gathered as-is. The
// port's own attributes are read from sysfs and are 0 when unavailable, which still compares equal
// across ranks that all failed to read them.
struct rasDiagnosticsNetDeviceData {
  uint64_t hostHash; // reporting rank's node identity, for counting nodes per device group
  int speed;         // the port's rate now, falling back to what the plugin reported at open
  int port;
  int8_t netDeviceType;
  int8_t linkLayer; // RAS_DIAG_NET_LINK_*
  int8_t portState; // IB port state, e.g. 4 for ACTIVE
  int8_t physState; // IB physical port state, e.g. 5 for LinkUp
  // The plugin exposed this port as its "<device>_dma" data-direct variant. The record is grouped
  // under the underlying device name, so this is what keeps a node using a different DMA path
  // from comparing equal to its peers.
  int8_t dataDirect;
  // Marker records only: the rank used a recognized IB verbs plugin, so it was expected to report a
  // device and its NICs went uninspected, as opposed to a plugin this check does not inspect at all.
  int8_t ibVerbsWithoutDevice;
  char name[RAS_DIAG_NET_DEVICE_NAME_LEN];
};

int rasDiagnosticsNetDevicePrepare(const struct ncclTopoNetPropertiesSnapshot* netProps, int nNetProps,
                                   uint64_t hostHash, char* out, size_t outStride, int capacity);
// The root exists so that the test can point the reader at a tree it built; callers use the default.
void rasDiagnosticsNetReadPortAttrs(struct rasDiagnosticsNetDeviceData* netData,
                                    const char* sysfsRoot = "/sys/class/infiniband");

#endif // NCCL_RAS_DIAGNOSTICS_CHECKS_H_
