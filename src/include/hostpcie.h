/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_HOSTPCIE_H_
#define NCCL_HOSTPCIE_H_

#include "nccl.h"
#include <stdint.h>

// Defined by NCCL_PARAM in src/misc/ibvwrap.cc. Both the built-in IB transport and the
// net plugin path below obey it, so one setting covers a run whichever one supplies the
// network.
//   -1 (default) report the network port rate rather than the bandwidth of the host
//                PCIe link, which NCCL already models on its own
//    0           never call ibv_query_port_speed()
//    1           keep whatever the device reports, as it is
int64_t ncclParamIbQueryPortSpeed();

// Bandwidth, in Mbps, of the PCIe path between a device and the host root complex:
// the device's own link in *ownLink, the slowest link of the whole path in *pathMin.
// sysDir is the device's directory under /sys; label names it in log messages.
ncclResult_t ncclHostPciePath(const char* sysDir, const char* label, uint64_t* ownLink, uint64_t* pathMin);

// True when value is within 1% of reference, which absorbs rounding differences between
// what a device reports and what sysfs states.
bool ncclBwMatches(uint64_t value, uint64_t reference);

// Rate of an RDMA port, in Mbps, read from <pciPath>/infiniband/<dev>/ports/<port>/rate.
ncclResult_t ncclNetSysfsPortRate(const char* pciPath, int port, uint64_t* mbps);

// The speed NCCL should use for a NIC a net plugin has described, in Mbps. Returns the
// plugin's own figure unless that figure is the bandwidth of the host PCIe link, in which
// case the network port rate replaces it. See the comment on the definition.
int ncclNetAdjustSpeed(const char* name, const char* pciPath, int port, int speed);

#endif
