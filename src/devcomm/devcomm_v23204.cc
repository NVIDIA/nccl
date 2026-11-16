/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime.h"

// From 2.32.4, device code strides GIN barrier signals by ncclGinBarrierSlots() of the selected backend's barrier
// preference (two signals per barrier on signal-efficient backends such as EFA_GDA) instead of by the team size.
struct ncclDevCommCompat ncclDevCommCompat_v23204 = {
  NCCL_VERSION(2, 32, 4), // minVersion
  NCCL_VERSION_CODE,      // maxVersion
  nullptr,           // commPropertiesFilter
  nullptr,           // devCommRequirementsFilter
  nullptr,           // devCommCopyNewToOld
  nullptr,           // devCommCopyOldToNew
  ncclDevCommGinSignalsPerBarrier_v23204, // ginSignalsPerBarrier
};
