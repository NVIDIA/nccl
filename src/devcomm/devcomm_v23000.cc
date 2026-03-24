/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime.h"

struct ncclDevCommCompat ncclDevCommCompat_v23000 = {
  NCCL_VERSION(2, 30, 0), NCCL_VERSION_CODE, // minVersion, maxVersion
  nullptr,                                   // commPropertiesFilter
  nullptr,                                   // devCommRequirementsFilter
  nullptr,                                   // devCommCopyNewToOld
  nullptr,                                   // devCommCopyOldToNew
};
