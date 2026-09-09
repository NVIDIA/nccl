/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "dev_runtime.h"

struct ncclDevCommCompat ncclDevCommCompat_v23100 = {
  NCCL_VERSION(2, 31, 0), // minVersion
  NCCL_VERSION_CODE, // maxVersion
  nullptr,           // commPropertiesFilter
  nullptr,           // devCommRequirementsFilter
  nullptr,           // devCommCopyNewToOld
  nullptr,           // devCommCopyOldToNew
};
