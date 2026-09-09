/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_INSPECTOR_OTEL_H_
#define INSPECTOR_INSPECTOR_OTEL_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "inspector.h"

struct inspectorOtelAttribute {
  std::string key;
  std::string value;
};

struct inspectorOtelMetricPoint {
  std::string name;
  std::string description;
  std::string unit;
  std::vector<inspectorOtelAttribute> attributes;
  double value;
  uint64_t timestampUsec;
};

void inspectorOtelInitFromEnv();
bool inspectorOtelIsEnabled();
bool inspectorOtelVerboseEnabled();
const char* inspectorOtelEndpoint();
inspectorResult_t inspectorOtelCommInfoListDump(struct inspectorCommInfoList* commList);

inspectorResult_t inspectorOtelExportMetrics(
    const std::vector<inspectorOtelMetricPoint>& points,
    uint64_t timestampUsec);

#endif  // INSPECTOR_INSPECTOR_OTEL_H_
