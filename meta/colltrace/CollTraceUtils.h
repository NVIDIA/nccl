// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "CollTraceColl.h"

namespace ncclx::colltrace {

std::string getNcclPatternStr(ncclPattern_t pattern);

void reportCollToScuba(
    const std::string reportReason,
    const CollTraceColl& coll,
    const CommLogData& logMetaData);

} // namespace ncclx::colltrace
