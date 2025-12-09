// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CollTraceUtils.h"

#include "meta/logger/ScubaLogger.h"
#include <iostream>

namespace ncclx::colltrace {

static std::unordered_map<ncclPattern_t, std::string> ncclPatternStr = {
    {ncclPatternRing, "Ring"},
    {ncclPatternRingTwice, "RingTwice"},
    {ncclPatternPipelineFrom, "PipelineFrom"},
    {ncclPatternPipelineTo, "PipelineTo"},
    {ncclPatternTreeUp, "TreeUp"},
    {ncclPatternTreeDown, "TreeDown"},
    {ncclPatternTreeUpDown, "TreeUpDown"},
    {ncclPatternCollnetChain, "CollnetChain"},
    {ncclPatternCollnetDirect, "CollnetDirect"},
    {ncclPatternNvls, "Nvls"},
    {ncclPatternNvlsTree, "NvlsTree"},
    {ncclPatternSend, "Send"},
    {ncclPatternRecv, "Recv"}};

std::string getNcclPatternStr(ncclPattern_t pattern) {
  return ncclPatternStr.count(pattern) ? ncclPatternStr[pattern] : "N/A";
}

void reportCollToScuba(
    const std::string reportReason,
    const CollTraceColl& coll,
    const CommLogData& logMetaData) {
  // There is a lot of logger dependency on the legacy ScubaEntry, it takes a
  // lot of effor to remove all the dpendencied. So we keep the scuba entry API
  // and do the conversion here for now :(
  auto collLegacySample = coll.toScubaEntry();

  NcclScubaSample scubaSample(reportReason);
  for (const auto& [key, value] : collLegacySample.getIntMap()) {
    scubaSample.addInt(key, value);
  }
  for (const auto& [key, value] : collLegacySample.getNormalMap()) {
    scubaSample.addNormal(key, value);
  }
  for (const auto& [key, value] : collLegacySample.getDoubleMap()) {
    scubaSample.addDouble(key, value);
  }
  scubaSample.setCommunicatorMetadata(&logMetaData);
  SCUBA_nccl_coll_trace.addSample(std::move(scubaSample));
}

} // namespace ncclx::colltrace
