// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <csignal>
#include <stdexcept>

#include "comm.h"
#include "info.h"
#include "meta/colltrace/CollTrace.h"

namespace ncclx::colltrace {

class CollTraceError : public std::runtime_error {
 public:
  explicit CollTraceError(const std::string& what) : std::runtime_error(what) {}
};

ncclResult_t collTraceInit(ncclComm* comm);

ncclResult_t collTraceDestroy(ncclComm* comm);

// For baseline transport
std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream);

ncclResult_t collTraceRecordStartEvent(
    cudaStream_t launchStream,
    CollTraceEvent* event);

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event);

CpuWaitEvent* collTraceGetCpuStartWaitEvent(CollTraceEvent& event);

CpuWaitEvent* collTraceGetCpuEndWaitEvent(CollTraceEvent& event);
} // namespace ncclx::colltrace
