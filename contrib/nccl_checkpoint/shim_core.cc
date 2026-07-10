/*
 * shim_core.cc — checkpoint shim state.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "shim_core.h"

thread_local int ncclDebugNoWarn = 0;

#include <atomic>

namespace nccl_checkpoint {

CommHandleMap g_commHandles;
RegHandleMap g_regHandles;
WindowHandleMap g_windowHandles;
static std::atomic<bool> g_checkpointPrepared{false};

uint64_t nextRegistrationSequence() {
  static std::atomic<uint64_t> next{1};
  return next++;
}

bool isCheckpointPrepared() {
  return g_checkpointPrepared.load(std::memory_order_acquire);
}

void markCheckpointPrepared() {
  g_checkpointPrepared.store(true, std::memory_order_release);
}

void clearCheckpointPrepared() {
  g_checkpointPrepared.store(false, std::memory_order_release);
}

}  // namespace nccl_checkpoint
