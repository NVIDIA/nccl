/*************************************************************************
 * Device-side profiler hook support.
 *
 * The device hook itself is provided by the profiler plugin .so and resolved
 * by NCCL through the profiler v7 getDeviceHook method (ncclProfiler_v7_t, see
 * plugin/profiler/profiler_v7.h and plugin/profiler.cc); this file only holds
 * the funcId -> name resolver that a plugin can call to label events.
 *************************************************************************/
#include "nccl.h"
#include "profiler_dev.h"

// Generated name table for primary func ids (device/host_table.cc); lets a
// consumer resolve a hook event's funcId to a name without reproducing the
// codegen's variant enumeration.
extern int const ncclDevFuncIdCount;
extern const char* const ncclDevFuncName[];

extern "C" __attribute__((visibility("default"))) const char* ncclProfilerDevFuncName(int funcId) {
  if (funcId >= 0 && funcId < ncclDevFuncIdCount) return ncclDevFuncName[funcId];
  return "unknown";
}
