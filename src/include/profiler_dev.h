/*************************************************************************
 * Device-side profiler hook ABI.
 *
 * Lets a profiler plugin provide a __device__ callback that ncclKernelMain's
 * profiler() invokes at each kernel work-item boundary. The plugin writes its
 * own telemetry (its own format, buffer, atomicity, depth) and drains it itself
 * -- an alternative to the fixed workStarted/workCompleted ring + proxy delivery.
 * The hook is additive and needs no proxy op or host callback: a plugin that
 * uses it without enabling ncclProfileKernelCh gets kernel timing with nothing
 * added to captured graphs. See plugins/profiler/example for a reference consumer.
 *************************************************************************/
#ifndef NCCL_PROFILER_DEV_H_
#define NCCL_PROFILER_DEV_H_

#include <stdint.h>

#define NCCL_PROFILER_DEV_START 0
#define NCCL_PROFILER_DEV_STOP 1

// One kernel work-item boundary event. POD + fixed layout for ABI stability.
typedef struct {
  uint64_t timestamp;   // globaltimer() ns (device clock)
  uint64_t workCounter; // monotonic per-channel work counter
  uint32_t funcId;      // collective identity (index into ncclDevFuncTable)
  uint8_t channelId;
  uint8_t phase;       // NCCL_PROFILER_DEV_START / _STOP
  uint16_t rsvd;
} ncclProfilerDevEvent_t;

// Plugin device hook. Called on lane 0, once per profiler-enabled work item,
// when registered. Must be cheap and must not touch the collective's data/sync.
typedef void (*ncclProfilerDevHook_t)(const ncclProfilerDevEvent_t* ev, void* devCtx);

#ifdef __cplusplus
extern "C" {
#endif
// Resolve an event's funcId to a human-readable name (e.g.
// "AllReduce_Sum_f32_RING_LL"); returns a static NCCL-owned string, or
// "unknown" if out of range. Backed by the generated device-func name table.
const char* ncclProfilerDevFuncName(int funcId);
#ifdef __cplusplus
}
#endif

#endif // NCCL_PROFILER_DEV_H_
