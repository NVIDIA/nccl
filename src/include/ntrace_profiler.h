/* (c) Facebook, Inc. and its affiliates. Confidential and proprietary. */

#ifndef NCCL_NTRACE_PROFILER_H_
#define NCCL_NTRACE_PROFILER_H_

#ifdef ENABLE_NTRACE
#include "ntrace_rt.h"

#define NTRACE_PROFILING_RECORD(profile_state, ...) \
  do {                                              \
    ntrace_log_##profile_state(__VA_ARGS__);        \
  } while (0)

static inline void ntraceProfilingDump(void) {
  ntrace_dump();
}

#else
#define NTRACE_PROFILING_RECORD(profile_state, ...) \
  do { /* no op */                                  \
  } while (0)

static inline void ntraceProfilingDump(void){/* no op */};
#endif

#endif /* end of NCCL_NTRACE_PROFILER_H_ */
