// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef NCCL_CVARS_H_INCLUDED
#define NCCL_CVARS_H_INCLUDED

#include <string>
#include <unordered_map>
#include <vector>

extern std::vector<std::string> NCCL_COLLTRACE;
extern int64_t NCCL_COLLTRACE_CHECK_INTERVAL_MS;
extern bool NCCL_COLLTRACE_EVENT_BLOCKING_SYNC;
extern int NCCL_COLLTRACE_RECORD_MAX;

extern int NCCL_COLLTRACE_RECORD_MAX_ITERATIONS;

extern int64_t NCCL_COLLTRACE_REPORT_FIRST_N_COLL;
extern std::vector<std::string> NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG;
extern std::vector<std::string> NCCL_FILTER_ALGO_LOGGING_BY_RANKS;
extern std::vector<std::string> NCCL_FILTER_MEM_LOGGING_BY_RANKS;
extern std::vector<std::string> NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS;

extern int NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES;

extern std::vector<std::string> NCCL_PROXYTRACE;
extern int NCCL_PROXYTRACE_RECORD_MAX;
extern bool NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED;

namespace ncclx {
extern std::unordered_map<std::string, std::string> nccl_config;
void updateNcclConfig(std::string fname);
}; // namespace ncclx
void ncclCvarInit();

#endif /* NCCL_CVARS_H_INCLUDED */
