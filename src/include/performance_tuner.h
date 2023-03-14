#ifndef PERFORMANCE_TUNER_H_
#define PERFORMANCE_TUNER_H_

#include "nccl_performance_tuner.h"

// Performance tuner utility to be called by NCCL internal core.

// Attempts to load NCCL performance tuner from environmental variable.
// Returns ncclSuccess if the correct tuner symbol has been found and
// successully loaded.  Otherwise returns an error and also logs the error.
ncclResult_t ncclLoadPerformanceTuner(ncclPerformanceTuner_t** tuner);

// Cleans up NCCL performance tuner plugin.
ncclResult_t ncclClosePerformanceTuner(ncclPerformanceTuner_t** tuner);
#endif
