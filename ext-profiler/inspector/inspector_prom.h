#ifndef INSPECTOR_INSPECTOR_PROM_H_
#define INSPECTOR_INSPECTOR_PROM_H_

#include <stdio.h>
#include "inspector.h"

// Forward declarations
struct inspectorCommInfoList;
struct inspectorDumpThread;

// Prometheus-related function declarations
inspectorResult_t inspectorPromCacheStaticLabels(struct inspectorCommInfo* commInfo);

inspectorResult_t inspectorPromCommInfoListDump(struct inspectorCommInfoList* commList,
                                                const char* output_root,
                                                struct inspectorDumpThread* dumpThread);

// Prometheus-specific configuration
uint64_t inspectorPromValidateInterval(uint64_t interval);

#endif  // INSPECTOR_INSPECTOR_PROM_H_
