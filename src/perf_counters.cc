/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "perf_counters.h"
#include "debug.h"

// Global performance counters instance
struct ncclPerfCounters ncclPerfStats = {};

void ncclPerfCountersReset(void) {
  ncclPerfStats.initTimeUs.store(0, std::memory_order_relaxed);
  ncclPerfStats.topoDiscoveryUs.store(0, std::memory_order_relaxed);
  ncclPerfStats.transportSetupUs.store(0, std::memory_order_relaxed);
  ncclPerfStats.channelInitUs.store(0, std::memory_order_relaxed);
  ncclPerfStats.proxyInitUs.store(0, std::memory_order_relaxed);

  ncclPerfStats.cudaMallocCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.cudaMallocBytes.store(0, std::memory_order_relaxed);
  ncclPerfStats.hostAllocCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.hostAllocBytes.store(0, std::memory_order_relaxed);

  ncclPerfStats.ipcHandleGetCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.ipcHandleOpenCount.store(0, std::memory_order_relaxed);

  ncclPerfStats.kernelLaunchCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.collectiveCount.store(0, std::memory_order_relaxed);

  ncclPerfStats.proxyWakeups.store(0, std::memory_order_relaxed);
  ncclPerfStats.proxyIdleUs.store(0, std::memory_order_relaxed);

  ncclPerfStats.netSendCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.netRecvCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.netSendBytes.store(0, std::memory_order_relaxed);
  ncclPerfStats.netRecvBytes.store(0, std::memory_order_relaxed);

  ncclPerfStats.retryCount.store(0, std::memory_order_relaxed);
  ncclPerfStats.errorCount.store(0, std::memory_order_relaxed);
}

void ncclPerfCountersPrint(void) {
  INFO(NCCL_INIT, "=== NCCL Performance Counters ===");
  INFO(NCCL_INIT, "Initialization:");
  INFO(NCCL_INIT, "  Total init time:     %llu us", (unsigned long long)NCCL_PERF_COUNTER_GET(initTimeUs));
  INFO(NCCL_INIT, "  Topo discovery:      %llu us", (unsigned long long)NCCL_PERF_COUNTER_GET(topoDiscoveryUs));
  INFO(NCCL_INIT, "  Transport setup:     %llu us", (unsigned long long)NCCL_PERF_COUNTER_GET(transportSetupUs));
  INFO(NCCL_INIT, "  Channel init:        %llu us", (unsigned long long)NCCL_PERF_COUNTER_GET(channelInitUs));
  INFO(NCCL_INIT, "  Proxy init:          %llu us", (unsigned long long)NCCL_PERF_COUNTER_GET(proxyInitUs));

  INFO(NCCL_INIT, "Memory:");
  INFO(NCCL_INIT, "  cudaMalloc calls:    %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(cudaMallocCount));
  INFO(NCCL_INIT, "  cudaMalloc bytes:    %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(cudaMallocBytes));
  INFO(NCCL_INIT, "  Host alloc calls:    %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(hostAllocCount));
  INFO(NCCL_INIT, "  Host alloc bytes:    %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(hostAllocBytes));

  INFO(NCCL_INIT, "IPC:");
  INFO(NCCL_INIT, "  IPC handle get:      %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(ipcHandleGetCount));
  INFO(NCCL_INIT, "  IPC handle open:     %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(ipcHandleOpenCount));

  INFO(NCCL_INIT, "Operations:");
  INFO(NCCL_INIT, "  Kernel launches:     %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(kernelLaunchCount));
  INFO(NCCL_INIT, "  Collective ops:      %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(collectiveCount));

  INFO(NCCL_INIT, "Proxy:");
  INFO(NCCL_INIT, "  Proxy wakeups:       %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(proxyWakeups));
  INFO(NCCL_INIT, "  Proxy idle time:     %llu us", (unsigned long long)NCCL_PERF_COUNTER_GET(proxyIdleUs));

  INFO(NCCL_INIT, "Network:");
  INFO(NCCL_INIT, "  Net send ops:        %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(netSendCount));
  INFO(NCCL_INIT, "  Net recv ops:        %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(netRecvCount));
  INFO(NCCL_INIT, "  Net bytes sent:      %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(netSendBytes));
  INFO(NCCL_INIT, "  Net bytes recv:      %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(netRecvBytes));

  INFO(NCCL_INIT, "Errors:");
  INFO(NCCL_INIT, "  Retry count:         %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(retryCount));
  INFO(NCCL_INIT, "  Error count:         %llu", (unsigned long long)NCCL_PERF_COUNTER_GET(errorCount));
  INFO(NCCL_INIT, "=================================");
}
