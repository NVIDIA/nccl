/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PERF_COUNTERS_H_
#define NCCL_PERF_COUNTERS_H_

#include <stdint.h>
#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NCCL Performance Counters
 *
 * This structure tracks various performance metrics during NCCL operations.
 * It is primarily used for debugging and optimization purposes.
 *
 * Counters are thread-safe and use atomic operations for updates.
 */
struct ncclPerfCounters {
  // Initialization timings (in microseconds)
  std::atomic<uint64_t> initTimeUs;           // Total ncclCommInit time
  std::atomic<uint64_t> topoDiscoveryUs;      // Topology discovery time
  std::atomic<uint64_t> transportSetupUs;     // Transport initialization time
  std::atomic<uint64_t> channelInitUs;        // Channel allocation time
  std::atomic<uint64_t> proxyInitUs;          // Proxy thread initialization time

  // Memory allocation counts
  std::atomic<uint64_t> cudaMallocCount;      // Number of cudaMalloc calls
  std::atomic<uint64_t> cudaMallocBytes;      // Total bytes allocated via cudaMalloc
  std::atomic<uint64_t> hostAllocCount;       // Number of host memory allocations
  std::atomic<uint64_t> hostAllocBytes;       // Total bytes allocated on host

  // IPC handle operations
  std::atomic<uint64_t> ipcHandleGetCount;    // cudaIpcGetMemHandle calls
  std::atomic<uint64_t> ipcHandleOpenCount;   // cudaIpcOpenMemHandle calls

  // Kernel operations
  std::atomic<uint64_t> kernelLaunchCount;    // Total kernel launches
  std::atomic<uint64_t> collectiveCount;      // Total collective operations

  // Proxy operations
  std::atomic<uint64_t> proxyWakeups;         // Proxy thread wakeup count
  std::atomic<uint64_t> proxyIdleUs;          // Time spent idle in proxy threads

  // Network operations
  std::atomic<uint64_t> netSendCount;         // Network send operations
  std::atomic<uint64_t> netRecvCount;         // Network receive operations
  std::atomic<uint64_t> netSendBytes;         // Total bytes sent over network
  std::atomic<uint64_t> netRecvBytes;         // Total bytes received over network

  // Error counts
  std::atomic<uint64_t> retryCount;           // Operation retry count
  std::atomic<uint64_t> errorCount;           // Total error count
};

// Global performance counters instance
extern struct ncclPerfCounters ncclPerfStats;

// Helper macros for performance counter updates
#define NCCL_PERF_COUNTER_ADD(counter, value) \
  do { ncclPerfStats.counter.fetch_add(value, std::memory_order_relaxed); } while(0)

#define NCCL_PERF_COUNTER_INC(counter) \
  NCCL_PERF_COUNTER_ADD(counter, 1)

#define NCCL_PERF_COUNTER_SET(counter, value) \
  do { ncclPerfStats.counter.store(value, std::memory_order_relaxed); } while(0)

#define NCCL_PERF_COUNTER_GET(counter) \
  ncclPerfStats.counter.load(std::memory_order_relaxed)

// Timer helper for measuring durations
#ifdef __cplusplus
#include <chrono>

class NcclPerfTimer {
public:
  NcclPerfTimer() : start_(std::chrono::high_resolution_clock::now()) {}

  uint64_t elapsedUs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
  }

  void recordTo(std::atomic<uint64_t>& counter) {
    counter.fetch_add(elapsedUs(), std::memory_order_relaxed);
  }

private:
  std::chrono::high_resolution_clock::time_point start_;
};

// Scoped timer that automatically records elapsed time on destruction
class NcclScopedPerfTimer {
public:
  NcclScopedPerfTimer(std::atomic<uint64_t>& counter) : counter_(counter), timer_() {}
  ~NcclScopedPerfTimer() { timer_.recordTo(counter_); }

private:
  std::atomic<uint64_t>& counter_;
  NcclPerfTimer timer_;
};

#define NCCL_PERF_TIMER_SCOPE(counter) \
  NcclScopedPerfTimer _perfTimer##__LINE__(ncclPerfStats.counter)

#endif // __cplusplus

// Reset all performance counters
void ncclPerfCountersReset(void);

// Print performance counters summary to INFO log
void ncclPerfCountersPrint(void);

#ifdef __cplusplus
}
#endif

#endif // NCCL_PERF_COUNTERS_H_
