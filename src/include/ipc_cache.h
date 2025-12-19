/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_IPC_CACHE_H_
#define NCCL_IPC_CACHE_H_

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>

/*
 * IPC Handle Cache (O3 Optimization)
 *
 * Caches cudaIpcMemHandle_t for device pointers to avoid redundant
 * cudaIpcGetMemHandle calls when the same buffer is shared with multiple peers.
 *
 * The cache is keyed by device pointer address. When a handle is requested:
 * 1. Check cache for existing handle
 * 2. If found, return cached handle (fast path)
 * 3. If not found, call cudaIpcGetMemHandle and cache result
 *
 * Thread-safe: Uses mutex for concurrent access protection.
 */

struct ncclIpcHandleCache
{
private:
    std::mutex mutex_;
    std::unordered_map<void *, cudaIpcMemHandle_t> cache_;

    // Statistics for performance monitoring
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;

public:
    /*
     * Get or create an IPC handle for the given device pointer.
     * Returns cudaSuccess on success, or the error from cudaIpcGetMemHandle on failure.
     */
    cudaError_t getHandle(void *devPtr, cudaIpcMemHandle_t *handle)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(devPtr);
        if (it != cache_.end())
        {
            // Cache hit - return cached handle
            *handle = it->second;
            hits_++;
            return cudaSuccess;
        }

        // Cache miss - generate new handle
        cudaError_t err = cudaIpcGetMemHandle(handle, devPtr);
        if (err == cudaSuccess)
        {
            cache_[devPtr] = *handle;
        }
        misses_++;
        return err;
    }

    /*
     * Remove a cached handle (call when memory is freed).
     */
    void removeHandle(void *devPtr)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.erase(devPtr);
    }

    /*
     * Clear all cached handles.
     */
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        hits_ = 0;
        misses_ = 0;
    }

    /*
     * Get cache statistics.
     */
    void getStats(uint64_t *hits, uint64_t *misses) const
    {
        *hits = hits_;
        *misses = misses_;
    }

    /*
     * Get cache size.
     */
    size_t size() const
    {
        return cache_.size();
    }
};

// Global IPC handle cache instance
extern ncclIpcHandleCache ncclIpcCache;

/*
 * Convenience macro for getting IPC handle with caching.
 * Use this instead of direct cudaIpcGetMemHandle calls.
 */
#define NCCL_IPC_GET_HANDLE_CACHED(handle, devPtr) \
    ncclIpcCache.getHandle(devPtr, handle)

/*
 * Remove handle from cache when memory is freed.
 */
#define NCCL_IPC_REMOVE_HANDLE(devPtr) \
    ncclIpcCache.removeHandle(devPtr)

#endif // NCCL_IPC_CACHE_H_
