#ifndef __NCCL_COMM_CACHE_HPP__
#define __NCCL_COMM_CACHE_HPP__

#include <map>
#include <mutex>
#include <vector>
#include <nccl.h>

namespace pace {

// Process-wide cache for ncclComm_t, keyed by the sorted set of global ranks
// participating in the comm. Multiple Comm objects (RSComm/SGComm/AGComm)
// built on the same topology can share a single ncclComm_t — only window
// registration / dev_comm creation / buffer allocation are per-Comm.
class NcclCommRegistry {
public:
    static NcclCommRegistry& instance();

    // Returns true if a comm for this key already exists.
    bool has(const std::vector<int>& key);

    // If a cached comm exists for `key`, return it and bump refcount.
    // Otherwise return nullptr (caller must build the comm and call install()).
    bool try_acquire(const std::vector<int>& key, ncclComm_t* out_comm);

    // Insert a freshly-built comm into the cache with refcount 1.
    void install(const std::vector<int>& key, ncclComm_t comm);

    // Decrement refcount; if it hits zero, finalize+destroy the comm and erase
    // the entry. Safe to call with a key that was never installed (no-op).
    void release(const std::vector<int>& key);

    // For diagnostics / tests.
    size_t size();

private:
    NcclCommRegistry() = default;
    NcclCommRegistry(const NcclCommRegistry&) = delete;
    NcclCommRegistry& operator=(const NcclCommRegistry&) = delete;

    struct Entry {
        ncclComm_t comm;
        int refcount;
    };

    std::mutex mu_;
    std::map<std::vector<int>, Entry> map_;
};

}  // namespace pace

#endif
