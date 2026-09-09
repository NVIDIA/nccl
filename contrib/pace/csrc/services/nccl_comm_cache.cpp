#include "services/nccl_comm_cache.hpp"
#include "util/error.hpp"

#include <cstdio>

namespace pace {

NcclCommRegistry& NcclCommRegistry::instance() {
    static NcclCommRegistry inst;
    return inst;
}

bool NcclCommRegistry::has(const std::vector<int>& key) {
    std::lock_guard<std::mutex> g(mu_);
    return map_.find(key) != map_.end();
}

bool NcclCommRegistry::try_acquire(const std::vector<int>& key, ncclComm_t* out_comm) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    it->second.refcount++;
    *out_comm = it->second.comm;
    return true;
}

void NcclCommRegistry::install(const std::vector<int>& key, ncclComm_t comm) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        // Lost a race with another thread that already installed. Bump refcount
        // for the caller and ask them to discard `comm` (caller responsibility).
        it->second.refcount++;
        return;
    }
    map_.emplace(key, Entry{comm, 1});
}

void NcclCommRegistry::release(const std::vector<int>& key) {
    ncclComm_t to_destroy = nullptr;
    {
        std::lock_guard<std::mutex> g(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return;
        if (--it->second.refcount > 0) return;
        to_destroy = it->second.comm;
        map_.erase(it);
    }
    NCCLCHECK(ncclCommFinalize(to_destroy));
    NCCLCHECK(ncclCommDestroy(to_destroy));
}

size_t NcclCommRegistry::size() {
    std::lock_guard<std::mutex> g(mu_);
    return map_.size();
}

}  // namespace pace
