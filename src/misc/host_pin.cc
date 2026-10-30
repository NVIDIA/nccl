/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Device-mapped host memory that can be unpinned while every communicator of the
// process is suspended.
//
// NCCL allocates small host buffers that kernels and the proxy read through UVA
// (abort flag, work FIFO, profiler counters, proxy host buffers). A CUDA process
// checkpoint cannot restore host pages pinned by cudaHostAlloc or cudaHostRegister on
// some drivers (CUDA_ERROR_NOT_SUPPORTED from cuCheckpointProcessRestore), so these
// buffers are pageable mappings registered with the device. While every memory
// manager in the process is suspended the registrations are dropped; the pageable
// pages and their contents stay. Resume registers the same ranges again, so device
// addresses captured in kernels and CUDA graphs stay valid.

#include "host_pin.h"
#include "alloc.h"
#include "debug.h"
#include "param.h"

#include <mutex>
#include <unordered_map>

#if defined(NCCL_OS_LINUX)
#include <sys/mman.h>
#endif

namespace {

struct HostPinRange {
  size_t size;
  bool owned;  // allocated by ncclHostPinAlloc (unmapped on free) or registered for its owner
};

struct HostPinRegistry {
  std::mutex lock;
  std::unordered_map<void*, HostPinRange> ranges;
  int liveManagers = 0;
  int suspendedManagers = 0;
  bool registered = true;  // state of every range in `ranges`
};

HostPinRegistry& registry() {
  static HostPinRegistry* r = new HostPinRegistry();  // never destroyed: used at exit
  return *r;
}

// -1 unknown, 0 use cudaHostAlloc, 1 use registered pageable mappings. Guarded by the registry lock.
int hostPinMode = -1;

ncclResult_t registerRange(void* ptr, size_t size) {
  CUDACHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped | cudaHostRegisterPortable));
  return ncclSuccess;
}

// The registered range must be addressed by its host pointer on the device, as
// cudaHostAllocMapped memory is under UVA; otherwise keep cudaHostAlloc.
ncclResult_t resolveMode() {
  if (hostPinMode >= 0) return ncclSuccess;
#if defined(NCCL_OS_LINUX)
  int dev, sameAddress = 0;
  CUDACHECK(cudaGetDevice(&dev));
  CUDACHECK(cudaDeviceGetAttribute(&sameAddress, cudaDevAttrCanUseHostPointerForRegisteredMem, dev));
  hostPinMode = sameAddress ? 1 : 0;
#else
  hostPinMode = 0;
#endif
  if (hostPinMode == 0) {
    INFO(NCCL_ALLOC, "Host pin: registered host memory needs a separate device address; using cudaHostAlloc, "
                     "which stays pinned across suspend");
  }
  return ncclSuccess;
}

// Invariant: ranges are unregistered exactly when at least one memory manager exists
// and all of them are suspended.
ncclResult_t applyLocked(HostPinRegistry& r) {
  bool want = !(r.liveManagers > 0 && r.suspendedManagers == r.liveManagers);
  if (want == r.registered) return ncclSuccess;
  size_t bytes = 0;
  for (auto& [ptr, range] : r.ranges) {
    if (want) {
      NCCLCHECK(registerRange(ptr, range.size));
    } else {
      CUDACHECK(cudaHostUnregister(ptr));
    }
    bytes += range.size;
  }
  r.registered = want;
  INFO(NCCL_ALLOC, "Host pin: %s %zu ranges (%zu bytes)", want ? "registered" : "unregistered", r.ranges.size(),
       bytes);
  return ncclSuccess;
}

ncclResult_t trackLocked(HostPinRegistry& r, void* ptr, size_t size, bool owned) {
  if (r.registered) NCCLCHECK(registerRange(ptr, size));
  r.ranges.emplace(ptr, HostPinRange{size, owned});
  return ncclSuccess;
}

ncclResult_t untrackLocked(HostPinRegistry& r, void* ptr, bool owned) {
  auto it = r.ranges.find(ptr);
  if (it == r.ranges.end() || it->second.owned != owned) {
    WARN("Host pin: %s of unknown pointer %p", owned ? "free" : "unregister", ptr);
    return ncclInternalError;
  }
  if (r.registered) CUDACHECK(cudaHostUnregister(ptr));
#if defined(NCCL_OS_LINUX)
  if (owned) munmap(ptr, it->second.size);
#endif
  r.ranges.erase(it);
  return ncclSuccess;
}

}  // namespace

ncclResult_t ncclHostPinAlloc(void** ptr, size_t size) {
  *ptr = nullptr;
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  NCCLCHECK(resolveMode());
  if (hostPinMode == 0) {
    CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
    memset(*ptr, 0, size);
    return ncclSuccess;
  }
#if defined(NCCL_OS_LINUX)
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  size_t mapped = (size + page - 1) / page * page;
  // Anonymous mappings are zero-filled and page-aligned, and are never shared
  // with malloc, so unregistering one cannot affect unrelated allocations.
  void* p = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    WARN("Host pin: mmap of %zu bytes failed", mapped);
    return ncclSystemError;
  }
  ncclResult_t ret = trackLocked(r, p, mapped, /*owned=*/true);
  if (ret != ncclSuccess) {
    munmap(p, mapped);
    return ret;
  }
  *ptr = p;
  return ncclSuccess;
#else
  return ncclInternalError;
#endif
}

ncclResult_t ncclHostPinRegister(void* ptr, size_t size) {
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  NCCLCHECK(resolveMode());
  if (hostPinMode == 0) return registerRange(ptr, size);
  return trackLocked(r, ptr, size, /*owned=*/false);
}

ncclResult_t ncclHostPinFree(void* ptr) {
  if (ptr == nullptr) return ncclSuccess;
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  if (hostPinMode == 1) return untrackLocked(r, ptr, /*owned=*/true);
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

ncclResult_t ncclHostPinUnregister(void* ptr) {
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  if (hostPinMode == 1) return untrackLocked(r, ptr, /*owned=*/false);
  CUDACHECK(cudaHostUnregister(ptr));
  return ncclSuccess;
}

ncclResult_t ncclHostPinManagerCreated() {
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  r.liveManagers++;
  return applyLocked(r);
}

ncclResult_t ncclHostPinManagerDestroyed(bool suspended) {
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  r.liveManagers--;
  if (suspended) r.suspendedManagers--;
  return applyLocked(r);
}

ncclResult_t ncclHostPinManagerSuspended() {
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  r.suspendedManagers++;
  return applyLocked(r);
}

ncclResult_t ncclHostPinManagerResumed() {
  HostPinRegistry& r = registry();
  std::lock_guard<std::mutex> guard(r.lock);
  r.suspendedManagers--;
  return applyLocked(r);
}
