#pragma once
/*
 * Included by shim.cc, shim_checkpoint.cc, and shim_auto.cc.
 */

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Internal NCCL headers used:
 *   comm.h  — ncclComm::commHash (uint64_t) */
#include "comm.h"
#undef WARN
#undef INFO
#undef TRACE

#define NCCL_CHECKPOINT_MAJOR 0
#define NCCL_CHECKPOINT_MINOR 1
#define NCCL_CHECKPOINT_PATCH 0
#define NCCL_CHECKPOINT_VERSION_CODE NCCL_VERSION(NCCL_CHECKPOINT_MAJOR, NCCL_CHECKPOINT_MINOR, NCCL_CHECKPOINT_PATCH)

namespace nccl_checkpoint {

enum NcclCheckpointLogLevel {
  ncclCheckpointLogInfo = 1,
  ncclCheckpointLogTrace = 2,
};

static inline bool ncclCheckpointShouldLog(NcclCheckpointLogLevel level) {
  static int enabled = -1;
  if (__builtin_expect(enabled != -1, 1)) return enabled >= static_cast<int>(level);

  enabled = 0;
  const char* debug = getenv("NCCL_DEBUG");
  if (debug != nullptr && strcmp(debug, "INFO") == 0) {
    enabled = ncclCheckpointLogInfo;
  } else if (debug != nullptr && strcmp(debug, "TRACE") == 0) {
    const char* subsys = getenv("NCCL_DEBUG_SUBSYS");
    if (subsys != nullptr && (strstr(subsys, "ALL") != nullptr || strstr(subsys, "CHECKPOINT") != nullptr)) {
      enabled = ncclCheckpointLogTrace;
    }
  }

  return enabled >= static_cast<int>(level);
}

static inline uint64_t ncclCheckpointGetTid() {
  return (uint64_t)syscall(SYS_gettid);
}

static inline const char* ncclCheckpointHostName() {
  static char hostname[1024] = "";
  if (hostname[0] == '\0') {
    if (gethostname(hostname, sizeof(hostname)) != 0) {
      strncpy(hostname, "unknown", sizeof(hostname));
    }
    hostname[sizeof(hostname) - 1] = '\0';
    for (int i = 0; hostname[i] != '\0'; i++) {
      if (hostname[i] == '.') {
        hostname[i] = '\0';
        break;
      }
    }
  }
  return hostname;
}

static inline double ncclCheckpointTimestampMs() {
  static auto epoch = std::chrono::steady_clock::now();
  auto delta = std::chrono::steady_clock::now() - epoch;
  return std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
}

static inline void ncclCheckpointLog(NcclCheckpointLogLevel level, const char* func, int line, const char* fmt, ...) {
  if (!ncclCheckpointShouldLog(level)) return;

  int cudaDev = 0;
  (void)cudaGetDevice(&cudaDev);

  flockfile(stderr);
  if (level == ncclCheckpointLogInfo) {
    fprintf(stderr, "%s:%d:%d [%d] NCCL INFO NCCL Checkpoint: ", ncclCheckpointHostName(), getpid(),
            (int)ncclCheckpointGetTid(), cudaDev);
  } else {
    fprintf(stderr, "%s:%d:%d [%d] %f %s:%d NCCL TRACE NCCL Checkpoint: ", ncclCheckpointHostName(), getpid(),
            (int)ncclCheckpointGetTid(), cudaDev, ncclCheckpointTimestampMs(), func, line);
  }

  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  funlockfile(stderr);
}

static inline void ncclCheckpointWarn(const char* file, const char* func, int line, const char* fmt, ...) {
  int cudaDev = 0;
  (void)cudaGetDevice(&cudaDev);

  flockfile(stderr);
  fprintf(stderr, "%s:%d:%d [%d] %s:%d (%s) NCCL WARN NCCL Checkpoint: ", ncclCheckpointHostName(), getpid(),
          (int)ncclCheckpointGetTid(), cudaDev, file, line, func);

  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  funlockfile(stderr);
}

#define WARN(...) ::nccl_checkpoint::ncclCheckpointWarn(__FILE__, __func__, __LINE__, __VA_ARGS__)
#define INFO(_dummy, ...) \
  ::nccl_checkpoint::ncclCheckpointLog(::nccl_checkpoint::ncclCheckpointLogInfo, nullptr, 0, __VA_ARGS__)
#define TRACE(_dummy, ...) \
  ::nccl_checkpoint::ncclCheckpointLog(::nccl_checkpoint::ncclCheckpointLogTrace, __func__, __LINE__, __VA_ARGS__)

template <typename FnT>
static inline ncclResult_t resolveRealFunction(const char* name, FnT* fn) {
  if (*fn != nullptr) return ncclSuccess;

  dlerror();
  void* sym = dlsym(RTLD_NEXT, name);
  if (sym == nullptr) {
    const char* error = dlerror();
    WARN("failed to resolve %s with dlsym(RTLD_NEXT): %s", name, error != nullptr ? error : "unknown error");
    return ncclInternalError;
  }
  *fn = reinterpret_cast<FnT>(sym);
  return ncclSuccess;
}

#define NCCLCHECK_WAIT(cmd, comm) \
  do { \
    ncclResult_t r = cmd; \
    while (r == ncclInProgress) { \
      usleep(10); \
      NCCLCHECK(ncclCommGetAsyncError(comm, &r)); \
    } \
    NCCLCHECK(r); \
  } while (0)

enum CommCreationKind {
  create_via_init = 1,
  create_via_split = 2,
  create_via_shrink = 3,
  create_via_grow = 4,
};

enum CommUserState {
  comm_user_active = 1,
  comm_user_finalized = 2,
  comm_user_destroyed = 3,
  comm_user_nocolor = 4,
};

struct CommInitParams {
  CommCreationKind creation = create_via_init;
  int nranks;
  int rank;
  uint64_t commHash;
  int cudaDev;
  ncclConfig_t config;
  char net_name[64];
  char comm_name[64];
  ncclComm_t commParent = nullptr;
  int splitColor = NCCL_SPLIT_NOCOLOR;
  int splitKey = 0;
  std::vector<int> shrinkExcludeRanks;
  int shrinkFlags = NCCL_SHRINK_DEFAULT;
  int growRankArg = -1;
  bool growUniqueIdProvided = false;
  bool configProvided = false;

  ncclConfig_t* getConfig() {
    return configProvided ? &config : nullptr;
  }
};

struct NoConfig {};

struct RegConfig {
  void* base = nullptr;
  size_t sz = 0;
  uint64_t sequence = 0;
};

struct WindowConfig {
  void* base = nullptr;
  size_t sz = 0;
  int flags = 0;
  uint64_t sequence = 0;
};

bool isCheckpointPrepared();
void markCheckpointPrepared();
void clearCheckpointPrepared();
uint64_t nextRegistrationSequence();

template <typename HandleT, typename ConfigT>
struct HandleEntry {
  using Handle = HandleT;
  using Config = ConfigT;

  Handle realHandle = nullptr;
  Config* config = nullptr;
  static constexpr const char* handleTypeString = "UNKNOWN";
};

template <typename Entry>
class HandleMapT {
protected:
  using Handle = typename Entry::Handle;
  using Config = typename Entry::Config;
  static constexpr uintptr_t MAX_SYNTH_HANDLE_VALUE = 65535;
  std::atomic<uintptr_t> nextSynth{1};

public:
  static constexpr const char* handleTypeString = Entry::handleTypeString;

  Handle makeSynthetic(Handle realHandle, Config* config = nullptr) {
    // Synthetic handles are allocated from a single monotonically increasing
    // counter. That makes std::map iteration over synthetic keys match handle
    // creation order, which the checkpoint replay path relies on.
    uintptr_t synthValue = allocSyntheticHandleValue();
    if (synthValue == 0) return nullptr;
    Handle synth = reinterpret_cast<Handle>(synthValue);
    std::lock_guard<std::mutex> lock(mtx_);
    auto [it, inserted] = entries_.try_emplace(synth);
    if (!inserted) return nullptr;
    it->second.realHandle = realHandle;
    it->second.config = config;
    TRACE(NCCL_ALL, "makeSynthetic(%s) %p => %p", handleTypeString, synth, realHandle);
    return synth;
  }

  void remap(Handle synthHandle, Handle newRealHandle) {
    std::lock_guard<std::mutex> lock(mtx_);
    Handle oldHandle = entries_[synthHandle].realHandle;
    entries_[synthHandle].realHandle = newRealHandle;
    TRACE(NCCL_ALL, "remap(%s) %p => %p now (was %p)", handleTypeString, synthHandle, newRealHandle, oldHandle);
  }

  /* lookup the synthHandle and return the realHandle */
  ncclResult_t toReal(Handle synthHandle, Handle* outRealHandle) const {
    *outRealHandle = reinterpret_cast<Handle>(synthHandle);
    if ((uintptr_t)synthHandle > MAX_SYNTH_HANDLE_VALUE) {
      /* handles cannot be greater than this value, it must already be the real value.*/
      return ncclSuccess;
    }
    if (synthHandle == nullptr) {
      TRACE(NCCL_ALL, "toReal(%s) resolving null pointer", handleTypeString);
      return ncclSuccess;
    }
    if (isCheckpointPrepared()) {
      return ncclInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(synthHandle);
    if (it == entries_.end()) {
      WARN("toReal(%s): missing synthetic handle %p", handleTypeString, synthHandle);
      return ncclInvalidArgument;
    }
    *outRealHandle = it->second.realHandle;
    return ncclSuccess;
  }

  void remove(Handle synthHandle) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(synthHandle);
    if (it != entries_.end()) {
      delete it->second.config;
      entries_.erase(it);
    }
  }

  Entry& operator[](Handle synthHandle) {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_[synthHandle];
  }

  ncclResult_t find(Handle synthHandle, const Entry** outEntry, Config* expectedConfig = nullptr) const {
    *outEntry = nullptr;
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(synthHandle);
    if (it == entries_.end()) {
      WARN("find(%s): missing handle %p", handleTypeString, synthHandle);
      return ncclInternalError;
    }
    if (expectedConfig != nullptr && it->second.config != expectedConfig) {
      WARN("find(%s): handle %p config mismatch: expected %p, found %p", handleTypeString, synthHandle, expectedConfig,
           it->second.config);
      return ncclInternalError;
    }
    *outEntry = &it->second;
    return ncclSuccess;
  }

  ncclResult_t find(Handle synthHandle, Entry** outEntry, Config* expectedConfig = nullptr) {
    *outEntry = nullptr;
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(synthHandle);
    if (it == entries_.end()) {
      WARN("find(%s): missing handle %p", handleTypeString, synthHandle);
      return ncclInternalError;
    }
    if (expectedConfig != nullptr && it->second.config != expectedConfig) {
      WARN("find(%s): handle %p config mismatch: expected %p, found %p", handleTypeString, synthHandle, expectedConfig,
           it->second.config);
      return ncclInternalError;
    }
    *outEntry = &it->second;
    return ncclSuccess;
  }

  bool checkHandle(Handle synthHandle) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.find(synthHandle) != entries_.end();
  }

  template <typename Fn>
  ncclResult_t forEachHandle(Fn&& fn) const {
    std::vector<std::pair<Handle, const Entry*>> work;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (const auto& [synth, entry] : entries_) {
        if (entry.config != nullptr) work.emplace_back(synth, &entry);
      }
    }
    for (const auto& [synth, entry] : work) {
      ncclResult_t ret = fn(synth, entry);
      if (ret != ncclSuccess) return ret;
    }
    return ncclSuccess;
  }

  uintptr_t peekNextHandle() {
    /* query without modifying */
    return nextSynth;
  }

protected:
  uintptr_t allocSyntheticHandleValue() {
    if (nextSynth > MAX_SYNTH_HANDLE_VALUE) return 0;
    return nextSynth++;
  }

  mutable std::mutex mtx_;
  std::map<Handle, Entry> entries_;
};

struct RegHandleEntry : public HandleEntry<void*, RegConfig> {
  static constexpr const char* handleTypeString = "ncclMR";
  ncclComm_t synthComm = nullptr;
};
using RegHandleMap = HandleMapT<RegHandleEntry>;

struct WindowHandleEntry : public HandleEntry<ncclWindow_t, WindowConfig> {
  static constexpr const char* handleTypeString = "ncclWindow_t";
  ncclComm_t synthComm = nullptr;
};
using WindowHandleMap = HandleMapT<WindowHandleEntry>;

struct CommHandleEntry : public HandleEntry<ncclComm_t, CommInitParams> {
  static constexpr const char* handleTypeString = "ncclComm_t";
  int liveChildCommCount = 0;
  CommUserState userState = comm_user_active;
  bool restoreUnsafe = false;
  const char* restoreUnsafeReason = nullptr;
  std::map<uint64_t, void*> registrations;
  std::map<uint64_t, ncclWindow_t> windows;
};

using CommHandleMap = HandleMapT<CommHandleEntry>;

extern CommHandleMap g_commHandles;
extern RegHandleMap g_regHandles;
extern WindowHandleMap g_windowHandles;

static inline void markCommRestoreUnsafe(ncclComm_t synthComm, const char* reason) {
  if (!g_commHandles.checkHandle(synthComm)) return;

  CommHandleEntry& entry = g_commHandles[synthComm];
  if (!entry.restoreUnsafe) {
    entry.restoreUnsafe = true;
    entry.restoreUnsafeReason = reason;
    WARN("communicator %p is not checkpoint-restore safe: %s", synthComm, reason);
  }
}

static inline void markWindowCommRestoreUnsafe(ncclWindow_t synthWin, const char* reason) {
  if (!g_windowHandles.checkHandle(synthWin)) return;

  WindowHandleEntry& entry = g_windowHandles[synthWin];
  markCommRestoreUnsafe(entry.synthComm, reason);
}

}  // namespace nccl_checkpoint

extern "C" ncclResult_t ncclCheckpointPrepare(void);
extern "C" ncclResult_t ncclCheckpointRestore(void);
extern "C" ncclResult_t ncclCheckpointGetVersion(int* checkpointVersion, int* ncclVersion);
