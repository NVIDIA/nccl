/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_H_INCLUDED
#define PARAM_H_INCLUDED

#include "nccl.h"
#include "param/common.h"
#include "param/parsers.h"
#include "debug.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <mutex>
// using C++ atomic because we don't include compiler.h here
#include <atomic>
#include <memory>

// Compiler detection macros
#if defined(__GNUC__) || defined(__clang__)
#define NCCL_PARAM_COMPILER_EXPORT_SYMBOL __attribute__((visibility("default")))
#define NCCL_PARAM_COMPILER_EXPECT(x, v) __builtin_expect((x), (v))
#elif defined(_MSC_VER)
#define NCCL_PARAM_COMPILER_EXPORT_SYMBOL
#define NCCL_PARAM_COMPILER_EXPECT(x, v) (x)
#else
  #error "Unsupported compiler"
#endif

// Exported helper wrapping ncclInitEnv() + ncclEnvPluginGetEnv() so that
// plugins only need to resolve a single symbol from libnccl.so.
// Defined in src/param/param.cc.
extern "C" NCCL_PARAM_COMPILER_EXPORT_SYMBOL const char* ncclParamEnvPluginGet(const char* key);

// Returns true if the given param key should have caching disabled
// (appears in NCCL_NO_CACHE env var). Defined in param.cc.
extern "C" NCCL_PARAM_COMPILER_EXPORT_SYMBOL bool ncclParamIsCacheDisabled(const char* key);

// ============================================================================
// Flags
// ============================================================================

enum NcclParamFlags {
  NCCL_PARAM_FLAG_NONE       = 0,
  NCCL_PARAM_FLAG_PUBLISHED  = 1ULL << 0,
  NCCL_PARAM_FLAG_DEPRECATED = 1ULL << 1,
  NCCL_PARAM_FLAG_CACHED     = 1ULL << 2,
  NCCL_PARAM_FLAG_UNUSED     = 1ULL << 3
};

// ============================================================================
// Helper Functions
// ============================================================================

namespace nccl {
namespace param {
namespace utils {

  // Convert flags to comma-separated string
  inline std::string flagsStr(uint64_t flags) {
    std::string result = "";
    // If Published is NOT set, display "Private"
    if (!(flags & NCCL_PARAM_FLAG_PUBLISHED)) {
      result += "Private";
    }
    // If Published IS set, don't display anything for it
    if (flags & NCCL_PARAM_FLAG_DEPRECATED) {
      if (!result.empty()) result += ", ";
      result += "Deprecated";
    }
    if (flags & NCCL_PARAM_FLAG_CACHED) {
      if (!result.empty()) result += ", ";
      result += "Cached";
    }
    if (flags & NCCL_PARAM_FLAG_UNUSED) {
      if (!result.empty()) result += ", ";
      result += "Unused";
    }
    return result;
  }

  // Convert source enum to string
  inline const char* sourceStr(int32_t source) {
    switch (source) {
      case NCCL_PARAM_SOURCE_DEFAULT:     return "Default";
      case NCCL_PARAM_SOURCE_ENV_VAR:     return "EnvVar";
      case NCCL_PARAM_SOURCE_ENV_PLUGIN:  return "EnvPlugin";
      case NCCL_PARAM_SOURCE_CONFIG_FILE: return "ConfigFile";
      default:                            return "Unknown";
    }
  }

} // namespace utils
} // namespace param
} // namespace nccl

// ============================================================================
// NcclParam Base Class (Interface)
// ============================================================================

struct NcclParamBase {
  virtual ~NcclParamBase() = default;
  virtual std::string toString() const = 0;
  virtual std::string dump(bool showAll = false) const = 0;
  virtual uint64_t getFlags() const = 0;

  // C API typed accessors - return 0 (OK) or 2 (TYPE_MISMATCH)
  virtual int getI8(int8_t* out) const = 0;
  virtual int getI16(int16_t* out) const = 0;
  virtual int getI32(int32_t* out) const = 0;
  virtual int getI64(int64_t* out) const = 0;
  virtual int getU8(uint8_t* out) const = 0;
  virtual int getU16(uint16_t* out) const = 0;
  virtual int getU32(uint32_t* out) const = 0;
  virtual int getU64(uint64_t* out) const = 0;
  virtual int getRawData(void* out, int maxLen, int* len) const = 0;
};

// ============================================================================
// Global Registry
// ============================================================================

// C-linkage singleton accessor — defined in param.cc, exported as "ncclParamRegistryInstance"
// Returns a process-wide RegistryState so map and mutex share identity across DSOs.
extern "C" NCCL_PARAM_COMPILER_EXPORT_SYMBOL void* ncclParamRegistryInstance();

class NcclParamRegistry {
public:
  using MapType = std::unordered_map<std::string, NcclParamBase*>;
  struct RegistryState {
    MapType map;
    std::mutex mtx;
  };

  static RegistryState& state() {
    return *static_cast<RegistryState*>(ncclParamRegistryInstance());
  }

  static MapType& instance() {
    return state().map;
  }

  static std::mutex& mutex() {
    return state().mtx;
  }

  // Register a parameter
  static ncclResult_t add(std::string key, NcclParamBase* param) {
    std::lock_guard<std::mutex> lock(mutex());
    auto& map = instance();
    if (map.find(key) != map.end()) {
      WARN("Duplicate parameter key: %s", key.c_str());
      return ncclInternalError;
    }
    map[key] = param;
    return ncclSuccess;
  }

  // Find a parameter by key
  static NcclParamBase* find(std::string key) {
    std::lock_guard<std::mutex> lock(mutex());
    auto& map = instance();
    auto it = map.find(key);
    return (it != map.end()) ? it->second : nullptr;
  }

  // Unregister a parameter
  static ncclResult_t remove(std::string key) {
    std::lock_guard<std::mutex> lock(mutex());
    auto& map = instance();
    map.erase(key);
    return ncclSuccess;
  }

  // Prevent instantiation
  NcclParamRegistry() = delete;
};

// ============================================================================
// NcclParam Template
// ============================================================================

template <typename T>
struct NcclParam : public NcclParamBase {
  const char* key;
  const T defaultValue;;

  const uint64_t flags;
  mutable T value;
  mutable ncclParamSource_t source = NCCL_PARAM_SOURCE_DEFAULT;

  // Storage for const char* parameters (so returned pointers remain valid).
  // For non-const-char* types this is an empty struct (no overhead).
  struct NoStorage {};
  using CStrStorage = typename std::conditional<std::is_same<T, const char*>::value, std::string, NoStorage>::type;
  mutable CStrStorage cstrStorage{};

  mutable std::mutex mtx;
  mutable std::atomic<bool> loaded{false};
  mutable std::atomic<int8_t> cacheDisabled{-1};

  // Metadata for dump()
  std::string typeName;
  std::string description;

  // Runtime parser storage
  std::function<ncclResult_t(const char*, T&)> runtimeResolve;
  std::function<ncclResult_t(const T&)> runtimeValidate;
  std::function<std::string(const T&)> runtimeToString;
  std::function<std::string()> runtimeDesc;

  ~NcclParam() override = default;

  // Constructor for default parser
  NcclParam(const char* key, T defVal, const char* typeStr = "",
            uint64_t flags = NCCL_PARAM_FLAG_NONE, const char* desc = "")
    : key(key), defaultValue(defVal), flags(flags), value(defVal), typeName(typeStr), description(desc) {
      NcclParamRegistry::add(key, this);
    }

  // Constructor for custom parser
  NcclParam(const char* key, T defVal, NcclParserFuncs<T> parser, const char* typeStr = "",
            uint64_t flags = NCCL_PARAM_FLAG_NONE, const char* desc = "")
    : key(key), defaultValue(defVal), flags(flags), value(defVal), typeName(typeStr), description(desc),
    runtimeResolve(std::move(parser.resolve)),
    runtimeValidate(std::move(parser.validate)),
    runtimeToString(std::move(parser.toString)),
    runtimeDesc(std::move(parser.desc))
    {
      NcclParamRegistry::add(key, this);
    }

  // A wrapper for calling the resolve function of the parser
  // configured of the parameter. Making it public to allow NCCL code to use
  // the parser function do (input -> value) conversion.
  // Useful for parameters with complex parsers such as OneOf or BitsetOf.
  ncclResult_t parserResolve(const char* input, T& out) const {
    ncclResult_t rc = runtimeResolve ? runtimeResolve(input, out)
                                     : NcclDefaultParser<T>::resolve(input, out);
    if (rc == ncclSuccess) rc = parserValidate(out);
    return rc;
  }

  // A wrapper for calling the validate function of the parser
  // configured of the parameter. Making it public to allow NCCL code to use
  // the parser function do (is val valid) check.
  // Useful for parameters with complex parsers such as OneOf or BitsetOf.
  ncclResult_t parserValidate(const T& val) const {
    if (runtimeValidate) return runtimeValidate(val);
    return NcclDefaultParser<T>::validate(val);
  }

  // A wrapper for calling the resolve function of the parser
  // configured of the parameter. Making it public to allow NCCL code to use
  // the parser function do (val -> string) conversion.
  // Useful for parameters with complex parsers such as OneOf or BitsetOf.
  std::string parserToString(const T& val) const {
    // Try runtime parser first
    if (runtimeToString) {
      return runtimeToString(val);
    }
    // Fallback to default parser
    return NcclDefaultParser<T>::toString(val);
  }

  // Virtual override for registry access
  // This convert current value stored to string for printing
  std::string toString() const override {
    auto lock = ensureLoaded();
    return parserToString(value);
  }

  uint64_t getFlags() const override {
    return flags;
  }

  // Return a string of how the current value is set, used in printing debug message
  const char* sourceCStr() const {
    auto lock = ensureLoaded();
    return nccl::param::utils::sourceStr(source);
  }

  // Prevent copy/move assignment (defaultValue is const)
  NcclParam& operator=(const NcclParam&) = delete;
  NcclParam& operator=(NcclParam&&) = delete;

  // Main access function for parameter value through function call-like interface
  T operator()() const {
    return accessValue<T>();
  }

  // Dump parameter information (disabled for non-Published params unless showAll is true)
  std::string dump(bool showAll = false) const override {
    if (!showAll && !(flags & NCCL_PARAM_FLAG_PUBLISHED)) {
      return "";
    }

    // not reading value directly here, using accessValue<T>() to make sure the value is set before reading
    std::string currentStr = parserToString(accessValue<T>());
    std::string result;
    std::string defaultStr = parserToString(defaultValue);
    // Line 1: Key (type) (flags) (desc)
    result += key;
    if (!typeName.empty()) {
      result += " (" + typeName + ")";
    }
    std::string flagStr = nccl::param::utils::flagsStr(flags);
    if (!flagStr.empty()) {
      result += " [" + flagStr + "]";
    }
    if (!description.empty()) {
      result += " " + description;
    }
    // Line 2: Current value, value source and default value
    result += "\n    Current value=" + (currentStr.empty() ? "<unset>" : currentStr);
    result += "  set_by=" + std::string(nccl::param::utils::sourceStr(static_cast<int32_t>(source)));
    result += ",  default=" + (defaultStr.empty() ? "<unset>" : defaultStr);
    // Line 3+: Accepted values
    std::string acceptedDesc;
    // Try runtime parser description first, then fall back to default parser
    if (runtimeDesc) {
      acceptedDesc = runtimeDesc();
    } else {
      acceptedDesc = NcclDefaultParser<T>::desc();
    }
    if (!acceptedDesc.empty()) {
      result += "\n    Accepted value: ";
      result += acceptedDesc;
    }
    // If accepted_desc is multiline, put default/source on its own line
    if (!acceptedDesc.empty() && acceptedDesc.find('\n') != std::string::npos) {
      result += "\n    ";
    }
    return result;
  }

  // C API typed accessor overrides
  int getI8(int8_t* out) const override { return doGet<int8_t>(out); }
  int getI16(int16_t* out) const override { return doGet<int16_t>(out); }
  int getI32(int32_t* out) const override { return doGet<int32_t>(out); }
  int getI64(int64_t* out) const override { return doGet<int64_t>(out); }
  int getU8(uint8_t* out) const override { return doGet<uint8_t>(out); }
  int getU16(uint16_t* out) const override { return doGet<uint16_t>(out); }
  int getU32(uint32_t* out) const override { return doGet<uint32_t>(out); }
  int getU64(uint64_t* out) const override { return doGet<uint64_t>(out); }
  int getRawData(void* out, int maxLen, int* len) const override {
    if (!out || !len || maxLen <= 0) return NCCL_PARAM_BAD_ARGUMENT;
    return doGetRaw(out, maxLen, len);
  }

private:
  // Core function to make sure value is loaded, check against all conditions including
  // NCCL_NO_CACHE
  std::unique_lock<std::mutex> ensureLoaded() const {
    // Fast path for Cached params: lock-free after first load.
    // Only take fast path if we know caching is not suppressed.
    if ((flags & NCCL_PARAM_FLAG_CACHED)
        && NCCL_PARAM_COMPILER_EXPECT(cacheDisabled.load(std::memory_order_acquire) == 0, true)
        && NCCL_PARAM_COMPILER_EXPECT(loaded.load(std::memory_order_acquire), true)) {
      return {};  // value is immutable after caching; no lock needed
    }
    std::unique_lock<std::mutex> lock(mtx);

    if (flags & NCCL_PARAM_FLAG_CACHED) {
      switch (cacheDisabled) {
      case -1: // first time access: cacheDisable status is not initialized
        cacheDisabled.store(ncclParamIsCacheDisabled(key) ? 1 : 0, std::memory_order_release);
        break;
      case 0: // second or later access: cache not disabled
        if (loaded) {
          return lock;
        }
        break;
      case 1: // cache has already been disabled for this param, proceed to load
        break;
      default: // should never reach here
        WARN("Invalid cacheDisabled state for key: %s", key);
        return {};
      }
    }

    loadValue();

    // Only need to update loaded_ if caching is allowed for this param
    if ((flags & NCCL_PARAM_FLAG_CACHED)
        && cacheDisabled == 0) {
      loaded.store(true, std::memory_order_release);
    }
    return lock;
  }

  // SFINAE: for non-const-char-* type
  template <typename U, std::enable_if_t<!std::is_same<U, const char*>::value, int> = 0>
  U accessValue() const {
    auto lock = ensureLoaded();
    return value;
  }

  // SFINAE: for const char *
  template <typename U, std::enable_if_t<std::is_same<U, const char*>::value, int> = 0>
  U accessValue() const {
    auto lock = ensureLoaded();
    if (value == nullptr) return nullptr;
    static thread_local std::string tlsCstrCopy;
    tlsCstrCopy = cstrStorage;
    return tlsCstrCopy.c_str();
  }

  // SFINAE: Load implementation for const char* type
  void loadValueImpl(const char* envPtr, std::true_type /*is_cstr*/) const {
    // const char* requires no parsing: we always store the raw env string (owned copy).
    if (envPtr) {
      cstrStorage = envPtr;
      value = cstrStorage.c_str();
    } else {
      if (defaultValue) {
        cstrStorage = defaultValue;
        value = cstrStorage.c_str();
      } else {
        cstrStorage.clear();
        value = nullptr;
      }
    }
  }

  // SFINAE: Load implementation for non-const-char* types
  void loadValueImpl(const char* envPtr, std::false_type /*is_cstr*/) const {
    if (envPtr) {
      T resolvedValue;
      ncclResult_t resolved = parserResolve(envPtr, resolvedValue);
      if (resolved == ncclSuccess) {
        value = resolvedValue;
      } else {
        // INFO(NCCL_ENV, "Warning: Invalid value '%s' for %s, using default\n", env_ptr, key);
        value = defaultValue;
      }
    } else {
      // No env var set - resolve default value string
      value = defaultValue;
    }
  }

  // Load value from environment variable via EnvPlugin chain
  void loadValue() const {
    // Check both sources independently to determine provenance
    const char* envPluginValue = ncclParamEnvPluginGet(key);  // Inits env + queries plugin chain

    // plugin_env is the effective value (same as ncclGetEnv would return)
    if (envPluginValue != nullptr) {
      source = NCCL_PARAM_SOURCE_ENV_PLUGIN;
      loadValueImpl(envPluginValue, std::is_same<T, const char*>{});
    } else {
      source = NCCL_PARAM_SOURCE_DEFAULT;
      loadValueImpl(nullptr, std::is_same<T, const char*>{}); // compiler should drop branch on env_ptr
    }
  }

  // SFINAE helper for typed get - type matches
  template<typename Req, typename Act = T, std::enable_if_t<std::is_same<Req, Act>::value, int> = 0>
    int doGet(Req* out) const {
      *out = (*this)();  // Uses operator()()
      return NCCL_PARAM_OK;
    }

  // SFINAE helper for typed get - type mismatch
  template<typename Req, typename Act = T, std::enable_if_t<!std::is_same<Req, Act>::value, int> = 0>
    int doGet(Req*) const {
      return NCCL_PARAM_TYPE_MISMATCH;
    }

  // SFINAE: trivially-copyable non-const-char* (int, bool, enum, etc.): memcpy sizeof(T) bytes
  template<typename U = T,
           std::enable_if_t<!std::is_same<U, const char*>::value
                            && std::is_trivially_copyable<U>::value, int> = 0>
  int doGetRaw(void* out, int maxLen, int* len) const {
    auto lock = ensureLoaded();
    if (static_cast<int>(sizeof(T)) > maxLen) {
      *len = 0;
      return NCCL_PARAM_BAD_ARGUMENT;
    }
    std::memcpy(out, &value, sizeof(T));
    *len = static_cast<int>(sizeof(T));
    return NCCL_PARAM_OK;
  }

  // SFINAE: non-trivially-copyable (e.g. std::unordered_set<>): raw copy is meaningless
  template<typename U = T,
           std::enable_if_t<!std::is_same<U, const char*>::value
                            && !std::is_trivially_copyable<U>::value, int> = 0>
  int doGetRaw(void* out, int maxLen, int* len) const {
    *len = 0;
    return NCCL_PARAM_TYPE_MISMATCH;
  }

  // SFINAE: const char*: copy null-terminated string bytes from cstrStorage
  template<typename U = T, std::enable_if_t<std::is_same<U, const char*>::value, int> = 0>
  int doGetRaw(void* out, int maxLen, int* len) const {
    auto lock = ensureLoaded();
    // cstrStorage holds the owned copy; empty when value is nullptr.
    const std::string& src = cstrStorage;
    int sz = static_cast<int>(src.size()) + 1;  // include null terminator
    if (sz > maxLen) {
      *len = 0;
      return NCCL_PARAM_BAD_ARGUMENT;
    }
    std::memcpy(out, src.c_str(), static_cast<size_t>(sz));
    *len = sz;
    return NCCL_PARAM_OK;
  }
};

// ============================================================================
// Macros
// ============================================================================

// Defining and Using (Including) a NcclParam
// Usage: DEFINE_NCCL_PARAM(name, type, key, default, flags, parser, desc)
// Generate global symbols for key to make compiler check for the uniqueness of the key.
// The check will happen at both compile and link time.
// Define parameter in .cc files
// Note: NCCL_DEFINE_PARAM is not designed to be put inside of a namespace. This can be
// changed if there is a need.
#define DEFINE_NCCL_PARAM(name, type, key, default, flags, parser, desc)                                  \
  namespace key_guards { struct guard_##key {}; };                                                        \
  NCCL_PARAM_COMPILER_EXPORT_SYMBOL extern constexpr char name##Key[] = #key;                             \
  NCCL_PARAM_COMPILER_EXPORT_SYMBOL NcclParam<type> name{ name##Key, default, parser, #type, flags, desc };

// Usage: USE_NCCL_PARAM(name, type)
// name and type must match the DEFINE_NCCL_PARAM.
#define USE_NCCL_PARAM(name, type) \
  extern NcclParam<type> name;

#endif /* PARAM_H_INCLUDED */
