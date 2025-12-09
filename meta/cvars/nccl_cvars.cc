// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/cvars/nccl_cvars.h"

#include <pwd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include <folly/String.h>
#include <folly/logging/xlog.h>

static int cudaDev = -1;
static bool logInfoLog = false;
namespace ncclx {
std::unordered_map<std::string, std::string> nccl_config;

void initEnvSet(std::unordered_set<std::string>& env);
void readCvarEnv();

#define CVAR_INFO(fmt, ...)                 \
  XLOGF_IF(                                 \
      INFO,                                 \
      logInfoLog,                           \
      "[CudaDev: {}] NCCL INFO CVAR: " fmt, \
      cudaDev,                              \
      __VA_ARGS__);

#define CVAR_ERROR(fmt, ...) \
  XLOGF(FATAL, "[CudaDev: {}] NCCL ERROR CVAR: " fmt, cudaDev, __VA_ARGS__);

#define CVAR_INFO_UNKNOWN_VALUE(name, value)               \
  do {                                                     \
    CVAR_INFO("Unknown value {} for env {}", value, name); \
  } while (0)

static bool env2bool(
    const char* str,
    const char* def); // Declear env2bool so we can use it in initCvarLogger

static void initCvarLogger() {
  // Used for ncclCvarInit time warning only
  auto err = cudaGetDevice(&cudaDev);
  if (err != cudaSuccess) {
    CVAR_INFO(
        "Error getting cuda device. Error: {}, ErrorStr: {}",
        (int)err,
        cudaGetErrorString(err));
  }
  logInfoLog = env2bool("NCCL_CVARS_LOG_INFO", "false");
}

static std::vector<std::string> tokenizer(std::string str) {
  // Split input string by comma
  std::vector<std::string> tokens;
  folly::split(",", str, tokens, true /* ignore empty */);

  // Trim white space & check for duplicates
  std::unordered_set<std::string> uniqueTokens;
  for (auto& token : tokens) {
    token = folly::trimWhitespace(token);
    if (not uniqueTokens.insert(token).second) {
      CVAR_INFO("Duplicate token {} found", token);
    }
  }

  return tokens;
}

static std::string readenv(const char* str, const char* def) {
  std::string s;
  if (getenv(str)) {
    s = std::string(getenv(str));
  } else if (ncclx::nccl_config.contains(str)) {
    s = ncclx::nccl_config[str];
  } else if (def) {
    s = std::string(def);
  } else {
    s = std::string("");
  }

  return s;
}

static bool env2bool(const char* str, const char* def) {
  std::string s = readenv(str, def);
  std::transform(s.cbegin(), s.cend(), s.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  if (s == "y")
    return true;
  else if (s == "n")
    return false;
  else if (s == "yes")
    return true;
  else if (s == "no")
    return false;
  else if (s == "t")
    return true;
  else if (s == "f")
    return false;
  else if (s == "true")
    return true;
  else if (s == "false")
    return false;
  else if (s == "1")
    return true;
  else if (s == "0")
    return false;
  else
    CVAR_INFO_UNKNOWN_VALUE(str, s.c_str());
  return true;
}

template <typename T>
static T env2num(const char* str, const char* def) {
  std::string s = readenv(str, def);

  if (std::find_if(s.begin(), s.end(), ::isdigit) != s.end()) {
    /* if the string contains a digit, try converting it normally */
    std::stringstream sstream(s);
    T ret;
    sstream >> ret;
    return ret;
  } else {
    /* if there are no digits, see if its a special string such as
     * "MAX" or "MIN". */
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    if (s == "MAX") {
      return std::numeric_limits<T>::max();
    } else if (s == "MIN") {
      return std::numeric_limits<T>::min();
    } else {
      CVAR_INFO("Unrecognized numeral {}", s);
      return 0;
    }
  }
}

static std::string env2str(const char* str, const char* def) {
  return std::string(folly::trimWhitespace(readenv(str, def)));
}

static std::vector<std::string> env2strlist(const char* str, const char* def) {
  return tokenizer(std::string(folly::trimWhitespace(readenv(str, def))));
}

static std::tuple<std::string, std::vector<std::string>> env2prefixedStrlist(
    const char* str,
    const char* def,
    const std::vector<std::string>& prefixes) {
  std::string s = readenv(str, def);

  // search if any prefix is specified
  for (auto prefix : prefixes) {
    if (!s.compare(0, prefix.size(), prefix)) {
      // if prefix is found, convert the remaining string to stringList
      std::string slist_s = s.substr(prefix.size());
      return std::make_tuple(prefix, tokenizer(slist_s));
    }
  }
  // if no prefix is found, convert entire string to stringList
  return std::make_tuple("", tokenizer(s));
}

inline void updateNcclConfig(std::string fname) {
  std::ifstream in(fname);
  if (!in) {
    CVAR_INFO("NCCL config file {} doesn't exists, skipping", fname);
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    // Trim the string starting with first `#`
    auto n = line.find("#");
    if (n != std::string::npos) {
      line.erase(n);
    }

    line = folly::trimWhitespace(line);
    if (line.empty()) {
      continue;
    }

    std::vector<std::string> tokens;
    folly::split("=", line, tokens, false /* ignore-empty */);
    if (tokens.size() != 2) {
      //CVAR_INFO("Ignoring invalid config option: {}", line);
      continue;
    }

    CVAR_INFO(
        "NCCL Config - Overriding CVAR {}={} from {}",
        tokens.at(0).c_str(),
        tokens.at(1).c_str(),
        fname.c_str());

    ncclx::nccl_config.emplace(tokens.at(0), tokens.at(1));
  }
}
}; // namespace ncclx

extern char** environ;
void ncclCvarInit() {
  std::unordered_set<std::string> env;
  ncclx::initEnvSet(env);

  ncclx::initCvarLogger();

  // Check if any NCCL_ env var is not in allow list
  char** s = environ;
  for (; *s; s++) {
    if (!strncmp(*s, "NCCL_", strlen("NCCL_"))) {
      std::string str(*s);
      str = str.substr(0, str.find("="));
      if (env.find(str) == env.end()) {
       // CVAR_INFO("Unknown env {} in the NCCL namespace", str);
      }
    }
  }

  ncclx::updateNcclConfig("/etc/nccl.conf");
  struct passwd* pwUser = getpwuid(getuid());
  if (pwUser) {
    std::string fname = std::string(pwUser->pw_dir) + "/.nccl.conf";
    ncclx::updateNcclConfig(fname);
  }

  ncclx::readCvarEnv();
}

std::vector<std::string> NCCL_COLLTRACE;
int64_t NCCL_COLLTRACE_CHECK_INTERVAL_MS;
bool NCCL_COLLTRACE_EVENT_BLOCKING_SYNC;
int NCCL_COLLTRACE_RECORD_MAX;
int NCCL_COLLTRACE_RECORD_MAX_ITERATIONS;

int64_t NCCL_COLLTRACE_REPORT_FIRST_N_COLL;
std::vector<std::string> NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG;
std::vector<std::string> NCCL_FILTER_ALGO_LOGGING_BY_RANKS;
std::vector<std::string> NCCL_FILTER_MEM_LOGGING_BY_RANKS;
std::vector<std::string> NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS;
int NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES;

std::vector<std::string> NCCL_PROXYTRACE;
int NCCL_PROXYTRACE_RECORD_MAX;
bool NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED;

namespace ncclx {
void initEnvSet(std::unordered_set<std::string>& env) {
  env.insert("NCCL_COLLTRACE");
  env.insert("NCCL_COLLTRACE_CHECK_INTERVAL_MS");
  env.insert("NCCL_COLLTRACE_EVENT_BLOCKING_SYNC");
  env.insert("NCCL_COLLTRACE_RECORD_MAX");
  env.insert("NCCL_COLLTRACE_RECORD_MAX_ITERATIONS");
  env.insert("NCCL_COLLTRACE_REPORT_FIRST_N_COLL");
  env.insert("NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG");
  env.insert("NCCL_FILTER_ALGO_LOGGING_BY_RANKS");
  env.insert("NCCL_FILTER_MEM_LOGGING_BY_RANKS");
  env.insert("NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS");
  env.insert("NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES");
  env.insert("NCCL_PROXYTRACE");
  env.insert("NCCL_PROXYTRACE_RECORD_MAX");
  env.insert("NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED");
}

void readCvarEnv() {
    NCCL_COLLTRACE = env2strlist("NCCL_COLLTRACE", "trace");
    NCCL_COLLTRACE_CHECK_INTERVAL_MS =
        env2num<int64_t>("NCCL_COLLTRACE_CHECK_INTERVAL_MS", "10");
    NCCL_COLLTRACE_EVENT_BLOCKING_SYNC =
        env2bool("NCCL_COLLTRACE_EVENT_BLOCKING_SYNC", "False");
    NCCL_COLLTRACE_RECORD_MAX = env2num<int>("NCCL_COLLTRACE_RECORD_MAX", "2000");
    NCCL_COLLTRACE_RECORD_MAX_ITERATIONS =
        env2num<int>("NCCL_COLLTRACE_RECORD_MAX_ITERATIONS", "2");
    NCCL_COLLTRACE_REPORT_FIRST_N_COLL =
        env2num<int64_t>("NCCL_COLLTRACE_REPORT_FIRST_N_COLL", "10");
    NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG =
        env2strlist("NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG", "ANY:0");
    NCCL_FILTER_ALGO_LOGGING_BY_RANKS =
        env2strlist("NCCL_FILTER_ALGO_LOGGING_BY_RANKS", "");
    NCCL_FILTER_MEM_LOGGING_BY_RANKS =
        env2strlist("NCCL_FILTER_MEM_LOGGING_BY_RANKS", "0");
    NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS =
        env2strlist("NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS", "");
    NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES =
        env2num<int>("NCCL_PROCESS_GLOBAL_ERRORS_MAX_STACK_TRACES", "20");
    NCCL_PROXYTRACE = env2strlist("NCCL_PROXYTRACE", "");
    NCCL_PROXYTRACE_RECORD_MAX = env2num<int>("NCCL_PROXYTRACE_RECORD_MAX", "20");
    NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED =
        env2bool("NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED", "True");
}
}; // namespace ncclx
