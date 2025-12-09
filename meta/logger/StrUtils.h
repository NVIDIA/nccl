// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <folly/container/F14Map.h>

#include "nccl.h"

inline std::string hashToHexStr(const uint64_t hash) {
  std::stringstream ss;
  ss << std::hex << hash;
  return ss.str();
}

template <typename T>
inline std::string vecToStr(
    const std::vector<T>& vec,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (auto it : vec) {
    if (!first) {
      ss << delim;
    }
    ss << it;
    first = false;
  }
  return ss.str();
}

template <typename T1, typename T2>
inline std::string f14FastMapToStr(
    const folly::F14FastMap<T1, T2>& map,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (const auto& [key, value] : map) {
    if (!first) {
      ss << delim;
    }
    ss << "(" << key.first << "," << key.second << "):" << value;
    first = false;
  }
  return ss.str();
}

template <typename T1, typename T2>
inline std::string unorderedMapToStr(
    const std::unordered_map<T1, T2>& map,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (const auto& [key, value] : map) {
    if (!first) {
      ss << delim;
    }
    ss << key << ":" << value;
    first = false;
  }
  return ss.str();
}

inline int64_t getTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto since_epoch = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
  int64_t timestamp = seconds.count();
  return timestamp;
}

inline std::string readFromFile(const std::string& filename) {
  if (filename.empty()) {
    return "";
  }
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return "";
  }
  return std::string(
      (std::istreambuf_iterator<char>(inFile)),
      std::istreambuf_iterator<char>());
}

inline std::string getThreadUniqueId(const std::string& tag = "") {
  auto threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return std::to_string(threadHash) + (tag.empty() ? "" : "-" + tag);
}

inline std::string uint64ToHexStr(
    const uint64_t val,
    const std::string& prefix = "") {
  std::stringstream ss;
  ss << prefix << std::hex << val;
  return ss.str();
}

template <typename T>
inline std::string unorderedSetToStr(
    const std::unordered_set<T>& vec,
    const std::string& delim = ", ") {
  std::stringstream ss;
  bool first = true;
  for (const auto& it : vec) {
    if (!first) {
      ss << delim;
    }
    ss << it;
    first = false;
  }
  return ss.str();
}

inline std::string getUniqueFileSuffix() {
  std::time_t t = std::time(nullptr);
  std::ostringstream time_str;
  struct tm nowTm {};
  localtime_r(&t, &nowTm);
  time_str << std::put_time(&nowTm, "%Y%m%d-%H%M%S");
  auto threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());

  return time_str.str() + "-" + std::to_string(threadHash);
}

template <typename T>
void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline std::string readFromFileUtil(const std::string& filename) {
  if (filename.empty()) {
    return "";
  }
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return "";
  }
  return std::string(
      (std::istreambuf_iterator<char>(inFile)),
      std::istreambuf_iterator<char>());
}

// Overwrite std::toString to support std::string type
template <typename T>
inline std::string toString(const T& obj) {
  std::ostringstream oss{};
  oss << obj;
  return oss.str();
}

template <typename T>
inline std::string toQuotedString(const T& obj) {
  return "\"" + toString(obj) + "\"";
}

/**
 * Serialize a map object to json string
 * Input arguments:
 *   keys: a vector of keys in the order of insertion
 *   map: the map object to be serialized
 *   quoted_key: whether to quote string key
 *   quoted_value: whether to quote string value
 */
template <typename T>
inline std::string serializeMap(
    const std::vector<std::string>& keys,
    const std::unordered_map<std::string, T>& map,
    bool quoted_key = false,
    bool quoted_value = false,
    const std::set<std::string>& skipKeys = {}) {
  std::string final_string = "{";
  // unordered_map doesn't maintain insertion order. Use keys to ensure
  // serialize in the same order as program defined
  for (auto& key : keys) {
    // skip if key doesn't exist in map
    if (map.find(key) == map.end()) {
      continue;
    }
    const T& val = map.at(key);
    final_string += quoted_key ? toQuotedString(key) : key; // always quote key
    final_string += ": ";
    // only string value needs to be quoted; set when creating map
    if (skipKeys.find(key) != skipKeys.end()) {
      final_string += toString(val);
    } else {
      final_string += quoted_value ? toQuotedString(val) : toString(val);
    }
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "}";
  return final_string;
}

/**
 * Serialize a unordered set object to json string
 * Input arguments:
 *   set: the unordered set object to be serialized
 */
template <typename T>
inline std::string serializeSet(std::unordered_set<T>& set) {
  std::string final_string = "[";
  for (auto& it : set) {
    final_string += toString(it);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

/**
 * Serialize a vector object to json string
 * Input arguments:
 *   vec: the vector object to be serialized
 */
template <typename T>
inline std::string serializeVec(const std::vector<T>& vec) {
  std::string final_string = "[";
  for (auto& it : vec) {
    final_string += toString(it);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

/**
 * Serialize a list object to json string
 * Input arguments:
 *   list: the list object to be serialized
 */
template <typename T>
inline std::string serializeList(std::list<T>& list) {
  std::string final_string = "[";
  for (auto& it : list) {
    final_string += toString(it);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}
/**
 * Serialize an iterable of objects to json string. Require the object type has
 * serialize function. Input arguments: deque: the deque object to be serialized
 * TODO: Check T is genuinely an iterable
 */
template <typename T>
inline std::string serializeObjects(T& objs) {
  std::string final_string = "[";
  for (auto& obj : objs) {
    final_string += obj.serialize(true /*quote*/);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

/**
 * Flatten a map object to plain string
 * Input arguments:
 *   keys: a vector of keys in the order of insertion
 *   map: the map object to be serialized
 *   kvdelim: delimiter between key and value
 *   kdelim: delimiter between key-value pairs
 */
template <typename T>
inline std::string mapToString(
    const std::vector<std::string>& keys,
    std::unordered_map<std::string, T>& map,
    const std::string& kvdelim = " ",
    const std::string& kdelim = " ") {
  std::string final_string;
  // unordered_map doesn't maintain insertion order. Use keys to ensure
  // serialize in the same order as program defined
  for (auto& key : keys) {
    // skip if key doesn't exist in map
    if (map.find(key) == map.end()) {
      continue;
    }
    T& val = map[key];
    final_string += key; // always quote key
    final_string += kvdelim;
    final_string += toString(val);
    final_string += kdelim;
  }
  if (final_string.size() > 1) {
    final_string = final_string.substr(
        0, final_string.size() - std::string(kdelim).size());
  }
  return final_string;
}

/**
 * Flatten a int to int map object to Json format
 */
inline std::string mapToJson(std::unordered_map<int, int>& map) {
  std::string final_string = "{";
  for (auto& [key, val] : map) {
    final_string += toQuotedString(std::to_string(key));
    final_string += ": ";
    final_string += std::to_string(val);
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string =
        final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "}";
  return final_string;
}


inline std::string getDatatypeStr(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
      return "ncclInt8";
    case ncclUint8:
      return "ncclUint8";
    case ncclInt32:
      return "ncclInt32";
    case ncclUint32:
      return "ncclUint32";
    case ncclInt64:
      return "ncclInt64";
    case ncclUint64:
      return "ncclUint64";
    case ncclFloat16:
      return "ncclFloat16";
    case ncclFloat32:
      return "ncclFloat32";
    case ncclFloat64:
      return "ncclFloat64";
    case ncclBfloat16:
      return "ncclBfloat16";
    default:
      return "Unknown type";
  }
}

inline std::string getRedOpStr(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:
      return "ncclSum";
    case ncclProd:
      return "ncclProd";
    case ncclMax:
      return "ncclMax";
    case ncclMin:
      return "ncclMin";
    case ncclAvg:
      return "ncclAvg";
    default:
      return "Unknown op";
  }
}
