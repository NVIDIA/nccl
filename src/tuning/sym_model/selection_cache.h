/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_SELECTION_CACHE_H_
#define NCCL_INT_SYM_MODEL_SELECTION_CACHE_H_

#include <array>
#include <cstddef>
#include <map>

// Bounded FIFO storage for model selections. Callers own keys and validity checks
// and keep each instance thread-local. Hits do not change replacement order.
template <typename Key, typename Value>
struct ncclSymkSelectionCache {
  using Map = std::map<Key, Value>;
  Map entries;
  std::array<typename Map::iterator, 256> insertionOrder{};
  size_t next = 0;

  // The insertion order holds iterators into this cache's map.
  ncclSymkSelectionCache() = default;
  ncclSymkSelectionCache(const ncclSymkSelectionCache&) = delete;
  ncclSymkSelectionCache& operator=(const ncclSymkSelectionCache&) = delete;
};

template <typename Key, typename Value>
const Value* ncclSymkSelectionCacheFind(const ncclSymkSelectionCache<Key, Value>& cache, Key const& key) {
  const auto entry = cache.entries.find(key);
  return entry == cache.entries.end() ? nullptr : &entry->second;
}

// Insert only after a miss. New keys replace the oldest insertion when full.
template <typename Key, typename Value>
void ncclSymkSelectionCacheInsert(ncclSymkSelectionCache<Key, Value>& cache, Key const& key, Value const& value) {
  const auto entry = cache.entries.emplace(key, value).first;
  if (cache.entries.size() > cache.insertionOrder.size()) cache.entries.erase(cache.insertionOrder[cache.next]);
  cache.insertionOrder[cache.next] = entry;
  cache.next = (cache.next + 1) % cache.insertionOrder.size();
}

#endif // NCCL_INT_SYM_MODEL_SELECTION_CACHE_H_
