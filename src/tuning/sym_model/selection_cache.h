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

// Bounded FIFO storage for model selections. Callers own keys and validity checks
// and keep each instance thread-local. Hits do not change replacement order.
template <typename Key, typename Value>
struct ncclSymkSelectionCache {
  struct Entry {
    Key key{};
    Value value{};
  };
  std::array<Entry, 64> entries{};
  size_t next = 0, count = 0;
};

template <typename Key, typename Value>
const Value* ncclSymkSelectionCacheFind(const ncclSymkSelectionCache<Key, Value>& cache, Key const& key) {
  for (size_t i = 0; i < cache.count; i++)
    if (cache.entries[i].key == key) return &cache.entries[i].value;
  return nullptr;
}

// Insert only after a miss. New keys replace the oldest insertion when full.
template <typename Key, typename Value>
void ncclSymkSelectionCacheInsert(ncclSymkSelectionCache<Key, Value>& cache, Key const& key, Value const& value) {
  cache.entries[cache.next] = {key, value};
  cache.next = (cache.next + 1) % cache.entries.size();
  if (cache.count < cache.entries.size()) cache.count++;
}

#endif // NCCL_INT_SYM_MODEL_SELECTION_CACHE_H_
