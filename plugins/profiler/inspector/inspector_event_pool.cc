#include "inspector_event_pool.h"
#include <stdlib.h>
#include <string.h>

// Global event pool
struct inspectorEventPool g_eventPool;

/*
 * Description:
 *   Helper function to allocate a new chunk for a pool.
 *
 * Parameters:
 *
 *   entrySize - Size of each entry in bytes.
 *   chunkSize - Number of entries to allocate in the chunk.
 *
 * Return:
 *
 *   struct inspectorPoolChunk* - Newly allocated chunk, or nullptr on
 *   failure.
 *
 */
static struct inspectorPoolChunk* allocatePoolChunk(size_t entrySize,
                                                    uint32_t chunkSize) {
  struct inspectorPoolChunk* chunk
    = (struct inspectorPoolChunk*)calloc(1, sizeof(struct inspectorPoolChunk));
  if (chunk == nullptr) {
    return nullptr;
  }

  chunk->entries = calloc(chunkSize, entrySize);
  if (chunk->entries == nullptr) {
    free(chunk);
    return nullptr;
  }

  chunk->chunkSize = chunkSize;
  chunk->next = nullptr;
  return chunk;
}


/*
 * Description:
 *   Initialize collective info pool with first chunk.
 *
 * Parameters:
 *
 *   strideSize - Number of entries to allocate in the initial chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
static inspectorResult_t initCollectivePool(uint32_t strideSize) {
  g_eventPool.collStrideSize = strideSize;
  g_eventPool.collTotalSize = 0;
  g_eventPool.collAllocCount = 0;
  g_eventPool.collChunkCount = 0;
  g_eventPool.collChunkList = nullptr;
  g_eventPool.collFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.collPoolLock, nullptr) != 0) {
    return inspectorLockError;
  }

  struct inspectorPoolChunk* chunk
    = allocatePoolChunk(sizeof(struct inspectorCollInfoPoolEntry),
                        strideSize);
  if (chunk == nullptr) {
    INFO_INSPECTOR(
      "NCCL Inspector: Failed to allocate initial collective info pool chunk");
    pthread_mutex_destroy(&g_eventPool.collPoolLock);
    return inspectorMemoryError;
  }

  g_eventPool.collChunkList = chunk;
  g_eventPool.collChunkCount = 1;
  g_eventPool.collTotalSize = strideSize;

  struct inspectorCollInfoPoolEntry* entries
    = (struct inspectorCollInfoPoolEntry*)chunk->entries;
  g_eventPool.collFreeList = &entries[0];
  for (uint32_t i = 0; i < strideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[strideSize - 1].next = nullptr;
  entries[strideSize - 1].inUse = false;

  INFO_INSPECTOR(
    "NCCL Inspector: Initialized collective pool with stride size %u",
    strideSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Grow collective pool by allocating a new chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorMemoryError - Failed to allocate new chunk.
 *
 */
static inspectorResult_t growCollectivePool() {
  struct inspectorPoolChunk* newChunk
    = allocatePoolChunk(sizeof(struct inspectorCollInfoPoolEntry),
                        g_eventPool.collStrideSize);

  if (newChunk == nullptr) {
    WARN_INSPECTOR("NCCL Inspector: Failed to grow collective info pool (current chunks: %u, total entries: %u)",
                   g_eventPool.collChunkCount, g_eventPool.collTotalSize);
    return inspectorMemoryError;
  }

  newChunk->next = g_eventPool.collChunkList;
  g_eventPool.collChunkList = newChunk;
  g_eventPool.collChunkCount++;
  g_eventPool.collTotalSize += g_eventPool.collStrideSize;

  struct inspectorCollInfoPoolEntry* entries
    = (struct inspectorCollInfoPoolEntry*)newChunk->entries;
  for (uint32_t i = 0; i < g_eventPool.collStrideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[g_eventPool.collStrideSize - 1].next = g_eventPool.collFreeList;
  entries[g_eventPool.collStrideSize - 1].inUse = false;
  g_eventPool.collFreeList = &entries[0];

  INFO_INSPECTOR(
    "NCCL Inspector: Grew collective pool to %u chunks (%u total entries)",
    g_eventPool.collChunkCount, g_eventPool.collTotalSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Initialize P2P info pool with first chunk.
 *
 * Parameters:
 *
 *   strideSize - Number of entries to allocate in the initial chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
static inspectorResult_t initP2pPool(uint32_t strideSize) {
  g_eventPool.p2pStrideSize = strideSize;
  g_eventPool.p2pTotalSize = 0;
  g_eventPool.p2pAllocCount = 0;
  g_eventPool.p2pChunkCount = 0;
  g_eventPool.p2pChunkList = nullptr;
  g_eventPool.p2pFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.p2pPoolLock, nullptr) != 0) {
    return inspectorLockError;
  }

  struct inspectorPoolChunk* chunk
    = allocatePoolChunk(sizeof(struct inspectorP2pInfoPoolEntry),
                        strideSize);
  if (chunk == nullptr) {
    INFO_INSPECTOR(
      "NCCL Inspector: Failed to allocate initial P2P info pool chunk");
    pthread_mutex_destroy(&g_eventPool.p2pPoolLock);
    return inspectorMemoryError;
  }

  g_eventPool.p2pChunkList = chunk;
  g_eventPool.p2pChunkCount = 1;
  g_eventPool.p2pTotalSize = strideSize;

  struct inspectorP2pInfoPoolEntry* entries
    = (struct inspectorP2pInfoPoolEntry*)chunk->entries;
  g_eventPool.p2pFreeList = &entries[0];
  for (uint32_t i = 0; i < strideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[strideSize - 1].next = nullptr;
  entries[strideSize - 1].inUse = false;

  INFO_INSPECTOR(
    "NCCL Inspector: Initialized P2P pool with stride size %u",
    strideSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Grow P2P pool by allocating a new chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorMemoryError - Failed to allocate new chunk.
 *
 */
static inspectorResult_t growP2pPool() {
  struct inspectorPoolChunk* newChunk
    = allocatePoolChunk(sizeof(struct inspectorP2pInfoPoolEntry),
                        g_eventPool.p2pStrideSize);

  if (newChunk == nullptr) {
    WARN_INSPECTOR("NCCL Inspector: Failed to grow P2P info pool (current chunks: %u, total entries: %u)",
                   g_eventPool.p2pChunkCount, g_eventPool.p2pTotalSize);
    return inspectorMemoryError;
  }

  newChunk->next = g_eventPool.p2pChunkList;
  g_eventPool.p2pChunkList = newChunk;
  g_eventPool.p2pChunkCount++;
  g_eventPool.p2pTotalSize += g_eventPool.p2pStrideSize;

  struct inspectorP2pInfoPoolEntry* entries
    = (struct inspectorP2pInfoPoolEntry*)newChunk->entries;
  for (uint32_t i = 0; i < g_eventPool.p2pStrideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[g_eventPool.p2pStrideSize - 1].next = g_eventPool.p2pFreeList;
  entries[g_eventPool.p2pStrideSize - 1].inUse = false;
  g_eventPool.p2pFreeList = &entries[0];

  INFO_INSPECTOR( "NCCL Inspector: Grew P2P pool to %u chunks (%u total entries)",
                  g_eventPool.p2pChunkCount, g_eventPool.p2pTotalSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Initialize the fixed-capacity ProxyOp info pool.
 *
 * Parameters:
 *
 *   proxyOpPoolSize - Maximum number of concurrently active ProxyOps.
 *
 * Return:
 *
 *   inspectorSuccess     - Success.
 *   inspectorMemoryError - Invalid capacity or failed allocation.
 *   inspectorLockError   - Failed to initialize the pool mutex.
 */
static inspectorResult_t initProxyOpPool(uint32_t proxyOpPoolSize) {
  if (g_eventPool.collChunkList == nullptr) {
    return inspectorUninitializedError;
  }
  if (g_eventPool.proxyOpChunkList != nullptr) {
    return inspectorSuccess;
  }
  if (proxyOpPoolSize == 0) {
    return inspectorMemoryError;
  }

  g_eventPool.proxyOpCapacity = proxyOpPoolSize;
  g_eventPool.proxyOpAllocCount = 0;
  g_eventPool.proxyOpChunkList = nullptr;
  g_eventPool.proxyOpFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.proxyOpPoolLock, nullptr) != 0) {
    g_eventPool.proxyOpCapacity = 0;
    return inspectorLockError;
  }

  struct inspectorPoolChunk* chunk = allocatePoolChunk(
    sizeof(struct inspectorProxyOpInfoPoolEntry), proxyOpPoolSize);
  if (chunk == nullptr) {
    pthread_mutex_destroy(&g_eventPool.proxyOpPoolLock);
    g_eventPool.proxyOpCapacity = 0;
    return inspectorMemoryError;
  }

  g_eventPool.proxyOpChunkList = chunk;

  struct inspectorProxyOpInfoPoolEntry* entries
    = (struct inspectorProxyOpInfoPoolEntry*)chunk->entries;
  g_eventPool.proxyOpFreeList = &entries[0];
  for (uint32_t i = 0; i < proxyOpPoolSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[proxyOpPoolSize - 1].next = nullptr;
  entries[proxyOpPoolSize - 1].inUse = false;

  INFO_INSPECTOR(
    "NCCL Inspector: Initialized fixed-capacity ProxyOp pool with %u entries",
    proxyOpPoolSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Initialize the fixed-capacity ProxyStep info pool.
 *
 * Parameters:
 *
 *   proxyStepPoolSize - Maximum number of concurrently active ProxySteps.
 *
 * Return:
 *
 *   inspectorSuccess     - Success.
 *   inspectorMemoryError - Invalid capacity or failed allocation.
 *   inspectorLockError   - Failed to initialize the pool mutex.
 */
static inspectorResult_t initProxyStepPool(uint32_t proxyStepPoolSize) {
  if (g_eventPool.proxyOpChunkList == nullptr) {
    return inspectorUninitializedError;
  }
  if (g_eventPool.proxyStepChunkList != nullptr) {
    return inspectorSuccess;
  }
  if (proxyStepPoolSize == 0) {
    return inspectorMemoryError;
  }

  g_eventPool.proxyStepCapacity = proxyStepPoolSize;
  g_eventPool.proxyStepAllocCount = 0;
  g_eventPool.proxyStepChunkList = nullptr;
  g_eventPool.proxyStepFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.proxyStepPoolLock, nullptr) != 0) {
    g_eventPool.proxyStepCapacity = 0;
    return inspectorLockError;
  }

  struct inspectorPoolChunk* chunk = allocatePoolChunk(
    sizeof(struct inspectorProxyStepInfoPoolEntry), proxyStepPoolSize);
  if (chunk == nullptr) {
    pthread_mutex_destroy(&g_eventPool.proxyStepPoolLock);
    g_eventPool.proxyStepCapacity = 0;
    return inspectorMemoryError;
  }

  g_eventPool.proxyStepChunkList = chunk;

  struct inspectorProxyStepInfoPoolEntry* entries
    = (struct inspectorProxyStepInfoPoolEntry*)chunk->entries;
  g_eventPool.proxyStepFreeList = &entries[0];
  for (uint32_t i = 0; i < proxyStepPoolSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[proxyStepPoolSize - 1].next = nullptr;
  entries[proxyStepPoolSize - 1].inUse = false;

  INFO_INSPECTOR(
    "NCCL Inspector: Initialized fixed-capacity ProxyStep pool with %u entries",
    proxyStepPoolSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Initialize comm info pool with first chunk.
 *
 * Parameters:
 *
 *   strideSize - Number of entries to allocate in the initial chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
static inspectorResult_t initCommPool(uint32_t strideSize) {
  g_eventPool.commStrideSize = strideSize;
  g_eventPool.commTotalSize = 0;
  g_eventPool.commAllocCount = 0;
  g_eventPool.commChunkCount = 0;
  g_eventPool.commChunkList = nullptr;
  g_eventPool.commFreeList = nullptr;

  if (pthread_mutex_init(&g_eventPool.commPoolLock, nullptr) != 0) {
    return inspectorLockError;
  }

  // Allocate first chunk
  struct inspectorPoolChunk* chunk = allocatePoolChunk(sizeof(struct inspectorCommInfoPoolEntry), strideSize);
  if (chunk == nullptr) {
    INFO_INSPECTOR( "NCCL Inspector: Failed to allocate initial comm info pool chunk");
    pthread_mutex_destroy(&g_eventPool.commPoolLock);
    return inspectorMemoryError;
  }

  g_eventPool.commChunkList = chunk;
  g_eventPool.commChunkCount = 1;
  g_eventPool.commTotalSize = strideSize;

  // Initialize free list for this chunk
  struct inspectorCommInfoPoolEntry* entries = (struct inspectorCommInfoPoolEntry*)chunk->entries;
  g_eventPool.commFreeList = &entries[0];
  for (uint32_t i = 0; i < strideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[strideSize - 1].next = nullptr;
  entries[strideSize - 1].inUse = false;

  INFO_INSPECTOR( "NCCL Inspector: Initialized comm pool with stride size %u", strideSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Grow comm pool by allocating a new chunk.
 *
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorMemoryError - Failed to allocate new chunk.
 *
 */
static inspectorResult_t growCommPool() {
  struct inspectorPoolChunk* newChunk = allocatePoolChunk(
    sizeof(struct inspectorCommInfoPoolEntry),
    g_eventPool.commStrideSize);

  if (newChunk == nullptr) {
    WARN_INSPECTOR("NCCL Inspector: Failed to grow comm info pool (current chunks: %u, total entries: %u)",
                   g_eventPool.commChunkCount, g_eventPool.commTotalSize);
    return inspectorMemoryError;
  }

  // Add chunk to the list
  newChunk->next = g_eventPool.commChunkList;
  g_eventPool.commChunkList = newChunk;
  g_eventPool.commChunkCount++;
  g_eventPool.commTotalSize += g_eventPool.commStrideSize;

  // Add entries from new chunk to free list
  struct inspectorCommInfoPoolEntry* entries = (struct inspectorCommInfoPoolEntry*)newChunk->entries;
  for (uint32_t i = 0; i < g_eventPool.commStrideSize - 1; i++) {
    entries[i].next = &entries[i + 1];
    entries[i].inUse = false;
  }
  entries[g_eventPool.commStrideSize - 1].next = g_eventPool.commFreeList;
  entries[g_eventPool.commStrideSize - 1].inUse = false;
  g_eventPool.commFreeList = &entries[0];

  INFO_INSPECTOR( "NCCL Inspector: Grew comm pool to %u chunks (%u total entries)",
                  g_eventPool.commChunkCount, g_eventPool.commTotalSize);
  return inspectorSuccess;
}

/*
 * Description:
 *   Free all chunks in a chunk list.
 *
 * Parameters:
 *
 *   chunkList - Head of the chunk list to free.
 *
 * Return:
 *
 *   None.
 *
 */
static void freeChunkList(struct inspectorPoolChunk* chunkList) {
  struct inspectorPoolChunk* chunk = chunkList;
  while (chunk != nullptr) {
    struct inspectorPoolChunk* next = chunk->next;
    free(chunk->entries);
    free(chunk);
    chunk = next;
  }
}

/*
 * Description:
 *   Cleanup all pools and destroy their associated locks.
 *
 * Return:
 *
 *   None.
 *
 */
static void cleanupPartialPoolInit() {
  if (g_eventPool.collChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.collPoolLock);
    freeChunkList(g_eventPool.collChunkList);
    g_eventPool.collChunkList = nullptr;
  }
  if (g_eventPool.p2pChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.p2pPoolLock);
    freeChunkList(g_eventPool.p2pChunkList);
    g_eventPool.p2pChunkList = nullptr;
  }
  if (g_eventPool.proxyStepChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.proxyStepPoolLock);
    freeChunkList(g_eventPool.proxyStepChunkList);
    g_eventPool.proxyStepChunkList = nullptr;
    g_eventPool.proxyStepFreeList = nullptr;
    g_eventPool.proxyStepCapacity = 0;
    g_eventPool.proxyStepAllocCount = 0;
  }
  if (g_eventPool.proxyOpChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.proxyOpPoolLock);
    freeChunkList(g_eventPool.proxyOpChunkList);
    g_eventPool.proxyOpChunkList = nullptr;
    g_eventPool.proxyOpFreeList = nullptr;
    g_eventPool.proxyOpCapacity = 0;
    g_eventPool.proxyOpAllocCount = 0;
  }
  if (g_eventPool.commChunkList != nullptr) {
    pthread_mutex_destroy(&g_eventPool.commPoolLock);
    freeChunkList(g_eventPool.commChunkList);
    g_eventPool.commChunkList = nullptr;
  }
}

/*
 * Description:
 *   Initialize all event pools with specified sizes.
 *
 * Parameters:
 *
 *   config - Sizes and enablement settings for all event pools.
 * Return:
 *
 *   inspectorSuccess    - Success.
 *   inspectorLockError  - Failed to initialize mutex.
 *   inspectorMemoryError - Failed to allocate initial chunk.
 *
 */
inspectorResult_t inspectorEventPoolInit(
    const struct inspectorEventPoolConfig& config) {
  inspectorResult_t res;

  memset(&g_eventPool, 0, sizeof(struct inspectorEventPool));

  g_eventPool.growEnabled = config.growEnabled;

  res = initCollectivePool(config.collPoolSize);
  if (res != inspectorSuccess) {
    cleanupPartialPoolInit();
    return res;
  }

  res = initP2pPool(config.p2pPoolSize);
  if (res != inspectorSuccess) {
    cleanupPartialPoolInit();
    return res;
  }

  res = initCommPool(config.commPoolSize);
  if (res != inspectorSuccess) {
    cleanupPartialPoolInit();
    return res;
  }

  if (config.enableProxy) {
    res = initProxyOpPool(config.proxyOpPoolSize);
    if (res != inspectorSuccess) {
      cleanupPartialPoolInit();
      return res;
    }

    res = initProxyStepPool(config.proxyStepPoolSize);
    if (res != inspectorSuccess) {
      cleanupPartialPoolInit();
      return res;
    }
  }

  INFO_INSPECTOR(
    "NCCL Inspector: Memory pools initialized (stride-based) - Coll: %u, P2P: %u, Comms: %u, pool grow: %s",
    config.collPoolSize, config.p2pPoolSize, config.commPoolSize,
    g_eventPool.growEnabled ? "enabled" : "disabled");

  return inspectorSuccess;
}

/*
 * Description:
 *   Finalize and cleanup all event pools.
 *
 * Return:
 *
 *   inspectorSuccess - Success.
 *
 */
inspectorResult_t inspectorEventPoolFinalize() {
  cleanupPartialPoolInit();
  return inspectorSuccess;
}

/*
 * Description:
 *   Allocate a collective info object from the pool. Grows the pool if
 *   necessary.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorCollInfo* - Allocated collective info, or nullptr on
 *                                failure.
 *
 */
struct inspectorCollInfo* inspectorEventPoolAllocColl() {
  pthread_mutex_lock(&g_eventPool.collPoolLock);

  // If free list is empty, try to grow the pool
  if (g_eventPool.collFreeList == nullptr) {
    if (!g_eventPool.growEnabled) {
      pthread_mutex_unlock(&g_eventPool.collPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Collective pool exhausted and pool grow is disabled (NCCL_INSPECTOR_POOL_GROW=0) - allocation failed!");
      return nullptr;
    }
    INFO_INSPECTOR(
      "NCCL Inspector: Collective pool exhausted, growing pool (current: %u chunks, %u entries)",
      g_eventPool.collChunkCount, g_eventPool.collTotalSize);

    inspectorResult_t res = growCollectivePool();
    if (res != inspectorSuccess) {
      pthread_mutex_unlock(&g_eventPool.collPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Failed to grow collective pool - allocation failed!");
      return nullptr;
    }
  }

  struct inspectorCollInfoPoolEntry* entry = g_eventPool.collFreeList;
  g_eventPool.collFreeList = entry->next;
  entry->inUse = true;
  entry->next = nullptr;
  g_eventPool.collAllocCount++;

  pthread_mutex_unlock(&g_eventPool.collPoolLock);

  memset(&entry->obj, 0, sizeof(struct inspectorCollInfo));

  return &entry->obj;
}

/*
 * Description:
 *   Allocate a P2P info object from the pool. Grows the pool if
 *   necessary.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorP2pInfo* - Allocated P2P info, or nullptr on failure.
 *
 */
struct inspectorP2pInfo* inspectorEventPoolAllocP2p() {
  pthread_mutex_lock(&g_eventPool.p2pPoolLock);

  // If free list is empty, try to grow the pool
  if (g_eventPool.p2pFreeList == nullptr) {
    if (!g_eventPool.growEnabled) {
      pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
      WARN_INSPECTOR("NCCL Inspector: P2P pool exhausted and pool grow is disabled (NCCL_INSPECTOR_POOL_GROW=0) - allocation failed!");
      return nullptr;
    }
    INFO_INSPECTOR( "NCCL Inspector: P2P pool exhausted, growing pool (current: %u chunks, %u entries)",
                    g_eventPool.p2pChunkCount, g_eventPool.p2pTotalSize);

    inspectorResult_t res = growP2pPool();
    if (res != inspectorSuccess) {
      pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Failed to grow P2P pool - allocation failed!");
      return nullptr;
    }
  }

  struct inspectorP2pInfoPoolEntry* entry = g_eventPool.p2pFreeList;
  g_eventPool.p2pFreeList = entry->next;
  entry->inUse = true;
  entry->next = nullptr;
  g_eventPool.p2pAllocCount++;

  pthread_mutex_unlock(&g_eventPool.p2pPoolLock);

  // Clear the structure before returning
  memset(&entry->obj, 0, sizeof(struct inspectorP2pInfo));

  return &entry->obj;
}

/*
 * Description:
 *   Allocate a ProxyOp info object from its fixed-capacity pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorProxyOpInfo* - Allocated ProxyOp info, or nullptr when
 *                                  the pool is exhausted.
 */
struct inspectorProxyOpInfo* inspectorEventPoolAllocProxyOp() {
  if (g_eventPool.proxyOpChunkList == nullptr) {
    return nullptr;
  }

  pthread_mutex_lock(&g_eventPool.proxyOpPoolLock);

  if (g_eventPool.proxyOpFreeList == nullptr) {
    uint32_t allocated = g_eventPool.proxyOpAllocCount;
    uint32_t capacity = g_eventPool.proxyOpCapacity;
    pthread_mutex_unlock(&g_eventPool.proxyOpPoolLock);
    static bool warned = false;
    if (!__atomic_exchange_n(&warned, true, __ATOMIC_RELAXED)) {
      WARN_INSPECTOR("NCCL Inspector: ProxyOp pool exhausted (%u/%u allocated); "
                     "failed Ops are counted in dump_stats. Further exhaustion messages are suppressed.",
                     allocated, capacity);
    }
    return nullptr;
  }

  struct inspectorProxyOpInfoPoolEntry* entry
    = g_eventPool.proxyOpFreeList;
  g_eventPool.proxyOpFreeList = entry->next;
  entry->inUse = true;
  entry->next = nullptr;
  g_eventPool.proxyOpAllocCount++;

  pthread_mutex_unlock(&g_eventPool.proxyOpPoolLock);

  memset(&entry->obj, 0, sizeof(struct inspectorProxyOpInfo));
  return &entry->obj;
}

/*
 * Description:
 *   Allocate a ProxyStep info object from its fixed-capacity pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorProxyStepInfo* - Allocated ProxyStep info, or nullptr
 *                                    when the pool is exhausted.
 */
struct inspectorProxyStepInfo* inspectorEventPoolAllocProxyStep() {
  if (g_eventPool.proxyStepChunkList == nullptr) {
    return nullptr;
  }

  pthread_mutex_lock(&g_eventPool.proxyStepPoolLock);

  if (g_eventPool.proxyStepFreeList == nullptr) {
    uint32_t allocated = g_eventPool.proxyStepAllocCount;
    uint32_t capacity = g_eventPool.proxyStepCapacity;
    pthread_mutex_unlock(&g_eventPool.proxyStepPoolLock);
    static bool warned = false;
    if (!__atomic_exchange_n(&warned, true, __ATOMIC_RELAXED)) {
      WARN_INSPECTOR("NCCL Inspector: ProxyStep pool exhausted (%u/%u allocated); "
                     "failed Steps are counted in their Op. Further exhaustion messages are suppressed.",
                     allocated, capacity);
    }
    return nullptr;
  }

  struct inspectorProxyStepInfoPoolEntry* entry
    = g_eventPool.proxyStepFreeList;
  g_eventPool.proxyStepFreeList = entry->next;
  entry->inUse = true;
  entry->next = nullptr;
  g_eventPool.proxyStepAllocCount++;

  pthread_mutex_unlock(&g_eventPool.proxyStepPoolLock);

  memset(&entry->obj, 0, sizeof(struct inspectorProxyStepInfo));
  return &entry->obj;
}

/*
 * Description:
 *   Allocate a comm info object from the pool. Grows the pool if
 *   necessary.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Return:
 *
 *   struct inspectorCommInfo* - Allocated comm info, or nullptr on
 *                                failure.
 *
 */
struct inspectorCommInfo* inspectorEventPoolAllocComm() {
  pthread_mutex_lock(&g_eventPool.commPoolLock);

  // If free list is empty, try to grow the pool
  if (g_eventPool.commFreeList == nullptr) {
    if (!g_eventPool.growEnabled) {
      pthread_mutex_unlock(&g_eventPool.commPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Comm pool exhausted and pool grow is disabled (NCCL_INSPECTOR_POOL_GROW=0) - allocation failed!");
      return nullptr;
    }
    INFO_INSPECTOR( "NCCL Inspector: Comm pool exhausted, growing pool (current: %u chunks, %u entries)",
                    g_eventPool.commChunkCount, g_eventPool.commTotalSize);

    inspectorResult_t res = growCommPool();
    if (res != inspectorSuccess) {
      pthread_mutex_unlock(&g_eventPool.commPoolLock);
      WARN_INSPECTOR("NCCL Inspector: Failed to grow comm pool - allocation failed!");
      return nullptr;
    }
  }

  struct inspectorCommInfoPoolEntry* entry = g_eventPool.commFreeList;
  g_eventPool.commFreeList = entry->next;
  entry->inUse = true;
  entry->next = nullptr;
  g_eventPool.commAllocCount++;

  pthread_mutex_unlock(&g_eventPool.commPoolLock);

  // Clear the structure before returning
  entry->obj = inspectorCommInfo{};

  return &entry->obj;
}

/*
 * Description:
 *   Release a collective info object back to the pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   collInfo - Collective info object to release.
 *
 * Return:
 *
 *   None.
 *
 */
void inspectorEventPoolReleaseColl(struct inspectorCollInfo* collInfo) {
  if (collInfo == nullptr) {
    return;
  }

  // Calculate the pool entry address from the object address
  struct inspectorCollInfoPoolEntry* entry =
    (struct inspectorCollInfoPoolEntry*)((char*)collInfo -
                                         offsetof(struct inspectorCollInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.collPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.collPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for collective info!");
    return;
  }

  entry->inUse = false;
  entry->next = g_eventPool.collFreeList;
  g_eventPool.collFreeList = entry;
  g_eventPool.collAllocCount--;

  pthread_mutex_unlock(&g_eventPool.collPoolLock);
}

/*
 * Description:
 *   Release a P2P info object back to the pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   p2pInfo - P2P info object to release.
 *
 * Return:
 *
 *   None.
 *
 */
void inspectorEventPoolReleaseP2p(struct inspectorP2pInfo* p2pInfo) {
  if (p2pInfo == nullptr) {
    return;
  }

  // Calculate the pool entry address from the object address
  struct inspectorP2pInfoPoolEntry* entry =
    (struct inspectorP2pInfoPoolEntry*)((char*)p2pInfo -
                                        offsetof(struct inspectorP2pInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.p2pPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for P2P info!");
    return;
  }

  entry->inUse = false;
  entry->next = g_eventPool.p2pFreeList;
  g_eventPool.p2pFreeList = entry;
  g_eventPool.p2pAllocCount--;

  pthread_mutex_unlock(&g_eventPool.p2pPoolLock);
}

/*
 * Description:
 *   Release a ProxyOp info object back to its fixed-capacity pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   proxyOpInfo - ProxyOp info object to release.
 *
 * Return:
 *
 *   None.
 */
void inspectorEventPoolReleaseProxyOp(struct inspectorProxyOpInfo* proxyOpInfo) {
  if (proxyOpInfo == nullptr || g_eventPool.proxyOpChunkList == nullptr) {
    return;
  }

  struct inspectorProxyOpInfoPoolEntry* entry
    = (struct inspectorProxyOpInfoPoolEntry*)((char*)proxyOpInfo -
        offsetof(struct inspectorProxyOpInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.proxyOpPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.proxyOpPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for ProxyOp info!");
    return;
  }

  entry->inUse = false;
  entry->next = g_eventPool.proxyOpFreeList;
  g_eventPool.proxyOpFreeList = entry;
  g_eventPool.proxyOpAllocCount--;

  pthread_mutex_unlock(&g_eventPool.proxyOpPoolLock);
}

/*
 * Description:
 *   Release a ProxyStep info object back to its fixed-capacity pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   proxyStepInfo - ProxyStep info object to release.
 *
 * Return:
 *
 *   None.
 */
void inspectorEventPoolReleaseProxyStep(
    struct inspectorProxyStepInfo* proxyStepInfo) {
  if (proxyStepInfo == nullptr || g_eventPool.proxyStepChunkList == nullptr) {
    return;
  }

  struct inspectorProxyStepInfoPoolEntry* entry
    = (struct inspectorProxyStepInfoPoolEntry*)((char*)proxyStepInfo -
        offsetof(struct inspectorProxyStepInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.proxyStepPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.proxyStepPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for ProxyStep info!");
    return;
  }

  entry->inUse = false;
  entry->next = g_eventPool.proxyStepFreeList;
  g_eventPool.proxyStepFreeList = entry;
  g_eventPool.proxyStepAllocCount--;

  pthread_mutex_unlock(&g_eventPool.proxyStepPoolLock);
}

/*
 * Description:
 *   Release a comm info object back to the pool.
 *
 * Thread Safety:
 *
 *   Thread-safe.
 *
 * Parameters:
 *
 *   commInfo - Comm info object to release.
 *
 * Return:
 *
 *   None.
 *
 */
void inspectorEventPoolReleaseComm(struct inspectorCommInfo* commInfo) {
  if (commInfo == nullptr) {
    return;
  }

  // Calculate the pool entry address from the object address
  struct inspectorCommInfoPoolEntry* entry =
    (struct inspectorCommInfoPoolEntry*)((char*)commInfo -
                                         offsetof(struct inspectorCommInfoPoolEntry, obj));

  pthread_mutex_lock(&g_eventPool.commPoolLock);

  if (!entry->inUse) {
    pthread_mutex_unlock(&g_eventPool.commPoolLock);
    WARN_INSPECTOR("NCCL Inspector: Double release detected for comm info!");
    return;
  }

  entry->inUse = false;
  entry->next = g_eventPool.commFreeList;
  g_eventPool.commFreeList = entry;
  g_eventPool.commAllocCount--;

  pthread_mutex_unlock(&g_eventPool.commPoolLock);
}
