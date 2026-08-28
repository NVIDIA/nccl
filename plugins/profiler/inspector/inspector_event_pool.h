#ifndef NCCL_INSPECTOR_EVENT_POOL_H_
#define NCCL_INSPECTOR_EVENT_POOL_H_

#include <pthread.h>
#include <stdint.h>

#include "inspector.h"

// Memory pool entry structures
struct inspectorCollInfoPoolEntry {
  struct inspectorCollInfo obj;
  struct inspectorCollInfoPoolEntry* next;
  bool inUse;
};

struct inspectorP2pInfoPoolEntry {
  struct inspectorP2pInfo obj;
  struct inspectorP2pInfoPoolEntry* next;
  bool inUse;
};

struct inspectorProxyOpInfoPoolEntry {
  struct inspectorProxyOpInfo obj;
  struct inspectorProxyOpInfoPoolEntry* next;
  bool inUse;
};

struct inspectorProxyStepInfoPoolEntry {
  struct inspectorProxyStepInfo obj;
  struct inspectorProxyStepInfoPoolEntry* next;
  bool inUse;
};

struct inspectorCommInfoPoolEntry {
  struct inspectorCommInfo obj;
  struct inspectorCommInfoPoolEntry* next;
  bool inUse;
};

// Chunk structure for stride-based pool growth
struct inspectorPoolChunk {
  void* entries;                    // Pointer to the array of entries in this chunk
  uint32_t chunkSize;               // Number of entries in this chunk
  struct inspectorPoolChunk* next;  // Next chunk in the list
};

struct inspectorEventPool {
  // Collective info pool
  struct inspectorPoolChunk* collChunkList;
  struct inspectorCollInfoPoolEntry* collFreeList;
  uint32_t collStrideSize;
  uint32_t collTotalSize;
  uint32_t collAllocCount;
  uint32_t collChunkCount;
  pthread_mutex_t collPoolLock;

  // P2P info pool
  struct inspectorPoolChunk* p2pChunkList;
  struct inspectorP2pInfoPoolEntry* p2pFreeList;
  uint32_t p2pStrideSize;
  uint32_t p2pTotalSize;
  uint32_t p2pAllocCount;
  uint32_t p2pChunkCount;
  pthread_mutex_t p2pPoolLock;

  // Proxy pools are fixed-capacity because ProxyStep events can be produced
  // at a much higher rate than collective and P2P events.
  struct inspectorPoolChunk* proxyOpChunkList;
  struct inspectorProxyOpInfoPoolEntry* proxyOpFreeList;
  uint32_t proxyOpCapacity;
  uint32_t proxyOpAllocCount;
  pthread_mutex_t proxyOpPoolLock;

  struct inspectorPoolChunk* proxyStepChunkList;
  struct inspectorProxyStepInfoPoolEntry* proxyStepFreeList;
  uint32_t proxyStepCapacity;
  uint32_t proxyStepAllocCount;
  pthread_mutex_t proxyStepPoolLock;

  // Comm info pool (keeping for future extensibility)
  struct inspectorPoolChunk* commChunkList;
  struct inspectorCommInfoPoolEntry* commFreeList;
  uint32_t commStrideSize;
  uint32_t commTotalSize;
  uint32_t commAllocCount;
  uint32_t commChunkCount;
  pthread_mutex_t commPoolLock;

  // Controls whether the Coll, P2P, and Comm pools are allowed to grow beyond
  // their initial size. Proxy pools remain fixed-capacity.
  // Proxy pool storage is allocated up front.
  // Its capacity does not change after initialization.
  // Exhaustion is reported to the caller.
  // Disabled via NCCL_INSPECTOR_POOL_GROW=0.
  bool growEnabled;
};

extern struct inspectorEventPool g_eventPool;

// Memory pool functions
inspectorResult_t inspectorEventPoolInit(uint32_t collPoolSize,
                                        uint32_t p2pPoolSize,
                                        uint32_t commPoolSize);
inspectorResult_t inspectorEventPoolFinalize();

struct inspectorCollInfo* inspectorEventPoolAllocColl();
struct inspectorP2pInfo* inspectorEventPoolAllocP2p();
struct inspectorProxyOpInfo* inspectorEventPoolAllocProxyOp();
struct inspectorProxyStepInfo* inspectorEventPoolAllocProxyStep();
struct inspectorCommInfo* inspectorEventPoolAllocComm();
void inspectorEventPoolReleaseColl(struct inspectorCollInfo* collInfo);
void inspectorEventPoolReleaseP2p(struct inspectorP2pInfo* p2pInfo);
void inspectorEventPoolReleaseProxyOp(struct inspectorProxyOpInfo* proxyOpInfo);
void inspectorEventPoolReleaseProxyStep(struct inspectorProxyStepInfo* proxyStepInfo);
void inspectorEventPoolReleaseComm(struct inspectorCommInfo* commInfo);

#endif // NCCL_INSPECTOR_EVENT_POOL_H_
