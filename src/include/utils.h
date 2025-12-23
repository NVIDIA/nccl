/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include "nccl.h"
#include "alloc.h"
#include "bitops.h"
#include "checks.h"
#include "compiler.h"
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <algorithm>
#include <new>
#include <type_traits>
#include <mutex>
#include <condition_variable>

int ncclCudaCompCap();

// PCI Bus ID <-> int64 conversion functions
ncclResult_t int64ToBusId(int64_t id, char* busId);
ncclResult_t busIdToInt64(const char* busId, int64_t* id);

ncclResult_t getBusId(int cudaDev, int64_t *busId);

ncclResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHostHash();
uint64_t getPidHash();
uint64_t hashCombine(uint64_t baseHash, uint64_t value);
ncclResult_t getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

static long log2i(long n) {
  return log2Down(n);
}

// Comparator function for qsort/bsearch to compare integers
static int compareInts(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

inline uint64_t clockNano() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec)*1000*1000*1000 + ts.tv_nsec;
}

/* get any bytes of random data from /dev/urandom, return ncclSuccess (0) if it succeeds. */
inline ncclResult_t getRandomData(void* buffer, size_t bytes) {
  ncclResult_t ret = ncclSuccess;
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE* fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) ret = ncclSystemError;
    if (fp) fclose(fp);
  }
  return ret;
}

static inline int gcd(int a, int b) {
  // use the euclidian algorithm
  while (b != 0) {
    int temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

////////////////////////////////////////////////////////////////////////////////

template<typename Int>
inline void ncclAtomicRefCountIncrement(Int* refs) {
  COMPILER_ATOMIC_FETCH_ADD(refs, 1, std::memory_order_relaxed);
}

template<typename Int>
inline Int ncclAtomicRefCountDecrement(Int* refs) {
  return COMPILER_ATOMIC_SUB_FETCH(refs, 1, std::memory_order_acq_rel);
}

////////////////////////////////////////////////////////////////////////////////
/* ncclMemoryStack: Pools memory for fast LIFO ordered allocation. Note that
 * granularity of LIFO is not per object, instead frames containing many objects
 * are pushed and popped. Therefor deallocation is extremely cheap since its
 * done at the frame granularity.
 *
 * The initial state of the stack is with one frame, the "nil" frame, which
 * cannot be popped. Therefor objects allocated in the nil frame cannot be
 * deallocated sooner than stack destruction.
 */
struct ncclMemoryStack;

void ncclMemoryStackConstruct(struct ncclMemoryStack* me);
void ncclMemoryStackDestruct(struct ncclMemoryStack* me);
void ncclMemoryStackPush(struct ncclMemoryStack* me);
void ncclMemoryStackPop(struct ncclMemoryStack* me);
void* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t size, size_t align);
template<typename T>
T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n=1);
template<typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt);

////////////////////////////////////////////////////////////////////////////////
/* ncclMemoryPool: A free-list of same-sized allocations. It is an invalid for
 * a pool instance to ever hold objects whose type have differing
 * (sizeof(T), alignof(T)) pairs. The underlying memory is supplied by
 * a backing `ncclMemoryStack` passed during Alloc(). If memory
 * backing any currently held object is deallocated then it is an error to do
 * anything other than reconstruct it, after which it is a valid empty pool.
 */
struct ncclMemoryPool;

// Equivalent to zero-initialization
void ncclMemoryPoolConstruct(struct ncclMemoryPool* me);
template<typename T>
T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing);
template<typename T>
void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj);
void ncclMemoryPoolTakeAll(struct ncclMemoryPool* me, struct ncclMemoryPool* from);

////////////////////////////////////////////////////////////////////////////////
/* ncclIntruQueue: A singly-linked list queue where the per-object next pointer
 * field is given via the `next` template argument.
 *
 * Example:
 *   struct Foo {
 *     struct Foo *next1, *next2; // can be a member of two lists at once
 *   };
 *   ncclIntruQueue<Foo, &Foo::next1> list1;
 *   ncclIntruQueue<Foo, &Foo::next2> list2;
 */
template<typename T, T *T::*next>
struct ncclIntruQueue;

template<typename T, T *T::*next>
void ncclIntruQueueConstruct(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
bool ncclIntruQueueEmpty(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
T* ncclIntruQueueHead(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
void ncclIntruQueueEnqueue(ncclIntruQueue<T,next> *me, T *x);
template<typename T, T *T::*next>
void ncclIntruQueueEnqueueFront(ncclIntruQueue<T,next> *me, T *x);
template<typename T, T *T::*next>
T* ncclIntruQueueDequeue(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
T* ncclIntruQueueTryDequeue(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
void ncclIntruQueueTransfer(ncclIntruQueue<T,next> *dst, ncclIntruQueue<T,next> *src);


////////////////////////////////////////////////////////////////////////////////
/* ncclThreadSignal: Couples a std::mutex and std::condition_variable together. The "mutex"
 * and "cond" fields are part of the public interface.
 */
struct ncclThreadSignal {
  std::mutex mutex;
  std::condition_variable cond;
};

// A convenience instance per-thread.
extern thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance;

////////////////////////////////////////////////////////////////////////////////

template<typename T, T *T::*next>
struct ncclIntruQueueMpsc;

template<typename T, T *T::*next>
void ncclIntruQueueMpscConstruct(struct ncclIntruQueueMpsc<T,next>* me);
template<typename T, T *T::*next>
bool ncclIntruQueueMpscEmpty(struct ncclIntruQueueMpsc<T,next>* me);
// Enqueue element. Returns true if queue is not abandoned. Even if queue is
// abandoned the element enqueued, so the caller needs to make arrangements for
// the queue to be tended.
template<typename T, T *T::*next>
bool ncclIntruQueueMpscEnqueue(struct ncclIntruQueueMpsc<T,next>* me, T* x);
// Dequeue all elements at a glance. If there aren't any and `waitSome` is
// true then this call will wait until it can return a non empty list.
template<typename T, T *T::*next>
T* ncclIntruQueueMpscDequeueAll(struct ncclIntruQueueMpsc<T,next>* me, bool waitSome);
// Dequeue all elements and set queue to abandoned state.
template<typename T, T *T::*next>
T* ncclIntruQueueMpscAbandon(struct ncclIntruQueueMpsc<T,next>* me);

////////////////////////////////////////////////////////////////////////////////

struct ncclMemoryStack {
  struct Hunk {
    struct Hunk* above; // reverse stack pointer
    size_t size; // size of this allocation (including this header struct)
  };
  struct Unhunk { // proxy header for objects allocated out-of-hunk
    struct Unhunk* next;
    void* obj;
  };
  struct Frame {
    struct Hunk* hunk; // top of non-empty hunks
    uintptr_t bumper, end; // points into top hunk
    struct Unhunk* unhunks;
    struct Frame* below;
  };

  static void* allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align);
  static void* allocate(struct ncclMemoryStack* me, size_t size, size_t align);

  struct Hunk stub;
  struct Frame topFrame;
};

inline void ncclMemoryStackConstruct(struct ncclMemoryStack* me) {
  me->stub.above = nullptr;
  me->stub.size = 0;
  me->topFrame.hunk = &me->stub;
  me->topFrame.bumper = 0;
  me->topFrame.end = 0;
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = nullptr;
}

inline void* ncclMemoryStack::allocate(struct ncclMemoryStack* me, size_t size, size_t align) {
  uintptr_t o = (me->topFrame.bumper + align-1) & -uintptr_t(align);
  void* obj;
  if (COMPILER_EXPECT(o + size <= me->topFrame.end, true)) {
    me->topFrame.bumper = o + size;
    obj = reinterpret_cast<void*>(o);
  } else {
    obj = allocateSpilled(me, size, align);
  }
  return obj;
}

inline void* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t size, size_t align) {
  void *obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return obj;
}

template<typename T>
inline T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n) {
  void *obj = ncclMemoryStack::allocate(me, n*sizeof(T), alignof(T));
  memset(obj, 0, n*sizeof(T));
  return (T*)obj;
}

template<typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt) {
  size_t size = sizeof(Header);
  size = (size + alignof(Element)-1) & -alignof(Element);
  size += nElt*sizeof(Element);
  size_t align = alignof(Header) < alignof(Element) ? alignof(Element) : alignof(Header);
  void *obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return (Header*)obj;
}

inline void ncclMemoryStackPush(struct ncclMemoryStack* me) {
  using Frame = ncclMemoryStack::Frame;
  Frame tmp = me->topFrame;
  Frame* snapshot = (Frame*)ncclMemoryStack::allocate(me, sizeof(Frame), alignof(Frame));
  *snapshot = tmp; // C++ struct assignment
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = snapshot;
}

inline void ncclMemoryStackPop(struct ncclMemoryStack* me) {
  ncclMemoryStack::Unhunk* un = me->topFrame.unhunks;
  while (un != nullptr) {
    free(un->obj);
    un = un->next;
  }
  me->topFrame = *me->topFrame.below; // C++ struct assignment
}


////////////////////////////////////////////////////////////////////////////////

struct ncclMemoryPool {
  struct Cell {
    Cell *next;
  };
  struct Cell* head;
  struct Cell* tail; // meaningful only when head != nullptr
};

inline void ncclMemoryPoolConstruct(struct ncclMemoryPool* me) {
  me->head = nullptr;
}

template<typename T>
inline T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell;
  if (COMPILER_EXPECT(me->head != nullptr, true)) {
    cell = me->head;
    me->head = cell->next;
  } else {
    // Use the internal allocate() since it doesn't memset to 0 yet.
    size_t cellSize = std::max(sizeof(Cell), sizeof(T));
    size_t cellAlign = std::max(alignof(Cell), alignof(T));
    cell = (Cell*)ncclMemoryStack::allocate(backing, cellSize, cellAlign);
  }
  memset(cell, 0, sizeof(T));
  return reinterpret_cast<T*>(cell);
}

template<typename T>
inline void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell = reinterpret_cast<Cell*>(obj);
  cell->next = me->head;
  if (me->head == nullptr) me->tail = cell;
  me->head = cell;
}

inline void ncclMemoryPoolTakeAll(struct ncclMemoryPool* me, struct ncclMemoryPool* from) {
  if (from->head != nullptr) {
    from->tail->next = me->head;
    if (me->head == nullptr) me->tail = from->tail;
    me->head = from->head;
    from->head = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, T *T::*next>
struct ncclIntruQueue {
  T *head, *tail;
};

template<typename T, T *T::*next>
inline void ncclIntruQueueConstruct(ncclIntruQueue<T,next> *me) {
  me->head = nullptr;
  me->tail = nullptr;
}

template<typename T, T *T::*next>
inline bool ncclIntruQueueEmpty(ncclIntruQueue<T,next> *me) {
  return me->head == nullptr;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueHead(ncclIntruQueue<T,next> *me) {
  return me->head;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueTail(ncclIntruQueue<T,next> *me) {
  return me->tail;
}

template<typename T, T *T::*next>
inline void ncclIntruQueueEnqueue(ncclIntruQueue<T,next> *me, T *x) {
  x->*next = nullptr;
  (me->head ? me->tail->*next : me->head) = x;
  me->tail = x;
}

template<typename T, T *T::*next>
inline void ncclIntruQueueEnqueueFront(ncclIntruQueue<T,next> *me, T *x) {
  if (me->head == nullptr) me->tail = x;
  x->*next = me->head;
  me->head = x;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueDequeue(ncclIntruQueue<T,next> *me) {
  T *ans = me->head;
  me->head = ans->*next;
  if (me->head == nullptr) me->tail = nullptr;
  return ans;
}

template<typename T, T *T::*next>
inline bool ncclIntruQueueDelete(ncclIntruQueue<T,next> *me, T *x) {
  T *prev = nullptr;
  T *cur = me->head;
  bool found = false;

  while (cur) {
    if (cur == x) {
      found = true;
      break;
    }
    prev = cur;
    cur = cur->*next;
  }

  if (found) {
    if (prev == nullptr)
      me->head = cur->*next;
    else
      prev->*next = cur->*next;
    if (cur == me->tail)
      me->tail = prev;
  }
  return found;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueTryDequeue(ncclIntruQueue<T,next> *me) {
  T *ans = me->head;
  if (ans != nullptr) {
    me->head = ans->*next;
    if (me->head == nullptr) me->tail = nullptr;
  }
  return ans;
}

template<typename T, T *T::*next>
void ncclIntruQueueTransfer(ncclIntruQueue<T,next> *dst, ncclIntruQueue<T,next> *src) {
  (dst->tail ? dst->tail->next : dst->head) = src->head;
  if (src->tail) dst->tail = src->tail;
  src->head = nullptr;
  src->tail = nullptr;
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, T *T::*next>
struct ncclIntruQueueMpsc {
  T* head;
  uintptr_t tail;
  struct ncclThreadSignal* waiting;
};

template<typename T, T *T::*next>
void ncclIntruQueueMpscConstruct(struct ncclIntruQueueMpsc<T,next>* me) {
  me->head = nullptr;
  me->tail = 0x0;
  me->waiting = nullptr;
}

template<typename T, T *T::*next>
bool ncclIntruQueueMpscEmpty(struct ncclIntruQueueMpsc<T,next>* me) {
  return COMPILER_ATOMIC_LOAD(&me->tail, std::memory_order_relaxed) <= 0x2;
}

template<typename T, T *T::*next>
bool ncclIntruQueueMpscEnqueue(ncclIntruQueueMpsc<T,next>* me, T* x) {
  COMPILER_ATOMIC_STORE(&(x->*next), nullptr, std::memory_order_relaxed);
  uintptr_t utail = COMPILER_ATOMIC_EXCHANGE(&me->tail, reinterpret_cast<uintptr_t>(x), std::memory_order_acq_rel);
  T* prev = reinterpret_cast<T*>(utail);
  T** prevNext = utail <= 0x2 ? &me->head : &(prev->*next);
  COMPILER_ATOMIC_STORE(prevNext, x, std::memory_order_relaxed);
  if (utail == 0x1) { // waiting
    std::atomic_thread_fence(std::memory_order_acquire); // to see me->waiting
    // This lock/unlock is essential to ensure we don't race ahead of the consumer
    // and signal the cond before they begin waiting on it.
    struct ncclThreadSignal* waiting = me->waiting;
    {
      std::unique_lock<std::mutex> lock(waiting->mutex);
    }
    waiting->cond.notify_all();
  }
  return utail != 0x2; // not abandoned
}

template<typename T, T *T::*next>
T* ncclIntruQueueMpscDequeueAll(ncclIntruQueueMpsc<T,next>* me, bool waitSome) {
  T* head = COMPILER_ATOMIC_LOAD(&me->head, std::memory_order_relaxed);
  if (head == nullptr) {
    if (!waitSome) return nullptr;
    uint64_t t0 = clockNano();
    bool sleeping = false;
    do {
      if (clockNano()-t0 >= 10*1000) { // spin for first 10us
        struct ncclThreadSignal* waitSignal = &ncclThreadSignalLocalInstance;
        std::unique_lock<std::mutex> lock(waitSignal->mutex);
        uintptr_t expected = sleeping ? 0x1 : 0x0;
        uintptr_t desired = 0x1;
        me->waiting = waitSignal; // release done by successful compare exchange
        if (COMPILER_ATOMIC_COMPARE_EXCHANGE(&me->tail, &expected, desired, std::memory_order_release, std::memory_order_relaxed)) {
          sleeping = true;
          waitSignal->cond.wait(lock);
        }
      }
      head = COMPILER_ATOMIC_LOAD(&me->head, std::memory_order_relaxed);
    } while (head == nullptr);
  }

  COMPILER_ATOMIC_STORE(&me->head, nullptr, std::memory_order_relaxed);
  uintptr_t utail = COMPILER_ATOMIC_EXCHANGE(&me->tail, 0x0, std::memory_order_acq_rel);
  T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
  T *x = head;
  while (x != tail) {
    T *x1;
    int spins = 0;
    while (true) {
      x1 = COMPILER_ATOMIC_LOAD(&(x->*next), std::memory_order_relaxed);
      if (x1 != nullptr) break;
      if (++spins == 1024) { spins = 1024-1; std::this_thread::yield(); }
    }
    x = x1;
  }
  return head;
}

template<typename T, T *T::*next>
T* ncclIntruQueueMpscAbandon(ncclIntruQueueMpsc<T,next>* me) {
  uintptr_t expected = 0x0;
  if (COMPILER_ATOMIC_COMPARE_EXCHANGE(&me->tail, &expected, /*desired=*/0x2, std::memory_order_relaxed, std::memory_order_relaxed)) {
    return nullptr;
  } else {
    int spins = 0;
    T* head;
    while (true) {
      head = COMPILER_ATOMIC_LOAD(&me->head, std::memory_order_relaxed);
      if (head != nullptr) break;
      if (++spins == 1024) { spins = 1024-1; std::this_thread::yield(); }
    }
    COMPILER_ATOMIC_STORE(&me->head, nullptr, std::memory_order_relaxed);
    uintptr_t utail = COMPILER_ATOMIC_EXCHANGE(&me->tail, 0x2, std::memory_order_acq_rel);
    T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
    T *x = head;
    while (x != tail) {
      T *x1;
      spins = 0;
      while (true) {
        x1 = COMPILER_ATOMIC_LOAD(&(x->*next), std::memory_order_relaxed);
        if (x1 != nullptr) break;
        if (++spins == 1024) { spins = 1024-1; std::this_thread::yield(); }
      }
      x = x1;
    }
    return head;
  }
}

ncclResult_t ncclBitsToString(uint32_t bits, uint32_t mask, const char* (*toStr)(int), char *buf, size_t bufLen, const char *wildcard);

////////////////////////////////////////////////////////////////////////////////
// Hash function for pointer types (shared by address map implementations)
uint64_t ncclHashPointer(int hbits, void* key);

////////////////////////////////////////////////////////////////////////////////
// Intrusive address map implementation (avoids per-entry allocations)

/*
 * ncclIntruAddressMap Usage Contract
 * ===================================
 *
 * OVERVIEW:
 *   - Intrusive map that stores next-pointers directly in user objects
 *   - Avoids separate malloc per entry (only allocates bucket table)
 *   - Uses C++ templates for type safety with type-erased implementation
 *
 * CONSTRUCTION:
 *   - POD type with automatic zero-initialization for static/global instances
 *   - Example (global): static ncclIntruAddressMap<Obj, void*, &Obj::key, &Obj::next> globalMap;
 *   - Example (local): ncclIntruAddressMap<Obj, void*, &Obj::key, &Obj::next> localMap = {};
 *   - Zero-initialization works: ncclIntruAddressMap<...> map = {};
 *
 * DESTRUCTION:
 *   - NO explicit destructor function is provided
 *   - The map automatically cleans up internal memory when the last entry is removed
 *   - When count reaches 0, the internal table is freed and the map returns to
 *     zero-initialized state, making destruction trivial (no-op)
 *
 * USER RESPONSIBILITY:
 *   - Caller MUST remove all inserted objects before abandoning the map
 *   - Failure to remove all objects will leak memory (the internal bucket table)
 *   - Objects must outlive their presence in the map
 *   - Key and next-pointer fields are modified by the map
 *   - Delete/free objects separately after removing from map
 *
 * THREAD SAFETY:
 *   - This data structure is NOT thread-safe
 *   - Caller must provide external synchronization
 *
 * USAGE VALIDATION:
 *   Compile-time checks (via static_assert):
 *   - Key type size must be <= sizeof(uintptr_t)
 *
 *   Runtime checks (returns ncclInvalidUsage with WARN):
 *   - Map pointer must not be NULL
 *   - Object pointer must not be NULL (for Insert/Find operations)
 *   - Key size must be valid (0 < keySize <= sizeof(uintptr_t))
 */

// Untyped internal structure
struct ncclIntruAddressMap_untyped {
  int hbits;  // log2 of table size
  int count;  // number of entries
  void** table;
};

// Typed wrapper (uses composition for C compatibility)
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
struct ncclIntruAddressMap {
  // Compile-time checks for valid usage
  static_assert(sizeof(Key) <= sizeof(uintptr_t),
    "ncclIntruAddressMap: Key type size must be <= sizeof(uintptr_t). "
    "Keys larger than a pointer cannot be safely converted to uintptr_t.");

  ncclIntruAddressMap_untyped base;
};

// Destructor (optional - only needed if entries remain in map)
// Note: Map auto-cleans when last entry is removed, so this is only needed
// if abandoning a non-empty map to avoid leaking the bucket table.
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline void ncclIntruAddressMapDestruct(struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map) {
  if (map->base.table != nullptr) {
    free(map->base.table);
    map->base.table = nullptr;
  }
  map->base.hbits = 0;
  map->base.count = 0;
}

// Internal untyped function prototypes
ncclResult_t ncclIntruAddressMapInsert_untyped(
  struct ncclIntruAddressMap_untyped* map,
  int keySize, int keyFieldOffset, int nextFieldOffset,
  uintptr_t key, void* object);

ncclResult_t ncclIntruAddressMapFind_untyped(
  struct ncclIntruAddressMap_untyped* map,
  int keySize, int keyFieldOffset, int nextFieldOffset,
  uintptr_t key, void** object);

ncclResult_t ncclIntruAddressMapRemove_untyped(
  struct ncclIntruAddressMap_untyped* map,
  int keySize, int keyFieldOffset, int nextFieldOffset,
  uintptr_t key);

// Typed template implementations (type-erasing wrappers)
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline ncclResult_t ncclIntruAddressMapInsert(
    struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
    Key key, Obj* object) {
  Obj dummy;
  // Using offsetof macro would be better except it won't work with non-C types,
  // like those that involve inheritance.
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  return ncclIntruAddressMapInsert_untyped(
    &map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
    reinterpret_cast<uintptr_t>(key), object);
}

template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline ncclResult_t ncclIntruAddressMapFind(
    struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
    Key key, Obj** object) {
  Obj dummy;
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  void* tmp;
  ncclResult_t ret = ncclIntruAddressMapFind_untyped(
    &map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
    reinterpret_cast<uintptr_t>(key), &tmp);
  *object = (Obj*)tmp;
  return ret;
}

template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline ncclResult_t ncclIntruAddressMapRemove(
    struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
    Key key) {
  Obj dummy;
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  return ncclIntruAddressMapRemove_untyped(
    &map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
    reinterpret_cast<uintptr_t>(key));
}

inline ncclResult_t ncclThreadJoin(std::thread& thread) {
  try {
    thread.join();
    return ncclSuccess;
  } catch (const std::exception& e) {
    WARN("Thread join failed: %s", e.what());
    return ncclSystemError;
  }
}

#endif
