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
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <algorithm>
#include <new>
#include <type_traits>

int ncclCudaCompCap();

// PCI Bus ID <-> int64 conversion functions
ncclResult_t int64ToBusId(int64_t id, char* busId);
ncclResult_t busIdToInt64(const char* busId, int64_t* id);

ncclResult_t getBusId(int cudaDev, int64_t *busId);

ncclResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHostHash();
uint64_t getPidHash();
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

////////////////////////////////////////////////////////////////////////////////

template<typename Int>
inline void ncclAtomicRefCountIncrement(Int* refs) {
  __atomic_fetch_add(refs, 1, __ATOMIC_RELAXED);
}

template<typename Int>
inline Int ncclAtomicRefCountDecrement(Int* refs) {
  return __atomic_sub_fetch(refs, 1, __ATOMIC_ACQ_REL);
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
/* ncclThreadSignal: Couples a pthread mutex and cond together. The "mutex"
 * and "cond" fields are part of the public interface.
 */
struct ncclThreadSignal {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

// returns {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER}
constexpr ncclThreadSignal ncclThreadSignalStaticInitializer();

void ncclThreadSignalConstruct(struct ncclThreadSignal* me);
void ncclThreadSignalDestruct(struct ncclThreadSignal* me);

// A convenience instance per-thread.
extern __thread struct ncclThreadSignal ncclThreadSignalLocalInstance;

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
  if (__builtin_expect(o + size <= me->topFrame.end, true)) {
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
  if (__builtin_expect(me->head != nullptr, true)) {
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

constexpr ncclThreadSignal ncclThreadSignalStaticInitializer() {
  return {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
}

inline void ncclThreadSignalConstruct(struct ncclThreadSignal* me) {
  pthread_mutex_init(&me->mutex, nullptr);
  pthread_cond_init(&me->cond, nullptr);
}

inline void ncclThreadSignalDestruct(struct ncclThreadSignal* me) {
  pthread_mutex_destroy(&me->mutex);
  pthread_cond_destroy(&me->cond);
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
  return __atomic_load_n(&me->tail, __ATOMIC_RELAXED) <= 0x2;
}

template<typename T, T *T::*next>
bool ncclIntruQueueMpscEnqueue(ncclIntruQueueMpsc<T,next>* me, T* x) {
  __atomic_store_n(&(x->*next), nullptr, __ATOMIC_RELAXED);
  uintptr_t utail = __atomic_exchange_n(&me->tail, reinterpret_cast<uintptr_t>(x), __ATOMIC_ACQ_REL);
  T* prev = reinterpret_cast<T*>(utail);
  T** prevNext = utail <= 0x2 ? &me->head : &(prev->*next);
  __atomic_store_n(prevNext, x, __ATOMIC_RELAXED);
  if (utail == 0x1) { // waiting
    __atomic_thread_fence(__ATOMIC_ACQUIRE); // to see me->waiting
    // This lock/unlock is essential to ensure we don't race ahead of the consumer
    // and signal the cond before they begin waiting on it.
    struct ncclThreadSignal* waiting = me->waiting;
    pthread_mutex_lock(&waiting->mutex);
    pthread_mutex_unlock(&waiting->mutex);
    pthread_cond_broadcast(&waiting->cond);
  }
  return utail != 0x2; // not abandoned
}

template<typename T, T *T::*next>
T* ncclIntruQueueMpscDequeueAll(ncclIntruQueueMpsc<T,next>* me, bool waitSome) {
  T* head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
  if (head == nullptr) {
    if (!waitSome) return nullptr;
    uint64_t t0 = clockNano();
    bool sleeping = false;
    do {
      if (clockNano()-t0 >= 10*1000) { // spin for first 10us
        struct ncclThreadSignal* waitSignal = &ncclThreadSignalLocalInstance;
        pthread_mutex_lock(&waitSignal->mutex);
        uintptr_t expected = sleeping ? 0x1 : 0x0;
        uintptr_t desired = 0x1;
        me->waiting = waitSignal; // release done by successful compare exchange
        if (__atomic_compare_exchange_n(&me->tail, &expected, desired, /*weak=*/true, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
          sleeping = true;
          pthread_cond_wait(&waitSignal->cond, &waitSignal->mutex);
        }
        pthread_mutex_unlock(&waitSignal->mutex);
      }
      head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
    } while (head == nullptr);
  }

  __atomic_store_n(&me->head, nullptr, __ATOMIC_RELAXED);
  uintptr_t utail = __atomic_exchange_n(&me->tail, 0x0, __ATOMIC_ACQ_REL);
  T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
  T *x = head;
  while (x != tail) {
    T *x1;
    int spins = 0;
    while (true) {
      x1 = __atomic_load_n(&(x->*next), __ATOMIC_RELAXED);
      if (x1 != nullptr) break;
      if (++spins == 1024) { spins = 1024-1; sched_yield(); }
    }
    x = x1;
  }
  return head;
}

template<typename T, T *T::*next>
T* ncclIntruQueueMpscAbandon(ncclIntruQueueMpsc<T,next>* me) {
  uintptr_t expected = 0x0;
  if (__atomic_compare_exchange_n(&me->tail, &expected, /*desired=*/0x2, /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    return nullptr;
  } else {
    int spins = 0;
    T* head;
    while (true) {
      head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
      if (head != nullptr) break;
      if (++spins == 1024) { spins = 1024-1; sched_yield(); }
    }
    __atomic_store_n(&me->head, nullptr, __ATOMIC_RELAXED);
    uintptr_t utail = __atomic_exchange_n(&me->tail, 0x2, __ATOMIC_ACQ_REL);
    T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
    T *x = head;
    while (x != tail) {
      T *x1;
      spins = 0;
      while (true) {
        x1 = __atomic_load_n(&(x->*next), __ATOMIC_RELAXED);
        if (x1 != nullptr) break;
        if (++spins == 1024) { spins = 1024-1; sched_yield(); }
      }
      x = x1;
    }
    return head;
  }
}
#endif
