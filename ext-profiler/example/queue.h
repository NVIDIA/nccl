/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef QUEUE_H
#define QUEUE_H

template<typename T, T *T::*next>
struct profilerQueue {
  T *head, *tail;
};

template<typename T, T *T::*next>
 inline void profilerQueueConstruct(profilerQueue<T,next> *me) {
  me->head = nullptr;
  me->tail = nullptr;
}

template<typename T, T *T::*next>
 inline bool profilerQueueEmpty(profilerQueue<T,next> *me) {
  return me->head == nullptr;
}

template<typename T, T *T::*next>
inline T* profilerQueueHead(profilerQueue<T,next> *me) {
  return me->head;
}

template<typename T, T *T::*next>
 inline T* profilerQueueTail(profilerQueue<T,next> *me) {
  return me->tail;
}

template<typename T, T *T::*next>
 inline void profilerQueueEnqueue(profilerQueue<T,next> *me, T *x) {
  x->*next = nullptr;
  (me->head ? me->tail->*next : me->head) = x;
  me->tail = x;
}

template<typename T, T *T::*next>
 inline T* profilerQueueDequeue(profilerQueue<T,next> *me) {
  T *ans = me->head;
  me->head = ans->*next;
  if (me->head == nullptr) me->tail = nullptr;
  return ans;
}

#endif
