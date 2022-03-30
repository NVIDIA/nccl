/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include "nccl.h"
#include "checks.h"
#include <stdint.h>

int ncclCudaCompCap();

// PCI Bus ID <-> int64 conversion functions
ncclResult_t int64ToBusId(int64_t id, char* busId);
ncclResult_t busIdToInt64(const char* busId, int64_t* id);

ncclResult_t getBusId(int cudaDev, int64_t *busId);

ncclResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}

// Recyclable list that avoids frequent malloc/free
template<typename T>
struct ncclListElem {
  T data;
  struct ncclListElem* next;
};

template<typename T>
class ncclRecyclableList {
 private:
  struct ncclListElem<T>* head;
  struct ncclListElem<T>* tail;
  struct ncclListElem<T>* cursor;
  int n;

 public:
  ncclRecyclableList() {
    tail = cursor = head = NULL;
    n = 0;
  }

  int count() const { return n; }

  // Get a new element from the list and return pointer
  ncclResult_t getNewElem(T** dataOut) {
    if (tail != NULL) {
      *dataOut = &tail->data;
      memset(*dataOut, 0, sizeof(T));
    } else {
      NCCLCHECK(ncclCalloc(&tail, 1));
      *dataOut = &tail->data;
      cursor = head = tail;
    }
    if (tail->next == NULL) {
      NCCLCHECK(ncclCalloc(&tail->next, 1));
    }
    tail = tail->next;
    n += 1;
    return ncclSuccess;
  }

  T* begin() {
    if (head == NULL || head == tail) return NULL;
    cursor = head->next;
    return &head->data;
  }

  // Get next element from the list during an iteration
  T* getNext() {
    // tail always points to the next element to be enqueued
    // hence does not contain valid data
    if (cursor == NULL || cursor == tail) return NULL;
    T* rv = &cursor->data;
    cursor = cursor->next;
    return rv;
  }

  T* peakNext() {
    if (cursor == NULL || cursor == tail) return NULL;
    return &cursor->data;
  }

  // Recycle the list without freeing the space
  void recycle() {
    tail = cursor = head;
    n = 0;
  }

  ~ncclRecyclableList() {
    while (head != NULL) {
      struct ncclListElem<T>* temp = head;
      head = head->next;
      free(temp);
    }
  }
};

#endif
