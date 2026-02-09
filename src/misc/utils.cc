/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "core.h"
#include "os.h"

#include "nvmlwrap.h"

#include <stdlib.h>
#include <mutex>

// Get current Compute Capability
int ncclCudaCompCap() {
  int cudaDev;
  if (!CUDASUCCESS(cudaGetDevice(&cudaDev))) return 0;
  int ccMajor, ccMinor;
  if (!CUDASUCCESS(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev))) return 0;
  if (!CUDASUCCESS(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev))) return 0;
  return ccMajor*10+ccMinor;
}

ncclResult_t int64ToBusId(int64_t id, char* busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
  char hexStr[17];  // Longest possible int64 hex string + null terminator.
  int hexOffset = 0;
  for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return ncclSuccess;
}

// Convert a logical cudaDev index to the NVML device minor number
ncclResult_t getBusId(int cudaDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  CUDACHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));
  NCCLCHECK(busIdToInt64(busIdStr, busId));
  return ncclSuccess;
}

ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

static uint64_t hostHashValue = 0;
/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the NCCL_HOSTID env var.
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static void getHostHashOnce() {
  char hostHash[1024];
  const char *hostId;

  // Fall back is the full hostname if something fails
  (void) getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = ncclGetEnv("NCCL_HOSTID")) != NULL) {
    INFO(NCCL_ENV, "NCCL_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash)-1);
    hostHash[sizeof(hostHash)-1] = '\0';
  } else {
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
      }
      fclose(file);
    }
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  TRACE(NCCL_INIT,"unique hostname '%s'", hostHash);

  hostHashValue = getHash(hostHash, strlen(hostHash));
}
uint64_t getHostHash(void) {
  static std::once_flag once;
  std::call_once(once, getHostHashOnce);
  return hostHashValue;
}

uint64_t hashCombine(uint64_t baseHash, uint64_t value) {
  uint64_t hacc[2] = {1, 1};
  eatHash(hacc, &baseHash);
  eatHash(hacc, &value);
  return digestHash(hacc);
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long) ncclOsGetpid());
  int plen = strlen(pname);
  int len = readlink("/proc/self/ns/pid", pname+plen, sizeof(pname)-1-plen);
  if (len < 0) len = 0;

  pname[plen+len]='\0';
  TRACE(NCCL_INIT,"unique PID '%s'", pname);

  return getHash(pname, strlen(pname));
}

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchIf(const char* string, const char* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}


bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact)
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance;

void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) {
  // `me->hunks` points to the top of the stack non-empty hunks. Hunks above
  // this (reachable via `->above`) are empty.
  struct Hunk* top = me->topFrame.hunk;
  size_t mallocSize = 0;

  // If we have lots of space left in hunk but that wasn't enough then we'll
  // allocate the object unhunked.
  if (me->topFrame.end - me->topFrame.bumper >= 8<<10)
    goto unhunked;

  // If we have another hunk (which must be empty) waiting above this one and
  // the object fits then use that.
  if (top && top->above) {
    struct Hunk* top1 = top->above;
    uintptr_t uobj = (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align-1) & -uintptr_t(align);
    if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
      me->topFrame.hunk = top1;
      me->topFrame.bumper = uobj + size;
      me->topFrame.end = reinterpret_cast<uintptr_t>(top1) + top1->size;
      return reinterpret_cast<void*>(uobj);
    }
  }

  { // If the next hunk we're going to allocate wouldn't be big enough but the
    // Unhunk proxy fits in the current hunk then go allocate as unhunked.
    size_t nextSize = (top ? top->size : 0) + (64<<10);
    constexpr size_t maxAlign = 64;
    if (nextSize < sizeof(struct Hunk) + maxAlign + size) {
      uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk)-1) & -uintptr_t(alignof(Unhunk));
      if (uproxy + sizeof(struct Unhunk) <= me->topFrame.end)
        goto unhunked;
    }

    // At this point we must need another hunk, either to fit the object
    // itself or its Unhunk proxy.
    mallocSize = nextSize;
    INFO(NCCL_ALLOC, "%s:%d memory stack hunk malloc(%llu)", __FILE__, __LINE__, (unsigned long long)mallocSize);
    struct Hunk *top1 = (struct Hunk*)malloc(mallocSize);
    if (top1 == nullptr) goto malloc_exhausted;
    top1->size = nextSize;
    top1->above = nullptr;
    if (top) top->above = top1;
    top = top1;
    me->topFrame.hunk = top;
    me->topFrame.end = reinterpret_cast<uintptr_t>(top) + nextSize;
    me->topFrame.bumper = reinterpret_cast<uintptr_t>(top) + sizeof(struct Hunk);
  }

  { // Try to fit object in the new top hunk.
    uintptr_t uobj = (me->topFrame.bumper + align-1) & -uintptr_t(align);
    if (uobj + size <= me->topFrame.end) {
      me->topFrame.bumper = uobj + size;
      return reinterpret_cast<void*>(uobj);
    }
  }

unhunked:
  { // We need to allocate the object out-of-band and put an Unhunk proxy in-band
    // to keep track of it.
    uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk)-1) & -uintptr_t(alignof(Unhunk));
    Unhunk* proxy = reinterpret_cast<Unhunk*>(uproxy);
    me->topFrame.bumper = uproxy + sizeof(Unhunk);
    proxy->next = me->topFrame.unhunks;
    me->topFrame.unhunks = proxy;
    mallocSize = size;
    proxy->obj = malloc(mallocSize);
    INFO(NCCL_ALLOC, "%s:%d memory stack non-hunk malloc(%llu)", __FILE__, __LINE__, (unsigned long long)mallocSize);
    if (proxy->obj == nullptr) goto malloc_exhausted;
    return proxy->obj;
  }

malloc_exhausted:
  WARN("%s:%d Unrecoverable error detected: malloc(size=%llu) returned null.", __FILE__, __LINE__, (unsigned long long)mallocSize);
  abort();
}

void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  // Free unhunks first because both the frames and unhunk proxies lie within the hunks.
  struct ncclMemoryStack::Frame* f = &me->topFrame;
  while (f != nullptr) {
    struct ncclMemoryStack::Unhunk* u = f->unhunks;
    while (u != nullptr) {
      free(u->obj);
      u = u->next;
    }
    f = f->below;
  }
  // Free hunks
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk *h1 = h->above;
    free(h);
    h = h1;
  }
}

/* return concatenated string representing each set bit */
ncclResult_t ncclBitsToString(uint32_t bits, uint32_t mask, const char* (*toStr)(int), char *buf, size_t bufLen, const char *wildcard) {
  if (!buf || !bufLen)
    return ncclInvalidArgument;

  bits &= mask;

  // print wildcard value if all bits set
  if (wildcard && bits == mask) {
    snprintf(buf, bufLen, "%s", wildcard);
    return ncclSuccess;
  }

  // Add each set bit to string
  int pos = 0;
  for (int i = 0; bits; i++, bits >>= 1) {
    if (bits & 1) {
      if (pos > 0) pos += snprintf(buf + pos, bufLen - pos, "|");
      pos += snprintf(buf + pos, bufLen - pos, "%s", toStr(i));
    }
  }

  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// Hash function for pointer types (shared by address map implementations)
// Uses shadowpool's algorithm
uint64_t ncclHashPointer(int hbits, void* key) {
  uintptr_t h = reinterpret_cast<uintptr_t>(key);
  h ^= h>>32;
  h *= 0x9e3779b97f4a7c13;
  return (uint64_t)h >> (64-hbits);
}

////////////////////////////////////////////////////////////////////////////////
// Intrusive address map implementation (untyped core functions)

// Helper: Read key from object at given offset
// Key must be convertible to uintptr_t, so we read keySize bytes and zero-extend
static inline uintptr_t readKey(void* object, int keySize, int keyFieldOffset) {
  void* keyPtr = (char*)object + keyFieldOffset;
  uintptr_t result = 0;
  memcpy(&result, keyPtr, keySize);
  return result;
}

// Helper: Read next pointer from object at given offset
// Uses memcpy to avoid strict aliasing violations when actual type is T*, not void*
static inline void* readNextPtr(void* object, int nextFieldOffset) {
  void* nextPtr = (char*)object + nextFieldOffset;
  void* result = nullptr;
  memcpy(&result, nextPtr, sizeof(void*));
  return result;
}

// Helper: Write next pointer to object at given offset
// Uses memcpy to avoid strict aliasing violations when actual type is T*, not void*
static inline void writeNextPtr(void* object, int nextFieldOffset, void* value) {
  void* nextPtr = (char*)object + nextFieldOffset;
  memcpy(nextPtr, &value, sizeof(void*));
}

ncclResult_t ncclIntruAddressMapInsert_untyped(
    struct ncclIntruAddressMap_untyped* map,
    int keySize, int keyFieldOffset, int nextFieldOffset,
    uintptr_t key, void* object) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Lazy initialization - create table on first insert
  if (map->hbits == 0) {
    map->hbits = 4;
    map->table = (void**)calloc(1<<map->hbits, sizeof(void*));
    if (map->table == nullptr) {
      map->hbits = 0; // Reset on failure
      WARN("Intrusive address map initialization failed: calloc(%d entries) returned null", 1<<map->hbits);
      return ncclSystemError;
    }
  }

  int hbits = map->hbits;

  // Check for address map size increase before inserting. Maintain 2:1 object:bucket ratio.
  if (map->count+1 > 2<<hbits) {
    int oldHbits = hbits;
    int oldSize = 1<<oldHbits;
    int newHbits = hbits + 1;
    int newSize = 1<<newHbits;

    // Allocate a new table (don't use realloc to avoid data corruption during rehashing)
    void** newTable = (void**)malloc(newSize * sizeof(void*));
    if (newTable == nullptr) {
      WARN("Intrusive address map resize failed: malloc(%d entries) returned null", newSize);
      return ncclSystemError;
    }

    // Initialize all new buckets to nullptr
    for (int i = 0; i < newSize; i++) {
      newTable[i] = nullptr;
    }

    // Rehash all existing entries from old table to new table
    for (int i = 0; i < oldSize; i++) {
      void* obj = map->table[i];

      while (obj) {
        void* next = readNextPtr(obj, nextFieldOffset);
        uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
        uint64_t b = ncclHashPointer(newHbits, (void*)objKey);
        writeNextPtr(obj, nextFieldOffset, newTable[b]);
        newTable[b] = obj;
        obj = next;
      }
    }

    // Free old table and update to new table
    free(map->table);
    map->table = newTable;
    map->hbits = newHbits;
  }

  // Insert object into appropriate bucket
  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* currentNext = readNextPtr(object, nextFieldOffset);

  // Check if next pointer is already non-NULL (object might already be in a list)
  if (currentNext != nullptr) {
    INFO(NCCL_INIT, "Intrusive map: inserting object %p with non-NULL next pointer %p (key=0x%lx). "
         "Object may already be in another list or this is intentional reuse.",
         object, currentNext, (unsigned long)key);
  }

  writeNextPtr(object, nextFieldOffset, map->table[b]);
  map->table[b] = object;
  map->count += 1;

  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapFind_untyped(
    struct ncclIntruAddressMap_untyped* map,
    int keySize, int keyFieldOffset, int nextFieldOffset,
    uintptr_t key, void** object) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Output object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  *object = nullptr;

  // Empty map is not an error - just means key not found
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      *object = obj;
      return ncclSuccess;
    }
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key not found is not an error - *object is already nullptr
  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapRemove_untyped(
    struct ncclIntruAddressMap_untyped* map,
    int keySize, int keyFieldOffset, int nextFieldOffset,
    uintptr_t key) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Removing from empty map is not an error - it's idempotent
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* prev = nullptr;
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      void* next = readNextPtr(obj, nextFieldOffset);

      // Update the previous pointer to skip the current object
      if (prev == nullptr) {
        // Removing from head of bucket
        map->table[b] = next;
      } else {
        // Removing from middle/end of list
        writeNextPtr(prev, nextFieldOffset, next);
      }

      map->count -= 1;

      // If this was the last entry, clean up the table (same pattern as non-intrusive map)
      if (map->count == 0) {
        free(map->table);
        map->hbits = 0;
        map->count = 0;
        map->table = nullptr;
      }

      return ncclSuccess;
    }
    prev = obj;
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key not found is not an error - remove is idempotent
  return ncclSuccess;
}
