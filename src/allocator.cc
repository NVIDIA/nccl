/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "transport.h"
#include "group.h"
#include "nvtx.h"
#include "utils.h"

NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc(void **ptr, size_t size) {
  NCCL_NVTX3_FUNC_RANGE;
  ncclResult_t ret = ncclSuccess;

#if CUDART_VERSION >= 11030
  size_t memGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1;
  int cudaDev;
  int flag;
  int dcnt;

  if (ptr == NULL || size == 0) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));

  if (ncclCuMemEnable()) {
    size_t handleSize = size;
    int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#if CUDART_VERSION >= 12030
    // Query device to see if FABRIC handle support is available
    flag = 0;
    (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
    if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
#endif
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
    memprop.location.id = currentDev;
    // Query device to see if RDMA support is available
    flag = 0;
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    CUDACHECK(cudaGetDeviceCount(&dcnt));
    ALIGN_SIZE(handleSize, memGran);

#if CUDART_VERSION >= 12030
    if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
      /* First try cuMemCreate() with FABRIC handle support and then remove if it fails */
      CUresult err = CUPFN(cuMemCreate(&handle, handleSize, &memprop, 0));
      if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
        requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
        memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
        /* Allocate the physical memory on the device */
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      } else if (err != CUDA_SUCCESS) {
        // Catch and report any error from above
        CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
      }
    } else
#endif
    {
      /* Allocate the physical memory on the device */
      CUCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
    }
    /* Reserve a virtual address range */
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;
      if (i == cudaDev || (CUDASUCCESS(cudaDeviceCanAccessPeer(&p2p, i, cudaDev)) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = i;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1));
      }
      if (0 == p2p && i != cudaDev) INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i);
    }
    goto exit;
  }

fallback:
#endif
  // Coverity is right to complain that we may pass a NULL ptr to cudaMalloc.  That's deliberate though:
  // we want CUDA to return an error to the caller.
  // coverity[var_deref_model]
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
ncclResult_t  ncclMemFree(void *ptr) {
  NCCL_NVTX3_FUNC_RANGE;
  ncclResult_t ret = ncclSuccess;
  int saveDevice;

  CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 11030
  CUdevice ptrDev = 0;

  if (ptr == NULL) goto fallback;
  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);

exit:
  CUDACHECK(cudaSetDevice(saveDevice));
  return ret;
fail:
  goto exit;
}

////////////////////////////////////////////////////////////////////////////////
// ncclSpace:
//
// This datastructure "cuts" the line of non-negative integers into segments
// which alternate between "full" (allocated) and "empty" (not allocated). The
// cuts are sorted ascending. The segment after the last cut must be empty
// (the unallocated frontier). Knwoing this we can deduce whether the segment
// ending at cut[i] is full or empty with this formula:
//   isFull(i) = (i%2 != ncuts%2)

void ncclSpaceConstruct(struct ncclSpace* a) {
  memset(a, 0, sizeof(*a));
}

void ncclSpaceDestruct(struct ncclSpace* a) {
  free(a->cuts);
}

static void insertSegment(struct ncclSpace* a, int index, int64_t lo, int64_t hi) {
  // Insert space for two cuts in `a->cuts[]` before `index`.
  if (a->count + 2 > a->capacity) {
    a->capacity *= 2;
    if (a->capacity == 0) a->capacity = 16;
    int64_t* cuts1 = (int64_t*)malloc(a->capacity*sizeof(int64_t));
    for (int i=0; i < index; i++) cuts1[i] = a->cuts[i];
    for (int i=index; i < a->count; i++) cuts1[i+2] = a->cuts[i];
    free(a->cuts);
    a->cuts = cuts1;
  } else {
    for (int i=a->count-1; index <= i; i--) a->cuts[i+2] = a->cuts[i];
  }
  a->cuts[index+0] = lo;
  a->cuts[index+1] = hi;
  a->count += 2;

  // Filter pairs of adjacent repeated values from cuts[]. Since these mark
  // boundaries where segments transition between full<->empty, dropping such a
  // pair fuses two adjacent segments together. Examples:
  //   [1,2,3,3,4] -> [1,2,4]
  //   [1,2,3,3,3,4] -> [1,2,3,4] // have to leave one 3 because its a full<->empty transition
  //   [1,2,3,3,3,3,4] -> [1,2,4]
  // Leading zeros don't have to be in pairs, they are always dropped:
  //   [0,1,2] -> [1,2]
  //   [0,0,1,2] -> [1,2]
  int r = index, w = index; // Read and write cursors.
  int64_t prev = r==0 ? 0 : a->cuts[r-1];
  while (r < a->count) {
    int64_t cur = a->cuts[r++];
    a->cuts[w++] = cur;
    if (prev == cur) { // Repeated value is an empty segment which can be deleted.
      // Erase last two cuts or just one if we're at the start.
      w -= w==1 ? 1 : 2;
      // Zeros can only occur at the beginning (due to being sorted). We want to
      // drop any number of zeros, but only even numbers of other repeated values.
      // So set to zero here, which will make prev=0, thus if next value is zero
      // it will be dropped but if its not zero then it will need to begin a new
      // pair to be dropped.
      cur = 0;
    }
    prev = cur;
  }
  a->count = w;
}

ncclResult_t ncclSpaceAlloc(
    struct ncclSpace* a, int64_t limit, int64_t size, int align,
    int64_t* outOffset
  ) {
  // When allocating we try to locate the first empty segment which can hold
  // the allocation and move its lower cut upward.
  int i = a->count%2; // First empty segment ends at cuts[i]
  size_t off;
  while (i <= a->count) {
    size_t lo = i == 0 ? 0 : a->cuts[i-1];
    size_t hi = i == a->count ? limit : a->cuts[i];
    off = alignUp(lo, align);
    if (off + size <= hi) {
      *outOffset = off;
      if (i == 0 || off + size == hi) { // Slow path required.
        insertSegment(a, i, off, off+size);
      } else { // We can just append to the end of a full segment.
        a->cuts[i-1] = off + size;
      }
      return ncclSuccess;
    }
    i += 2; // Next empty segment
  }
  WARN("Allocation failed. No suitable space found to accommodate size=0x%lx within limit=0x%lx", (long)size, (long)limit);
  return ncclInternalError;
}

ncclResult_t ncclSpaceFree(struct ncclSpace* a, int64_t offset, int64_t size) {
  if (a->count == 0 || a->cuts[a->count-1] <= offset) {
    WARN("No allocation found at offset=0x%lx", (long)offset);
    return ncclInternalError;
  }

  // This could be binary search, but since allocate is linear there's no point.
  int i = 1 - a->count%2; // First full segment ends at cuts[i]
  while (a->cuts[i] <= offset) i += 2;

  int64_t lo = i==0 ? 0 : a->cuts[i-1];
  int64_t hi = a->cuts[i];

  if (offset < lo || hi < offset + size) {
    WARN("Given size=0x%lx extends beyond allocation.", (long)size);
    return ncclInternalError;
  }

  // First try the two fast cases which just shrink a segment from one side.
  if (i != 0 && lo == offset && offset + size != hi) {
    a->cuts[i-1] = offset + size; // Bring bottom up.
  } else if (lo != offset && offset + size == hi) {
    a->cuts[i] = offset; // Bring top down.
  } else { // Slow path.
    insertSegment(a, i, offset, offset+size);
  }
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// ncclShadowPool:

struct ncclShadowPage { // A contiguous block of (at most) 64 objects
  struct ncclShadowPage* next;
  int objSize;
  uint64_t freeMask;
  void* devObjs;
};
struct ncclShadowObject {
  struct ncclShadowObject* next;
  void* devObj;
  void* hostObj;
  struct ncclShadowPage* page; // null if not allocated in page but directly in CUDA mempool.
};

void ncclShadowPoolConstruct(struct ncclShadowPool* pool) {
  pool->hbits = 0;
  pool->count = 0;
  pool->table = nullptr;
  pool->pages = nullptr;
}

ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool* pool) {
  if (pool->hbits != 0) {
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (pool->count != 0) {
      for (int i=0; i < 1<<pool->hbits; i++) {
        struct ncclShadowObject* obj = pool->table[i];
        while (obj != nullptr) {
          struct ncclShadowPage* page = obj->page;
          if (page != nullptr) {
            if (page->freeMask == 0) { // Put full pages back into page list.
              page->freeMask = 1;
              page->next = pool->pages;
              pool->pages = page;
            }
          } else {
            cudaFreeAsync(obj->devObj, stream);
          }
          struct ncclShadowObject* next = obj->next;
          free(obj);
          obj = next;
        }
      }
    }
    free(pool->table);

    while (pool->pages != nullptr) {
      cudaFreeAsync(pool->pages->devObjs, stream);
      struct ncclShadowPage* next = pool->pages->next;
      free(pool->pages);
      pool->pages = next;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaMemPoolDestroy(pool->memPool);
  }
  return ncclSuccess;
}

static void hashInsert(struct ncclShadowPool* pool, struct ncclShadowObject* obj) {
  uint64_t b = ncclHashPointer(pool->hbits, obj->devObj);
  obj->next = pool->table[b];
  pool->table[b] = obj;
}

ncclResult_t ncclShadowPoolAlloc(
    struct ncclShadowPool* pool, size_t size, void** outDevObj, void** outHostObj,
    cudaStream_t stream
  ) {
  if (size == 0) {
    if (outDevObj) *outDevObj = nullptr;
    if (outHostObj) *outHostObj = nullptr;
    return ncclSuccess;
  }

  int hbits = pool->hbits;
  if (hbits == 0) {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    cudaGetDevice(&props.location.id);
    CUDACHECK(cudaMemPoolCreate(&pool->memPool, &props));

    pool->hbits = hbits = 4;
    pool->table = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*)<<hbits);
    for (int i=0; i < 1<<hbits; i++) pool->table[i] = nullptr;
  }

  // Check for hash table size increase before inserting. Maintain 2:1 object:bucket ratio.
  if (pool->count+1 > 2<<hbits) {
    struct ncclShadowObject** table0 = pool->table;
    struct ncclShadowObject** table1 = (struct ncclShadowObject**)malloc(sizeof(struct ncclShadowObject*)<<(hbits+1));
    pool->table = table1;
    pool->hbits = hbits+1;
    for (int i1=0; i1 < 2<<hbits; i1++) table1[i1] = nullptr;
    for (int i0=0; i0 < 1<<hbits; i0++) {
      struct ncclShadowObject* obj = table0[i0];
      while (obj) {
        struct ncclShadowObject* next = obj->next;
        hashInsert(pool, obj);
        obj = next;
      }
    }
    hbits += 1; // match pool->hbits
    free(table0);
  }

  struct ncclShadowPage* page;
  void *devObj;
  if ((64<<10)/size >= 3) {
    int shift = std::max<int>(0, (int)log2Down(size) + 1 - 4);
    int pageObjSize = ((size + (1<<shift)-1)>>shift)<<shift;
    struct ncclShadowPage** pagePtr = &pool->pages;
    while (true) {
      page = *pagePtr;
      if (page == nullptr) {
        size_t pageSize = std::min<size_t>(64<<10, 64*pageObjSize);
        page = (struct ncclShadowPage*)malloc(sizeof(struct ncclShadowPage));
        page->objSize = pageObjSize;
        page->freeMask = uint64_t(-1)>>(64 - pageSize/pageObjSize);
        page->next = pool->pages;
        pool->pages = page;
        CUDACHECK(cudaMallocFromPoolAsync(&page->devObjs, pageSize, pool->memPool, stream));
        CUDACHECK(cudaMemsetAsync(page->devObjs, 0, pageSize, stream));
        // fall through...
      }
      if (page->objSize == pageObjSize) {
        int slot = popFirstOneBit(&page->freeMask);
        devObj = (char*)page->devObjs + slot*pageObjSize;
        if (page->freeMask == 0) *pagePtr = page->next; // Remove full page from list.
        break;
      }
      pagePtr = &page->next;
    }
  } else {
    page = nullptr;
    CUDACHECK(cudaMallocFromPoolAsync(&devObj, size, pool->memPool, stream));
    CUDACHECK(cudaMemsetAsync(devObj, 0, size, stream));
  }

  struct ncclShadowObject* obj = (struct ncclShadowObject*)malloc(
    sizeof(struct ncclShadowObject) + /*padding=*/alignof(max_align_t)-1 + size
  );
  obj->page = page;
  obj->devObj = devObj;
  obj->hostObj = alignUp((char*)(obj+1), alignof(max_align_t));
  memset(obj->hostObj, 0, size);
  hashInsert(pool, obj);
  pool->count += 1;
  if (outDevObj) *outDevObj = devObj;
  if (outHostObj) *outHostObj = obj->hostObj;
  return ncclSuccess;
}

ncclResult_t ncclShadowPoolFree(struct ncclShadowPool* pool, void* devObj, cudaStream_t stream) {
  if (devObj == nullptr) return ncclSuccess;

  uint64_t b = ncclHashPointer(pool->hbits, devObj);
  struct ncclShadowObject** pobj = &pool->table[b];
  while (true) {
    if (*pobj == nullptr) {
      WARN("Device object does not exist in shadow pool.");
      return ncclInternalError;
    }
    if ((*pobj)->devObj == devObj) break;
    pobj = &(*pobj)->next;
  }
  struct ncclShadowObject* obj = *pobj;
  *pobj = obj->next;
  if (obj->page != nullptr) {
    if (obj->page->freeMask == 0) {
      obj->page->next = pool->pages;
      pool->pages = obj->page;
    }
    int slot = ((char*)obj->devObj - (char*)obj->page->devObjs)/obj->page->objSize;
    obj->page->freeMask |= uint64_t(1)<<slot;
  } else {
    CUDACHECK(cudaFreeAsync(devObj, stream));
  }
  free(obj);
  pool->count -= 1;
  return ncclSuccess;
}

ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** hostObj) {
  if (devObj == nullptr) {
    *hostObj = nullptr;
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(pool->hbits, devObj);
  struct ncclShadowObject* obj = pool->table[b];
  while (true) {
    if (obj == nullptr) {
      WARN("Device object does not exist in shadow pool.");
      return ncclInternalError;
    }
    if (obj->devObj == devObj) break;
    obj = obj->next;
  }
  *hostObj = obj->hostObj;
  return ncclSuccess;
}
