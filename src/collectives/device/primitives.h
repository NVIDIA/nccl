/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs


/* Defines primitive operations: Copy, Reduce, DoubleCopy, and ReduceCopy.
 *
 * In order to reduce the reptetion of template arguments, the operations
 * are bundled as static methods of the Primitives class.
 *
 * Each primitive operation copies/reduces a contiguous buffer and syncs
 * an optional set of flags against a sub-step counter. The sync value is
 * based on the step parameter. Sync flags must be of type WaitFlag or
 * PostFlag. The primitive routines wait for all WaitFlag args to attain
 * at least a value of SUBSTEPS*(step-1)+substep+1 (i.e. completion of
 * corresponding substep by previous step) before executing the transfer.
 * After each substep is transfered, all PostFlag arguments get updated to
 * the value SUBSTEPS*step+substep+1.
 */


class WaitFlag {
  volatile uint64_t * const flag;
  const int shift;
 public:
  __device__ __forceinline__
  WaitFlag(volatile uint64_t * const flag, const int shift) : flag(flag), shift(shift) { }
  __device__ __forceinline__
  void wait(uint64_t val) { while ((*flag + shift) < val) /*SPIN*/; }
};


class PostFlag {
  volatile uint64_t * const flag;
  const int shift;
  volatile int * const fifo;
  const int fifo_size;
 public:
  __device__ __forceinline__
  PostFlag(volatile uint64_t* const flag, const int shift, volatile int* const fifo, const int fifo_size) : flag(flag), shift(shift), fifo(fifo), fifo_size(fifo_size) { }
  __device__ __forceinline__
  void post(uint64_t val) { *flag = (val - shift); }
  __device__ __forceinline__
  void postSize(uint64_t step, int size) { if (fifo != NULL) fifo[step%fifo_size] = size; };
};


// Helper to check if any argument is of type T.
// e.g. AnyAre<WaitFlag>(Flag1, Flag2, ...)
template<typename T> __device__ __forceinline__
bool AnyAre() { return false; }

template<typename T, typename FIRST_T, typename... TAIL_Ts>
__device__ __forceinline__
bool AnyAre(FIRST_T first, TAIL_Ts... tail) {
  return std::is_same<T, FIRST_T>::value || AnyAre<T>(tail...);
}


// Wait on all WaitFlags, ignore PostFlags
__device__ __forceinline__
void WaitOnFlags(uint64_t val) { }

template <typename... TAIL_Ts> __device__ __forceinline__
void WaitOnFlags(uint64_t val, WaitFlag flag, TAIL_Ts... tail) {
  flag.wait(val);
  WaitOnFlags(val, tail...);
}

template <typename... TAIL_Ts> __device__ __forceinline__
void WaitOnFlags(uint64_t val, PostFlag, TAIL_Ts... tail) {
  WaitOnFlags(val, tail...);
}


// Post all PostFlags, ignore WaitFlags
__device__ __forceinline__
void PostToFlags(uint64_t val) { }

template <typename... TAIL_Ts> __device__ __forceinline__
void PostToFlags(uint64_t val, WaitFlag flag, TAIL_Ts... tail) {
  PostToFlags(val, tail...);
}

template <typename... TAIL_Ts> __device__ __forceinline__
void PostToFlags(uint64_t val, PostFlag flag, TAIL_Ts... tail) {
  flag.post(val);
  PostToFlags(val, tail...);
}


// Post sizes for PostFlags, ignore WaitFlags
__device__ __forceinline__
void PostSizeToFlags(uint64_t step, int size) { }

template <typename... TAIL_Ts> __device__ __forceinline__
void PostSizeToFlags(uint64_t step, int size, WaitFlag flag, TAIL_Ts... tail) {
  PostSizeToFlags(step, size, tail...);
}

template <typename... TAIL_Ts> __device__ __forceinline__
void PostSizeToFlags(uint64_t step, int size, PostFlag flag, TAIL_Ts... tail) {
  flag.postSize(step, size);
  PostSizeToFlags(step, size, tail...);
}


// Create pointer arithmetic syntax that doesn't break for nullptr_t
template <typename Tptr> __device__ __forceinline__
Tptr ptradd(Tptr ptr, int i) {
  return ptr + i;
}

__device__ __forceinline__
nullptr_t ptradd(nullptr_t ptr, int i) {
  return nullptr;
}


// Implementation of primitive types
template <int UNROLL, int SUBSTEPS, typename T, typename REDOP=FuncSum<T> >
class Primitives {
 private:
  template <typename SRC2_T, // either T* or nullptr_t
      typename DST2_T, // either T* or nullptr_t
      typename... SYNC_Ts> // either WaitFunc or PostFunc
  static __device__ __forceinline__ void
  GenericOp(const int tid, const int nthreads,
      const T*     src1,
      const SRC2_T src2,
      T*     dst1,
      DST2_T dst2,
      int len, int maxoffset, uint64_t step, SYNC_Ts... flags) {

    enum { noSrc2 = std::is_same<SRC2_T, nullptr_t>::value };
    enum { noDst2 = std::is_same<DST2_T, nullptr_t>::value };
    static_assert(noSrc2 || std::is_same<SRC2_T, const T*>::value,
        "src2 must be of type T* or nullptr_t");
    static_assert(noDst2 || std::is_same<DST2_T, T*>::value,
        "dst2 must be of type T* or nullptr_t");

    using OpType = typename std::conditional<noSrc2, FuncSum<T>, REDOP>::type;

    int sliceSize = len / SUBSTEPS;
    int sliceOffset = 0;

#pragma unroll 1
    for (int sub=0; sub<SUBSTEPS; ++sub) {
      int realSize = max(0, min(sliceSize, maxoffset-sliceOffset));
      if (tid < nthreads) {
        if (AnyAre<WaitFlag>(flags...)) {
          if (tid == 0) {
            WaitOnFlags(SUBSTEPS*step + sub + 1, flags...);
          }
          asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
        }
        ReduceOrCopy
        <
        UNROLL,
        OpType,
        T,
        !std::is_same<DST2_T, nullptr_t>::value, // HAS_DEST1
        !std::is_same<SRC2_T, nullptr_t>::value  // HAS_SRC1
        >
        (
            tid, nthreads,
            ptradd(dst1, sliceOffset),
            ptradd(dst2, sliceOffset),
            ptradd(src1, sliceOffset),
            ptradd(src2, sliceOffset),
            realSize
        );
        if (AnyAre<PostFlag>(flags...)) {
          __syncthreads();
        }
      } else {
        if (AnyAre<PostFlag>(flags...)) {
          __syncthreads();
          PostSizeToFlags(SUBSTEPS*step+sub, realSize*sizeof(T), flags...);
          __threadfence_system();
          PostToFlags(SUBSTEPS*step + sub + 1, flags...);
        }
      }
      sliceOffset += sliceSize;
    }
  }

 public:
  template <typename... SYNC_Ts>
  static __device__ __forceinline__ void
  Copy(const int tid, const int nthreads, const T* src, T* dst,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp(tid, nthreads, src, nullptr, dst, nullptr, len, maxOffset, step, flags...);
  }

  template <typename... SYNC_Ts>
  static __device__ __forceinline__ void
  DoubleCopy(const int tid, const int nthreads, const T* src, T* dst1, T* dst2,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp(tid, nthreads, src, nullptr, dst1, dst2, len, maxOffset, step, flags...);
  }

  template <typename... SYNC_Ts>
  static __device__ __forceinline__ void
  Reduce(const int tid, const int nthreads, const T* src1, const T* src2, T* dst,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp(tid, nthreads, src1, src2, dst, nullptr, len, maxOffset, step, flags...);
  }

  template <typename... SYNC_Ts>
  static __device__ __forceinline__ void
  ReduceCopy(const int tid, const int nthreads, const T* src1, const T* src2, T* dst1, T* dst2,
      int len, int maxOffset, uint64_t step, SYNC_Ts... flags) {
    GenericOp(tid, nthreads, src1, src2, dst1, dst2, len, maxOffset, step, flags...);
  }
};

#endif // end include guard
