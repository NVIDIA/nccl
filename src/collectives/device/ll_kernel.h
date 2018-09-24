/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LL_KERNEL_H_
#define NCCL_LL_KERNEL_H_

static __device__ uint64_t readLL(union ncclLLFifoLine* src, uint32_t flag) {
  uint32_t data1, flag1, data2, flag2;
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
  } while ((flag1 != flag) || (flag2 != flag));
  uint64_t val64 = data1 + (((uint64_t)data2) << 32);
  return val64;
}

static __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
}

// Using memcpy handles misaligned pointers.
static __device__ uint64_t readAL(uint64_t* src) {
  uint64_t val;
  memcpy((char*)&val, (char*)src, sizeof(uint64_t));
  return val;
}
static __device__ void storeAL(uint64_t* dst, uint64_t val) {
  memcpy((char*)dst, (char*)&val, sizeof(uint64_t));
}

template <typename T, class FUNC>
class LLPrimitives {
 private:
  template <int HAS_SRC1, int HAS_SRC2, int HAS_DST1, int HAS_DST2>
  static __device__ void ReduceCopyGeneric(const T* src1, union ncclLLFifoLine* src2, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t iflag, uint32_t oflag, int nthreads) {
    if (size <= 0) return;
    size_t size64 = size * sizeof(T) / sizeof(uint64_t);
    uint64_t* src1A = (uint64_t*)src1;
    uint64_t* dst1A = (uint64_t*)dst1;
    int offset = threadIdx.x;
    // Do multiples of 64 bits
#pragma unroll 1
    for (; offset < size64; offset += nthreads) {
      uint64_t val;
      if (HAS_SRC1) {
        val = readAL(src1A+offset);
        if (HAS_SRC2) val = MULTI<FUNC, T>()(readLL(src2+offset, iflag), val);
      } else if (HAS_SRC2) {
        val = readLL(src2+offset, iflag);
      }
      if (HAS_DST1) storeAL(dst1A+offset, val);
      if (HAS_DST2) storeLL(dst2+offset, val, oflag);
    }
    // Finish last word
    int sizeDone = size64*(sizeof(uint64_t)/sizeof(T));
    int sizeRem = size - sizeDone;
    if (threadIdx.x == 0 && sizeRem) {
      const T* src1B = src1 + sizeDone;
      T* dst1B = dst1 + sizeDone;

      uint64_t lastVal;
      T* vals = (T*)&lastVal;

      if (HAS_SRC2) {
        uint64_t lastVal2 = readLL(src2+size64, iflag);
        T* src2B = (T*)&lastVal2;
        for (int offset = 0; offset < sizeRem; offset++) {
          vals[offset] = HAS_SRC1 ? FUNC()(src2B[offset], src1B[offset]) : src2B[offset];
        }
      } else if (HAS_SRC1) {
        for (int offset = 0; offset < sizeRem; offset++) {
          vals[offset] = src1B[offset];
        }
      }
      if (HAS_DST2) storeLL(dst2+size64, lastVal, oflag);
      if (HAS_DST1) {
        for (int offset = 0; offset < sizeRem; offset++) {
          dst1B[offset] = vals[offset];
        }
      }
    }
  }
 public:
  static __device__ void ReduceCopy(const T* src, union ncclLLFifoLine* dst, int size, uint32_t oflag, int nthreads) {
    return ReduceCopyGeneric<1, 0, 0, 1>(src, NULL, NULL, dst, size, 0, oflag, nthreads);
  }

  static __device__ void ReduceCopy(union ncclLLFifoLine* src, T* dst, int size, uint32_t iflag, int nthreads) {
    return ReduceCopyGeneric<0, 1, 1, 0>(NULL, src, dst, NULL, size, iflag, 0, nthreads);
  }

  static __device__ void ReduceCopy(const T* src1, union ncclLLFifoLine* src2, union ncclLLFifoLine* dst, int size, uint32_t iflag, uint32_t oflag, int nthreads) {
    return ReduceCopyGeneric<1, 1, 0, 1>(src1, src2, NULL, dst, size, iflag, oflag, nthreads);
  }

  static __device__ void ReduceCopy(const T* src1, union ncclLLFifoLine* src2, T* dst, int size, uint32_t iflag, int nthreads) {
    return ReduceCopyGeneric<1, 1, 1, 0>(src1, src2, dst, NULL, size, iflag, 0, nthreads);
  }

  static __device__ void ReduceCopy(const T* src, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t oflag, int nthreads) {
    return ReduceCopyGeneric<1, 0, 1, 1>(src, NULL, dst1, dst2, size, 0, oflag, nthreads);
  }

  static __device__ void ReduceCopy(union ncclLLFifoLine* src, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t iflag, uint32_t oflag, int nthreads) {
    return ReduceCopyGeneric<0, 1, 1, 1>(NULL, src, dst1, dst2, size, iflag, oflag, nthreads);
  }

  static __device__ void ReduceCopy(const T* src1, union ncclLLFifoLine* src2, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t iflag, uint32_t oflag, int nthreads) {
    return ReduceCopyGeneric<1, 1, 1, 1>(src1, src2, dst1, dst2, size, iflag, oflag, nthreads);
  }
};

// Common macros

#define STEP_TO_SLOT(step) \
  (step % NCCL_LL_CHUNKS)

#define WAIT_NEXT \
  if (tid == 0) { \
    while (sendHead + NCCL_LL_CHUNKS <= step) { \
      sendHead = sendHeadPtr[0]; \
    } \
  } \
  asm volatile ("bar.sync 1, %0;" :: "r"(llNthreads));

#define POST_SIZE \
  if (tid == 0 && sizesFifo) sizesFifo[step % NCCL_LL_CHUNKS] = (maxOffset <= 0) ? -1 : (maxOffset*2*(int)sizeof(T));

#define ACK_PREV \
  asm volatile ("bar.sync 1, %0;" :: "r"(llNthreads)); \
  if (tid == 0) recvHeadPtr[0] = step;

#define FIFO_CLEANING_AND_SAVE_STEP(flag) do { \
  if (step > ring->send.conn.llLastCleaning + NCCL_LL_CLEAN_FREQ) { \
    /* Reset all flags */ \
    static_assert((NCCL_LL_BUFF_SIZE % NCCL_LL_MAX_NTHREADS) == 0, "NCCL_LL_BUFF_SIZE must be a multiple of THREADS"); \
    static_assert(NCCL_LL_BUFF_SIZE/(sizeof(union ncclLLFifoLine)*NCCL_LL_MAX_NTHREADS) > 0, "NCCL_LL_BUFF_SIZE is less than 16 bytes*THREADS"); \
    const union ncclLLFifoLine resetLine = { 0, flag, 0, flag }; \
    for (int i=0; i<NCCL_LL_BUFF_SIZE/(sizeof(union ncclLLFifoLine)*llNthreads); i++) { \
      prevInput[tid+i*llNthreads].i4 = resetLine.i4; \
    } \
    __threadfence_system(); \
    /* Restart from the same slot, only make sure sender waits for data to be reset */ \
    step += NCCL_LL_CHUNKS; \
    ACK_PREV; \
    while (sendHeadPtr[0] < step); \
    if (tid == 0) ring->send.conn.llLastCleaning = step; \
  } \
  ring->send.conn.llStep = step; \
} while (0);

#endif
