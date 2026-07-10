/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_GPI_DEVICE_HOST_COMMON_H_
#define _NCCL_DEVICE_GIN_GPI_DEVICE_HOST_COMMON_H_

#include <cstdint>

#define NCCL_GIN_GPI_VERSION 100

typedef uint8_t gpi_gfd_op_t;

typedef struct {
  uint64_t value;
} __attribute__((aligned(64))) gpi_ci_t;

/* Flag to indicate if a counted operation has been issued to the signal. */
#define GPI_SIGNAL_COUNTED_FLAG (1UL << 0)

typedef struct {
  uint64_t value;
  uint64_t flags;
} __attribute__((aligned(64))) gpi_signal_t;

#define GPI_COUNTER_SUCCESS_BITS 56
#define GPI_COUNTER_SUCCESS_MASK ((1ULL << GPI_COUNTER_SUCCESS_BITS) - 1)

/* User can set this flag in the counter to indicate if a writeback has occurred or not.
 * When cleared, the writeback has happened.
 */
#define GPI_COUNTER_FLAG_WRITEBACK_PENDING (1UL << 63)

/* When set, a GFD operation associated with a counter has experienced an error. This flag
 * is cleared by resetting the counter.
 */
#define GPI_COUNTER_FLAG_ERROR (1UL << 62)

typedef struct {
  uint64_t value;
} __attribute__((aligned(64))) gpi_counter_t;

// typedef struct {
//  uint8_t value;
//} __attribute__((aligned(64))) gpi_counter_pending_writeback_t;
/* Bit used to indicate that this is a control vs. data operation. */
#define GPI_GFD_OP_CTRL (1U << 7)

/* Supported control operations. */
enum gpi_gfd_ctrl_op {
  GPI_GFD_CTRL_OP_COUNTER_RESET = 0,
  GPI_GFD_CTRL_OP_COUNTER_RESET_NO_WRITEBACK = 1,
  GPI_GFD_CTRL_OP_COUNTER_WRITEBACK = 2,
  GPI_GFD_CTRL_OP_SIGNAL_RESET = 3,
};

/* Supported data operations. */
enum gpi_gfd_data_op {
  GPI_GFD_DATA_OP_READ = 0,
  GPI_GFD_DATA_OP_WRITE = 1,
  GPI_GFD_DATA_OP_WRITE_SIGNAL_ADD = 2,
  GPI_GFD_DATA_OP_WRITE_SIGNAL_COUNTED = 3,
  GPI_GFD_DATA_OP_WRITE_INLINE = 4,
  GPI_GFD_DATA_OP_WRITE_INLINE_SIGNAL_ADD = 5,
  GPI_GFD_DATA_OP_WRITE_INLINE_SIGNAL_COUNTED = 6,
  GPI_GFD_DATA_OP_AMO_ADD = 7,
  GPI_GFD_DATA_OP_PE_FLUSH = 8,
};

/* Data modifiers which can be OR'ed with gpi_gfd_op_data. */
#define GPI_GFD_DATA_OP_WITH_COUNTER_FLAG (1 << 0)
#define GPI_GFD_DATA_OP_WITH_COUNTER_COUNTED (1 << 1)
#define GPI_GFD_DATA_OP_WITH_COUNTER_WRITEBACK (1 << 2)

typedef union {
  uint64_t raw;
  struct {
    uint64_t owner:1;
    uint64_t resv:63;
  } __attribute__((packed)) flag;
  struct {
    uint32_t owner:1;
    uint32_t resv:31;
    uint32_t data;
  } __attribute__((packed)) inline_data;
  struct {
    uint64_t owner:1;
    uint64_t offset:63;
  } __attribute__((packed)) handle_offset;
  struct {
    uint16_t owner:1;
    uint16_t resv:15;
    uint16_t handle;
    uint32_t signal_value_high;
  } __attribute__((packed)) src_handle;
  struct {
    uint16_t owner:1;
    uint16_t resv:15;
    uint16_t handle;
    uint32_t signal_value_low;
  } __attribute__((packed)) dst_handle;
  struct {
    uint32_t owner:1;
    uint32_t pe:31;
    uint32_t size;
  } __attribute__((packed)) dst;
  struct {
    uint16_t owner:1;
    uint16_t resv:15;
    uint8_t op;
    uint8_t op_flags;
    uint16_t counter;
    uint16_t signal;
  } __attribute__((packed)) header;
} gpi_gfd_segment_t;

enum gpi_gfd_segment_id {
  GPI_GFD_SEG_HEADER = 0,

  /* Required data transfer fields. */
  GPI_GFD_DATA_DST = 1,
  GPI_GFD_DATA_DST_MEM_HANDLE = 2,
  GPI_GFD_DATA_SRC_MEM_HANDLE = 3,
  GPI_GFD_DATA_DST_MEM_HANDLE_OFFSET = 4,
  GPI_GFD_DATA_SRC_MEM_HANDLE_OFFSET = 5,
  GPI_GFD_DATA_INLINE_DATA_LOW = 6,
  GPI_GFD_DATA_INLINE_DATA_HIGH = 7,

  GPI_GFD_SEG_MAX = 8,
};

typedef union {
  gpi_gfd_segment_t segments[GPI_GFD_SEG_MAX];
} __attribute__((aligned(64))) gpi_gfd_t;

#ifdef __cplusplus
static_assert(sizeof(gpi_gfd_t) == 64, "gpi_gfd_t must be 64 bytes");
#else
_Static_assert(sizeof(gpi_gfd_t) == 64, "gpi_gfd_t must be 64 bytes");
#endif

/* Data types that are also used in the host code */
typedef struct {
  uintptr_t* gpu_memic_ptr;
  uint64_t pi_;
  gpi_ci_t* ci_;
  uint64_t ci_value_;
  uint32_t log_depth;
} Queue_t;

typedef struct {
  gpi_counter_t* gpu_counter_ptr_;
  gpi_signal_t* gpu_signal_ptr_;
  Queue_t queue_;
} gpi_gpu_channel_t;

enum gpi_resource_sharing_mode {
  GPI_RESOURCE_SHARING_MODE_EXCLUSIVE = 0, /* Resource is exclusive to one CUDA thread */
  GPI_RESOURCE_SHARING_MODE_CTA = 1, /* Resource is shared by all CUDA threads in a CTA */
  GPI_RESOURCE_SHARING_MODE_GPU = 2, /* Resource is shared by all CUDA threads in a GPU */
};
enum gpi_post_mode {
  /* Use a single thread in a warp to post a GFD. */
  GPI_POST_MODE_THREAD,

  /* Use a single thread in a warp to post a GFD using TMA copy. */
  GPI_POST_MODE_TMA,
};

#ifndef GPI_ACCESS_ONCE
#define GPI_ACCESS_ONCE(x) (*(volatile typeof(x)*)&(x))
#endif

#ifndef GPI_READ_ONCE
#define GPI_READ_ONCE(x) GPI_ACCESS_ONCE(x)
#endif

#ifndef GPI_WRITE_ONCE
#define GPI_WRITE_ONCE(x, v) (GPI_ACCESS_ONCE(x) = (v))
#endif

#endif /* _NCCL_DEVICE_GIN_GPI_DEVICE_HOST_COMMON_H_ */
