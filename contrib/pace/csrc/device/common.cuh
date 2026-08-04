#ifndef PACE_KERNELS_INCLUDE_COMMON_CUH
#define PACE_KERNELS_INCLUDE_COMMON_CUH

#include "nccl.h"
#include "nccl_device.h"
#include <cstddef>
#include <cstdint>

#define GIN_PHASE_SEND 1
#define GIN_PHASE_RECV 2
#define GIN_PHASE_ALL  (GIN_PHASE_SEND | GIN_PHASE_RECV)

#define GIN_CTA_THREADS 1024

#endif
