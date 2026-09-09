#ifndef PACE_KERNELS_INCLUDE_SYNC_CUH
#define PACE_KERNELS_INCLUDE_SYNC_CUH

#include "device/common.cuh"

namespace pace {

void first_sync_func(size_t counts, uint32_t nctxs, struct ncclDevComm devComm, cudaStream_t stream);

}

#endif
