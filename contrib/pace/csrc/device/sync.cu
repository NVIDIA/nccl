#include "device/sync.cuh"
#include <cuda.h>
#include "device/configs.cuh"
#include "device/mem.cuh"
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <sys/types.h>
#include "util/error.hpp"
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>

namespace pace {

__global__ void FirstSync(size_t counts, uint32_t nctxs, struct ncclDevComm devComm) {
    // Allow nctxs > blockDim.x (e.g. per-SM exclusive contexts on H100 with 132 SMs).
    for (uint32_t ctx = threadIdx.x; ctx < nctxs; ctx += blockDim.x) {
        ncclGin gin {devComm, int(ctx)};
        for (size_t i = 0; i < counts; ++i) {
            gin.resetSignal(i);
        }
    }
}

void first_sync_func(size_t counts, uint32_t nctxs, struct ncclDevComm devComm, cudaStream_t stream) {
    const uint32_t block = 256;
    FirstSync<<<1, block, 0, stream>>>(counts, nctxs, devComm);
    CUDACHECK(cudaGetLastError());
}

}