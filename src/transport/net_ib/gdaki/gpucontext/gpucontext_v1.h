#include <linux/types.h>
#include <stdint.h>
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"

struct ncclGinGdakiGPUContext_v1 {
  struct doca_gpu_dev_verbs_qp *gdqp;
  struct doca_gpu_dev_verbs_qp *companion_gdqp;
  struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table;
  struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table;

  // Local buffer we don't consume but is required for some operations.
  __be32 sink_buffer_lkey;
};

static_assert(sizeof(ncclGinGdakiGPUContext_v1) == 72);

void ncclGinGdakiGPUContext_v1_init(void* ctxArray, int idx,
                                    struct doca_gpu_dev_verbs_qp* gdqp,
                                    struct doca_gpu_dev_verbs_qp* companion_gdqp,
                                    struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table,
                                    struct ncclGinGdakiGlobalGPUBufferTable<uint64_t> signals_table,
                                    __be32 sink_buffer_lkey) {
  ncclGinGdakiGPUContext_v1* ctx = (ncclGinGdakiGPUContext_v1*)ctxArray + idx;
  ctx->gdqp            = gdqp;
  ctx->companion_gdqp  = companion_gdqp;
  ctx->counters_table  = counters_table;
  ctx->signals_table   = signals_table;
  ctx->sink_buffer_lkey = sink_buffer_lkey;
}
