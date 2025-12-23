/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "nvtx.h"
#include "param.h"

static constexpr const nvtxPayloadEnum_t NvtxEnumRedSchema[] = {
  {"Sum", ncclSum, 0},
  {"Product", ncclProd, 0},
  {"Max", ncclMax, 0},
  {"Min", ncclMin, 0},
  {"Avg", ncclAvg, 0}
};

NCCL_PARAM(NvtxDisable, "NVTX_DISABLE", 0);

// Must be called before the first call to any reduction operation.
void initNvtxRegisteredEnums() {
  // Register schemas and strings
  if (ncclParamNvtxDisable()) {
    return;
  }

  constexpr const nvtxPayloadEnumAttr_t eAttr {
    NVTX_PAYLOAD_ENUM_ATTR_ENTRIES | NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES |
      NVTX_PAYLOAD_ENUM_ATTR_SIZE | NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID,
    NULL,
    NvtxEnumRedSchema,
    std::extent<decltype(NvtxEnumRedSchema)>::value,
    sizeof(ncclRedOp_t),
    NVTX_PAYLOAD_ENTRY_NCCL_REDOP,
    nullptr
  };

  nvtxPayloadEnumRegister(nvtx3::domain::get<nccl_domain>(), &eAttr);
}
