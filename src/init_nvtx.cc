#include "nccl.h"
#include "nvtx.h"

static constexpr const nvtxPayloadEnum_t NvtxEnumRedSchema[] = {
  {"Sum", ncclSum},
  {"Product", ncclProd},
  {"Max", ncclMax},
  {"Min", ncclMin},
  {"Avg", ncclAvg}
};

// Must be called before the first call to any reduction operation.
void initNvtxRegisteredEnums() {
  // Register schemas and strings
  constexpr const nvtxPayloadEnumAttr_t eAttr {
    .fieldMask = NVTX_PAYLOAD_ENUM_ATTR_ENTRIES | NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES |
      NVTX_PAYLOAD_ENUM_ATTR_SIZE | NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID,
    .name = NULL,
    .entries = NvtxEnumRedSchema,
    .numEntries = std::extent<decltype(NvtxEnumRedSchema)>::value,
    .sizeOfEnum = sizeof(ncclRedOp_t),
    .schemaId = NVTX_PAYLOAD_ENTRY_NCCL_REDOP
  };

  nvtxPayloadEnumRegister(nvtx3::domain::get<nccl_domain>(), &eAttr);
}
