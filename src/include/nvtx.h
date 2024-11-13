/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVTX_H_
#define NCCL_NVTX_H_

#include "nvtx3/nvtx3.hpp"

#if __cpp_constexpr >= 201304L && !defined(NVTX3_CONSTEXPR_IF_CPP14)
#define NVTX3_CONSTEXPR_IF_CPP14 constexpr
#else
#define NVTX3_CONSTEXPR_IF_CPP14
#endif

// Define all NCCL-provided static schema IDs here (avoid duplicates).
#define NVTX_SID_CommInitRank         0
#define NVTX_SID_CommInitAll          1
#define NVTX_SID_CommDestroy          2 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommAbort            3 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_AllGather            4
#define NVTX_SID_AllReduce            5
#define NVTX_SID_Broadcast            6
#define NVTX_SID_ReduceScatter        7
#define NVTX_SID_Reduce               8
#define NVTX_SID_Send                 9
#define NVTX_SID_Recv                 10
#define NVTX_SID_CommInitRankConfig   11 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommInitRankScalable 12 // same schema as NVTX_SID_CommInitRank
#define NVTX_SID_CommSplit            13

// Define static schema ID for the reduction operation.
#define NVTX_PAYLOAD_ENTRY_NCCL_REDOP 14 + NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START

extern const nvtxDomainHandle_t ncclNvtxDomainHandle;

struct nccl_domain{static constexpr char const* name{"NCCL"};};

class payload_schema {
 public:
  explicit payload_schema(const nvtxPayloadSchemaEntry_t entries[], size_t numEntries, const uint64_t schemaId, const char* schemaName = nullptr) noexcept
  {
    schema_attr.name = schemaName;
    schema_attr.entries = entries;
    schema_attr.numEntries = numEntries;
    schema_attr.schemaId = schemaId;
    nvtxPayloadSchemaRegister(nvtx3::domain::get<nccl_domain>(), &schema_attr);
  }

  payload_schema() = delete;
  ~payload_schema() = default;
  payload_schema(payload_schema const&) = default;
  payload_schema& operator=(payload_schema const&) = default;
  payload_schema(payload_schema&&) = default;
  payload_schema& operator=(payload_schema&&) = default;

 private:
  nvtxPayloadSchemaAttr_t schema_attr{
    NVTX_PAYLOAD_SCHEMA_ATTR_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE |
    NVTX_PAYLOAD_SCHEMA_ATTR_SCHEMA_ID,
    nullptr,
    NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
    NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
    nullptr, 0, 0, 0, 0, nullptr};
};

// Create NVTX push/pop range with parameters
// @param name of the operation (see `NVTX_SID_*`)
// @param N  schema name
// @param S  schema (entries)
// @param P  payload (struct)
#define NVTX3_FUNC_WITH_PARAMS(ID, S, P) \
  static const payload_schema schema{S, std::extent<decltype(S)>::value, \
    NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##ID, #ID}; \
  static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
  nvtxPayloadData_t nvtx3_bpl__[] = { \
    {NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##ID, sizeof(P), &(P)}}; \
  ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__, nvtx3_bpl__}; \
  ::nvtx3::v1::scoped_range_in<nccl_domain> const nvtx3_range__{nvtx3_func_attr__};

extern void initNvtxRegisteredEnums();

#endif
