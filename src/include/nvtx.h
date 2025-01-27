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
#define NVTX_SID_CommFinalize         14

// Define static schema ID for the reduction operation.
#define NVTX_PAYLOAD_ENTRY_NCCL_REDOP 14 + NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START

extern const nvtxDomainHandle_t ncclNvtxDomainHandle;

struct nccl_domain{static constexpr char const* name{"NCCL"};};

/// @brief Register an NVTX payload schema for static-size payloads.
class payload_schema {
 public:
  explicit payload_schema(const nvtxPayloadSchemaEntry_t entries[], size_t numEntries,
    const uint64_t schemaId, const size_t size) noexcept
  {
    schema_attr.payloadStaticSize = size;
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
    nullptr, /* schema name is not needed */
    NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
    NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
    nullptr, 0, 0, 0, 0, nullptr};
};

// Convenience macro to give the payload parameters a scope.
#define NVTX3_PAYLOAD(...) __VA_ARGS__

// Create NVTX push/pop range with parameters
// @param N NCCL API name without the `nccl` prefix.
// @param T name of the used NVTX payload schema without "Schema" suffix.
// @param P payload parameters/entries
#define NVTX3_FUNC_WITH_PARAMS(N, T, P) \
  constexpr uint64_t schemaId = NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##N; \
  static const payload_schema schema{T##Schema, std::extent<decltype(T##Schema)>::value - 1, \
    schemaId, sizeof(T)}; \
  static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
  const T _payload = {P}; \
  nvtxPayloadData_t nvtx3_bpl__[] = {{schemaId, sizeof(_payload), &_payload}}; \
  ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__, nvtx3_bpl__}; \
  ::nvtx3::v1::scoped_range_in<nccl_domain> const nvtx3_range__{nvtx3_func_attr__};

/// @brief Creates an NVTX range with extended payload using the RAII pattern.
/// @tparam PayloadType Data type of the payload.
template <typename PayloadType>
class ncclNvtxRange {
 public:
  explicit ncclNvtxRange(const nvtxEventAttributes_t* evtAttr) noexcept {
    nvtxDomainRangePushEx(nvtx3::domain::get<nccl_domain>(), evtAttr);
  }

  ~ncclNvtxRange() noexcept {
    if (payloadData.payload) {
      nvtxRangePopPayload(nvtx3::domain::get<nccl_domain>(), &payloadData, 1);
    } else {
      nvtxDomainRangePop(nvtx3::domain::get<nccl_domain>());
    }
  }

  void setPayloadData(const uint64_t schemaId) noexcept
  {
    payloadData = {schemaId, sizeof(PayloadType), &payload};
  }

  ncclNvtxRange() = delete;
  ncclNvtxRange(ncclNvtxRange const&) = default;
  ncclNvtxRange& operator=(ncclNvtxRange const&) = default;
  ncclNvtxRange(ncclNvtxRange&&) = default;
  ncclNvtxRange& operator=(ncclNvtxRange&&) = default;

  // Holds the payload data.
  PayloadType payload{};

 private:
  nvtxPayloadData_t payloadData = {NVTX_PAYLOAD_ENTRY_TYPE_INVALID, 0, NULL};
};

// Create an NVTX range with the function name as the range name. Use RAII pattern.
// @param T Type ID of the NVTX payload (pointer for variable-size payloads).
#define NVTX3_RANGE(T) \
  static ::nvtx3::v1::registered_string_in<nccl_domain> const nvtx3_func_name__{__func__}; \
  ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
  ncclNvtxRange<T> nvtx3_range__{nvtx3_func_attr__.get()};

// Add static-size payload to the NVTX range created with `NVTX3_RANGE()`,
// which must be in this or an outer scope.
// @param N NCCL API name without the `nccl` prefix.
// @param S name of the used NVTX payload schema.
// @param P payload parameters/entries
#define NVTX3_RANGE_ADD_PAYLOAD(N, S, P) do { \
  constexpr uint64_t schema_id = NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START + NVTX_SID_##N; \
  static const payload_schema schema{S, std::extent<decltype(S)>::value - 1, schema_id, \
    sizeof(nvtx3_range__.payload)}; \
  nvtx3_range__.payload = {P}; \
  nvtx3_range__.setPayloadData(schema_id); \
} while (0)

extern void initNvtxRegisteredEnums();

#endif
