/**
 * The NVTX counters extension is intended to collect counter values of various
 * sources. It uses the NVTX payload extension to specify the data layout a
 * counter group.
 *
 * A counter group is a set of counters that are collected together (at the same
 * time). Counters are always registered as a group. Hence, a single counter is
 * represented by a group with one counter.
 *
 * A sample refers to all values for a given timestamp. These values must
 * include counter values and may include multiple instances of a counter group.
 *
 * The NVTX domain handle is the first argument to all counter collect
 * functions. 0/NULL/nullptr represents the default domain (no domain).
 */

#include "nvToolsExtPayload.h"

#ifndef NVTOOLSEXT_COUNTERS_H
#define NVTOOLSEXT_COUNTERS_H

/**
 * \brief The compatibility ID is used for versioning of this extension.
 */
#ifndef NVTX_EXT_COUNTERS_COMPATID
#define NVTX_EXT_COUNTERS_COMPATID 0x0101
#endif

/**
 * \brief The module ID identifies the payload extension. It has to be unique
 * among the extension modules.
 */
#ifndef NVTX_EXT_COUNTERS_MODULEID
#define NVTX_EXT_COUNTERS_MODULEID 4
#endif


/** Identifies an invalid scope and indicates an error if returned by `nvtxScopeRegister`. */
#define NVTX_SCOPE_NONE                   0 /* no scope */

#define NVTX_SCOPE_ROOT                   1

#define NVTX_SCOPE_CURRENT_HW_MACHINE     2 /* Node/machine name, Device? */
#define NVTX_SCOPE_CURRENT_HW_SOCKET      3
#define NVTX_SCOPE_CURRENT_HW_CPU         4
#define NVTX_SCOPE_CURRENT_HW_CPU_LOGICAL 5
/* Innermost HW execution context at registration time */
#define NVTX_SCOPE_CURRENT_HW_INNERMOST   6

/* Virtualized hardware, virtual machines, OS (if you don't know any better) */
#define NVTX_SCOPE_CURRENT_HYPERVISOR     7
#define NVTX_SCOPE_CURRENT_VM             8
#define NVTX_SCOPE_CURRENT_KERNEL         9
#define NVTX_SCOPE_CURRENT_CONTAINER     10
#define NVTX_SCOPE_CURRENT_OS	         11

/* Software scopes */
#define NVTX_SCOPE_CURRENT_SW_PROCESS 	 12 /* Process scope */
#define NVTX_SCOPE_CURRENT_SW_THREAD  	 13 /* Thread scope */
#define NVTX_SCOPE_CURRENT_SW_FIBER      14
/* Innermost SW execution context at registration time */
#define NVTX_SCOPE_CURRENT_SW_INNERMOST  15

/** Static (user-provided) scope IDs (feed forward) */
#define NVTX_SCOPE_ID_STATIC_START  (1 << 24)

/** Dynamically (tool) generated scope IDs */
#define NVTX_SCOPE_ID_DYNAMIC_START 4294967296  /* 1 << 32 */


/** Identifier of the semantic extension for counters. */
#define NVTX_SEMANTIC_ID_COUNTERS_V1 5

/***  Flags to augment the counter value. ***/
#define NVTX_COUNTERS_FLAG_NONE       0

/**
 * Convert the fixed point value to a normalized floating point.
 * Use the sign/unsign from the underlying type this flag is applied to.
 * Unsigned [0f : 1f] or signed [-1f : 1f]
 */
#define NVTX_COUNTERS_FLAG_NORM       (1 << 1)

/**
 * Tools should apply scale and limits when graphing, ideally in a "soft" way to
 * to see when limits are exceeded.
 */
#define NVTX_COUNTERS_FLAG_LIMIT_MIN  (1 << 2)
#define NVTX_COUNTERS_FLAG_LIMIT_MAX  (1 << 3)
#define NVTX_COUNTERS_FLAG_LIMITS \
    (NVTX_COUNTERS_FLAG_LIMIT_MIN | NVTX_COUNTERS_FLAG_LIMIT_MAX)

/** Counter time scope **/
#define NVTX_COUNTERS_FLAG_TIME_POINT       (1 << 5)
#define NVTX_COUNTERS_FLAG_TIME_SINCE_LAST  (2 << 5)
#define NVTX_COUNTERS_FLAG_TIME_UNTIL_NEXT  (3 << 5)
#define NVTX_COUNTERS_FLAG_TIME_SINCE_START (4 << 5)

/** Counter value type **/
#define NVTX_COUNTERS_FLAG_VALUE_ABSOLUTE   (1 << 10)
#define NVTX_COUNTERS_FLAG_VALUE_DELTA      (2 << 10) // delta to previous counter sample

/** Counter visualization hints **/
#define NVTX_COUNTERS_FLAG_INTERPOLATE      (1 << 14)

/** Datatypes for limits union (value of `limitType`). */
#define NVTX_COUNTERS_LIMIT_I64 0
#define NVTX_COUNTERS_LIMIT_U64 1
#define NVTX_COUNTERS_LIMIT_F64 2

/** Reasons for the missing sample value. */
#define NVTX_COUNTERS_SAMPLE_ZERO        0
#define NVTX_COUNTERS_SAMPLE_UNCHANGED   1
#define NVTX_COUNTERS_SAMPLE_UNAVAILABLE 2

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * \brief Specify additional properties of a counter or counter group.
 */
typedef struct nvtxSemanticsCounter_v1
{
    /** Header of the semantic extension (with identifier, version, etc.). */
    struct nvtxSemanticsHeader_v1 header;

    /**
     * Flag if normalization, scale limits, etc. should be applied to counter
     * values.
     */
    uint64_t flags;

    /** Unit of the counter value (case insensitive) */
    const char* unit;

    /** Should be 1 if not used. */
    uint64_t unitScaleNumerator;

    /** Should be 1 if not used. */
    uint64_t unitScaleDenominator;

    /** Determines the used union member. Use defines `NVTX_COUNTERS_LIMIT_*`. */
    int64_t limitType;

    /** Soft graph limit. */
    union limits_t {
        int64_t i64[2];
        uint64_t u64[2];
        double d[2];
    } limits;
} nvtxSemanticsCounter_t;

typedef struct nvtxCountersAttr_v1
{
    size_t structSize;

    /**
     * A schema ID referring to the data layout of the counter group or a
     * predefined NVTX payloads number type.
     */
    uint64_t schemaId;

    /** Name of the counter group. */
    const char* name;

    /** Identifier of the scope of the counters. */
    uint64_t scopeId;

    /**
     * (Optional) Specify additional semantics for a counter (group). The
     * semantics provided are applied to the all counters in a group. If the
     * semantics should only refer to a single counter in a group, the semantics
     * field of the payload entry has to be used. Accepted semantics are
     * `nvtxSemanticsCounter_t` and `nvtxSemanticsTime_t`.
     */
    const nvtxSemanticsHeader_t* semantics;
} nvtxCountersAttr_t;

/* Forward declaration of opaque counter group registration structure */
struct nvtxCountersRegistration_st;
typedef struct nvtxCountersRegistration_st nvtxCountersRegistration;

/* \brief Counters Handle Structure.
* \anchor COUNTERS_HANDLE_STRUCTURE
*
* This structure is opaque to the user and is used as a handle to reference a counter group.
* This type is returned from tools when using the NVTX API to create a counters group.
*/
typedef nvtxCountersRegistration* nvtxCountersHandle_t;

typedef struct nvtxCountersBatch_v1
{
    /** Handle to attributes (data layout, scope, etc.) of a counter (group). */
    nvtxCountersHandle_t hCounter;

    /** Array of counter samples. */
    const void* counters;

    /** Size of the `counters` array (in bytes). */
    size_t cntArrSize;

    /** Array of timestamps or reference-time plus delta pair. `NULL` is used, if
    timestamps are part of the counter (group) layout.) */
    const void* timestamps;

    /** Size of the `timestamps` array or definition (in bytes). */
    size_t tsSize;
} nvtxCountersBatch_t;

/**
 * \brief Register a counter group.
 *
 * @param hDomain NVTX domain handle.
 * @param attr Pointer to the attributes of the counter (group).
 *
 * @return Counter handle identifying a counter or counter (group).
 *         The counter handle is unique within the NVTX domain.
 */
NVTX_DECLSPEC nvtxCountersHandle_t NVTX_API nvtxCountersRegister(
    nvtxDomainHandle_t hDomain,
    const nvtxCountersAttr_t* attr);

/**
 * \brief Sample one integer counter by value immediately (the NVTX tool determines the timestamp).
 *
 * @param hDomain handle of the NVTX domain.
 * @param hCounter handle of the NVTX counter (group).
 * @param value 64-bit integer counter value.
 */
NVTX_DECLSPEC void NVTX_API nvtxCountersSampleInt64(
    nvtxDomainHandle_t hDomain,
    nvtxCountersHandle_t hCounter,
    int64_t value);

/**
 * \brief Sample one floating point counter by value immediately (the NVTX tool determines the timestamp).
 *
 * @param hDomain handle of the NVTX domain.
 * @param hCounter handle of the NVTX counter (group).
 * @param value 64-bit floating-point counter value.
 */
NVTX_DECLSPEC void NVTX_API nvtxCountersSampleFloat64(
    nvtxDomainHandle_t hDomain,
    nvtxCountersHandle_t hCounter,
    double value);

/**
 * \brief Sample a counter group by reference immediately (the NVTX tool determines the timestamp).
 *
 * @param hDomain handle of the NVTX domain.
 * @param hCounter handle of the NVTX counter (group).
 * @param counters pointer to one or more counter values.
 * @param size size of the counter value(s) in bytes.
 */
NVTX_DECLSPEC void NVTX_API nvtxCountersSample(
    nvtxDomainHandle_t hDomain,
    nvtxCountersHandle_t hCounter,
    void* values,
    size_t size);

/**
 * \brief Sample without value.
 *
 * @param hDomain handle of the NVTX domain.
 * @param hCounter handle of the NVTX counter (group).
 * @param reason reason for the missing sample value.
 */
NVTX_DECLSPEC void NVTX_API nvtxCountersSampleNoValue(
    nvtxDomainHandle_t hDomain,
    nvtxCountersHandle_t hCounter,
    uint8_t reason);

/**
 * \brief Submit a batch of counters in the given domain.
 *        Timestamps are part of the counter sample data.
 *
 * The size of a data sampling point is defined by the `staticSize` field of the
 * payload schema. An NVTX tool can assume that the counter samples are stored
 * as an array with each entry being `staticSize` bytes.
 *
 * @param hDomain handle of the NVTX domain
 * @param hCounter handle of the counter group (includes counter data decoding schema)
 * @param counters blob containing counter data and timestamps
 * @param size size of the counter data blob in bytes
 */
NVTX_DECLSPEC void NVTX_API nvtxCountersSubmitBatch(
    nvtxDomainHandle_t hDomain,
    nvtxCountersHandle_t hCounter,
    const void* counters,
    size_t size);

/**
 * \brief Submit a batch of counters in the given domain.
 *        Timestamps are separated from the counter data.
 *
 * @param hDomain handle of the NVTX domain
 * @param counterBatch Pointer to the counter data to be submitted.
 */
NVTX_DECLSPEC void NVTX_API nvtxCountersSubmitBatchEx(
    nvtxDomainHandle_t hDomain,
    const nvtxCountersBatch_t* counterBatch);


#define NVTX3EXT_CBID_nvtxCountersRegister           0
#define NVTX3EXT_CBID_nvtxCountersSampleInt64        1
#define NVTX3EXT_CBID_nvtxCountersSampleFloat64      2
#define NVTX3EXT_CBID_nvtxCountersSample             3
#define NVTX3EXT_CBID_nvtxCountersSampleNoValue      4
#define NVTX3EXT_CBID_nvtxCountersSubmitBatch        5
#define NVTX3EXT_CBID_nvtxCountersSubmitBatchEx      6

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

#define NVTX_EXT_TYPES_GUARD /* Ensure other headers cannot be included directly */
#include "nvtxDetail/nvtxExtTypes.h"
#undef NVTX_EXT_TYPES_GUARD

#ifndef NVTX_NO_IMPL
#define NVTX_EXT_IMPL_COUNTERS_GUARD /* Ensure other headers cannot be included directly */
#include "nvtxDetail/nvtxExtImplCounters_v1.h"
#undef NVTX_EXT_IMPL_COUNTERS_GUARD
#endif /*NVTX_NO_IMPL*/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NVTOOLSEXT_COUNTERS_H */