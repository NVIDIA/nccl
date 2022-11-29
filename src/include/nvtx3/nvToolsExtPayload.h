/*
* Copyright 2021  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "nvtx3/nvToolsExt.h"

#ifndef NVTOOLSEXT_PAYLOAD_H
#define NVTOOLSEXT_PAYLOAD_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * \brief A compatibility ID value used in initialization to identify version
 * differences.
 */
#define NVTX_EXT_COMPATID_PAYLOAD 0x0103

/**
 * \brief This module ID identifies the payload extension. It has to be unique
 * among the extension modules.
 */
#define NVTX_EXT_MODULEID_PAYLOAD 2

/**
 * \brief Additional values for the enum @ref nvtxPayloadType_t
 */
#define NVTX_PAYLOAD_TYPE_BINARY ((int32_t)0xDFBD0009)


/** ---------------------------------------------------------------------------
 * Payload schema entry flags.
 * ------------------------------------------------------------------------- */
#define NVTX_PAYLOAD_ENTRY_FLAG_UNUSED 0

/**
 * Absolute pointer into a payload (entry) of the same event.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_POINTER          (1 << 1)

/**
 * Offset from base address of the payload.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_OFFSET_FROM_BASE (1 << 2)

/**
 * Offset from the end of this payload entry.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_OFFSET_FROM_HERE (1 << 3)

/**
 * The value is an array with fixed length, set with the field `arrayLength`.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE      (1 << 4)

/**
 * The value is a zero-/null-terminated array.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED (2 << 4)

/**
 * \brief A single or multi-dimensional array of variable length.
 *
 * The field `arrayLength` contains the index of the schema entry that holds the
 * length(s). If the other field points to a scalar entry then this will be the
 * 1D array. If the other field points to a FIXED_SIZE array, then the number of
 * dimensions is defined with the registration of the scheme. If the other field
 * is ZERO_TERMINATED, the array the dimensions can be determined at runtime.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX    (3 << 4)

/**
 * A tool may not support deep copy and just ignore this flag.
 * See @ref NVTX_PAYLOAD_SCHEMA_FLAG_DEEP_COPY for more details.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY             (1 << 9)

/**
 * The entry specifies the message in a deferred event. The entry type can be
 * any string type. The flag is ignored for schemas that are not flagged with
 * `NVTX_PAYLOAD_SCHEMA_FLAG_RANGE*` or `NVTX_PAYLOAD_SCHEMA_FLAG_MARK`.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE         (1 << 10)

/**
 * @note The ‘array’ flags assume that the array is embedded. Otherwise,
 * @ref NVTX_PAYLOAD_ENTRY_FLAG_POINTER has to be additionally specified. Some
 * combinations may be invalid based on the `NVTX_PAYLOAD_SCHEMA_TYPE_*` this
 * entry is enclosed. For instance, variable length embedded arrays are valid
 * within @ref NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC but invalid with
 * @ref NVTX_PAYLOAD_SCHEMA_TYPE_STATIC. See `NVTX_PAYLOAD_SCHEMA_TYPE_*` for
 * additional details.
 */

/* Helper macro to check if an entry represents an array. */
#define NVTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY (\
    NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE | \
    NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED | \
    NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX)

/** ---------------------------------------------------------------------------
 * Types of entries in a payload schema.
 * ------------------------------------------------------------------------- */

/**
 * @note Several of the predefined types contain the size (in bits) in their
 * names. For some data types the size (in bytes) is not fixed and may differ
 * for different platforms/operating systems/compilers. To provide portability,
 * an array of sizes (in bytes) for type 1 to 28 ( @ref
 * NVTX_PAYLOAD_ENTRY_TYPE_CHAR to @ref NVTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE)
 * is passed to the NVTX extension initialization function
 * @ref InitializeInjectionNvtxExtension via the `extInfo` field of
 * @ref nvtxExtModuleInfo_t.
 */

#define NVTX_PAYLOAD_ENTRY_TYPE_INVALID 0

/**
 * Basic integer types.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_CHAR        1
#define NVTX_PAYLOAD_ENTRY_TYPE_UCHAR       2
#define NVTX_PAYLOAD_ENTRY_TYPE_SHORT       3
#define NVTX_PAYLOAD_ENTRY_TYPE_USHORT      4
#define NVTX_PAYLOAD_ENTRY_TYPE_INT         5
#define NVTX_PAYLOAD_ENTRY_TYPE_UINT        6
#define NVTX_PAYLOAD_ENTRY_TYPE_LONG        7
#define NVTX_PAYLOAD_ENTRY_TYPE_ULONG       8
#define NVTX_PAYLOAD_ENTRY_TYPE_LONGLONG    9
#define NVTX_PAYLOAD_ENTRY_TYPE_ULONGLONG  10

/**
 * Integer types with explicit size.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_INT8       11
#define NVTX_PAYLOAD_ENTRY_TYPE_UINT8      12
#define NVTX_PAYLOAD_ENTRY_TYPE_INT16      13
#define NVTX_PAYLOAD_ENTRY_TYPE_UINT16     14
#define NVTX_PAYLOAD_ENTRY_TYPE_INT32      15
#define NVTX_PAYLOAD_ENTRY_TYPE_UINT32     16
#define NVTX_PAYLOAD_ENTRY_TYPE_INT64      17
#define NVTX_PAYLOAD_ENTRY_TYPE_UINT64     18

/**
 * C floating point types
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_FLOAT      19
#define NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE     20
#define NVTX_PAYLOAD_ENTRY_TYPE_LONGDOUBLE 21

/**
 * Size type (`size_t`)
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_SIZE       22

/**
 * Any address, e.g. `void*`. If the pointer type matters, use the flag @ref
 * NVTX_PAYLOAD_ENTRY_FLAG_POINTER and the respective type instead.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS    23

/**
 * Special character types.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_WCHAR      24 /* wide character (since C90) */
#define NVTX_PAYLOAD_ENTRY_TYPE_CHAR8      25 /* since C2x and C++20 */
#define NVTX_PAYLOAD_ENTRY_TYPE_CHAR16     26
#define NVTX_PAYLOAD_ENTRY_TYPE_CHAR32     27

/**
 * There is type size and alignment information for all previous types.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE (NVTX_PAYLOAD_ENTRY_TYPE_CHAR32 + 1)

/**
 * Store raw 8-bit binary data. As with `char`, 1-byte alignment is assumed.
 * Typically a tool will display this as hex or binary.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_BYTE       32

/**
 * These types do not have standardized equivalents. It is assumed that the
 * number at the end corresponds to the bits used to store the value and that
 * the alignment corresponds to standardized types of the same size.
 * A tool may not support these types.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_INT128     33
#define NVTX_PAYLOAD_ENTRY_TYPE_UINT128    34

#define NVTX_PAYLOAD_ENTRY_TYPE_FLOAT16    42
#define NVTX_PAYLOAD_ENTRY_TYPE_FLOAT32    43
#define NVTX_PAYLOAD_ENTRY_TYPE_FLOAT64    44
#define NVTX_PAYLOAD_ENTRY_TYPE_FLOAT128   45

#define NVTX_PAYLOAD_ENTRY_TYPE_BF16       50
#define NVTX_PAYLOAD_ENTRY_TYPE_TF32       52

/**
 * These types are normalized numbers stored in integers. UNORMs represent 0.0
 * to 1.0 and SNORMs represent -1.0 to 1.0. The number after represents the
 * number of integer bits. Alignment is take from equivalent types INT# matching
 * to SNORM# and UINT# matching to UNORM#.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_SNORM8     61
#define NVTX_PAYLOAD_ENTRY_TYPE_UNORM8     62
#define NVTX_PAYLOAD_ENTRY_TYPE_SNORM16    63
#define NVTX_PAYLOAD_ENTRY_TYPE_UNORM16    64
#define NVTX_PAYLOAD_ENTRY_TYPE_SNORM32    65
#define NVTX_PAYLOAD_ENTRY_TYPE_UNORM32    66
#define NVTX_PAYLOAD_ENTRY_TYPE_SNORM64    67
#define NVTX_PAYLOAD_ENTRY_TYPE_UNORM64    68

/**
 * String types.
 *
 * If `arrayOrUnionDetail` is greater than `0`, the entry is a fixed-size string
 * with the provided length.
 *
 * `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE` is ignored for string types. It
 * just specifies once more that the entry is a fixed-size string.
 *
 * Setting the flag `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED` indicates a
 * zero-terminated string. If `arrayOrUnionDetail` is greater than `0`, a zero-
 * terminated array of fixed-size strings is assumed.
 *
 * Setting the flag `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX` specifies the
 * entry index of the entry which contains the string length. It is not possible
 * to describe a variable length array of strings.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING       75 /* `char*`, system LOCALE */
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8  76
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF16 77
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32 78

/**
 * @ref nvtxStringHandle_t returned by @ref nvtxDomainRegisterString
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE 80

/**
 * Entry types to be used in deferred events. Data types are as defined by
 * NVTXv3 core: category -> uint32_t, color -> uint32_t, color type -> int32_t.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_NVTX_CATEGORY    90
#define NVTX_PAYLOAD_ENTRY_TYPE_NVTX_COLORTYPE   91
#define NVTX_PAYLOAD_ENTRY_TYPE_NVTX_COLOR       92

/**
 * This type marks the union selector member (entry index) in schemas used by
 * a union with internal internal selector.
 * See @ref NVTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_UNION_SELECTOR 100

/**
 * Timestamp types occupy the range from 128 to 255
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP64 128 /* data type is uint64_t */

/**
 * CPU timestamp sources.
 * \todo All 64 bits?
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_TSC                              129
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_TSC_NONVIRTUALIZED               130
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_REALTIME           131
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_REALTIME_COARSE    132
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_MONOTONIC          133
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_MONOTONIC_RAW      134
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_MONOTONIC_COARSE   135
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_BOOTTIME           136
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_PROCESS_CPUTIME_ID 137
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPU_CLOCK_GETTIME_THREAD_CPUTIME_ID  138

#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_WIN_QPC     160
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_WIN_GSTAFT  161
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_WIN_GSTAFTP 162

#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_C_TIME         163
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_C_CLOCK        164
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_C_TIMESPEC_GET 165

#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_STEADY_CLOCK          166
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_HIGH_RESOLUTION_CLOCK 167
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_SYSTEM_CLOCK          168
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_UTC_CLOCK             169
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_TAI_CLOCK             170
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_GPS_CLOCK             171
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_CPP_FILE_CLOCK            172

/**
 * \brief GPU timestamp sources.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_GPU_GLOBALTIMER 192
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_GPU_SM_CLOCK    193
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_GPU_SM_CLOCK64  194
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_GPU_CUPTI       195

/**
 * The timestamp was provided by the NVTX handler’s timestamp routine.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_TIMESTAMP_TOOL_PROVIDED 224

/**
 * This predefined schema ID can be used in `nvtxPayloadData_t` to indicate that
 * the payload is a blob of memory which other payload entries may point into.
 * A tool will not expose this payload directly.
 */
#define NVTX_TYPE_PAYLOAD_SCHEMA_REFERENCED 1022

/**
 * This predefined schema ID can be used in `nvtxPayloadData_t` to indicate that
 * the payload is a blob which can be shown with an arbitrary data viewer.
 */
#define NVTX_TYPE_PAYLOAD_SCHEMA_RAW        1023

/* Custom (static) schema IDs. */
#define NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START  (1 << 24)

/* Dynamic schema IDs (generated by the tool) start here. */
#define NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_DYNAMIC_START 4294967296  // 1 << 32


/**
 * \brief Size and alignment information for predefined payload entry types.
 *
 * The struct contains the size and the alignment size in bytes. A respective
 * array for the predefined types is passed via nvtxExtModuleInfo_t to the NVTX
 * client/handler. The type (ID) is used as index into this array.
 */
typedef struct nvtxPayloadEntryTypeInfo_t
{
    uint16_t size;
    uint16_t align;
} nvtxPayloadEntryTypeInfo_t;

/**
 * \brief Entry in a schema.
 *
 * A payload schema consists of an array of payload schema entries. It is
 * registered with @ref nvtxPayloadSchemaRegister. `flag` can be set to `0` for
 * simple values, 'type' is the only "required" field. If not set explicitly,
 * all other fields are zero-initialized, which means that the entry has no name
 * and the offset is determined based on self-alignment rules.
 *
 * Example schema:
 *  nvtxPayloadSchemaEntry_t desc[] = {
 *      {0, NVTX_EXT_PAYLOAD_TYPE_UINT8, "one byte"},
 *      {0, NVTX_EXT_PAYLOAD_TYPE_INT32, "four bytes"}
 *  };
 */
typedef struct nvtxPayloadSchemaEntry_t
{
    /**
     * \brief Flags to augment the basic type.
     *
     * This field allows additional properties of the payload entry to be
     * specified. Valid values are `NVTX_PAYLOAD_ENTRY_FLAG_*`.
     */
    uint64_t       flags;

    /**
     * \brief Predefined payload schema entry type or ID of a registered payload
     * schema.
     */
    uint64_t       type;

    /**
     * \brief Name of the payload entry. (Optional)
     *
     * Providing a name is useful to give a meaning to the associated value.
     */
    const char*    name;

    /**
     * \brief Description of the payload entry. (Optional)
     */
    const char*    description;

    /**
     * \brief String or array length or union selector for union types.
     *
     * If @ref type is a C string type, this defines the length of the string.
     *
     * If @ref flags specify that the entry is an array, this field defines the
     * length of the array. See `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_*` for more
     * details.
     *
     * If @ref type implies that the entry is a union with schema type
     * @ref NVTX_PAYLOAD_SCHEMA_TYPE_UNION (external selection of the union
     * member), this field contains the index (starting with 0) to an entry of
     * integer type in the same schema. The associated field contains the
     * selected union member.
     *
     * @note An array of schema type @ref NVTX_PAYLOAD_SCHEMA_TYPE_UNION is not
     * supported. @ref NVTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR can
     * be used instead.
     */
    uint64_t       arrayOrUnionDetail;

    /**
     * \brief Offset in the binary payload data (in bytes).
     *
     * This field specifies the byte offset from the base address of the actual
     * binary data (blob) to the data of this entry.
     *
     * This is an optional field, but it is recommended to specify this field to
     * avoid issues in the automatic detection of the offset by a tool/handler.
     */
    uint64_t       offset;

    /**
     * Semantics are not yet defined.
     */
    void*          semantics;

    /**
     * Reserved for future use. Do not use it!
     */
    void*          reserved;
} nvtxPayloadSchemaEntry_t;

/**
 * \brief Binary payload data, size and decoding information.
 *
 * An array of nvtxPayloadData_t is passed to the NVTX event attribute payload
 * member. To attach a single payload the macro @ref NVTX_EXT_PAYLOAD_SET_ATTR
 * can be used.
 */
typedef struct nvtxPayloadData_t
{
    /**
     * The schema ID, which defines the layout of the binary data.
     */
    uint64_t    schemaId;

    /**
     * Size of the binary payload (blob) in bytes.
     */
    size_t      size;

    /**
     * Pointer to the binary payload data.
     */
    const void* payload;
} nvtxPayloadData_t;

/* Helper macros for safe double-cast of pointer to uint64_t value */
#ifndef NVTX_POINTER_AS_PAYLOAD_ULLVALUE
# ifdef __cplusplus
# define NVTX_POINTER_AS_PAYLOAD_ULLVALUE(p) \
    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))
# else
#define NVTX_POINTER_AS_PAYLOAD_ULLVALUE(p) ((uint64_t)(uintptr_t)p)
# endif
#endif


#define NVTX_PAYLOAD_CONCAT2(a,b) a##b
#define NVTX_PAYLOAD_CONCAT(a,b) NVTX_PAYLOAD_CONCAT2(a,b)
#define NVTX_DATA_VAR NVTX_PAYLOAD_CONCAT(nvtxDFDB,__LINE__)

/**
 * \brief Helper macro to attach a single payload to an NVTX event attribute.
 *
 * @note The NVTX push, start or mark operation must not be in the same or a
 * nested scope.
 */
#define NVTX_PAYLOAD_EVTATTR_SET(EVTATTR, SCHEMA_ID, PAYLOAD_ADDR, SIZE) \
    nvtxPayloadData_t NVTX_DATA_VAR[] = {{SCHEMA_ID, SIZE, PAYLOAD_ADDR}}; \
    (EVTATTR).payload.ullValue = \
        NVTX_POINTER_AS_PAYLOAD_ULLVALUE(NVTX_DATA_VAR); \
    (EVTATTR).payloadType = NVTX_PAYLOAD_TYPE_BINARY; \
    (EVTATTR).reserved0 = 1;

/**
 * \brief Helper macro to attach multiple payloads to an NVTX event attribute.
 *
 * The payload data array (`nvtxPayloadData_t`) is passed as first argument to
 * this macro.
 */
#define NVTX_PAYLOAD_EVTATTR_SET_MULTIPLE(EVTATTR, PAYLOADS) \
    (EVTATTR).payloadType = NVTX_PAYLOAD_TYPE_BINARY; \
    (EVTATTR).reserved0 = sizeof(PAYLOADS)/sizeof(nvtxPayloadData_t); \
    (EVTATTR).payload.ullValue = NVTX_POINTER_AS_PAYLOAD_ULLVALUE(PAYLOADS);


/**
 * \brief The payload schema type.
 *
 * A schema can be either of these types.
 */
enum nvtxPayloadSchemaType
{
    NVTX_PAYLOAD_SCHEMA_TYPE_INVALID = 0,

    NVTX_PAYLOAD_SCHEMA_TYPE_STATIC  = 1,
    NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC = 2,

    NVTX_PAYLOAD_SCHEMA_TYPE_UNION   = 3,
    NVTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR = 4
};

/**
 * \brief Flags for static and dynamic schemas.
 */
enum nvtxPayloadSchemaFlags
{
    NVTX_PAYLOAD_SCHEMA_FLAG_NONE = 0,

    /**
     * This flag indicates that a schema and the corresponding payloads can
     * contain fields which require a deep copy.
     */
    NVTX_PAYLOAD_SCHEMA_FLAG_DEEP_COPY  = (1 << 1),

    /**
     * This flag indicates that a schema and the corresponding payloads can
     * be referenced by another payload of the same event.
     */
    NVTX_PAYLOAD_SCHEMA_FLAG_REFERENCED = (1 << 2),

    /**
     * The schema describes a deferred event/marker. Such a schema requires one
     * timestamp entry and one string entry with the flag
     * `NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE`. Category and color can be
     * optionally specified with the respective entry types. The deferred event
     * can contain a binary payload itself by using a custom schema ID as type
     * its schema description. Multiple occurrences of the same event can be
     * described by specifying an array timestamps.
     */
    NVTX_PAYLOAD_SCHEMA_FLAG_DEFERRED_EVENT = (1 << 3),
    /**
     * The schema describes a deferred event/marker. Such a schema requires
     * one start timestamp, one end timestamp and one string entry with the flag
     * `NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE`. Category and color can be
     * optionally specified with the respective entry types. The deferred range
     * can contain a binary payload itself by using a custom schema ID as type
     * its schema description.
     *
     * Timestamps can be provided in different ways:
     *  - A single range has two timestamp entries with the first (smaller entry
     *    index) being used as the start/push timestamp.
     *  - If the range schema contains one array of timestamps, the tool assumes
     *    that the array contains alternating start and end timestamps.
     *  - If two timestamp arrays are specified the first entry (with the
     *    smaller entry index) is assumed to contain the start timestamps. Both
     *    arrays have to be of the same size.
     */
    NVTX_PAYLOAD_SCHEMA_FLAG_DEFERRED_RANGE = (2 << 3)
};

/**
 * The values allow the valid fields in @ref nvtxPayloadSchemaAttr_t to be
 * specified via setting the field `fieldMask`.
 */
#define NVTX_PAYLOAD_SCHEMA_ATTR_NAME        (1 << 1)
#define NVTX_PAYLOAD_SCHEMA_ATTR_TYPE        (1 << 2)
#define NVTX_PAYLOAD_SCHEMA_ATTR_FLAGS       (1 << 3)
#define NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES     (1 << 4)
#define NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES (1 << 5)
#define NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE (1 << 6)
#define NVTX_PAYLOAD_SCHEMA_ATTR_ALIGNMENT   (1 << 7)
#define NVTX_PAYLOAD_SCHEMA_ATTR_SCHEMA_ID   (1 << 8)

/**
 * NVTX payload schema attributes.
 */
typedef struct nvtxPayloadSchemaAttr_t
{
    /**
     * \brief Mask of valid fields in this structure.
     *
     * The values from `enum nvtxPayloadSchemaAttributes` have to be used.
     */
    uint64_t                        fieldMask;

    /**
     * \brief Name of the payload schema. (Optional)
     */
    const char*                     name;

    /**
     * \brief Payload schema type. (Mandatory) \anchor PAYLOAD_TYPE_FIELD
     *
     * A value from `enum nvtxPayloadSchemaType` has to be used.
     */
    uint64_t                        type;

    /**
     * \brief Payload schema flags. (Optional)
     *
     * Flags defined in `enum nvtxPayloadSchemaFlags` can be used to set
     * additional properties of the schema.
     */
    uint64_t                        flags;

    /**
     * \brief Entries of a payload schema. (Mandatory) \anchor ENTRIES_FIELD
     *
     * This field is a pointer to an array of schema entries, each describing a
     * field in a data structure, e.g. in a C struct or union.
     */
    const nvtxPayloadSchemaEntry_t* entries;

    /**
     * \brief Number of entries in the payload schema. (Mandatory)
     *
     * Number of entries in the array of payload entries \ref ENTRIES_FIELD.
     */
    size_t                          numEntries;

    /**
     * \brief The binary payload size in bytes for static payload schemas.
     *
     * If \ref PAYLOAD_TYPE_FIELD is @ref NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC this
     * value is ignored. If this field is not specified for a schema of type
     * @ref NVTX_PAYLOAD_SCHEMA_TYPE_STATIC, the size can be automatically
     * determined by a tool.
     */
    size_t                          payloadStaticSize;

    /**
     * \brief The byte alignment for packed structures.
     *
     * If not specified, this field defaults to `0`, which means that the fields
     * in the data structure are not packed and natural alignment rules can be
     * applied.
     */
    size_t                          packAlign;

    /* Static/custom schema ID must be
       >= NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START and
       < NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_DYNAMIC_START */
    uint64_t                        schemaId;
} nvtxPayloadSchemaAttr_t;

/**
 * \brief Register a payload schema.
 *
 * @param domain NVTX domain handle.
 * @param attr NVTX payload schema attributes.
 */
NVTX_DECLSPEC uint64_t NVTX_API nvtxPayloadSchemaRegister(
    nvtxDomainHandle_t domain, const nvtxPayloadSchemaAttr_t* attr);

/**
 * \brief Enumeration entry.
 *
 * Since the value of an enum entry might not be meaningful for the analysis,
 * a tool can show the name of enum entry instead.
 *
 * @note EXPERIMENTAL
 */
typedef struct nvtxPayloadEnum_t
{
    /**
     * Name of the enum value.
     */
    const char* name;

    /**
     * Value of the enum entry.
     */
    uint64_t    value;

    /**
     * Indicates that this entry sets a specific set of bits, which can be used
     * to easily define bitsets.
     */
    int8_t      isFlag;
} nvtxPayloadEnum_t;

/**
 * The values are used to set the field `fieldMask` and specify which fields in
 * `nvtxPayloadEnumAttr_t` are set.
 */
#define NVTX_PAYLOAD_ENUM_ATTR_NAME        (1 << 1)
#define NVTX_PAYLOAD_ENUM_ATTR_ENTRIES     (1 << 2)
#define NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES (1 << 3)
#define NVTX_PAYLOAD_ENUM_ATTR_SIZE        (1 << 4)
#define NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID   (1 << 5)

/**
 * NVTX payload enumeration type attributes.
 */
typedef struct nvtxPayloadEnumAttr_t {
    /**
     * Mask of valid fields in this struct.
     * The values from `enum nvtxPayloadSchemaAttributes` have to be used.
     */
    uint64_t                 fieldMask;

    /**
     * Name of the enum. (Optional)
     */
    const char*              name;

    /**
     * Entries of the enum. (Mandatory)
     */
    const nvtxPayloadEnum_t* entries;

    /**
     * Number of entries in the enum. (Mandatory)
     */
    size_t                   numEntries;

    /**
     * Size of enumeration type in bytes
     */
    size_t                   sizeOfEnum;

    /**
     * Static/custom schema ID must be
     * >= NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START and
     *  < NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_DYNAMIC_START
     */
    uint64_t                 schemaId;
} nvtxPayloadEnumAttr_t;

/**
 * \brief Register an enumeration type with the payload extension.
 *
 * @param domain NVTX domain handle
 * @param attr NVTX payload enumeration type attributes.
 */
NVTX_DECLSPEC uint64_t nvtxPayloadEnumRegister(nvtxDomainHandle_t domain,
    const nvtxPayloadEnumAttr_t* attr);

/**
 * \brief Callback Ids of API functions in the payload extension.
 *
 * The NVTX handler can use these values to register a handler function. When
 * InitializeInjectionNvtxExtension(nvtxExtModuleInfo_t* moduleInfo) is
 * executed, a handler routine 'handlenvtxPayloadRegisterSchema' can be
 * registered as follows:
 *      moduleInfo->segments->slots[NVTX3EXT_CBID_nvtxPayloadSchemaRegister] =
 *          (intptr_t)handlenvtxPayloadRegisterSchema;
 */
typedef enum NvtxExtPayloadCallbackId
{
    NVTX3EXT_CBID_nvtxPayloadSchemaRegister = 0,
    NVTX3EXT_CBID_nvtxPayloadEnumRegister   = 1,
    NVTX3EXT_CBID_PAYLOAD_FN_NUM            = 2
} NvtxExtPayloadCallbackId;

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

#define NVTX_EXT_TYPES_GUARD /* Ensure other headers cannot include directly */
#include "nvtxExtDetail/nvtxExtTypes.h"
#undef NVTX_EXT_TYPES_GUARD

#ifndef NVTX_NO_IMPL
#define NVTX_EXT_IMPL_PAYLOAD_GUARD /* Ensure other headers cannot included directly */
#include "nvtxExtDetail/nvtxExtPayloadTypeInfo.h"
#include "nvtxExtDetail/nvtxExtImplPayload_v1.h"
#undef NVTX_EXT_IMPL_PAYLOAD_GUARD
#endif /*NVTX_NO_IMPL*/

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NVTOOLSEXT_PAYLOAD_H */
