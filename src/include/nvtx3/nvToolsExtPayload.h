/*
* Copyright 2021-2024  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "nvToolsExt.h"

/* Optionally include helper macros. */
/* #include "nvToolsExtPayloadHelper.h" */

/**
 * If needed, semantic extension headers can be included after this header.
 */

/**
 * \brief The compatibility ID is used for versioning of this extension.
 */
#ifndef NVTX_EXT_PAYLOAD_COMPATID
#define NVTX_EXT_PAYLOAD_COMPATID 0x0103
#endif

/**
 * \brief The module ID identifies the payload extension. It has to be unique
 * among the extension modules.
 */
#ifndef NVTX_EXT_PAYLOAD_MODULEID
#define NVTX_EXT_PAYLOAD_MODULEID 2
#endif

/**
 * \brief Additional value for the enum @ref nvtxPayloadType_t
 */
#ifndef NVTX_PAYLOAD_TYPE_EXT
#define NVTX_PAYLOAD_TYPE_EXT ((int32_t)0xDFBD0009)
#endif

/** ---------------------------------------------------------------------------
 * Payload schema entry flags. Used for @ref nvtxPayloadSchemaEntry_t::flags.
 * ------------------------------------------------------------------------- */
#ifndef NVTX_PAYLOAD_ENTRY_FLAGS_V1
#define NVTX_PAYLOAD_ENTRY_FLAGS_V1

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
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE           (1 << 4)

/**
 * The value is a zero-/null-terminated array.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED      (2 << 4)

/**
 * \brief A single or multi-dimensional array of variable length.
 *
 * The field `arrayOrUnionDetail` contains the index of the schema entry that
 * holds the length(s). If the length entry is a scalar, then this entry is a 1D
 * array. If the length entry is a fixed-size array, then the number of
 * dimensions is defined with the registration of the schema. If the length
 * entry is a zero-terminated array, then the array of the dimensions can be
 * determined at runtime.
 * For multidimensional arrays, values are stored in row-major order, with rows
 * being stored consecutively in contiguous memory. The size of the entry (in
 * bytes) is the product of the dimensions multiplied with size of the array
 * element.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX         (3 << 4)

/**
 * \brief A single or multi-dimensional array of variable length, where the
 * dimensions are stored in a different payload (index) of the same event.
 *
 * This enables an existing address to an array to be directly passed, while the
 * dimensions are defined in a separate payload (with only one payload entry).
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_PAYLOAD_INDEX (4 << 4)

/**
 * \brief The value or data that is pointed to by this payload entry value shall
 * be copied by the NVTX handler.
 *
 * A tool may not support deep copy and just ignore this flag.
 * See @ref NVTX_PAYLOAD_SCHEMA_FLAG_DEEP_COPY for more details.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_DEEP_COPY          (1 << 8)

/**
 * Notifies the NVTX handler to hide this entry in case of visualization.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_HIDE               (1 << 9)

/**
 * The entry specifies the event message. Any string type can be used.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE      (1 << 10)

/**
 * \brief The entry contains an event timestamp.
 *
 * The time source might be provided via the entry semantics field. In most
 * cases, the timestamp (entry) type is @ref NVTX_PAYLOAD_ENTRY_TYPE_UINT64.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_EVENT_TIMESTAMP    (2 << 10)

/**
 * These flags specify the NVTX event type to which an entry refers.
 */
#define NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN        (1 << 12)
#define NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END          (2 << 12)
#define NVTX_PAYLOAD_ENTRY_FLAG_MARK               (3 << 12)
#define NVTX_PAYLOAD_ENTRY_FLAG_COUNTER            (4 << 12)

#endif /* NVTX_PAYLOAD_ENTRY_FLAGS_V1 */
/** ---------------------------------------------------------------------------
 * END: Payload schema entry flags.
 * ------------------------------------------------------------------------- */

/** \todo: Keep this in the header? */
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

#define NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_TYPE(F) \
    (F & NVTX_PAYLOAD_ENTRY_FLAG_IS_ARRAY)
/** \todo end */


/** ---------------------------------------------------------------------------
 * Types of entries in a payload schema.
 *
 * @note Several of the predefined types contain the size (in bits) in their
 * names. For some data types the size (in bytes) is not fixed and may differ
 * for different platforms/operating systems/compilers. To provide portability,
 * an array of sizes (in bytes) for type 1 to 28 ( @ref
 * NVTX_PAYLOAD_ENTRY_TYPE_CHAR to @ref NVTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE)
 * is passed to the NVTX extension initialization function
 * @ref InitializeInjectionNvtxExtension via the `extInfo` field of
 * @ref nvtxExtModuleInfo_t.
 * ------------------------------------------------------------------------- */
#ifndef NVTX_PAYLOAD_ENTRY_TYPES_V1
#define NVTX_PAYLOAD_ENTRY_TYPES_V1

#define NVTX_PAYLOAD_ENTRY_TYPE_INVALID     0

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
 * Floating point types
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_FLOAT      19
#define NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE     20
#define NVTX_PAYLOAD_ENTRY_TYPE_LONGDOUBLE 21

/**
 * Size type (`size_t` in C).
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
 * Typically, a tool will display this as hex or binary.
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
 * Data types are as defined by NVTXv3 core.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_CATEGORY   68 /* uint32_t */
#define NVTX_PAYLOAD_ENTRY_TYPE_COLOR_ARGB 69 /* uint32_t */

/**
 * The scope of events or counters (see `nvtxScopeRegister`).
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_SCOPE_ID   70 /* uint64_t */

/**
 * Thread ID as scope.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT32 73
#define NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT64 74

/**
 * \brief String types.
 *
 * If no flags are set for the entry and `arrayOrUnionDetail > 0`, the entry is
 * assumed to be a fixed-size string with the given length, embedded in the payload.
 * `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_FIXED_SIZE` is redundant for fixed-size strings.
 *
 * \todo(Revise the following paragraph.)
 * Setting the flag `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED` specifies a
 * zero-terminated string. If `arrayOrUnionDetail > 0`, the entry is handled as
 * a zero-terminated array of fixed-size strings.
 *
 * Setting the flag `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_LENGTH_INDEX` specifies a
 * variable-length string with the length given in the entry specified by the
 * field `arrayOrUnionDetail`.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING       75 /* `char*`, system LOCALE */
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8  76
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF16 77
#define NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32 78

/**
 * The entry value is of type @ref nvtxStringHandle_t returned by
 * @ref nvtxDomainRegisterString.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE 80

/**
 * This type marks the union selector member (entry index) in schemas used by
 * a union with internal selector.
 * See @ref NVTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_UNION_SELECTOR 100

/**
 * \brief Predefined schema ID for payload data that is referenced in another payload.
 *
 * This schema ID can be used in @ref nvtxPayloadData_t::schema_id to indicate that the
 * payload is a blob of memory which other payload entries may point into.
 * A tool will not expose this payload directly.
 *
 * This schema ID cannot be used as schema entry type!
 */
#define NVTX_TYPE_PAYLOAD_SCHEMA_REFERENCED 1022

/**
 * \brief Predefined schema ID for raw payload data.
 *
 * This schema ID can be used in @ref nvtxPayloadData_t::schema_id to indicate
 * that the payload is a blob, which can be shown with an arbitrary data viewer.
 * This schema ID cannot be used as schema entry type!
 */
#define NVTX_TYPE_PAYLOAD_SCHEMA_RAW        1023

/**
 * \deprecated: Remove for official release!
 * In the initial version of this header custom schema IDs started
 * here. Unless predefined types require more than 16 bits we can keep this
 * value to preserve backwards compatibility. The value is not used as first
 * ID for custom schemas any more, but in the analysis every entry type >= this
 * value is assumed to be a custom schema.
 */
#define NVTX_PAYLOAD_ENTRY_TYPE_CUSTOM_BASE 65536

/* Custom (static) schema IDs. */
#define NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_STATIC_START  (1 << 24)

/* Dynamic schema IDs (generated by the tool) start here. */
#define NVTX_PAYLOAD_ENTRY_TYPE_SCHEMA_ID_DYNAMIC_START 4294967296 /* 1 << 32 */

#endif /* NVTX_PAYLOAD_ENTRY_TYPES_V1 */
/** ---------------------------------------------------------------------------
 * END: Payload schema entry types.
 * ------------------------------------------------------------------------- */


#ifndef NVTX_PAYLOAD_SCHEMA_TYPES_V1
#define NVTX_PAYLOAD_SCHEMA_TYPES_V1

/**
 * \brief The payload schema type.
 *
 * A schema can be either of the following types. It is set with
 * @ref nvtxPayloadSchemaAttr_t::type.
 */
#define NVTX_PAYLOAD_SCHEMA_TYPE_INVALID                      0
#define NVTX_PAYLOAD_SCHEMA_TYPE_STATIC                       1
#define NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC                      2
#define NVTX_PAYLOAD_SCHEMA_TYPE_UNION                        3
#define NVTX_PAYLOAD_SCHEMA_TYPE_UNION_WITH_INTERNAL_SELECTOR 4

#endif /* NVTX_PAYLOAD_SCHEMA_TYPES_V1 */


#ifndef NVTX_PAYLOAD_SCHEMA_FLAGS_V1
#define NVTX_PAYLOAD_SCHEMA_FLAGS_V1

/**
 * \brief Flags for static and dynamic schemas.
 *
 * The schema flags are used with @ref nvtxPayloadSchemaAttr_t::flags.
 */
#define NVTX_PAYLOAD_SCHEMA_FLAG_NONE           0

/**
 * This flag indicates that a schema and the corresponding payloads can
 * contain fields which require a deep copy.
 */
#define NVTX_PAYLOAD_SCHEMA_FLAG_DEEP_COPY      (1 << 1)

/**
 * This flag indicates that a schema and the corresponding payload can be
 * referenced by another payload of the same event. If the schema is not
 * intended to be visualized directly, it is possible use
 * @ref NVTX_TYPE_PAYLOAD_SCHEMA_REFERENCED instead.
 */
#define NVTX_PAYLOAD_SCHEMA_FLAG_REFERENCED     (1 << 2)

/**
 * The schema defines a counter group. An NVTX handler can expect that the schema
 * contains entries with counter semantics.
 */
#define NVTX_PAYLOAD_SCHEMA_FLAG_COUNTER_GROUP  (1 << 3)


#endif /* NVTX_PAYLOAD_SCHEMA_FLAGS_V1 */


#ifndef NVTX_PAYLOAD_SCHEMA_ATTRS_V1
#define NVTX_PAYLOAD_SCHEMA_ATTRS_V1

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
#define NVTX_PAYLOAD_SCHEMA_ATTR_EXTENSION   (1 << 9)

#endif /* NVTX_PAYLOAD_SCHEMA_ATTRS_V1 */


#ifndef NVTX_PAYLOAD_ENUM_ATTRS_V1
#define NVTX_PAYLOAD_ENUM_ATTRS_V1

/**
 * The values are used to set the field `fieldMask` and specify which fields in
 * @ref nvtxPayloadEnumAttr_t are set.
 */
#define NVTX_PAYLOAD_ENUM_ATTR_NAME        (1 << 1)
#define NVTX_PAYLOAD_ENUM_ATTR_ENTRIES     (1 << 2)
#define NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES (1 << 3)
#define NVTX_PAYLOAD_ENUM_ATTR_SIZE        (1 << 4)
#define NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID   (1 << 5)
#define NVTX_PAYLOAD_ENUM_ATTR_EXTENSION   (1 << 6)

#endif /* NVTX_PAYLOAD_ENUM_ATTRS_V1 */

/**
 * An NVTX scope specifies the execution scope or source of events or counters.
 */
#ifndef NVTX_SCOPES_V1
#define NVTX_SCOPES_V1

/** Identifies an invalid scope and indicates an error if returned by `nvtxScopeRegister`. */
#define NVTX_SCOPE_NONE                    0 /* no scope */

#define NVTX_SCOPE_ROOT                    1

#define NVTX_SCOPE_CURRENT_HW_MACHINE      2 /* Node/machine name */
#define NVTX_SCOPE_CURRENT_HW_SOCKET       3
#define NVTX_SCOPE_CURRENT_HW_CPU_PHYSICAL 4 /* Physical CPU core */
#define NVTX_SCOPE_CURRENT_HW_CPU_LOGICAL  5 /* Logical CPU core */
/* Innermost HW execution context at registration time */
#define NVTX_SCOPE_CURRENT_HW_INNERMOST   15

/* Virtualized hardware, virtual machines, OS (if you don't know any better)
\todo: Need to be more precise what information is expected for each of these scopes. */
#define NVTX_SCOPE_CURRENT_HYPERVISOR     16
#define NVTX_SCOPE_CURRENT_VM             17
#define NVTX_SCOPE_CURRENT_KERNEL         18
#define NVTX_SCOPE_CURRENT_CONTAINER      19
#define NVTX_SCOPE_CURRENT_OS             20

/* Software scopes */
#define NVTX_SCOPE_CURRENT_SW_PROCESS 	  21 /* Process scope */
#define NVTX_SCOPE_CURRENT_SW_THREAD  	  22 /* Thread scope */
/* Innermost SW execution context at registration time */
#define NVTX_SCOPE_CURRENT_SW_INNERMOST   31

/** Static (user-provided) scope IDs (feed forward) */
#define NVTX_SCOPE_ID_STATIC_START  (1 << 24)

/** Dynamically (tool) generated scope IDs */
#define NVTX_SCOPE_ID_DYNAMIC_START 4294967296  /* 1 << 32 */

#endif /* NVTX_SCOPES_V1 */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef NVTX_PAYLOAD_TYPEDEFS_V1
#define NVTX_PAYLOAD_TYPEDEFS_V1

/**
 * \brief Size and alignment information for predefined payload entry types.
 *
 * The struct contains the size and the alignment size in bytes. A respective
 * array for the predefined types is passed via nvtxExtModuleInfo_t to the NVTX
 * client/handler. The type (ID) is used as index into this array.
 */
typedef struct nvtxPayloadEntryTypeInfo_v1
{
    uint16_t size;
    uint16_t align;
} nvtxPayloadEntryTypeInfo_t;

/**
 * \brief Binary payload data, size and decoding information.
 *
 * An array of type `nvtxPayloadData_t` is passed to the NVTX event attached to
 * an NVTX event via the `payload.ullvalue` field of NVTX event attributes.
 *
 * The `schemaId` be a predefined schema entry type (`NVTX_PAYLOAD_ENTRY_TYPE*`),
 * a schema ID (statically specified or dynamically created) or one of
 * `NVTX_PAYLOAD_TYPE_REFERENCED` or `NVTX_PAYLOAD_TYPE_RAW`.
 *
 * Setting the size of a payload to `MAX_SIZE` can be useful to reduce the
 * overhead of NVTX instrumentation, when no NVTX handler is attached. However,
 * a tool might not be able to detect the size of a payload and thus skip it.
 * A reasonable use case is a payload that represents a null-terminated
 * C string, where the NVTX handler can call `strlen()`.
 */
typedef struct nvtxPayloadData_v1
{
    /**
     * The schema ID, which defines the layout of the binary data.
     */
    uint64_t    schemaId;

    /**
     * Size of the payload (blob) in bytes. `SIZE_MAX` (`-1`) indicates the tool
     * that it should figure out the size, which might not be possible.
     */
    size_t      size;

    /**
     * Pointer to the binary payload data.
     */
    const void* payload;
} nvtxPayloadData_t;


/**
 * \brief Header of the payload entry's semantic field.
 *
 * If the semantic field of the payload schema entry is set, the first four
 * fields (header) are defined with this type. A tool can iterate through the
 * extensions and check, if it supports (can handle) it.
 */
typedef struct nvtxSemanticsHeader_v1
{
    uint32_t structSize; /** Size of semantic extension struct. */
    uint16_t semanticId;
    uint16_t version;
    const struct nvtxSemanticsHeader_v1* next; /** linked list */
    /* Additional fields are defined by the specific semantic extension. */
} nvtxSemanticsHeader_t;

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
 *  nvtxPayloadSchemaEntry_t schema[] = {
 *      {0, NVTX_EXT_PAYLOAD_TYPE_UINT8, "one byte"},
 *      {0, NVTX_EXT_PAYLOAD_TYPE_INT32, "four bytes"}
 *  };
 */
typedef struct nvtxPayloadSchemaEntry_v1
{
    /**
     * \brief Flags to augment the basic type.
     *
     * This field allows additional properties of the payload entry to be
     * specified. Valid values are `NVTX_PAYLOAD_ENTRY_FLAG_*`.
     */
    uint64_t       flags;

    /**
     * \brief Predefined payload schema entry type or custom schema ID.
     *
     * Predefined types are `NVTX_PAYLOAD_ENTRY_TYPE_*`. Passing a schema ID
     * enables nesting of schemas.
     */
    uint64_t       type;

    /**
     * \brief Name or label of the payload entry. (Optional)
     *
     * A meaningful name or label can help organizing and interpreting the data.
     */
    const char*    name;

    /**
     * \brief Description of the payload entry. (Optional)
     *
     * A more detail description of the data that is stored with this entry.
     */
    const char*    description;

    /**
     * \brief String length, array length or member selector for union types.
     *
     * If @ref type is a C string type, this field specifies the string length.
     *
     * If @ref flags specify that the entry is an array, this field specifies
     * the array length. See `NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_*` for more details.
     *
     * If @ref type is a union with schema type @ref NVTX_PAYLOAD_SCHEMA_TYPE_UNION
     * (external selection of the union member), this field contains the index
     * (starting with 0) to an entry of integral type in the same schema. The
     * associated field value specifies the selected union member.
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
     * binary data (blob) to the start address of the data of this entry.
     *
     * It is recommended (but not required) to provide the offset it. Otherwise,
     * the NVTX handler will determine the offset from natural alignment rules.
     * In some cases, e.g. dynamic schema layouts, the offset cannot be set and
     * has to be determined based on the data of prior entries.
     *
     * Setting the offset can also be used to skip entries during payload parsing.
     */
    uint64_t       offset;

    /**
     * \brief Additional semantics of the payload entry.
     *
     * The field points to the first element in a linked list, which enables
     * multiple semantic extensions.
     */
    const nvtxSemanticsHeader_t* semantics;

    /**
     * \brief Reserved for future use. Do not use it!
     */
    const void*    reserved;
} nvtxPayloadSchemaEntry_t;


/**
 * \brief Header of the schema attribute extension field.
 */
typedef struct nvtxPayloadSchemaExtension_v1
{
    uint32_t structSize; /** Size of schema extension struct. */
    uint16_t schemaExtId;
    uint16_t version;
    const struct nvtxPayloadSchemaExtension_v1* next; /** linked list */
    /* Additional fields are defined by the specific schema extension. */
} nvtxPayloadSchemaExtension_t;

/**
 * \brief NVTX payload schema attributes.
 */
typedef struct nvtxPayloadSchemaAttr_v1
{
    /**
     * \brief Mask of valid fields in this struct.
     *
     * Use the `NVTX_PAYLOAD_SCHEMA_ATTR_*` defines.
     */
    uint64_t                        fieldMask;

    /**
     * \brief Name of the payload schema. (Optional)
     */
    const char*                     name;

    /**
     * \brief Payload schema type. (Mandatory) \anchor PAYLOAD_TYPE_FIELD
     *
     * Use the `NVTX_PAYLOAD_SCHEMA_TYPE_*` defines.
     */
    uint64_t                        type;

    /**
     * \brief Payload schema flags. (Optional)
     *
     * Flags defined by `NVTX_PAYLOAD_SCHEMA_FLAG_*` can be used to set
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

    /* Flexible extension for schema attributes. */
    void*                           extension;
} nvtxPayloadSchemaAttr_t;

/**
 * \brief This type is used to describe an enumeration.
 *
 * Since the value of an enum entry might not be meaningful for the analysis
 * and/or visualization, a tool can show the name of enum entry instead.
 *
 * An array of this struct is passed to @ref nvtxPayloadEnumAttr_t::entries to be
 * finally registered via @ref nvtxPayloadEnumRegister with the NVTX handler.
 *
 * @note EXPERIMENTAL
 */
typedef struct nvtxPayloadEnum_v1
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
     * to define bitsets.
     */
    int8_t      isFlag;
} nvtxPayloadEnum_t;

/**
 * \brief NVTX payload enumeration type attributes.
 *
 * A pointer to this struct is passed to @ref nvtxPayloadEnumRegister.
 */
typedef struct nvtxPayloadEnumAttr_v1
{
    /**
     * Mask of valid fields in this struct. See `NVTX_PAYLOAD_ENUM_ATTR_*`.
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

    /* Flexible extension for enumeration attributes. */
    void*                    extension;
} nvtxPayloadEnumAttr_t;

typedef struct nvtxScopeAttr_v1
{
    size_t structSize;

    /** Path delimited by '/' characters, relative to parentScope. Leading
    slashes are ignored. Nodes in the path may use name[key] syntax to indicate
    an array of sibling nodes, which may be combined with other non-array nodes
    or different arrays at the same scope. Node names should be UTF8 printable
    characters, excluding '/', '[', and ']' characters which have special
    meaning here. An empty C string "" and `NULL` are valid inputs and treated
    equivalently. */
    const char* path;

    uint64_t parentScope;

    /** The static scope ID must be unique within the domain,
        >= NVTX_EVENT_SCOPE_ID_STATIC_START, and
        < NVTX_EVENT_SCOPE_ID_DYNAMIC_START. */
    uint64_t scopeId;
} nvtxScopeAttr_t;


#endif /* NVTX_PAYLOAD_TYPEDEFS_V1 */

#ifndef NVTX_PAYLOAD_API_FUNCTIONS_V1
#define NVTX_PAYLOAD_API_FUNCTIONS_V1

/**
 * \brief Register a payload schema.
 *
 * @param domain NVTX domain handle.
 * @param attr NVTX payload schema attributes.
 */
NVTX_DECLSPEC uint64_t NVTX_API nvtxPayloadSchemaRegister(
    nvtxDomainHandle_t domain,
    const nvtxPayloadSchemaAttr_t* attr);

/**
 * \brief Register an enumeration type with the payload extension.
 *
 * @param domain NVTX domain handle
 * @param attr NVTX payload enumeration type attributes.
 */
NVTX_DECLSPEC uint64_t NVTX_API nvtxPayloadEnumRegister(
    nvtxDomainHandle_t domain,
    const nvtxPayloadEnumAttr_t* attr);

/**
 * \brief Register a scope.
 *
 * @param domain NVTX domain handle (0 for default domain)
 * @param attr Scope attributes.
 *
 * @return an identifier for the scope. If the operation was not successful,
 * `NVTX_SCOPE_NONE` is returned.
 */
NVTX_DECLSPEC uint64_t NVTX_API nvtxScopeRegister(
    nvtxDomainHandle_t domain,
    const nvtxScopeAttr_t* attr);

/**
 * \brief Marks an instantaneous event in the application with the attributes
 * being passed via the extended payload.
 *
 * An NVTX handler can assume that the payload contains the event message.
 * Otherwise, it might ignore the event.
 *
 * @param domain NVTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 */
NVTX_DECLSPEC void NVTX_API nvtxMarkPayload(
    nvtxDomainHandle_t domain,
    const nvtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief Begin a nested thread range with the attributes being passed via the
 * payload.
 *
 * @param domain NVTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 *
 * @return The level of the range being ended. If an error occurs a negative
 * value is returned on the current thread.
 */
NVTX_DECLSPEC int NVTX_API nvtxRangePushPayload(
    nvtxDomainHandle_t domain,
    const nvtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief End a nested thread range with an additional custom payload.
 *
 * NVTX event attributes passed to this function (via the payloads) overwrite
 * event attributes (message and color) that have been set in the push event.
 * Other payload entries extend the data of the range.
 *
 * @param domain NVTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 *
 * @return The level of the range being ended. If an error occurs a negative
 * value is returned on the current thread.
 */
NVTX_DECLSPEC int NVTX_API nvtxRangePopPayload(
    nvtxDomainHandle_t domain,
    const nvtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief Start a thread range with attributes passed via the extended payload.
 *
 * @param domain NVTX domain handle
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 *
 * @return The level of the range being ended. If an error occurs a negative
 * value is returned on the current thread.
 */
NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartPayload(
    nvtxDomainHandle_t domain,
    const nvtxPayloadData_t* payloadData,
    size_t count);

/**
 * \brief End a thread range and pass a custom payload.
 *
 * NVTX event attributes passed to this function (via the payloads) overwrite
 * event attributes (message and color) that have been set in the start event.
 * Other payload entries extend the data of the range.
 *
 * @param domain NVTX domain handle
 * @param id The correlation ID returned from a NVTX range start call.
 * @param payloadData pointer to an array of structured payloads.
 * @param count number of payload BLOBs.
 */
NVTX_DECLSPEC void NVTX_API nvtxRangeEndPayload(
    nvtxDomainHandle_t domain,
    nvtxRangeId_t id,
    const nvtxPayloadData_t* payloadData,
    size_t count);

/**
 * @brief Checks if an NVTX domain is enabled (unofficial and may not work)
 *
 * @param domain NVTX domain handle
 * @return 0 if the domain is not enabled.
 */
NVTX_DECLSPEC uint8_t NVTX_API nvtxDomainIsEnabled(
    nvtxDomainHandle_t domain);

#endif /* NVTX_PAYLOAD_API_FUNCTIONS_V1 */

#ifndef NVTX_PAYLOAD_CALLBACK_ID_V1
#define NVTX_PAYLOAD_CALLBACK_ID_V1
/**
 * \brief Callback Ids of API functions in the payload extension.
 *
 * The NVTX handler can use these values to register a handler function. When
 * InitializeInjectionNvtxExtension(nvtxExtModuleInfo_t* moduleInfo) is
 * executed, a handler routine 'handlenvtxPayloadRegisterSchema' can be
 * registered as follows:
 * \code{.c}
 *      moduleInfo->segments->slots[NVTX3EXT_CBID_nvtxPayloadSchemaRegister] =
 *          (intptr_t)YourPayloadRegisterSchemaHandlerFn;
 * \endcode
 */
#define NVTX3EXT_CBID_nvtxPayloadSchemaRegister      0
#define NVTX3EXT_CBID_nvtxPayloadEnumRegister        1
#define NVTX3EXT_CBID_nvtxMarkPayload                2
#define NVTX3EXT_CBID_nvtxRangePushPayload           3
#define NVTX3EXT_CBID_nvtxRangePopPayload            4
#define NVTX3EXT_CBID_nvtxRangeStartPayload          5
#define NVTX3EXT_CBID_nvtxRangeEndPayload            6
#define NVTX3EXT_CBID_nvtxDomainIsEnabled            7
#define NVTX3EXT_CBID_nvtxScopeRegister             12
#endif /* NVTX_PAYLOAD_CALLBACK_ID_V1 */

/*** Helper utilities ***/

/** \brief  Helper macro for safe double-cast of pointer to uint64_t value. */
#ifndef NVTX_POINTER_AS_PAYLOAD_ULLVALUE
# ifdef __cplusplus
# define NVTX_POINTER_AS_PAYLOAD_ULLVALUE(p) \
    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p))
# else
#define NVTX_POINTER_AS_PAYLOAD_ULLVALUE(p) ((uint64_t)(uintptr_t)p)
# endif
#endif

#ifndef NVTX_PAYLOAD_EVTATTR_SET_DATA
/**
 * \brief Helper macro to attach a single payload to an NVTX event attribute.
 *
 * @param evtAttr NVTX event attribute (variable name)
 * @param pldata_addr Adress of `nvtxPayloadData_t` variable.
 * @param schema_id NVTX binary payload schema ID.
 * @param pl_addr Address of the (actual) payload.
 * @param sz size of the (actual) payload.
 */
#define NVTX_PAYLOAD_EVTATTR_SET_DATA(evtAttr, pldata_addr, schema_id, pl_addr, sz) \
    (pldata_addr)->schemaId = schema_id; \
    (pldata_addr)->size = sz; \
    (pldata_addr)->payload = pl_addr; \
    (evtAttr).payload.ullValue = NVTX_POINTER_AS_PAYLOAD_ULLVALUE(pldata_addr); \
    (evtAttr).payloadType = NVTX_PAYLOAD_TYPE_EXT; \
    (evtAttr).reserved0 = 1;
#endif /* NVTX_PAYLOAD_EVTATTR_SET_DATA */

#ifndef NVTX_PAYLOAD_EVTATTR_SET_MULTIPLE
/**
 * \brief Helper macro to attach multiple payloads to an NVTX event attribute.
 *
 * @param evtAttr NVTX event attribute (variable name)
 * @param pldata Payload data array (of type `nvtxPayloadData_t`)
 */
#define NVTX_PAYLOAD_EVTATTR_SET_MULTIPLE(evtAttr, pldata) \
    (evtAttr).payloadType = NVTX_PAYLOAD_TYPE_EXT; \
    (evtAttr).reserved0 = sizeof(pldata)/sizeof(nvtxPayloadData_t); \
    (evtAttr).payload.ullValue = NVTX_POINTER_AS_PAYLOAD_ULLVALUE(pldata);
#endif /* NVTX_PAYLOAD_EVTATTR_SET_MULTIPLE */

#ifndef NVTX_PAYLOAD_EVTATTR_SET
/*
 * Do not use this macro directly! It is a helper to attach a single payload to
 * an NVTX event attribute.
 * @warning The NVTX push, start or mark operation must not be in an outer scope.
 */
#define NVTX_PAYLOAD_EVTATTR_SET(evtAttr, schema_id, pl_addr, sz) \
    nvtxPayloadData_t _NVTX_PAYLOAD_DATA_VAR[] = \
        {{schema_id, sz, pl_addr}}; \
    (evtAttr)->payload.ullValue = \
        NVTX_POINTER_AS_PAYLOAD_ULLVALUE(_NVTX_PAYLOAD_DATA_VAR); \
    (evtAttr)->payloadType = NVTX_PAYLOAD_TYPE_EXT; \
    (evtAttr)->reserved0 = 1;
#endif /* NVTX_PAYLOAD_EVTATTR_SET */

#ifndef nvtxPayloadRangePush
/**
 * \brief Helper macro to push a range with extended payload.
 *
 * @param domain NVTX domain handle (0 for default domain)
 * @param evtAttr pointer to NVTX event attribute.
 * @param schemaId NVTX payload schema ID
 * @param plAddr Pointer to the binary data (actual payload)
 * @param size Size of the binary payload data in bytes.
 */
#define nvtxPayloadRangePush(domain, evtAttr, schemaId, plAddr, size) \
do { \
    NVTX_PAYLOAD_EVTATTR_SET(evtAttr, schemaId, plAddr, size) \
    nvtxDomainRangePushEx(domain, evtAttr); \
} while (0)
#endif /* nvtxPayloadRangePush */

#ifndef nvtxPayloadMark
/**
 * \brief Helper macro to set a marker with extended payload.
 *
 * @param domain NVTX domain handle (0 for default domain)
 * @param evtAttr pointer to NVTX event attribute.
 * @param schemaId NVTX payload schema ID
 * @param plAddr Pointer to the binary data (actual payload)
 * @param size Size of the binary payload data in bytes.
 */
#define nvtxPayloadMark(domain, evtAttr, schemaId, plAddr, size) \
do { \
    NVTX_PAYLOAD_EVTATTR_SET(evtAttr, schemaId, plAddr, size) \
    nvtxDomainMarkEx(domain, evtAttr); \
} while (0)
#endif /* nvtxPayloadMark */


#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

/* Extension types are required for the implementation and the NVTX handler. */
#define NVTX_EXT_TYPES_GUARD
#include "nvtxDetail/nvtxExtTypes.h"
#undef NVTX_EXT_TYPES_GUARD

#ifndef NVTX_NO_IMPL
#define NVTX_EXT_IMPL_PAYLOAD_GUARD
#include "nvtxDetail/nvtxExtImplPayload_v1.h"
#undef NVTX_EXT_IMPL_PAYLOAD_GUARD
#endif /* NVTX_NO_IMPL */

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
