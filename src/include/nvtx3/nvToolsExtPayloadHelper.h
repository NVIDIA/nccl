/*
* Copyright 2023  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#include "nvtxDetail/nvtxExtPayloadHelperInternal.h"


/* This is just an empty marker (for readability), which can be omitted. */
/* TODO: Fix issue with trailing comma at end of entry list. */
#define NCCL_NVTX_PAYLOAD_ENTRIES


/**
 * Use this macro for payload entries that are defined by a schema (nested
 * payload schema).
 */
#define NVTX_PAYLOAD_NESTED(schemaId) _NVTX_PAYLOAD_NESTED(schemaId)


/**
 * \brief Define a payload schema for an existing C `struct` definition.
 *
 *  This macro does
 *   1) create schema description (array of schema entries).
 *   2) set the schema attributes for a static data layout.
 *
 * It can be used in static code or within a function context.
 *
 * Example:
 *  NVTX_DEFINE_SCHEMA_FOR_STRUCT(your_struct, "SchemaName",
 *      NCCL_NVTX_PAYLOAD_ENTRIES(
 *          (index, TYPE_INT, "integer value"),
 *          (dpfloat, TYPE_DOUBLE, "fp64 value"),
 *          (text, TYPE_CSTRING, "text", NULL, 24)
 *      )
 *  )
 *
 * It is required to at least provide the struct name and the payload entries.
 * The first two fields (member name and NVTX entry type) of each payload entry
 * are required.
 *
 * The optional parameters are only allowed to be passed in the predefined order.
 * Hence, `payload_flags` requires `payload_schema` to be given and
 * `prefix` requires `payload_flags` and `payload_schema` to be given.
 * The payload entries are always the last parameter. A maximum of 16 schema
 * entries is supported.
 *
 * It is recommended to use `NVTX_PAYLOAD_SCHEMA_REGISTER` to register the schema.
 *
 * @param struct_id The name of the struct.
 * @param schema_name (Optional 1) name of the payload schema. Default is `NULL`.
 * @param prefix (Optional 2) prefix before the schema and attributes variables,
 *               e.g. `static const`. Leave this empty, if no prefix is desired.
 * @param schema_flags (Optional 2) flags to augment the payload schema.
 *                     Default is `NVTX_PAYLOAD_SCHEMA_FLAG_NONE`.
 * @param schema_id (Optional 4) User-defined payload schema ID.
 * @param entries (Mandatory) Payload schema entries. This is always the last
 *                parameter to the macro.
 */
#define NVTX_DEFINE_SCHEMA_FOR_STRUCT(struct_id, ...) \
    _NVTX_DEFINE_SCHEMA_FOR_STRUCT(struct_id, __VA_ARGS__)


/**
 * \brief Define a C struct together with a matching schema.
 *
 * This macro does
 *   1) define the payload type (typedef struct).
 *   2) create schema description (array of schema entries).
 *   3) set the schema attributes for a static data layout.
 *
 * The macro can be used in static code or within a function context.
 *
 * It defines the schema attributes in `struct_id##Attr`. Thus, it is recommended
 * to use `NVTX_PAYLOAD_SCHEMA_REGISTER(domain, struct_id)` to register the schema.
 *
 * Example:
 *  NVTX_DEFINE_STRUCT_WITH_SCHEMA(your_struct_name, "Your schema name",
 *      NCCL_NVTX_PAYLOAD_ENTRIES(
 *          (int, index, TYPE_INT, "integer value"),
 *          (double, dpfloat, TYPE_DOUBLE, "fp64 value"),
 *          (const char, (text, 24), TYPE_CSTRING, "text", NULL, 24)
 *      )
 *  )
 *
 * The first three fields (C type, member, entry type) of each entry are required.
 * A fixed-size array or string requires a special notation with the member
 * name and the size separated by comma and put into brackets (see last entry
 * in the example).
 *
 * The optional parameters are positional (only allowed to be passed in the
 * predefined order). A maximum of 16 schema entries is supported.
 *
 * @param struct_id The name of the struct.
 * @param schema_name (Optional 1) name of the payload schema. Default is `NULL`.
 * @param prefix (Optional 2) prefix before the schema and attributes variables,
 *               e.g. `static const`. Leave this empty, if no prefix is desired.
 * @param schema_flags (Optional 3) flags to augment the payload schema.
 *                     Default is `NVTX_PAYLOAD_SCHEMA_FLAG_NONE`.
 * @param schema_id (Optional 4) User-defined payload schema ID.
 * @param entries (Mandatory) The schema entries. This is always the last
 *                parameter to the macro.
 */
#define NVTX_DEFINE_STRUCT_WITH_SCHEMA(struct_id, ...) \
    _NVTX_DEFINE_STRUCT_WITH_SCHEMA(struct_id, __VA_ARGS__)

/**
 * \brief Initialize and register the NVTX binary payload schema.
 *
 * This does essentially the same as `NVTX_DEFINE_STRUCT_WITH_SCHEMA`, but in
 * addition the schema is registered. The schema ID will be defined as follows:
 * `const uint64_t struct_id##_schemaId`.
 *
 * @param domain The NVTX domain handle (0 for default domain).
 * All other parameters are similar to `NVTX_DEFINE_STRUCT_WITH_SCHEMA`.
 */
#define NVTX_DEFINE_STRUCT_WITH_SCHEMA_AND_REGISTER(domain, struct_id, ...) \
    _NVTX_DEFINE_STRUCT_WITH_SCHEMA(struct_id, __VA_ARGS__) \
    const uint64_t struct_id##_schemaId = nvtxPayloadSchemaRegister(domain, &struct_id##Attr);

/**
 * \brief Define payload schema for an existing `struct` and register the schema.
 *
 * This does essentially the same as `NVTX_PAYLOAD_STATIC_SCHEMA_DEFINE`, but in
 * addition, the schema is registered and `uint64_t struct_id##_schemaId` set.
 *
 * @param domain The NVTX domain handle (0 for default domain).
 * All other parameters are similar to `NVTX_PAYLOAD_STATIC_SCHEMA_DEFINE`.
 */
#define NVTX_DEFINE_SCHEMA_FOR_STRUCT_AND_REGISTER(domain, struct_id, ...) \
    _NVTX_DEFINE_SCHEMA_FOR_STRUCT(struct_id, __VA_ARGS__) \
    const uint64_t struct_id##_schemaId = nvtxPayloadSchemaRegister(domain, &struct_id##Attr);

/**
 * \brief Create a type definition for the given struct ID and members.
 *
 * This is a convenience macro. A normal `typedef` can be used instead.
 *
 * Example usage:
 *   NVTX_DEFINE_STRUCT(your_struct,
 *           (double, fp64),
 *           (uint8_t, u8),
 *           (float, fp32[3])
 *   )
 *
 * @param struct_id The name of the struct.
 * @param members The members of the struct.
 */
#define NVTX_DEFINE_STRUCT(struct_id, ...) \
    _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, __VA_ARGS__)

/**
 * \brief Register an NVTX binary payload schema.
 *
 * This is a convenience macro, which takes the same `struct_id` that has been
 * used in other helper macros. Instead, `nvtxPayloadSchemaRegister` can also be
 * used, but `&struct_id##Attr` has to be passed.
 *
 * @param domain The NVTX domain handle (0 for default domain).
 * @param struct_id The name of the struct.
 *
 * @return NVTX schema ID
 */
#define NVTX_PAYLOAD_SCHEMA_REGISTER(domain, struct_id) \
    nvtxPayloadSchemaRegister(domain, &struct_id##Attr);

