/*
* Copyright 2023  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_EXT_PAYLOAD_HELPER_INTERNAL_H
#define NVTX_EXT_PAYLOAD_HELPER_INTERNAL_H

/* General helper macros */
#include "nvtxExtHelperMacros.h"

/* Get variable name with line number (almost unique per file). */
#define _NVTX_PAYLOAD_DATA_VAR NVTX_EXT_CONCAT(nvtxDFDB,__LINE__)

/* Create real arguments from just pasting tokens next to each other. */
#define _NVTX_PAYLOAD_PASS_THROUGH(...) __VA_ARGS__

/* Avoid prefixing `NVTX_PAYLOAD_ENTRY_` for nested payloads. */
#define NVTX_PAYLOAD_ENTRY_THROWAWAY
#define _NVTX_PAYLOAD_NESTED(id) THROWAWAY id

/*
 * Create the NVTX binary payloads schema attributes.
 *
 * @param struct_id The name of the struct.
 * @param schema_name The name of the schema.
 * @param schema_flags Additional schema flags
 * @param mask_add Fields to be added to the mask.
 * @param num_entries The number schema entries.
 */
#define NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, schema_id, mask_add, num_entries) \
    nvtxPayloadSchemaAttr_t struct_id##Attr = { \
        /*.fieldMask = */NVTX_PAYLOAD_SCHEMA_ATTR_TYPE | mask_add \
            NVTX_PAYLOAD_SCHEMA_ATTR_ENTRIES | \
            NVTX_PAYLOAD_SCHEMA_ATTR_NUM_ENTRIES | \
            NVTX_PAYLOAD_SCHEMA_ATTR_STATIC_SIZE, \
        /*.name = */schema_name, \
        /*.type = */NVTX_PAYLOAD_SCHEMA_TYPE_STATIC, \
        /*.flags = */schema_flags, \
        /*.entries = */struct_id##Schema, /*.numEntries = */num_entries, \
        /*.payloadStaticSize = */sizeof(struct_id), \
        /*.packAlign = */0, /*.schemaId = */schema_id};


/*****************************************************************/
/*** Helper for `NVTX_DEFINE_SCHEMA_FOR_STRUCT[_AND_REGISTER]` ***/

/* First part of schema entry for different number of arguments. */
#define _NVTX_PAYLOAD_SCHEMA_EF2(member, etype) \
    0, NVTX_PAYLOAD_ENTRY_##etype, NULL, NULL, 0,
#define _NVTX_PAYLOAD_SCHEMA_EF3(member, etype, name) \
    0, NVTX_PAYLOAD_ENTRY_##etype, name, NULL, 0,
#define _NVTX_PAYLOAD_SCHEMA_EF4(member, etype, name, desc) \
    0, NVTX_PAYLOAD_ENTRY_##etype, name, desc, 0,
#define _NVTX_PAYLOAD_SCHEMA_EF5(member, etype, name, desc, arraylen) \
    0, NVTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,
#define _NVTX_PAYLOAD_SCHEMA_EF6(member, etype, name, desc, arraylen, flags) \
    NVTX_PAYLOAD_ENTRY_FLAG_##flags, NVTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,

#define _NVTX_PAYLOAD_SCHEMA_ENTRY_FRONT(...) \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_SCHEMA_EF, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

/* Second part of schema entry (append struct member).
   (At least two arguments are passed (`member` and `etype`). */
#define _NVTX_PAYLOAD_SCHEMA_ENTRY_END(member, ...) member

/* Resolve to schema entry. `entry` is `(ctype, name, ...)`. */
#define _NVTX_PAYLOAD_SCHEMA_ENTRY(struct_id, entry) \
    {_NVTX_PAYLOAD_SCHEMA_ENTRY_FRONT entry \
    offsetof(struct_id, _NVTX_PAYLOAD_SCHEMA_ENTRY_END entry)},

/* Handle up to 16 schema entries. */
#define _NVTX_PAYLOAD_SME1(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1)
#define _NVTX_PAYLOAD_SME2(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME1(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME3(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME2(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME4(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME3(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME5(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME4(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME6(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME5(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME7(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME6(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME8(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME7(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME9(s,e1,...)  _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME8(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME10(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME9(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME11(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME10(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME12(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME11(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME13(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME12(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME14(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME13(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME15(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME14(s,__VA_ARGS__)
#define _NVTX_PAYLOAD_SME16(s,e1,...) _NVTX_PAYLOAD_SCHEMA_ENTRY(s,e1) _NVTX_PAYLOAD_SME15(s,__VA_ARGS__)

#define _NVTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, ...) \
  nvtxPayloadSchemaEntry_t struct_id##Schema[] = { \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_SME, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(struct_id, __VA_ARGS__) \
    {0, 0} \
  };

/*
 * Handle optional parameters for `NVTX_DEFINE_SCHEMA_FOR_STRUCT[_AND_REGISTER]`.
 */
#define _NVTX_DEFINE_S4S_6(struct_id, schema_name, prefix, schema_flags, schema_id, entries) \
    prefix _NVTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
    prefix NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, schema_id, \
        NVTX_PAYLOAD_SCHEMA_ATTR_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FLAGS | NVTX_PAYLOAD_SCHEMA_ATTR_SCHEMA_ID |,\
        NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))
#define _NVTX_DEFINE_S4S_5(struct_id, schema_name, prefix, schema_flags, entries) \
    prefix _NVTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
    prefix NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, 0, \
        NVTX_PAYLOAD_SCHEMA_ATTR_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FLAGS |, \
        NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))
#define _NVTX_DEFINE_S4S_4(struct_id, schema_name, prefix, entries) \
    prefix _NVTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
    prefix NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, NVTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, \
        NVTX_PAYLOAD_SCHEMA_ATTR_NAME |, \
        NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))
#define _NVTX_DEFINE_S4S_3(struct_id, schema_name, entries) \
    _NVTX_DEFINE_S4S_4(struct_id, schema_name, /*prefix*/, entries)
#define _NVTX_DEFINE_S4S_2(struct_id, entries) \
    _NVTX_PAYLOAD_SCHEMA_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
    NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, NULL, NVTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, ,\
        NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))

#define _NVTX_DEFINE_SCHEMA_FOR_STRUCT(struct_id, ...) \
    NVTX_EXT_CONCAT(_NVTX_DEFINE_S4S_, \
        NVTX_EXT_NUM_ARGS(struct_id, __VA_ARGS__))(struct_id, __VA_ARGS__)

/*** END: Helper for `NVTX_PAYLOAD_STATIC_SCHEMA_{DEFINE,SETUP}` ***/


/******************************************************************/
/*** Helper for `NVTX_DEFINE_STRUCT_WITH_SCHEMA[_AND_REGISTER]` ***/

/* Extract struct member for fixed-size arrays. */
#define _NVTX_PAYLOAD_STRUCT_ARR_MEM1(name) name
#define _NVTX_PAYLOAD_STRUCT_ARR_MEM2(name, count) name[count]

/* Extract type and member name and handle special case of fixed-size array. */
#define _NVTX_PAYLOAD_STRUCT_E2(type, member) type member;
#define _NVTX_PAYLOAD_STRUCT_E3(type, member, etype) type member;
#define _NVTX_PAYLOAD_STRUCT_E4(type, member, etype, name) type member;
#define _NVTX_PAYLOAD_STRUCT_E5(type, member, etype, name, desc) type member;
#define _NVTX_PAYLOAD_STRUCT_E6(type, member, etype, name, desc, arraylen) \
    type NVTX_EXT_CONCAT(_NVTX_PAYLOAD_STRUCT_ARR_MEM, NVTX_EXT_NUM_ARGS member) member;
#define _NVTX_PAYLOAD_STRUCT_E7(type, member, etype, name, desc, arraylen, flags) \
    _NVTX_PAYLOAD_STRUCT_E6(type, member, etype, name, desc, arraylen)

/* Handle different number of arguments per struct entry. */
#define _NVTX_PAYLOAD_STRUCT_ENTRY_(...) \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_STRUCT_E, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

/* Handle up to 16 struct members. */
#define _NVTX_PAYLOAD_STRUCT_ENTRY(entry) _NVTX_PAYLOAD_STRUCT_ENTRY_ entry
#define _NVTX_PAYLOAD_STRUCT1(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1)
#define _NVTX_PAYLOAD_STRUCT2(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT1(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT3(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT2(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT4(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT3(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT5(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT4(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT6(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT5(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT7(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT6(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT8(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT7(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT9(e1, ...)  _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT8(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT10(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT9(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT11(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT10(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT12(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT11(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT13(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT12(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT14(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT13(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT15(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT14(__VA_ARGS__)
#define _NVTX_PAYLOAD_STRUCT16(e1, ...) _NVTX_PAYLOAD_STRUCT_ENTRY(e1) _NVTX_PAYLOAD_STRUCT15(__VA_ARGS__)

/* Generate the typedef. */
#define _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, ...) \
  typedef struct { \
      NVTX_EXT_CONCAT(_NVTX_PAYLOAD_STRUCT, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__) \
  } struct_id;

/* Generate first part of the schema entry. */
#define _NVTX_PAYLOAD_INIT_SCHEMA_N3(type, memberId, etype) \
    0, NVTX_PAYLOAD_ENTRY_##etype, NULL, NULL, 0,
#define _NVTX_PAYLOAD_INIT_SCHEMA_N4(type, memberId, etype, name) \
    0, NVTX_PAYLOAD_ENTRY_##etype, name, NULL, 0,
#define _NVTX_PAYLOAD_INIT_SCHEMA_N5(type, memberId, etype, name, desc) \
    0, NVTX_PAYLOAD_ENTRY_##etype, name, desc, 0,
#define _NVTX_PAYLOAD_INIT_SCHEMA_N6(type, memberId, etype, name, desc, arraylen) \
    0, NVTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,
#define _NVTX_PAYLOAD_INIT_SCHEMA_N7(type, memberId, etype, name, desc, arraylen, flags) \
    NVTX_PAYLOAD_ENTRY_FLAG_##flags, NVTX_PAYLOAD_ENTRY_##etype, name, desc, arraylen,

#define _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY_FRONT(...) \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_INIT_SCHEMA_N, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

#define _NVTX_PAYLOAD_ARRAY_MEMBER1(name) name
#define _NVTX_PAYLOAD_ARRAY_MEMBER2(name, count) name

/* Resolve to last part of schema entry (append struct member). */
#define _NVTX_PAYLOAD_INIT_SCHEMA_NX3(type, memberId, ...) memberId
#define _NVTX_PAYLOAD_INIT_SCHEMA_NX4(type, memberId, ...) memberId
#define _NVTX_PAYLOAD_INIT_SCHEMA_NX5(type, memberId, ...) memberId
#define _NVTX_PAYLOAD_INIT_SCHEMA_NX6(type, memberId, ...) \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_ARRAY_MEMBER, NVTX_EXT_NUM_ARGS memberId) memberId
#define _NVTX_PAYLOAD_INIT_SCHEMA_NX7(type, memberId, ...) \
    _NVTX_PAYLOAD_INIT_SCHEMA_NX6(type, memberId, __VA_ARGS__)

#define _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY_END(...) \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_INIT_SCHEMA_NX, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

/* Resolve to schema entry. `entry` is `(ctype, name, ...)`. */
#define _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(struct_id, entry) \
    {_NVTX_PAYLOAD_SCHEMA_INIT_ENTRY_FRONT entry \
    offsetof(struct_id, _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY_END entry)},

/* Handle up to 16 schema entries. */
#define _NVTX_PAYLOAD_INIT_SME1(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1)
#define _NVTX_PAYLOAD_INIT_SME2(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME1(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME3(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME2(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME4(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME3(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME5(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME4(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME6(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME5(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME7(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME6(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME8(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME7(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME9(s, e1, ...)  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME8(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME10(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME9(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME11(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME10(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME12(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME11(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME13(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME12(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME14(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME13(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME15(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME14(s, __VA_ARGS__)
#define _NVTX_PAYLOAD_INIT_SME16(s, e1, ...) _NVTX_PAYLOAD_SCHEMA_INIT_ENTRY(s, e1) _NVTX_PAYLOAD_INIT_SME15(s, __VA_ARGS__)

#define _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, ...) \
  nvtxPayloadSchemaEntry_t struct_id##Schema[] = { \
    NVTX_EXT_CONCAT(_NVTX_PAYLOAD_INIT_SME, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(struct_id, __VA_ARGS__) \
    {0, 0} \
  };

/*
 * Handle optional parameters for `NVTX_DEFINE_STRUCT_WITH_SCHEMA[_AND_REGISTER]`.
 */
#define _NVTX_DEFINE_SWS_6(struct_id, schema_name, prefix, schema_flags, schema_id, entries) \
  _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, schema_id, \
      NVTX_PAYLOAD_SCHEMA_ATTR_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FLAGS | \
      NVTX_PAYLOAD_SCHEMA_ATTR_SCHEMA_ID |, \
      NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))
#define _NVTX_DEFINE_SWS_5(struct_id, schema_name, prefix, schema_flags, entries) \
  _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, schema_flags, 0, \
      NVTX_PAYLOAD_SCHEMA_ATTR_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FLAGS |, \
      NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))
#define _NVTX_DEFINE_SWS_4(struct_id, schema_name, prefix, entries) \
  _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  prefix NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, schema_name, NVTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, \
      NVTX_PAYLOAD_SCHEMA_ATTR_NAME |, \
      NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))
#define _NVTX_DEFINE_SWS_3(struct_id, schema_name, entries) \
  _NVTX_DEFINE_SWS_4(struct_id, schema_name, /* no prefix */, entries)
#define _NVTX_DEFINE_SWS_2(struct_id, entries) \
  _NVTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  _NVTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _NVTX_PAYLOAD_PASS_THROUGH entries) \
  NVTX_PAYLOAD_SCHEMA_ATTR(struct_id, NULL, NVTX_PAYLOAD_SCHEMA_FLAG_NONE, 0, , \
      NVTX_EXT_NUM_ARGS(_NVTX_PAYLOAD_PASS_THROUGH entries))

#define _NVTX_DEFINE_STRUCT_WITH_SCHEMA(struct_id, ...) \
    NVTX_EXT_CONCAT(_NVTX_DEFINE_SWS_, \
        NVTX_EXT_NUM_ARGS(struct_id, __VA_ARGS__))(struct_id, __VA_ARGS__)

/*** END: Helper for `NVTX_PAYLOAD_STATIC_SCHEMA_{INIT,CREATE}` */

#endif /* NVTX_EXT_PAYLOAD_HELPER_INTERNAL_H */