#ifndef PACE_KERNELS_INCLUDE_RS_DEFS_CUH
#define PACE_KERNELS_INCLUDE_RS_DEFS_CUH

// HOST_ASSERT (SWITCH_RANKS / SWITCH_TYPES) + fprintf/exit (the fatal-default
// switch cases) come from the shared error header.
#include "util/error.hpp"

#define RS_OUT_CAST_NONE 0
#define RS_OUT_CAST_BF16 1

#define RS_OUT_MODE_DIRECT 0
#define RS_OUT_MODE_AVG 1
#define RS_OUT_MODE_CAST_BF16 2
#define RS_OUT_MODE_AVG_CAST_BF16 3

// ---------------------------------------------------------------------------
// Reduce-scatter kernel-dispatch DSL.
//
// Element type and reduction-op tags (host-side dispatch enums; distinct from
// the CUDA/NCCL runtime enums), plus the compile-time SWITCH_* dispatch macros
// that expand a runtime value into a templated launcher instantiation. Some of
// these generic dispatch macros (SWITCH_RANKS / SWITCH_TYPES / SWITCH_FUSE /
// SWITCH_RANKS_WITH_DTYPE) are also reused by the EP engine.
// ---------------------------------------------------------------------------
#define RS_TYPE_UNKNOWN 0
#define RS_TYPE_FLOAT32 1
#define RS_TYPE_BFLOAT16 2

#define RS_RED_OP_UNKNOWN 0
#define RS_RED_OP_SUM 1
#define RS_RED_OP_AVG 2

#define SWITCH_TYPE(type, type_macro) {\
    switch (type) {\
        case RS_TYPE_FLOAT32: type_macro(float); break;\
        case RS_TYPE_BFLOAT16: type_macro(__nv_bfloat16); break;\
        default: fprintf(stderr, "Unsupported data type %d\n", type); exit(EXIT_FAILURE);\
    }\
}

#define SWITCH_MUL(mul, post_mul, macro, ...) {\
    if (mul == 1.0 && post_mul == 1.0) {\
        macro(false, __VA_ARGS__);\
    } else {\
        macro(true, __VA_ARGS__);\
    }\
}

#define SWITCH_ALIGN(aligned, macro, ...) {\
    if (aligned) {\
        macro(true, __VA_ARGS__);\
    } else {\
        macro(false, __VA_ARGS__);\
    }\
}

#define SWITCH_OUT_MODE(mode, macro, ...) {\
    switch (mode) {\
        case RS_OUT_MODE_DIRECT: macro(RS_OUT_MODE_DIRECT, ##__VA_ARGS__); break;\
        case RS_OUT_MODE_AVG: macro(RS_OUT_MODE_AVG, ##__VA_ARGS__); break;\
        case RS_OUT_MODE_CAST_BF16: macro(RS_OUT_MODE_CAST_BF16, ##__VA_ARGS__); break;\
        case RS_OUT_MODE_AVG_CAST_BF16: macro(RS_OUT_MODE_AVG_CAST_BF16, ##__VA_ARGS__); break;\
        default: fprintf(stderr, "Unsupported out mode %d\n", mode); exit(EXIT_FAILURE);\
    }\
}

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 24: case_macro(24); \
        case 32: case_macro(32); \
        case 48: case_macro(48); \
        case 64: case_macro(64); \
        case 72: case_macro(72); \
        default: HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
switch (type) { \
    case CUDA_R_16BF: case_macro(nv_bfloat16); \
    default: HOST_ASSERT(false and "Unsupported type"); \
} while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        case 16: case_macro(dtype, 16); \
        case 24: case_macro(dtype, 24); \
        case 32: case_macro(dtype, 32); \
        case 48: case_macro(dtype, 48); \
        case 64: case_macro(dtype, 64); \
        case 72: case_macro(dtype, 72); \
        default: HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_FUSE(fuse, macro, ...) {\
    if (fuse) {\
        macro(true, __VA_ARGS__);\
    } else {\
        macro(false, __VA_ARGS__);\
    }\
}

#endif
