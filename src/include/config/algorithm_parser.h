/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef NCCL_CONFIG_ALGORITHM_PARSER_H_
#define NCCL_CONFIG_ALGORITHM_PARSER_H_

#include "nccl.h"
#include <stdint.h>

// Parse an algSelection string into an algorithmMask.
// Grammar: selection := term (',' term)* ; term := factor ('+' factor)* ;
//   factor := ['^'] tag.  ',' unions, '+' intersects, '^' negates.
//   Tags match registry names as case-insensitive name prefixes. Case-insensitive,
//   whitespace ignored, no parentheses.
// NULL/empty -> *outMask = 0 (automatic). Unknown tag, malformed term (no real
// tag, e.g. "+"), or contradictory term (empty intersection) -> ncclInvalidArgument.
// No logging here: the caller (ncclCollConfigGetAlgMask) decides WARN vs
// INFO based on forceAlgSelection.
// All the valid strings can be found in src/config/algorithm_registry.cc
ncclResult_t ncclAlgParse(const char* sel, uint64_t* outMask);

#endif
