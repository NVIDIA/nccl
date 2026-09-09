/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_CONFIG_H_
#define NIIN_CONFIG_H_

#include <cstdio>

// NIIN_ON_NOT_IMPLEMENTED controls what happens when a function that is not
// implementable on top of NCCL is called at runtime. Options:
//   NIIN_TRAP  (default) - __trap(), hard stop matching NVSHMEM behavior
//   NIIN_NOOP            - silent no-op (return 0 for value-returning functions)
//   NIIN_WARN            - printf a warning, then no-op/return 0

#define NIIN_TRAP 0
#define NIIN_NOOP 1
#define NIIN_WARN 2

#ifndef NIIN_ON_NOT_IMPLEMENTED
#define NIIN_ON_NOT_IMPLEMENTED NIIN_TRAP
#endif

#if NIIN_ON_NOT_IMPLEMENTED == NIIN_TRAP

#define NIIN_NOT_IMPLEMENTED_VOID(fname) \
  do { printf("NIIN: %s not implemented\n", fname); __trap(); } while(0)

#define NIIN_NOT_IMPLEMENTED_RETURN(fname, retval) \
  do { printf("NIIN: %s not implemented\n", fname); __trap(); return retval; } while(0)

#elif NIIN_ON_NOT_IMPLEMENTED == NIIN_NOOP

#define NIIN_NOT_IMPLEMENTED_VOID(fname) \
  do { (void)0; } while(0)

#define NIIN_NOT_IMPLEMENTED_RETURN(fname, retval) \
  do { return retval; } while(0)

#elif NIIN_ON_NOT_IMPLEMENTED == NIIN_WARN

#define NIIN_NOT_IMPLEMENTED_VOID(fname) \
  do { printf("NIIN warning: %s not implemented, no-op\n", fname); } while(0)

#define NIIN_NOT_IMPLEMENTED_RETURN(fname, retval) \
  do { printf("NIIN warning: %s not implemented, returning default\n", fname); return retval; } while(0)

#else
#error "NIIN_ON_NOT_IMPLEMENTED must be NIIN_TRAP, NIIN_NOOP, or NIIN_WARN"
#endif

#endif // NIIN_CONFIG_H_
