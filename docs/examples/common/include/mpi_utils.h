/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef MPI_UTILS_H_
#define MPI_UTILS_H_

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

// MPI error checking macro
#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int err = cmd;                                                             \
    if (err != MPI_SUCCESS) {                                                  \
      char error_string[MPI_MAX_ERROR_STRING];                                 \
      int length;                                                              \
      MPI_Error_string(err, error_string, &length);                            \
      fprintf(stderr, "MPI error at %s:%d - %s\n", __FILE__, __LINE__,         \
              error_string);                                                   \
      fprintf(stderr, "Failed MPI operation: %s\n", #cmd);                     \
      MPI_Abort(MPI_COMM_WORLD, err);                                          \
    }                                                                          \
  } while (0)

#endif
