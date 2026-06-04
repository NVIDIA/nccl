/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCLXFER_BENCH_COMMON_H_
#define NCCLXFER_BENCH_COMMON_H_

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

// ============================================================================
// CLI parsing helpers
// ============================================================================

// Pre-MPI_Init env propagation (CLI → library init env).
static inline void benchSetEnv(const char* name, const char* value) {
  setenv(name, value, 1);
}

// Parse an integer arg from argv, _Exit'ing on garbage so the bench
// fails loudly rather than silently treating "abc" as 0 (the atoi
// pitfall).
static inline int benchParseInt(const char* s, const char* what) {
  if (s == nullptr) {
    fprintf(stderr, "[bench] %s: missing value\n", what);
    _Exit(1);
  }
  char* end = nullptr;
  errno = 0;
  long v = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) {
    fprintf(stderr, "[bench] %s: invalid integer '%s'\n", what, s);
    _Exit(1);
  }
  return static_cast<int>(v);
}

// ============================================================================
// Error Checking Macros
// ============================================================================

#define MPICHECK(cmd)                                                           \
  do {                                                                          \
    int e = cmd;                                                                \
    if (e != MPI_SUCCESS) {                                                     \
      fprintf(stderr, "Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      abort();                                                                  \
    }                                                                           \
  } while (0)

#define CUDACHECK(cmd)                                                                               \
  do {                                                                                               \
    cudaError_t e = cmd;                                                                             \
    if (e != cudaSuccess) {                                                                          \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      abort();                                                                                       \
    }                                                                                                \
  } while (0)

#define NCCLCHECK(cmd)                                                                               \
  do {                                                                                               \
    ncclResult_t r = cmd;                                                                            \
    if (r != ncclSuccess) {                                                                          \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      abort();                                                                                       \
    }                                                                                                \
  } while (0)

// ============================================================================
// Argument Parsing Helpers
// ============================================================================

static inline int benchParseInt(const char* pStr) {
  if (pStr == nullptr) return 0;
  char* pEnd = nullptr;
  long value = strtol(pStr, &pEnd, 10);
  if (pEnd == pStr) return 0;
  if (value < INT_MIN) return INT_MIN;
  if (value > INT_MAX) return INT_MAX;
  return (int)value;
}

static inline size_t benchParseSize(const char* pStr) {
  char* pEnd;
  double value = strtod(pStr, &pEnd);
  if (*pEnd != '\0') {
    switch (*pEnd) {
    case 'k':
    case 'K':
      value *= 1024;
      break;
    case 'm':
    case 'M':
      value *= 1024 * 1024;
      break;
    case 'g':
    case 'G':
      value *= 1024 * 1024 * 1024;
      break;
    default:
      break;
    }
  }
  return (size_t)value;
}

static inline void benchParseMeshDims(const char* pStr, int pDims[2]) {
  char* pCopy = strdup(pStr);
  char* pSaveptr = nullptr;
  char* pToken = strtok_r(pCopy, ",x", &pSaveptr);
  pDims[0] = benchParseInt(pToken);
  pToken = strtok_r(nullptr, ",x", &pSaveptr);
  pDims[1] = benchParseInt(pToken);
  free(pCopy);
}

static inline int benchParseTensorDims(const char* pStr, size_t pDims[3]) {
  char* pCopy = strdup(pStr);
  int nDims = 0;
  char* pSaveptr = nullptr;
  char* pToken = strtok_r(pCopy, ",x", &pSaveptr);
  while (pToken != nullptr && nDims < 3) {
    pDims[nDims++] = benchParseSize(pToken);
    pToken = strtok_r(nullptr, ",x", &pSaveptr);
  }
  free(pCopy);
  return nDims;
}

static inline MPI_Comm benchMpiWorld() {
  return MPI_COMM_WORLD; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Datatype benchMpiByte() {
  return MPI_BYTE; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Datatype benchMpiInt() {
  return MPI_INT; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Datatype benchMpiDouble() {
  return MPI_DOUBLE; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Op benchMpiMin() {
  return MPI_MIN; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Op benchMpiMax() {
  return MPI_MAX; // NOLINT(bugprone-casting-through-void)
}

static inline MPI_Op benchMpiSum() {
  return MPI_SUM; // NOLINT(bugprone-casting-through-void)
}

#endif // NCCLXFER_BENCH_COMMON_H_
