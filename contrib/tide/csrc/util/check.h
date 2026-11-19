// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace tide {

[[noreturn]] inline void fail(const std::string& what, const char* file, int line) {
  std::ostringstream os;
  os << "[tide] " << what << " (" << file << ":" << line << ")";
  throw std::runtime_error(os.str());
}

}  // namespace tide

#define TIDE_CHECK(cond, msg)                             \
  do {                                                    \
    if (!(cond)) ::tide::fail(msg, __FILE__, __LINE__);   \
  } while (0)

#define NCCLCHECK(call)                                                          \
  do {                                                                           \
    ncclResult_t r_ = (call);                                                    \
    if (r_ != ncclSuccess)                                                       \
      ::tide::fail(std::string(#call) + " -> " + ncclGetErrorString(r_), __FILE__, __LINE__); \
  } while (0)

#define CUDACHECK(call)                                                          \
  do {                                                                           \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess)                                                       \
      ::tide::fail(std::string(#call) + " -> " + cudaGetErrorString(e_), __FILE__, __LINE__); \
  } while (0)

#define CUCHECK(call)                                                            \
  do {                                                                           \
    CUresult e_ = (call);                                                        \
    if (e_ != CUDA_SUCCESS) {                                                    \
      const char* s_ = nullptr;                                                  \
      cuGetErrorString(e_, &s_);                                                 \
      ::tide::fail(std::string(#call) + " -> " + (s_ ? s_ : "?"), __FILE__, __LINE__); \
    }                                                                            \
  } while (0)
