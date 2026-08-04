#ifndef PACE_UTIL_ERROR_HPP_
#define PACE_UTIL_ERROR_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <exception>
#include <string>

#include "cuda_runtime.h"

// ---------------------------------------------------------------------------
// CUDA / NCCL / CU error-check macros. Host-side; log to stderr and abort on
// failure. (Callers must have the relevant nccl.h / cuda.h types in scope.)
// ---------------------------------------------------------------------------
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUCHECK(cmd)                                                           \
  do {                                                                         \
    CUresult err = cmd;                                                        \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              errStr);                                                         \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Use inline function to clear CUDA error inside expressions
static inline cudaError_t cuda_clear(cudaError_t err) {
  if (err != cudaSuccess)
    (void)cudaGetLastError();
  return err;
}

// Check if cudaSuccess & clear CUDA error
#define CUDASUCCESS(cmd) cuda_clear(cmd) == cudaSuccess

// ---------------------------------------------------------------------------
// GIN assertions: HOST_ASSERT throws (host), DEVICE_ASSERT prints + traps.
// ---------------------------------------------------------------------------
#ifndef GIN_STATIC_ASSERT
#define GIN_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

/**
 * Exception class for Gin runtime errors. Detailed message with file/line.
 */
class GinException : public std::exception {
private:
    std::string message = {};

public:
    explicit GinException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#ifndef HOST_ASSERT
#define HOST_ASSERT(cond)                                               \
    do {                                                                \
        if (not(cond)) {                                                \
            throw GinException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)
#endif

#ifndef DEVICE_ASSERT
#define DEVICE_ASSERT(cond)                                                             \
    do {                                                                                \
        if (not(cond)) {                                                                \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;");                                                               \
        }                                                                               \
    } while (0)
#endif

#endif // PACE_UTIL_ERROR_HPP_
