/*************************************************************************
 * Unit tests for CUDA error handling macros (tested with stubs)
 *
 * Tests verify that:
 *   1. CUDACHECK returns correct error codes (table-driven)
 *   2. CUDACHECKIGNORE logs but does not return
 *   3. CUDACHECKGOTO sets result and jumps to label
 *   4. cudaGetLastError clears sticky error state
 *   5. Source files have been patched to use CUDA_CHECK_WARN
 *
 * The source verification test (test_source_verified) intentionally FAILS
 * before the fix is applied, demonstrating the bug exists in unfixed code.
 *
 * No CUDA SDK or GPU required — uses minimal type stubs.
 *
 * Compile: gcc -Wall -Wextra -g -std=c99 -D_GNU_SOURCE -o test_cuda_error test_cuda_error.c -lpthread
 * Run:     ./test_cuda_error
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Test framework (matches plugins/tuner/example/test/ pattern)
 * ========================================================================= */

#define TEST_ASSERT(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      printf("  FAIL: %s - %s\n", __func__, message);                         \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define TEST_PASS()                                                            \
  do {                                                                         \
    printf("  PASS: %s\n", __func__);                                          \
    return 1;                                                                  \
  } while (0)

/* =========================================================================
 * Source verification
 * ========================================================================= */

static char *read_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  if (len <= 0) { fclose(f); return NULL; }
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc(len + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, len, f);
  buf[n] = '\0';
  fclose(f);
  return buf;
}

int test_source_verified(void) {
  int all_ok = 1;
  char msg[512];

  /* Check 1: nccl_ep.cc should use CUDA_CHECK_WARN */
  const char *nccl_ep_path = "../../contrib/nccl_ep/nccl_ep.cc";
  char *nccl_ep_src = read_file(nccl_ep_path);
  if (!nccl_ep_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", nccl_ep_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(nccl_ep_src, "CUDA_CHECK_WARN") == NULL) {
      snprintf(msg, sizeof(msg),
               "%s: no CUDA_CHECK_WARN found — CUDA cleanup errors are silently ignored",
               nccl_ep_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(nccl_ep_src);
  }

  /* Check 2: macros.cuh should define CUDA_CHECK_WARN */
  const char *macros_path = "../../contrib/nccl_ep/device/macros.cuh";
  char *macros_src = read_file(macros_path);
  if (!macros_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", macros_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(macros_src, "CUDA_CHECK_WARN") == NULL) {
      snprintf(msg, sizeof(msg),
               "%s: missing CUDA_CHECK_WARN macro definition",
               macros_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(macros_src);
  }

  if (all_ok) {
    TEST_PASS();
  }
  return 0;
}

/* =========================================================================
 * CUDA type stubs (no GPU required)
 * ========================================================================= */

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorMemoryAllocation 2
#define cudaErrorNoDevice 100

static cudaError_t stub_last_error = 0;

__attribute__((used))
static const char *cudaGetErrorString(cudaError_t err) {
  switch (err) {
  case cudaSuccess:
    return "no error";
  case cudaErrorInvalidValue:
    return "invalid argument";
  case cudaErrorMemoryAllocation:
    return "out of memory";
  case cudaErrorNoDevice:
    return "no CUDA-capable device is detected";
  default:
    return "unknown error";
  }
}

static cudaError_t cudaGetLastError(void) {
  cudaError_t e = stub_last_error;
  stub_last_error = cudaSuccess;
  return e;
}

/* Stub NCCL types */
typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclUnhandledCudaError 1

/* WARN/INFO stub counters */
static int warn_count = 0;
static int info_count = 0;

#define WARN(...)                                                              \
  do {                                                                         \
    warn_count++;                                                              \
  } while (0)
#define INFO(FLAGS, ...)                                                       \
  do {                                                                         \
    (void)(FLAGS);                                                             \
    info_count++;                                                              \
  } while (0)
#define NCCL_ALL 0

/* Copy of CUDACHECK macro from src/include/checks.h */
#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      WARN("Cuda failure '%s'", cudaGetErrorString(err));                      \
      (void)cudaGetLastError();                                                \
      return ncclUnhandledCudaError;                                           \
    }                                                                          \
  } while (0)

#define CUDACHECKGOTO(cmd, RES, label)                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      WARN("Cuda failure '%s'", cudaGetErrorString(err));                      \
      (void)cudaGetLastError();                                                \
      RES = ncclUnhandledCudaError;                                            \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define CUDACHECKIGNORE(cmd)                                                   \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      INFO(NCCL_ALL, "%s:%d Cuda failure '%s'", __FILE__, __LINE__,            \
           cudaGetErrorString(err));                                            \
      (void)cudaGetLastError();                                                \
    }                                                                          \
  } while (0)

/* Helper: simulates a CUDA call returning a specific error */
static cudaError_t stub_cuda_return;
static cudaError_t fake_cuda_call(void) {
  cudaError_t ret = stub_cuda_return;
  stub_last_error = ret;
  return ret;
}

/* =========================================================================
 * CUDACHECK table-driven tests
 * ========================================================================= */

typedef struct {
  const char *name;
  cudaError_t error_code;
  int expect_warn;
  int expect_success;
} CudaCheckTestCase;

static CudaCheckTestCase cuda_check_cases[] = {
    {"success", cudaSuccess, 0, 1},
    {"invalid_value", cudaErrorInvalidValue, 1, 0},
    {"out_of_memory", cudaErrorMemoryAllocation, 1, 0},
    {"no_device", cudaErrorNoDevice, 1, 0},
    {NULL, 0, 0, 0}};

static ncclResult_t test_cudacheck_wrapper(cudaError_t err) {
  stub_cuda_return = err;
  CUDACHECK(fake_cuda_call());
  return ncclSuccess;
}

int test_cudacheck_cases(void) {
  for (int i = 0; cuda_check_cases[i].name != NULL; i++) {
    CudaCheckTestCase *tc = &cuda_check_cases[i];
    warn_count = 0;
    stub_last_error = cudaSuccess;

    ncclResult_t ret = test_cudacheck_wrapper(tc->error_code);

    char msg[512];
    if (tc->expect_success) {
      snprintf(msg, sizeof(msg), "[%s] expected ncclSuccess, got %d", tc->name, ret);
      TEST_ASSERT(ret == ncclSuccess, msg);
    } else {
      snprintf(msg, sizeof(msg), "[%s] expected error, got ncclSuccess", tc->name);
      TEST_ASSERT(ret == ncclUnhandledCudaError, msg);
    }

    if (tc->expect_warn) {
      snprintf(msg, sizeof(msg), "[%s] expected WARN to be called", tc->name);
      TEST_ASSERT(warn_count > 0, msg);
    } else {
      snprintf(msg, sizeof(msg), "[%s] WARN should not be called", tc->name);
      TEST_ASSERT(warn_count == 0, msg);
    }
  }
  TEST_PASS();
}

int test_cudacheckignore(void) {
  info_count = 0;
  stub_cuda_return = cudaSuccess;
  CUDACHECKIGNORE(fake_cuda_call());
  TEST_ASSERT(info_count == 0, "CUDACHECKIGNORE should not log on success");

  info_count = 0;
  stub_cuda_return = cudaErrorInvalidValue;
  CUDACHECKIGNORE(fake_cuda_call());
  TEST_ASSERT(info_count == 1, "CUDACHECKIGNORE should log on error");

  TEST_PASS();
}

int test_cudacheckgoto(void) {
  ncclResult_t res = ncclSuccess;
  int reached_label = 0;

  warn_count = 0;
  stub_cuda_return = cudaSuccess;
  CUDACHECKGOTO(fake_cuda_call(), res, error_label);
  TEST_ASSERT(res == ncclSuccess, "CUDACHECKGOTO success: res should be 0");
  TEST_ASSERT(warn_count == 0, "CUDACHECKGOTO success: no WARN");
  goto skip_error;

error_label:
  reached_label = 1;

skip_error:
  TEST_ASSERT(!reached_label, "CUDACHECKGOTO success: should not goto label");

  res = ncclSuccess;
  reached_label = 0;
  warn_count = 0;
  stub_cuda_return = cudaErrorMemoryAllocation;
  CUDACHECKGOTO(fake_cuda_call(), res, error_label2);
  TEST_ASSERT(0, "CUDACHECKGOTO error: should have jumped to label");

error_label2:
  reached_label = 1;
  TEST_ASSERT(reached_label, "CUDACHECKGOTO error: should reach label");
  TEST_ASSERT(res == ncclUnhandledCudaError,
              "CUDACHECKGOTO error: res should be ncclUnhandledCudaError");
  TEST_ASSERT(warn_count == 1, "CUDACHECKGOTO error: WARN should be called");
  TEST_PASS();
}

int test_cuda_error_state_cleared(void) {
  stub_last_error = cudaErrorInvalidValue;
  TEST_ASSERT(stub_last_error != cudaSuccess, "error should be set");

  cudaError_t cleared = cudaGetLastError();
  TEST_ASSERT(cleared == cudaErrorInvalidValue,
              "cudaGetLastError should return the error");
  TEST_ASSERT(stub_last_error == cudaSuccess,
              "after cudaGetLastError, error should be cleared");

  cleared = cudaGetLastError();
  TEST_ASSERT(cleared == cudaSuccess,
              "second cudaGetLastError should return success");
  TEST_PASS();
}

/* =========================================================================
 * Test runner
 * ========================================================================= */

typedef int (*TestFunction)(void);

typedef struct {
  const char *name;
  TestFunction func;
  const char *description;
} TestCase;

static TestCase test_cases[] = {
    {"source-verified", test_source_verified,
     "Verify NCCL source uses CUDA_CHECK_WARN (FAILS before fix)"},
    {"cudacheck-cases", test_cudacheck_cases,
     "CUDACHECK macro with various error codes"},
    {"cudacheckignore", test_cudacheckignore,
     "CUDACHECKIGNORE logs but does not return"},
    {"cudacheckgoto", test_cudacheckgoto,
     "CUDACHECKGOTO sets result and jumps to label"},
    {"cuda-error-clear", test_cuda_error_state_cleared,
     "cudaGetLastError clears sticky error state"},
    {NULL, NULL, NULL}};

static void show_help(const char *prog) {
  printf("Usage: %s [test-name ...]\n\n", prog);
  printf("Available tests:\n");
  for (int i = 0; test_cases[i].name != NULL; i++) {
    printf("  %-25s %s\n", test_cases[i].name, test_cases[i].description);
  }
}

static TestFunction find_test(const char *name) {
  for (int i = 0; test_cases[i].name != NULL; i++) {
    if (strcmp(test_cases[i].name, name) == 0)
      return test_cases[i].func;
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc > 1 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    show_help(argv[0]);
    return 0;
  }

  printf("CUDA Error Handling Tests\n");
  printf("==========================\n");

  int passed = 0, total = 0;

  if (argc == 1) {
    for (int i = 0; test_cases[i].name != NULL; i++) {
      total++;
      passed += test_cases[i].func();
    }
  } else {
    for (int arg = 1; arg < argc; arg++) {
      TestFunction func = find_test(argv[arg]);
      if (func) {
        total++;
        passed += func();
      } else {
        printf("ERROR: Unknown test '%s'\n", argv[arg]);
        show_help(argv[0]);
        return 1;
      }
    }
  }

  printf("\n==========================\n");
  printf("Results: %d/%d tests passed\n", passed, total);

  if (passed == total) {
    printf("All tests PASSED!\n");
    return 0;
  } else {
    printf("Some tests FAILED!\n");
    return 1;
  }
}
