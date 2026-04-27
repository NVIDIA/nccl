/*************************************************************************
 * Unit tests for thread-safe ncclStrerror wrapper
 *
 * Tests verify that:
 *   1. ncclStrerror returns correct error messages (table-driven)
 *   2. ncclStrerror handles edge cases (zero buffer, unknown errno)
 *   3. ncclStrerror is thread-safe under concurrent access
 *   4. Source files have been patched to use ncclStrerror / strerror_r
 *
 * The source verification test (test_source_verified) intentionally FAILS
 * before the fix is applied, demonstrating the bug exists in unfixed code.
 *
 * Compile: gcc -Wall -Wextra -g -std=c99 -D_GNU_SOURCE -o test_strerror test_strerror.c -lpthread
 * Run:     ./test_strerror
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>

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
 * Function under test: ncclStrerror (extracted from src/include/checks.h)
 * ========================================================================= */

static inline const char *ncclStrerror(int errnum, char *buf, size_t buflen) {
#if (_POSIX_C_SOURCE >= 200112L) && !defined(_GNU_SOURCE)
  /* POSIX variant: int strerror_r(int, char*, size_t) */
  if (strerror_r(errnum, buf, buflen) != 0)
    snprintf(buf, buflen, "Unknown error %d", errnum);
  return buf;
#else
  /* GNU variant: char* strerror_r(int, char*, size_t) */
  return strerror_r(errnum, buf, buflen);
#endif
}

/* =========================================================================
 * Source verification: check NCCL source for thread-safe strerror
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

  /* Check 1: checks.h should contain ncclStrerror function */
  const char *checks_path = "../../src/include/checks.h";
  char *checks_src = read_file(checks_path);
  if (!checks_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", checks_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(checks_src, "ncclStrerror") == NULL) {
      snprintf(msg, sizeof(msg),
               "%s: missing ncclStrerror() helper — strerror() still used directly",
               checks_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(checks_src);
  }

  /* Check 2: linux.cc should use ncclStrerror, not raw strerror(errno) */
  const char *linux_path = "../../src/os/linux.cc";
  char *linux_src = read_file(linux_path);
  if (!linux_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", linux_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(linux_src, "strerror(errno)") != NULL) {
      snprintf(msg, sizeof(msg),
               "%s: still uses strerror(errno) (thread-unsafe) — expected ncclStrerror",
               linux_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(linux_src);
  }

  if (all_ok) {
    TEST_PASS();
  }
  return 0;
}

/* =========================================================================
 * ncclStrerror table-driven tests
 * ========================================================================= */

typedef struct {
  const char *name;
  int errnum;
  size_t buflen;
  const char *expected_substring; /* NULL = just check non-empty */
  int expect_null_terminated;
} StrerrorTestCase;

static StrerrorTestCase strerror_cases[] = {
    /* Positive: known errno values */
    {"EINVAL", EINVAL, 256, "nvalid argument", 1},
    {"ENOENT", ENOENT, 256, "o such file", 1},
    {"ENOMEM", ENOMEM, 256, "emory", 1},
    {"EBADF", EBADF, 256, "ad file", 1},
    {"EACCES", EACCES, 256, "ermission", 1},
    {"EAGAIN", EAGAIN, 256, NULL, 1},
    {"EPERM", EPERM, 256, "peration not permitted", 1},
    {"success_zero", 0, 256, NULL, 1},

    /* Negative: unknown/invalid errno values */
    {"unknown_99999", 99999, 256, NULL, 1},
    {"negative_errno", -1, 256, NULL, 1},
    {"INT_MAX_errno", INT_MAX, 256, NULL, 1},
    {"INT_MIN_errno", INT_MIN, 256, NULL, 1},

    /* Boundary: small buffers */
    {"tiny_buffer_4", EINVAL, 4, NULL, 1},
    {"tiny_buffer_1", EINVAL, 1, NULL, 1},

    {NULL, 0, 0, NULL, 0} /* sentinel */
};

int test_strerror_known_values(void) {
  for (int i = 0; strerror_cases[i].name != NULL; i++) {
    StrerrorTestCase *tc = &strerror_cases[i];
    char buf[256];
    memset(buf, 'X', sizeof(buf));

    size_t buflen = tc->buflen < sizeof(buf) ? tc->buflen : sizeof(buf);
    const char *result = ncclStrerror(tc->errnum, buf, buflen);

    char msg[512];
    snprintf(msg, sizeof(msg), "[%s] ncclStrerror returned NULL", tc->name);
    TEST_ASSERT(result != NULL, msg);

    /* Result must be non-empty for valid errno values */
    if (tc->errnum >= 0 && tc->errnum < 200) {
      snprintf(msg, sizeof(msg), "[%s] result is empty", tc->name);
      TEST_ASSERT(strlen(result) > 0, msg);
    }

    /* Check expected substring if provided */
    if (tc->expected_substring != NULL) {
      snprintf(msg, sizeof(msg), "[%s] expected '%s' in '%s'", tc->name,
               tc->expected_substring, result);
      TEST_ASSERT(strstr(result, tc->expected_substring) != NULL, msg);
    }

    /* Check null termination via returned pointer */
    if (tc->expect_null_terminated && buflen > 0) {
      size_t result_len = strlen(result);
      snprintf(msg, sizeof(msg),
               "[%s] returned string not properly null-terminated (len=%zu)",
               tc->name, result_len);
      TEST_ASSERT(result_len < 1024, msg);

      /* If result points into buf, verify it's within bounds */
      if (result >= buf && result < buf + sizeof(buf)) {
        snprintf(msg, sizeof(msg),
                 "[%s] result in buf but extends past buflen", tc->name);
        TEST_ASSERT(result + result_len < buf + buflen, msg);
      }
    }
  }
  TEST_PASS();
}

/* Zero-buffer test: buflen=0 should not crash */
int test_strerror_zero_buffer(void) {
  char buf[4] = "XYZ";
  const char *result = ncclStrerror(EINVAL, buf, 0);
  TEST_ASSERT(result != NULL, "result should not be NULL even with buflen=0");
  TEST_PASS();
}

/* Thread safety test for ncclStrerror */
#define STRERROR_NUM_THREADS 8
#define STRERROR_ITERATIONS 10000

typedef struct {
  int thread_id;
  int errnum;
  int pass;
  char failure_msg[256];
} StrerrorThreadArg;

static int strerror_errnums[] = {EINVAL, ENOENT, ENOMEM, EBADF,
                                  EAGAIN, EPERM,  EACCES, EEXIST};

static void *strerror_thread_func(void *arg) {
  StrerrorThreadArg *ta = (StrerrorThreadArg *)arg;
  ta->pass = 1;
  ta->failure_msg[0] = '\0';

  for (int i = 0; i < STRERROR_ITERATIONS; i++) {
    char buf[256];
    const char *result = ncclStrerror(ta->errnum, buf, sizeof(buf));
    if (result == NULL) {
      snprintf(ta->failure_msg, sizeof(ta->failure_msg),
               "thread %d iter %d: NULL result", ta->thread_id, i);
      ta->pass = 0;
      return NULL;
    }
    if (strlen(result) == 0) {
      snprintf(ta->failure_msg, sizeof(ta->failure_msg),
               "thread %d iter %d: empty result", ta->thread_id, i);
      ta->pass = 0;
      return NULL;
    }
  }
  return NULL;
}

int test_strerror_thread_safety(void) {
  pthread_t threads[STRERROR_NUM_THREADS];
  StrerrorThreadArg args[STRERROR_NUM_THREADS];

  for (int i = 0; i < STRERROR_NUM_THREADS; i++) {
    args[i].thread_id = i;
    args[i].errnum = strerror_errnums[i];
    args[i].pass = 0;
    pthread_create(&threads[i], NULL, strerror_thread_func, &args[i]);
  }

  for (int i = 0; i < STRERROR_NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  for (int i = 0; i < STRERROR_NUM_THREADS; i++) {
    char msg[512];
    snprintf(msg, sizeof(msg), "thread %d (errno=%d) failed: %s", i,
             args[i].errnum, args[i].failure_msg);
    TEST_ASSERT(args[i].pass, msg);
  }
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
     "Verify NCCL source uses ncclStrerror/strerror_r (FAILS before fix)"},
    {"strerror-values", test_strerror_known_values,
     "ncclStrerror with known/unknown/boundary errno values"},
    {"strerror-zero-buf", test_strerror_zero_buffer,
     "ncclStrerror with zero-length buffer"},
    {"strerror-threads", test_strerror_thread_safety,
     "ncclStrerror concurrent thread safety"},
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

  printf("ncclStrerror Thread Safety Tests\n");
  printf("=================================\n");

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

  printf("\n=================================\n");
  printf("Results: %d/%d tests passed\n", passed, total);

  if (passed == total) {
    printf("All tests PASSED!\n");
    return 0;
  } else {
    printf("Some tests FAILED!\n");
    return 1;
  }
}
