/*************************************************************************
 * Unit tests for thread-safe strtok_r / gmtime_r replacements
 *
 * Tests verify that:
 *   1. strtok_r correctly tokenizes strings (table-driven)
 *   2. gmtime_r correctly converts timestamps (table-driven)
 *   3. Both functions are thread-safe under concurrent access
 *   4. Source files have been patched to use the thread-safe variants
 *
 * The source verification test (test_source_verified) intentionally FAILS
 * before the fix is applied, demonstrating the bug exists in unfixed code.
 *
 * Compile: gcc -Wall -Wextra -g -std=c99 -D_GNU_SOURCE -o test_strtok_gmtime test_strtok_gmtime.c -lpthread
 * Run:     ./test_strtok_gmtime
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
 * Source verification: read NCCL source and check for thread-safe variants
 * ========================================================================= */

/* Helper: read entire file into malloc'd buffer. Returns NULL on failure. */
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

  /* Check 1: plugin source should not use thread-unsafe strtok.
   * Try plugin.cc first (C++ modernization), fall back to plugin.c. */
  const char *plugin_cc_path = "../../plugins/tuner/example/plugin.cc";
  const char *plugin_c_path = "../../plugins/tuner/example/plugin.c";
  char *plugin_src = read_file(plugin_cc_path);
  const char *plugin_path = plugin_cc_path;
  if (!plugin_src) {
    plugin_src = read_file(plugin_c_path);
    plugin_path = plugin_c_path;
  }
  if (!plugin_src) {
    snprintf(msg, sizeof(msg), "Cannot read plugin source (run from tests/c/)");
    printf("  SKIP: %s - %s\n", __func__, msg);
    /* Don't fail if file not found — allows running outside repo */
  } else {
    /* strstr("strtok(") matches bare strtok but NOT strtok_r (different char before paren) */
    if (strstr(plugin_src, "strtok(") != NULL) {
      snprintf(msg, sizeof(msg),
               "%s: still uses strtok (thread-unsafe) — expected strtok_r or std::string",
               plugin_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(plugin_src);
  }

  /* Check 2: inspector.cc should use gmtime_r (not gmtime) */
  const char *inspector_path = "../../plugins/profiler/inspector/inspector.cc";
  char *inspector_src = read_file(inspector_path);
  if (!inspector_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", inspector_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(inspector_src, "gmtime_r") == NULL) {
      snprintf(msg, sizeof(msg),
               "%s: still uses gmtime (thread-unsafe) — expected gmtime_r",
               inspector_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(inspector_src);
  }

  if (all_ok) {
    TEST_PASS();
  }
  return 0;
}

/* =========================================================================
 * strtok_r table-driven tests
 * ========================================================================= */

#define MAX_TOKENS 16

typedef struct {
  const char *name;
  const char *input;
  size_t input_len;      /* 0 = use strlen, >0 = explicit (for embedded nulls) */
  const char *delimiters;
  const char *expected_tokens[MAX_TOKENS]; /* NULL-terminated list */
  int expected_count;
} TokenizeTestCase;

static TokenizeTestCase tokenize_cases[] = {
    /* Positive: normal tokenization */
    {"pipe_delimited", "INIT|COLL|P2P", 0, "|", {"INIT", "COLL", "P2P", NULL}, 3},
    {"comma_delimited", "A,B,C,D", 0, ",", {"A", "B", "C", "D", NULL}, 4},
    {"single_token", "INIT", 0, ",", {"INIT", NULL}, 1},

    /* Negative: edge cases */
    {"empty_string", "", 0, ",", {NULL}, 0},
    {"trailing_delim", "A,B,", 0, ",", {"A", "B", NULL}, 2},
    {"leading_delim", ",A,B", 0, ",", {"A", "B", NULL}, 2},
    {"consecutive_delim", "A,,B", 0, ",", {"A", "B", NULL}, 2},
    {"all_delimiters", ",,,,", 0, ",", {NULL}, 0},

    /* Security: embedded null (strtok_r stops at \0) */
    {"embedded_null", "A,B\0C,D", 7, ",", {"A", "B", NULL}, 2},

    /* Boundary: multi-char delimiters, high bytes */
    {"multi_char_delim", "A,;B;,C", 0, ",;", {"A", "B", "C", NULL}, 3},
    {"high_bytes", "\xff,\xfe,\x80", 0, ",", {"\xff", "\xfe", "\x80", NULL}, 3},

    {NULL, NULL, 0, NULL, {NULL}, 0} /* sentinel */
};

int test_strtok_r_cases(void) {
  for (int i = 0; tokenize_cases[i].name != NULL; i++) {
    TokenizeTestCase *tc = &tokenize_cases[i];

    /* strtok_r modifies the input, so we must copy it */
    size_t len = tc->input_len > 0 ? tc->input_len : strlen(tc->input);
    char *input = (char *)malloc(len + 1);
    memcpy(input, tc->input, len);
    input[len] = '\0';

    char *saveptr = NULL;
    const char *tokens[MAX_TOKENS];
    int count = 0;

    char *token = strtok_r(input, tc->delimiters, &saveptr);
    while (token != NULL && count < MAX_TOKENS) {
      tokens[count++] = token;
      token = strtok_r(NULL, tc->delimiters, &saveptr);
    }

    char msg[512];
    snprintf(msg, sizeof(msg), "[%s] expected %d tokens, got %d", tc->name,
             tc->expected_count, count);
    TEST_ASSERT(count == tc->expected_count, msg);

    for (int j = 0; j < count; j++) {
      snprintf(msg, sizeof(msg), "[%s] token[%d] expected '%s' got '%s'",
               tc->name, j, tc->expected_tokens[j], tokens[j]);
      TEST_ASSERT(strcmp(tokens[j], tc->expected_tokens[j]) == 0, msg);
    }

    free(input);
  }
  TEST_PASS();
}

/* Very long token test */
int test_strtok_r_very_long_token(void) {
  const int LEN = 4096;
  char *input = (char *)malloc(LEN + 1);
  memset(input, 'A', LEN);
  input[LEN] = '\0';

  char *saveptr = NULL;
  char *token = strtok_r(input, ",", &saveptr);

  TEST_ASSERT(token != NULL, "token should not be NULL");
  TEST_ASSERT((int)strlen(token) == LEN, "token should be full length");
  TEST_ASSERT(strtok_r(NULL, ",", &saveptr) == NULL,
              "should be only one token");

  free(input);
  TEST_PASS();
}

/* Thread safety: each thread tokenizes its own string */
#define TOKENIZE_NUM_THREADS 8
#define TOKENIZE_ITERATIONS 1000

typedef struct {
  int thread_id;
  int pass;
  char failure_msg[256];
} TokenizeThreadArg;

static void *tokenize_thread_func(void *arg) {
  TokenizeThreadArg *ta = (TokenizeThreadArg *)arg;
  ta->pass = 1;
  ta->failure_msg[0] = '\0';

  for (int i = 0; i < TOKENIZE_ITERATIONS; i++) {
    char input[64];
    /* Each thread creates a unique string to verify no cross-contamination */
    snprintf(input, sizeof(input), "T%d_A,T%d_B,T%d_C", ta->thread_id,
             ta->thread_id, ta->thread_id);

    char *saveptr = NULL;
    char *t1 = strtok_r(input, ",", &saveptr);
    char *t2 = strtok_r(NULL, ",", &saveptr);
    char *t3 = strtok_r(NULL, ",", &saveptr);
    char *t4 = strtok_r(NULL, ",", &saveptr);

    if (t1 == NULL || t2 == NULL || t3 == NULL || t4 != NULL) {
      snprintf(ta->failure_msg, sizeof(ta->failure_msg),
               "thread %d iter %d: wrong token count", ta->thread_id, i);
      ta->pass = 0;
      return NULL;
    }

    char expected[32];
    snprintf(expected, sizeof(expected), "T%d_A", ta->thread_id);
    if (strcmp(t1, expected) != 0) {
      snprintf(ta->failure_msg, sizeof(ta->failure_msg),
               "thread %d iter %d: t1='%s' expected '%s'", ta->thread_id, i,
               t1, expected);
      ta->pass = 0;
      return NULL;
    }
  }
  return NULL;
}

int test_strtok_r_thread_safety(void) {
  pthread_t threads[TOKENIZE_NUM_THREADS];
  TokenizeThreadArg args[TOKENIZE_NUM_THREADS];

  for (int i = 0; i < TOKENIZE_NUM_THREADS; i++) {
    args[i].thread_id = i;
    args[i].pass = 0;
    pthread_create(&threads[i], NULL, tokenize_thread_func, &args[i]);
  }

  for (int i = 0; i < TOKENIZE_NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  for (int i = 0; i < TOKENIZE_NUM_THREADS; i++) {
    char msg[512];
    snprintf(msg, sizeof(msg), "thread %d failed: %s", i, args[i].failure_msg);
    TEST_ASSERT(args[i].pass, msg);
  }
  TEST_PASS();
}

/* =========================================================================
 * gmtime_r table-driven tests
 * ========================================================================= */

typedef struct {
  const char *name;
  time_t timestamp;
  int expected_year;  /* tm_year + 1900 */
  int expected_month; /* tm_mon + 1 */
  int expected_day;   /* tm_mday */
} GmtimeTestCase;

static GmtimeTestCase gmtime_cases[] = {
    /* Positive: known timestamps */
    {"epoch", 0, 1970, 1, 1},
    {"y2k", 946684800, 2000, 1, 1},
    {"recent", 1700000000, 2023, 11, 14},

    /* Boundary: negative timestamp (before epoch) */
    {"before_epoch", -1, 1969, 12, 31},

    /* Boundary: Y2038 (32-bit time_t overflow) */
    {"y2038_boundary", 2147483647, 2038, 1, 19},

    /* Boundary: post-Y2038 (requires 64-bit time_t) */
    {"post_y2038", 4102444800L, 2100, 1, 1},

    {NULL, 0, 0, 0, 0} /* sentinel */
};

int test_gmtime_r_cases(void) {
  for (int i = 0; gmtime_cases[i].name != NULL; i++) {
    GmtimeTestCase *tc = &gmtime_cases[i];

    /* Skip post-Y2038 test on 32-bit systems */
    if (sizeof(time_t) <= 4 && tc->timestamp > 2147483647) {
      printf("  SKIP: [%s] (32-bit time_t)\n", tc->name);
      continue;
    }

    struct tm result;
    struct tm *ret = gmtime_r(&tc->timestamp, &result);

    char msg[512];
    snprintf(msg, sizeof(msg), "[%s] gmtime_r returned NULL", tc->name);
    TEST_ASSERT(ret != NULL, msg);
    TEST_ASSERT(ret == &result, "gmtime_r should return pointer to result buf");

    int year = result.tm_year + 1900;
    int month = result.tm_mon + 1;
    int day = result.tm_mday;

    snprintf(msg, sizeof(msg), "[%s] expected %d/%d/%d got %d/%d/%d", tc->name,
             tc->expected_year, tc->expected_month, tc->expected_day, year,
             month, day);
    TEST_ASSERT(year == tc->expected_year && month == tc->expected_month &&
                    day == tc->expected_day,
                msg);
  }
  TEST_PASS();
}

/* Thread safety: concurrent gmtime_r calls */
#define GMTIME_NUM_THREADS 8
#define GMTIME_ITERATIONS 10000

typedef struct {
  int thread_id;
  time_t timestamp;
  int expected_year;
  int pass;
  char failure_msg[256];
} GmtimeThreadArg;

static void *gmtime_thread_func(void *arg) {
  GmtimeThreadArg *ta = (GmtimeThreadArg *)arg;
  ta->pass = 1;
  ta->failure_msg[0] = '\0';

  for (int i = 0; i < GMTIME_ITERATIONS; i++) {
    struct tm result;
    struct tm *ret = gmtime_r(&ta->timestamp, &result);
    if (ret == NULL) {
      snprintf(ta->failure_msg, sizeof(ta->failure_msg),
               "thread %d iter %d: gmtime_r returned NULL", ta->thread_id, i);
      ta->pass = 0;
      return NULL;
    }
    int year = result.tm_year + 1900;
    if (year != ta->expected_year) {
      snprintf(ta->failure_msg, sizeof(ta->failure_msg),
               "thread %d iter %d: expected year %d got %d (cross-thread "
               "contamination!)",
               ta->thread_id, i, ta->expected_year, year);
      ta->pass = 0;
      return NULL;
    }
  }
  return NULL;
}

int test_gmtime_r_thread_safety(void) {
  pthread_t threads[GMTIME_NUM_THREADS];
  GmtimeThreadArg args[GMTIME_NUM_THREADS];

  /* Each thread gets a different timestamp with a different year */
  time_t timestamps[] = {0,          946684800,  1000000000, 1100000000,
                         1200000000, 1300000000, 1400000000, 1500000000};
  int years[] = {1970, 2000, 2001, 2004, 2008, 2011, 2014, 2017};

  for (int i = 0; i < GMTIME_NUM_THREADS; i++) {
    args[i].thread_id = i;
    args[i].timestamp = timestamps[i];
    args[i].expected_year = years[i];
    args[i].pass = 0;
    pthread_create(&threads[i], NULL, gmtime_thread_func, &args[i]);
  }

  for (int i = 0; i < GMTIME_NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  for (int i = 0; i < GMTIME_NUM_THREADS; i++) {
    char msg[512];
    snprintf(msg, sizeof(msg), "thread %d (year=%d) failed: %s", i,
             args[i].expected_year, args[i].failure_msg);
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
     "Verify NCCL source uses strtok_r/gmtime_r (FAILS before fix)"},
    {"strtok-cases", test_strtok_r_cases,
     "strtok_r tokenization (positive/negative/security)"},
    {"strtok-long", test_strtok_r_very_long_token,
     "strtok_r with 4096-byte token"},
    {"strtok-threads", test_strtok_r_thread_safety,
     "strtok_r concurrent thread safety"},
    {"gmtime-cases", test_gmtime_r_cases,
     "gmtime_r known timestamps and boundaries"},
    {"gmtime-threads", test_gmtime_r_thread_safety,
     "gmtime_r concurrent thread safety"},
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

  printf("strtok_r / gmtime_r Thread Safety Tests\n");
  printf("========================================\n");

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

  printf("\n========================================\n");
  printf("Results: %d/%d tests passed\n", passed, total);

  if (passed == total) {
    printf("All tests PASSED!\n");
    return 0;
  } else {
    printf("Some tests FAILED!\n");
    return 1;
  }
}
