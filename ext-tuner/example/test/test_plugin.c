/*************************************************************************
 * Unit tests for NCCL Tuner Plugin
 ************************************************************************/

#define _GNU_SOURCE  // Enable setenv/unsetenv and other GNU extensions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdarg.h>


// Include NCCL tuner header (which includes common.h and err.h)
#include "tuner.h"

// Include plugin source for testing
#include "../plugin.c"

// Test framework macros
#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      printf("FAIL: %s - %s\n", __func__, message); \
      return 0; \
    } \
  } while(0)

#define TEST_PASS() \
  do { \
    printf("PASS: %s\n", __func__); \
    return 1; \
  } while(0)

// Global test state
static int test_log_count = 0;

// Mock logger function
void mock_logger(ncclDebugLogLevel level, unsigned long flags,
                 const char* file, int line, const char* fmt, ...) {
  (void)flags; // Suppress unused parameter warning
  test_log_count++;

  // Check if we should print based on NCCL_DEBUG level
  const char* debug_level = getenv("NCCL_DEBUG");
  int should_print = 0;

  if (debug_level) {
    if (strcmp(debug_level, "TRACE") == 0) {
      should_print = 1; // Print everything
    } else if (strcmp(debug_level, "INFO") == 0 && level <= NCCL_LOG_INFO) {
      should_print = 1; // Print INFO and below
    } else if (strcmp(debug_level, "WARN") == 0 && level <= NCCL_LOG_WARN) {
      should_print = 1; // Print WARN and below
    }
  }

  if (!should_print) return;

  // Convert log level to string
  const char* level_str;
  switch(level) {
    case NCCL_LOG_NONE: level_str = "NONE"; break;
    case NCCL_LOG_VERSION: level_str = "VERSION"; break;
    case NCCL_LOG_WARN: level_str = "WARN"; break;
    case NCCL_LOG_INFO: level_str = "INFO"; break;
    case NCCL_LOG_ABORT: level_str = "ABORT"; break;
    case NCCL_LOG_TRACE: level_str = "TRACE"; break;
    default: level_str = "UNKNOWN"; break;
  }

  // Print log header
  printf("[TUNER:%s:%s:%d] ", level_str, file, line);

  // Print formatted message
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\n");
}

// Helper function to create test config file
void create_test_config(const char* filename, const char* content) {
  FILE* f = fopen(filename, "w");
  if (f) {
    fprintf(f, "%s", content);
    fclose(f);
  }
}

// Test 1: Plugin initialization
int test_plugin_init() {
  void* context = NULL;

  // Test successful initialization
  ncclResult_t result = pluginInit(&context, 0, 8, 2, mock_logger, NULL, NULL);
  TEST_ASSERT(result == ncclSuccess, "Plugin init should succeed");
  TEST_ASSERT(context != NULL, "Context should be allocated");

  // Clean up
  pluginFinalize(context);
  TEST_PASS();
}

// Test 2: Configuration file parsing - valid CSV
int test_config_parsing_valid() {
  const char* test_config =
    "# Test configuration\n"
    "allreduce,0,65536,tree,simple,2,1,-1,-1,-1\n"
    "broadcast,0,32768,ring,ll128,4,2,16,-1,-1\n"
    "# Comment line\n"
    "\n"  // Empty line
    "reduce,1024,2048,tree,simple,-1,-1,-1,-1,-1\n";

  create_test_config("test_valid.conf", test_config);

  // Set environment variable to use our test config
  setenv("NCCL_TUNER_CONFIG_FILE", "test_valid.conf", 1);

  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 16, 2, mock_logger, NULL, NULL);
  TEST_ASSERT(result == ncclSuccess, "Plugin init with valid config should succeed");

  // Clean up
  pluginFinalize(context);
  unlink("test_valid.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 3: Configuration file parsing - invalid CSV
int test_config_parsing_invalid() {
  const char* test_config =
    "allreduce,0,65536,tree,simple,2,1  # Missing nRanks and other fields\n"
    "invalid_collective,0,1024,ring,simple,1,1,1,-1,-1\n"
    "broadcast,abc,def,ring,simple,1,1,1,-1,-1\n";  // Invalid numbers

  create_test_config("test_invalid.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_invalid.conf", 1);

  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);
  // Should still succeed but with no valid configs loaded
  TEST_ASSERT(result == ncclSuccess, "Plugin init should succeed even with invalid config");

  // Clean up
  pluginFinalize(context);
  unlink("test_invalid.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 4: Collective type matching
int test_collective_matching() {
  const char* test_config =
    "allreduce,0,65536,tree,simple,8,1,-1,-1,-1\n"
    "broadcast,0,32768,ring,ll128,4,-1,-1,-1,-1\n";

  create_test_config("test_match.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_match.conf", 1);

  void* context = NULL;
  pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);

  // Create mock cost table
  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0; // Default high cost
    }
  }

  int nChannels;

  // Test allreduce matching (should match first config)
  ncclResult_t result = pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                                          cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                                          0, &nChannels);

  TEST_ASSERT(result == ncclSuccess, "GetCollInfo should succeed");
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "DEBUG: Checking cost_table[TREE][SIMPLE] (%p) = %.1f (expecting 0.0)",
              &cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE], cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  TEST_ASSERT(cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] == 0.0, "Tree/Simple should have low cost");
  TEST_ASSERT(nChannels == 8, "Should set 8 channels");

  // Test broadcast matching (should match second config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0; // Reset costs
    }
  }

  result = pluginGetCollInfo(context, ncclFuncBroadcast, 16384, 1,
                            cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                            0, &nChannels);
  TEST_ASSERT(result == ncclSuccess, "GetCollInfo should succeed");
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "DEBUG: Checking cost_table[RING][LL128] (%p) = %.1f (expecting 0.0)",
              &cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128], cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128]);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128] == 0.0, "Ring/LL128 should have low cost");
  TEST_ASSERT(nChannels == 4, "Should set 4 channels");

  // Clean up
  pluginFinalize(context);
  unlink("test_match.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 5: Size range matching
int test_size_matching() {
  const char* test_config =
    "allreduce,0,1024,tree,simple,2,-1,-1,-1,-1\n"
    "allreduce,1025,65536,ring,simple,4,-1,-1,-1,-1\n"
    "allreduce,65537,4294967295,ring,ll128,8,-1,-1,-1,-1\n";

  create_test_config("test_size.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_size.conf", 1);

  void* context = NULL;
  pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);

  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }
  int nChannels = 1;

  pluginGetCollInfo(context, ncclFuncAllReduce, 512, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "DEBUG: Small message - checking cost_table[TREE][SIMPLE] (%p) = %.1f (expecting 0.0)",
              &cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE], cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  TEST_ASSERT(cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] == 0.0, "Small: Tree/Simple should have low cost");
  TEST_ASSERT(nChannels == 2, "Small: Should set 2 channels");

  // Test medium message (should match second config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "DEBUG: Medium message - checking cost_table[RING][SIMPLE] (%p) = %.1f (expecting 0.0)",
              &cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE], cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] == 0.0, "Medium: Ring/Simple should have low cost");
  TEST_ASSERT(nChannels == 4, "Medium: Should set 4 channels");

  // Test large message (should match third config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 1048576, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "DEBUG: Large message - checking cost_table[RING][LL128] (%p) = %.1f (expecting 0.0)",
              &cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128], cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128]);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128] == 0.0, "Large: Ring/LL128 should have low cost");
  TEST_ASSERT(nChannels == 8, "Large: Should set 8 channels");

  // Clean up
  pluginFinalize(context);
  unlink("test_size.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 6: Topology matching
int test_topology_matching() {
  const char* test_config =
    "allreduce,0,65536,tree,simple,2,1,-1,-1,-1\n"      // Single node only
    "allreduce,0,65536,ring,simple,4,4,32,-1,-1\n"      // 4 nodes, 32 ranks exactly
    "allreduce,0,65536,ring,ll128,8,-1,-1,-1,-1\n";     // Any topology

  create_test_config("test_topo.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_topo.conf", 1);

  // Test with single node setup
  void* context1 = NULL;
  pluginInit(&context1, 0, 8, 1, mock_logger, NULL, NULL);  // 8 ranks, 1 node

  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  int nChannels;
  pluginGetCollInfo(context1, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  TEST_ASSERT(cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] == 0.0, "Single node: Should match tree config");
  TEST_ASSERT(nChannels == 2, "Single node: Should set 2 channels");

  pluginFinalize(context1);

  // Test with 4 nodes, 32 ranks setup
  void* context2 = NULL;
  pluginInit(&context2, 0, 32, 4, mock_logger, NULL, NULL);  // 32 ranks, 4 nodes

  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context2, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] == 0.0, "4-node: Should match ring/simple config");
  TEST_ASSERT(nChannels == 4, "4-node: Should set 4 channels");

  // Clean up
  unlink("test_topo.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 7: Default channels behavior (-1)
int test_default_channels() {
  const char* test_config =
    "allreduce,0,65536,tree,simple,-1,-1,-1,-1,-1\n";  // Use default channels

  create_test_config("test_default.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_default.conf", 1);

  void* context = NULL;
  pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);

  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  int nChannels = 99;  // Set to known value
  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);

  TEST_ASSERT(cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] == 0.0, "Should apply algorithm/protocol");
  TEST_ASSERT(nChannels == 1, "Should keep default channels (1) when config has -1");

  // Clean up
  pluginFinalize(context);
  unlink("test_default.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 8: regBuff matching
int test_regbuff_matching() {
  const char* test_config =
    "allreduce,0,65536,tree,simple,2,-1,-1,-1,1\n"      // Registered buffers only
    "allreduce,0,65536,ring,simple,4,-1,-1,-1,0\n"      // Non-registered buffers only
    "allreduce,0,65536,ring,ll128,8,-1,-1,-1,-1\n";     // Any buffer type (backward compatible)

  create_test_config("test_regbuff.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_regbuff.conf", 1);

  void* context = NULL;
  pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);

  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
  }

  int nChannels;

  // Test registered buffer (should match first config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    1, &nChannels);  // regBuff = 1 (registered)
  TEST_ASSERT(cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] == 0.0, "Registered buffer: Tree/Simple should have low cost");
  TEST_ASSERT(nChannels == 2, "Registered buffer: Should set 2 channels");

  // Test non-registered buffer (should match second config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);  // regBuff = 0 (non-registered)
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] == 0.0, "Non-registered buffer: Ring/Simple should have low cost");
  TEST_ASSERT(nChannels == 4, "Non-registered buffer: Should set 4 channels");

  // Test backward compatibility - config without regBuff should match any regBuff value
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  // First try with regBuff=2 (unusual value, should match third config)
  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    2, &nChannels);  // regBuff = 2 (only third config should match)
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128] == 0.0, "Any regBuff: Ring/LL128 should have low cost");
  TEST_ASSERT(nChannels == 8, "Any regBuff: Should set 8 channels");

  // Clean up
  pluginFinalize(context);
  unlink("test_regbuff.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 9: numPipeOps matching
int test_pipeops_matching() {
  const char* test_config =
    "allreduce,0,65536,tree,simple,2,-1,-1,1,-1\n"      // Single pipeline op
    "allreduce,0,65536,ring,simple,4,-1,-1,4,-1\n"      // Multiple pipeline ops
    "allreduce,0,65536,ring,ll128,8,-1,-1,-1,-1\n";     // Any pipeline ops (backward compatible)

  create_test_config("test_pipeops.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_pipeops.conf", 1);

  void* context = NULL;
  pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);

  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
  }

  int nChannels;

  // Test single pipeline op (should match first config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  TEST_ASSERT(cost_table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] == 0.0, "Single pipeOp: Tree/Simple should have low cost");
  TEST_ASSERT(nChannels == 2, "Single pipeOp: Should set 2 channels");

  // Test multiple pipeline ops (should match second config)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 4,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] == 0.0, "Multiple pipeOps: Ring/Simple should have low cost");
  TEST_ASSERT(nChannels == 4, "Multiple pipeOps: Should set 4 channels");

  // Test different number of pipeline ops (should match third config - backward compatible)
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 2,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_LL128] == 0.0, "Any pipeOps: Ring/LL128 should have low cost");
  TEST_ASSERT(nChannels == 8, "Any pipeOps: Should set 8 channels");

  // Clean up
  pluginFinalize(context);
  unlink("test_pipeops.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 10: No matching configuration (fallback behavior)
int test_no_match_fallback() {
  const char* test_config =
    "broadcast,0,1024,tree,simple,2,-1,-1,-1,-1\n";  // Only broadcast config

  create_test_config("test_fallback.conf", test_config);
  setenv("NCCL_TUNER_CONFIG_FILE", "test_fallback.conf", 1);

  void* context = NULL;
  pluginInit(&context, 0, 8, 1, mock_logger, NULL, NULL);

  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  int nChannels;
  // Try allreduce (should not match, use fallback)
  pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                    cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                    0, &nChannels);

  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "DEBUG: Fallback test - checking cost_table[RING][SIMPLE] (%p) = %.1f (expecting 0.0)",
              &cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE], cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  TEST_ASSERT(cost_table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] == 1.0, "Should use pass through unmodified");
  TEST_ASSERT(nChannels == 1, "Should use default channels");

  // Clean up
  pluginFinalize(context);
  unlink("test_fallback.conf");
  unsetenv("NCCL_TUNER_CONFIG_FILE");
  TEST_PASS();
}

// Test 11: Large configuration files (testing dynamic allocation)
int test_large_config() {
  const char* large_config_file = "test_large.conf";

  // Create a large configuration file with many entries
  // This tests the dynamic allocation functionality
  FILE* f = fopen(large_config_file, "w");
  TEST_ASSERT(f != NULL, "Should be able to create large config file");

  // Write header comment
  fprintf(f, "# Large configuration file for testing dynamic allocation\n");
  fprintf(f, "# This file contains many configurations to test memory allocation\n");

  // Generate a large number of configurations (much more than the old MAX_CONFIGS=100)
  const int num_configs = 500; // 5x the old static limit
  const char* collectives[] = {"allreduce", "broadcast", "reduce", "allgather", "reducescatter"};
  const char* algorithms[] = {"tree", "ring", "collnet_direct", "nvls"};
  const char* protocols[] = {"simple", "ll", "ll128"};

  for (int i = 0; i < num_configs; i++) {
    // Vary the configurations to create realistic test data
    const char* coll = collectives[i % 5];
    const char* algo = algorithms[i % 4];
    const char* proto = protocols[i % 3];

    size_t min_bytes = (i * 1024) % 1048576; // Vary from 0 to 1MB
    size_t max_bytes = min_bytes + 65536;    // 64KB range
    int channels = (i % 8) + 1;              // 1-8 channels
    int nodes = (i % 4) == 0 ? -1 : (i % 4); // Mix of -1 and 1-3 nodes
    int ranks = (i % 8) == 0 ? -1 : (i % 32) + 1; // Mix of -1 and 1-32 ranks
    int pipeOps = (i % 3) == 0 ? -1 : (i % 4) + 1; // Mix of -1 and 1-4 pipeOps
    int regBuff = (i % 3) == 0 ? -1 : (i % 2); // Mix of -1, 0, 1

    fprintf(f, "%s,%zu,%zu,%s,%s,%d,%d,%d,%d,%d\n",
            coll, min_bytes, max_bytes, algo, proto, channels, nodes, ranks, pipeOps, regBuff);
  }

  fclose(f);

  // Set environment to use our large config file
  setenv("NCCL_TUNER_CONFIG_FILE", large_config_file, 1);

  // Initialize plugin with large config
  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 16, 4, mock_logger, NULL, NULL);
  TEST_ASSERT(result == ncclSuccess, "Plugin init with large config should succeed");
  TEST_ASSERT(context != NULL, "Context should be allocated");

  // Verify that configurations were loaded
  TunerContext* ctx = (TunerContext*)context;
  TEST_ASSERT(ctx->numConfigs == num_configs, "Should load all configurations from large file");
  TEST_ASSERT(ctx->maxConfigs == num_configs, "maxConfigs should match allocated size");
  TEST_ASSERT(ctx->configs != NULL, "Configs array should be dynamically allocated");

  // Test that we can access configurations throughout the array
  // (This would have failed with the old static MAX_CONFIGS=100 limit)
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];
    // Basic sanity checks on the loaded configurations
    TEST_ASSERT(config->collType >= ncclFuncBroadcast && config->collType <= ncclFuncAllReduce,
                "Collective type should be valid");
    TEST_ASSERT(config->maxBytes >= config->minBytes, "maxBytes should be >= minBytes");
    TEST_ASSERT(config->nChannels > 0, "nChannels should be positive");
  }

  // Test specific configuration access at various indices
  // Index 0 (first config)
  TuningConfig* first_config = &ctx->configs[0];
  TEST_ASSERT(first_config != NULL, "First config should be accessible");

  // Index in middle
  TuningConfig* mid_config = &ctx->configs[num_configs / 2];
  TEST_ASSERT(mid_config != NULL, "Middle config should be accessible");

  // Index near end (this would have crashed with static array of 100)
  TuningConfig* late_config = &ctx->configs[num_configs - 1];
  TEST_ASSERT(late_config != NULL, "Last config should be accessible");

  // Test memory allocation size - verify we didn't over-allocate
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "Successfully loaded %d configurations (dynamic allocation)", ctx->numConfigs);
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "Memory allocated for %d configurations (%zu bytes total)",
              ctx->maxConfigs, ctx->maxConfigs * sizeof(TuningConfig));

  // Test that the plugin can still find matching configurations from the large set
  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0; // Default high cost
    }
  }

  int nChannels;
  // Try to find a matching configuration - should work with large config set
  result = pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                            cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                            0, &nChannels);
  TEST_ASSERT(result == ncclSuccess, "GetCollInfo should work with large config set");

  // Clean up
  pluginFinalize(context);
  unlink(large_config_file);
  unsetenv("NCCL_TUNER_CONFIG_FILE");

  TEST_PASS();
}

// Test 12: Very large configuration stress test
int test_very_large_config_stress() {
  const char* stress_config_file = "test_stress.conf";

  // Create an even larger configuration file to stress test the implementation
  FILE* f = fopen(stress_config_file, "w");
  TEST_ASSERT(f != NULL, "Should be able to create stress test config file");

  fprintf(f, "# Stress test configuration with very large number of entries\n");

  // Generate an extremely large number of configurations
  const int stress_configs = 2000; // 20x the old static limit

  for (int i = 0; i < stress_configs; i++) {
    // Create varied but valid configurations
    fprintf(f, "allreduce,%d,%d,ring,simple,4,-1,-1,-1,-1\n",
            i * 512, (i * 512) + 1024);
  }

  fclose(f);

  setenv("NCCL_TUNER_CONFIG_FILE", stress_config_file, 1);

  // Test initialization with stress config
  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 8, 2, mock_logger, NULL, NULL);
  TEST_ASSERT(result == ncclSuccess, "Plugin should handle very large config files");

  TunerContext* ctx = (TunerContext*)context;
  TEST_ASSERT(ctx->numConfigs == stress_configs, "Should load all stress test configurations");
  TEST_ASSERT(ctx->configs != NULL, "Stress test configs should be allocated");

  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "Stress test - loaded %d configurations successfully", stress_configs);
  mock_logger(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__,
              "Memory usage: %zu bytes for configuration array",
              stress_configs * sizeof(TuningConfig));

  // Verify we can access configurations throughout the entire range
  for (int i = 0; i < stress_configs; i += 100) { // Sample every 100th config
    TuningConfig* config = &ctx->configs[i];
    TEST_ASSERT(config->collType == ncclFuncAllReduce, "Config should have correct collective type");
    TEST_ASSERT(config->minBytes == (size_t)(i * 512), "Config should have correct minBytes");
  }

  // Clean up
  pluginFinalize(context);
  unlink(stress_config_file);
  unsetenv("NCCL_TUNER_CONFIG_FILE");

  TEST_PASS();
}

// Test 13: Edge case - empty config file
int test_empty_config() {
  const char* empty_config_file = "test_empty.conf";

  // Create empty config file (only comments)
  create_test_config(empty_config_file,
    "# Empty configuration file\n"
    "# No actual configurations\n"
    "\n"
    "\n");

  setenv("NCCL_TUNER_CONFIG_FILE", empty_config_file, 1);

  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 8, 2, mock_logger, NULL, NULL);
  TEST_ASSERT(result == ncclSuccess, "Plugin should handle empty config files");

  TunerContext* ctx = (TunerContext*)context;
  TEST_ASSERT(ctx->numConfigs == 0, "Should have zero configurations");
  TEST_ASSERT(ctx->maxConfigs == 0, "Should have zero max configurations");
  TEST_ASSERT(ctx->configs == NULL, "Should not allocate memory for empty config");

  // Test that plugin still works with no configurations (fallback behavior)
  float cost_table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float* cost_table_ptr[NCCL_NUM_ALGORITHMS];
  for (int i = 0; i < NCCL_NUM_ALGORITHMS; i++) {
    cost_table_ptr[i] = cost_table[i];
    for (int j = 0; j < NCCL_NUM_PROTOCOLS; j++) {
      cost_table[i][j] = 1.0;
    }
  }

  int nChannels;
  result = pluginGetCollInfo(context, ncclFuncAllReduce, 32768, 1,
                            cost_table_ptr, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                            0, &nChannels);
  TEST_ASSERT(result == ncclSuccess, "GetCollInfo should work with empty config");

  // Clean up
  pluginFinalize(context);
  unlink(empty_config_file);
  unsetenv("NCCL_TUNER_CONFIG_FILE");

  TEST_PASS();
}

// Test NVLink domain info handling
int test_nvl_domain_info() {
  printf("Testing NVLink domain info handling...\n");

  // Test NVLink domain structure with min/max ranks per domain
  ncclNvlDomainInfo_v5_t nvl_domain = {
    .nNvlDomains = 2, // 2 nodes = 2 domains
    .minRanksPerNvlDomain = 3, // minimum ranks across all domains (bottleneck)
    .maxRanksPerNvlDomain = 5  // maximum ranks across all domains (capacity)
  };

  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 8, 2, mock_logger, &nvl_domain, NULL);
  TEST_ASSERT(result == ncclSuccess, "Plugin init with NVLink domains should succeed");

  // Validate NVLD info structure
  TEST_ASSERT(nvl_domain.nNvlDomains == 2, "Should have 2 domains (nodes)");
  TEST_ASSERT(nvl_domain.minRanksPerNvlDomain == 3, "Should have minimum 3 ranks per domain");
  TEST_ASSERT(nvl_domain.maxRanksPerNvlDomain == 5, "Should have maximum 5 ranks per domain");

  // Clean up
  pluginFinalize(context);
  printf("NVLink domain info test passed!\n");
  TEST_PASS();
}

int test_tuner_constants() {
  // Initialize constants to -1.0 for testing purposes
  ncclTunerConstants_v5_t constants = {
    // Base latencies: [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS]
    .baseLatencies = {
      {-1.0, -1.0, -1.0},    // NCCL_ALGO_TREE: LL, LL128, Simple
      {-1.0, -1.0, -1.0},    // NCCL_ALGO_RING: LL, LL128, Simple
      {-1.0, -1.0, -1.0},   // NCCL_ALGO_COLLNET_DIRECT
      {-1.0, -1.0, -1.0},   // NCCL_ALGO_COLLNET_CHAIN
      {-1.0, -1.0, -1.0},    // NCCL_ALGO_NVLS
      {-1.0, -1.0, -1.0},    // NCCL_ALGO_NVLS_TREE
      {-1.0, -1.0, -1.0}     // NCCL_ALGO_PAT
    },

    // Hardware latencies: [NCCL_NUM_HW_LINKS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS]
    .hwLatencies = {
      // NCCL_HW_NVLINK
      {
        {-1.0, -1.0, -1.0},    // TREE
        {-1.0, -1.0, -1.0},    // RING
        {-1.0, -1.0, -1.0},    // COLLNET_DIRECT
        {-1.0, -1.0, -1.0},    // COLLNET_CHAIN
        {-1.0, -1.0, -1.0},    // NVLS
        {-1.0, -1.0, -1.0},    // NVLS_TREE
        {-1.0, -1.0, -1.0}     // PAT
      },
      // NCCL_HW_PCI
      {
        {-1.0, -1.0, -1.0},   // TREE
        {-1.0, -1.0, -1.0},    // RING
        {-1.0, -1.0, -1.0},  // COLLNET_DIRECT
        {-1.0, -1.0, -1.0},  // COLLNET_CHAIN
        {-1.0, -1.0, -1.0},     // NVLS
        {-1.0, -1.0, -1.0},   // NVLS_TREE
        {-1.0, -1.0, -1.0}   // PAT
      },
      // NCCL_HW_NET
      {
        {-1.0, -1.0, -1.0},  // TREE
        {-1.0, -1.0, -1.0},  // RING
        {-1.0, -1.0, -1.0},  // COLLNET_DIRECT
        {-1.0, -1.0, -1.0},  // COLLNET_CHAIN
        {-1.0, -1.0, -1.0},  // NVLS
        {-1.0, -1.0, -1.0},  // NVLS_TREE
        {-1.0, -1.0, -1.0}   // PAT
      }
    },

    // LL maximum bandwidths: [NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES]
    .llMaxBws = {
      {-1.0, -1.0, -1.0},  // Volta: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Ampere: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Hopper: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0}   // Blackwell: 1node, 2nodes, 4nodes
    },

    // Per-channel maximum Ring LL128 bandwidths: [NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES]
    .perChMaxRingLL128Bws = {
      {-1.0, -1.0, -1.0},   // Volta: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Ampere: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Hopper: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0}   // Blackwell: 1node, 2nodes, 4nodes
    },

    // Per-channel maximum Tree LL128 bandwidths: [NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES]
    .perChMaxTreeLL128Bws = {
      {-1.0, -1.0, -1.0},    // Volta: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},   // Ampere: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Hopper: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0}   // Blackwell: 1node, 2nodes, 4nodes
    },

    // Per-channel maximum Tree bandwidths: [NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES]
    .perChMaxTreeBws = {
      {-1.0, -1.0, -1.0},  // Volta: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Ampere: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0},  // Hopper: 1node, 2nodes, 4nodes
      {-1.0, -1.0, -1.0}   // Blackwell: 1node, 2nodes, 4nodes
    }
  };

  void* context = NULL;
  ncclResult_t result = pluginInit(&context, 0, 8, 2, mock_logger, NULL, &constants);
  TEST_ASSERT(result == ncclSuccess, "Plugin init with constants should succeed");

  // Test that the constants were set correctly
  TEST_ASSERT(constants.perChMaxTreeBws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] == 15.0, "Tree bandwidth should be 15GB/s");
  TEST_ASSERT(constants.perChMaxRingLL128Bws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] == 20.0, "Ring bandwidth should be 20GB/s");
  TEST_ASSERT(constants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] == 24.0, "NVLSTree base network latency should be 24us");

  // Clean up
  pluginFinalize(context);
  TEST_PASS();
}

// Test runner function pointer type
typedef int (*TestFunction)(void);

// Test registry
typedef struct {
  const char* name;
  TestFunction func;
  const char* description;
} TestCase;

// All available tests
TestCase test_cases[] = {
  {"init", test_plugin_init, "Plugin initialization"},
  {"config-valid", test_config_parsing_valid, "Valid configuration parsing"},
  {"config-invalid", test_config_parsing_invalid, "Invalid configuration parsing"},
  {"collective", test_collective_matching, "Collective type matching"},
  {"size", test_size_matching, "Size range matching"},
  {"topology", test_topology_matching, "Topology matching"},
  {"channels", test_default_channels, "Default channels behavior"},
  {"regbuff", test_regbuff_matching, "Registered buffer matching"},
  {"pipeops", test_pipeops_matching, "Pipeline operations matching"},
  {"fallback", test_no_match_fallback, "Fallback behavior"},
  {"large-config", test_large_config, "Large configuration files (dynamic allocation)"},
  {"stress-config", test_very_large_config_stress, "Very large configuration stress test"},
  {"empty-config", test_empty_config, "Empty configuration file handling"},
  {"nvl-domain", test_nvl_domain_info, "NVL domain info handling"},
  {"constants", test_tuner_constants, "Tuner constants initialization"},
  {NULL, NULL, NULL} // End marker
};

// Show help/usage information
void show_help(const char* program_name) {
  printf("Usage: %s [test_name ...]\n\n", program_name);
  printf("Available tests:\n");
  for (int i = 0; test_cases[i].name != NULL; i++) {
    printf("  %-15s - %s\n", test_cases[i].name, test_cases[i].description);
  }
  printf("\nExamples:\n");
  printf("  %s                    # Run all tests\n", program_name);
  printf("  %s init               # Run only initialization test\n", program_name);
  printf("  %s init collective    # Run initialization and collective tests\n", program_name);
  printf("  %s --help             # Show this help\n", program_name);
}

// Find test by name
TestFunction find_test(const char* name) {
  for (int i = 0; test_cases[i].name != NULL; i++) {
    if (strcmp(test_cases[i].name, name) == 0) {
      return test_cases[i].func;
    }
  }
  return NULL;
}

// Main test runner
int main(int argc, char* argv[]) {
  int passed = 0, total = 0;

  // Check for help
  if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    show_help(argv[0]);
    return 0;
  }

  printf("Running NCCL Tuner Plugin Unit Tests\n");
  printf("=====================================\n");

  if (argc == 1) {
    // No arguments - run all tests
    for (int i = 0; test_cases[i].name != NULL; i++) {
      printf("Running test: %s\n", test_cases[i].name);
      total++;
      passed += test_cases[i].func();
    }
  } else {
    // Run specific tests
    for (int arg = 1; arg < argc; arg++) {
      TestFunction test_func = find_test(argv[arg]);
      if (test_func) {
        total++;
        passed += test_func();
      } else {
        printf("ERROR: Unknown test '%s'\n", argv[arg]);
        printf("Use --help to see available tests\n");
        return 1;
      }
    }
  }

  printf("\n=====================================\n");
  printf("Test Results: %d/%d tests passed\n", passed, total);

  if (passed == total) {
    printf("All tests PASSED!\n");
    return 0;
  } else {
    printf("Some tests FAILED!\n");
    return 1;
  }
}
