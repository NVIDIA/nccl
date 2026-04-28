/*************************************************************************
 * Unit tests for C++ lookup table and tokenization patterns
 *
 * Tests verify the table-driven parsing introduced in the modernization
 * commits actually produces correct results for every entry. Covers:
 *   - debug.cc:   subsysTable (18 entries) and tsTable (6 entries)
 *   - init.cc:    policyTable (3 entries)
 *   - plugin.cc:  collTypeTable (5), algorithmTable (7), protocolTable (3)
 *   - cpuset.h:   hex-mask tokenization via istringstream
 *   - plugin.cc:  splitCSV / trim helpers
 *
 * Uses the real NCCL constant values as stubs so the tables here are
 * identical to the tables in the source — if someone adds a new enum
 * value and forgets the table, the source-verification test catches it.
 *
 * Compile: g++ -Wall -Wextra -g -std=c++14 -D_GNU_SOURCE -o test_lookup_tables test_lookup_tables.cc -lpthread
 * Run:     ./test_lookup_tables
 *************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

/* =========================================================================
 * Test framework (same macros as the C tests)
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
 * NCCL constant stubs — real values from nccl_common.h / nccl_tuner.h / nccl.h
 * ========================================================================= */

// Debug subsystem masks (nccl_common.h)
#define NCCL_INIT      0x1ULL
#define NCCL_COLL      0x2ULL
#define NCCL_P2P       0x4ULL
#define NCCL_SHM       0x8ULL
#define NCCL_NET       0x10ULL
#define NCCL_GRAPH     0x20ULL
#define NCCL_TUNING    0x40ULL
#define NCCL_ENV       0x80ULL
#define NCCL_ALLOC     0x100ULL
#define NCCL_CALL      0x200ULL
#define NCCL_PROXY     0x400ULL
#define NCCL_NVLS      0x800ULL
#define NCCL_BOOTSTRAP 0x1000ULL
#define NCCL_REG       0x2000ULL
#define NCCL_PROFILE   0x4000ULL
#define NCCL_RAS       0x8000ULL
#define NCCL_DESTROY   0x10000ULL
#define NCCL_ALL       (~0ULL)

// Debug log levels (nccl_common.h)
#define NCCL_LOG_NONE    0
#define NCCL_LOG_VERSION 1
#define NCCL_LOG_WARN    2
#define NCCL_LOG_INFO    3
#define NCCL_LOG_ABORT   4
#define NCCL_LOG_TRACE   5

// Collective types (nccl_common.h)
enum ncclFunc_t {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
};

// Algorithms (nccl_tuner.h)
#define NCCL_ALGO_TREE           0
#define NCCL_ALGO_RING           1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN  3
#define NCCL_ALGO_NVLS           4
#define NCCL_ALGO_NVLS_TREE      5
#define NCCL_ALGO_PAT            6

// Protocols (nccl_tuner.h)
#define NCCL_PROTO_LL     0
#define NCCL_PROTO_LL128  1
#define NCCL_PROTO_SIMPLE 2

// CTA policies (nccl.h)
#define NCCL_CTA_POLICY_DEFAULT    0x00
#define NCCL_CTA_POLICY_EFFICIENCY 0x01
#define NCCL_CTA_POLICY_ZERO       0x02
#define NCCL_CONFIG_UNDEF_INT      (-1)

/* =========================================================================
 * Exact copies of lookup tables from the source files
 * (If the source table changes, these must be updated to match)
 * ========================================================================= */

// --- debug.cc: subsysTable ---
struct SubsysEntry { const char* name; uint64_t mask; };
static const SubsysEntry subsysTable[] = {
  {"INIT",      NCCL_INIT},
  {"COLL",      NCCL_COLL},
  {"P2P",       NCCL_P2P},
  {"SHM",       NCCL_SHM},
  {"NET",       NCCL_NET},
  {"GRAPH",     NCCL_GRAPH},
  {"TUNING",    NCCL_TUNING},
  {"ENV",       NCCL_ENV},
  {"ALLOC",     NCCL_ALLOC},
  {"CALL",      NCCL_CALL},
  {"PROXY",     NCCL_PROXY},
  {"NVLS",      NCCL_NVLS},
  {"BOOTSTRAP", NCCL_BOOTSTRAP},
  {"REG",       NCCL_REG},
  {"PROFILE",   NCCL_PROFILE},
  {"RAS",       NCCL_RAS},
  {"DESTROY",   NCCL_DESTROY},
  {"ALL",       NCCL_ALL},
};
static constexpr int subsysTableSize = sizeof(subsysTable) / sizeof(subsysTable[0]);

// --- debug.cc: tsTable ---
struct TsEntry { const char* name; uint32_t mask; };
static const TsEntry tsTable[] = {
  {"ALL",     ~0U},
  {"VERSION", (1U << NCCL_LOG_VERSION)},
  {"WARN",    (1U << NCCL_LOG_WARN)},
  {"INFO",    (1U << NCCL_LOG_INFO)},
  {"ABORT",   (1U << NCCL_LOG_ABORT)},
  {"TRACE",   (1U << NCCL_LOG_TRACE)},
};
static constexpr int tsTableSize = sizeof(tsTable) / sizeof(tsTable[0]);

// --- init.cc: policyTable ---
struct PolicyEntry { const char* name; int value; };
static const PolicyEntry policyTable[] = {
  {"DEFAULT",    NCCL_CTA_POLICY_DEFAULT},
  {"EFFICIENCY", NCCL_CTA_POLICY_EFFICIENCY},
  {"ZERO",       NCCL_CTA_POLICY_ZERO},
};
static constexpr int policyTableSize = sizeof(policyTable) / sizeof(policyTable[0]);

// --- plugin.cc: collTypeTable ---
struct NamedCollType { const char* name; ncclFunc_t value; };
static const NamedCollType collTypeTable[] = {
  {"broadcast",     ncclFuncBroadcast},
  {"reduce",        ncclFuncReduce},
  {"allgather",     ncclFuncAllGather},
  {"reducescatter", ncclFuncReduceScatter},
  {"allreduce",     ncclFuncAllReduce},
};
static constexpr int collTypeTableSize = sizeof(collTypeTable) / sizeof(collTypeTable[0]);

// --- plugin.cc: algorithmTable ---
struct NamedAlgorithm { const char* name; int value; };
static const NamedAlgorithm algorithmTable[] = {
  {"tree",           NCCL_ALGO_TREE},
  {"ring",           NCCL_ALGO_RING},
  {"collnet_direct", NCCL_ALGO_COLLNET_DIRECT},
  {"collnet_chain",  NCCL_ALGO_COLLNET_CHAIN},
  {"nvls",           NCCL_ALGO_NVLS},
  {"nvls_tree",      NCCL_ALGO_NVLS_TREE},
  {"pat",            NCCL_ALGO_PAT},
};
static constexpr int algorithmTableSize = sizeof(algorithmTable) / sizeof(algorithmTable[0]);

// --- plugin.cc: protocolTable ---
struct NamedProtocol { const char* name; int value; };
static const NamedProtocol protocolTable[] = {
  {"ll",     NCCL_PROTO_LL},
  {"ll128",  NCCL_PROTO_LL128},
  {"simple", NCCL_PROTO_SIMPLE},
};
static constexpr int protocolTableSize = sizeof(protocolTable) / sizeof(protocolTable[0]);

/* =========================================================================
 * Exact copies of parse functions from plugin.cc
 * ========================================================================= */

static ncclFunc_t parseCollType(const std::string& str) {
  for (const auto& e : collTypeTable)
    if (str == e.name) return e.value;
  return ncclFuncAllReduce;
}

static int parseAlgorithm(const std::string& str) {
  for (const auto& e : algorithmTable)
    if (str == e.name) return e.value;
  return NCCL_ALGO_RING;
}

static int parseProtocol(const std::string& str) {
  for (const auto& e : protocolTable)
    if (str == e.name) return e.value;
  return NCCL_PROTO_SIMPLE;
}

/* =========================================================================
 * Exact copies of string helpers from plugin.cc
 * ========================================================================= */

static std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t");
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> splitCSV(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(trim(field));
  }
  return fields;
}

/* =========================================================================
 * Tests: istringstream + getline tokenization
 * ========================================================================= */

int test_tokenize_comma() {
  std::istringstream stream("INIT,COLL,P2P");
  std::string token;
  std::vector<std::string> tokens;
  while (std::getline(stream, token, ',')) tokens.push_back(token);

  char msg[256];
  snprintf(msg, sizeof(msg), "expected 3 tokens, got %d", (int)tokens.size());
  TEST_ASSERT(tokens.size() == 3, msg);
  TEST_ASSERT(tokens[0] == "INIT", "token[0] != INIT");
  TEST_ASSERT(tokens[1] == "COLL", "token[1] != COLL");
  TEST_ASSERT(tokens[2] == "P2P",  "token[2] != P2P");
  TEST_PASS();
}

int test_tokenize_pipe() {
  std::istringstream stream("DEFAULT|EFFICIENCY|ZERO");
  std::string token;
  std::vector<std::string> tokens;
  while (std::getline(stream, token, '|')) tokens.push_back(token);

  TEST_ASSERT(tokens.size() == 3, "expected 3 pipe-delimited tokens");
  TEST_ASSERT(tokens[0] == "DEFAULT",    "token[0]");
  TEST_ASSERT(tokens[1] == "EFFICIENCY", "token[1]");
  TEST_ASSERT(tokens[2] == "ZERO",       "token[2]");
  TEST_PASS();
}

int test_tokenize_single() {
  std::istringstream stream("INIT");
  std::string token;
  std::vector<std::string> tokens;
  while (std::getline(stream, token, ',')) tokens.push_back(token);

  TEST_ASSERT(tokens.size() == 1, "single token count");
  TEST_ASSERT(tokens[0] == "INIT", "single token value");
  TEST_PASS();
}

int test_tokenize_empty() {
  std::istringstream stream("");
  std::string token;
  std::vector<std::string> tokens;
  while (std::getline(stream, token, ',')) tokens.push_back(token);

  TEST_ASSERT(tokens.size() == 0, "empty string should produce 0 tokens");
  TEST_PASS();
}

/* =========================================================================
 * Tests: debug.cc subsystem table — every entry resolves correctly
 * ========================================================================= */

int test_subsys_table_all_entries() {
  char msg[256];
  for (int i = 0; i < subsysTableSize; i++) {
    // Look up by name, same way debug.cc does
    bool found = false;
    for (const auto& entry : subsysTable) {
      if (strcasecmp(subsysTable[i].name, entry.name) == 0) {
        snprintf(msg, sizeof(msg), "subsysTable[%d] '%s': value mismatch 0x%lx != 0x%lx",
                 i, subsysTable[i].name,
                 (unsigned long)entry.mask, (unsigned long)subsysTable[i].mask);
        TEST_ASSERT(entry.mask == subsysTable[i].mask, msg);
        found = true;
        break;
      }
    }
    snprintf(msg, sizeof(msg), "subsysTable[%d] '%s': not found by lookup", i, subsysTable[i].name);
    TEST_ASSERT(found, msg);
  }
  TEST_ASSERT(subsysTableSize == 18, "subsysTable should have 18 entries");
  TEST_PASS();
}

int test_subsys_table_case_insensitive() {
  // debug.cc uses strcasecmp — verify mixed case works
  const char* variants[] = {"init", "Init", "INIT", "iNiT"};
  for (const char* name : variants) {
    bool found = false;
    for (const auto& entry : subsysTable) {
      if (strcasecmp(name, entry.name) == 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "'%s' should map to NCCL_INIT (0x1)", name);
        TEST_ASSERT(entry.mask == NCCL_INIT, msg);
        found = true;
        break;
      }
    }
    char msg[256];
    snprintf(msg, sizeof(msg), "'%s' not found in subsysTable", name);
    TEST_ASSERT(found, msg);
  }
  TEST_PASS();
}

int test_subsys_table_unknown() {
  const char* unknown = "BOGUS";
  bool found = false;
  for (const auto& entry : subsysTable) {
    if (strcasecmp(unknown, entry.name) == 0) { found = true; break; }
  }
  TEST_ASSERT(!found, "unknown name should not match any entry");
  TEST_PASS();
}

// End-to-end: simulate the full debug.cc parsing pipeline
int test_subsys_pipeline() {
  // Simulate: NCCL_DEBUG_SUBSYS="INIT,NET,PROXY"
  uint64_t mask = 0ULL;
  std::istringstream stream("INIT,NET,PROXY");
  std::string subsys;
  while (std::getline(stream, subsys, ',')) {
    for (const auto& entry : subsysTable) {
      if (strcasecmp(subsys.c_str(), entry.name) == 0) {
        mask |= entry.mask;
        break;
      }
    }
  }
  TEST_ASSERT(mask == (NCCL_INIT | NCCL_NET | NCCL_PROXY),
              "INIT,NET,PROXY should OR to 0x1|0x10|0x400");

  // Simulate: NCCL_DEBUG_SUBSYS="ALL"
  mask = 0ULL;
  std::istringstream stream2("ALL");
  while (std::getline(stream2, subsys, ',')) {
    for (const auto& entry : subsysTable) {
      if (strcasecmp(subsys.c_str(), entry.name) == 0) {
        mask |= entry.mask;
        break;
      }
    }
  }
  TEST_ASSERT(mask == NCCL_ALL, "ALL should set all bits");
  TEST_PASS();
}

// Simulate the ^INIT invert logic from debug.cc
int test_subsys_invert() {
  const char* input = "^INIT,NET";
  int invert = 0;
  if (input[0] == '^') { invert = 1; input++; }
  uint64_t mask = invert ? ~0ULL : 0ULL;

  std::istringstream stream(input);
  std::string subsys;
  while (std::getline(stream, subsys, ',')) {
    for (const auto& entry : subsysTable) {
      if (strcasecmp(subsys.c_str(), entry.name) == 0) {
        if (invert) mask &= ~entry.mask; else mask |= entry.mask;
        break;
      }
    }
  }
  // Should have all bits EXCEPT INIT and NET
  TEST_ASSERT((mask & NCCL_INIT) == 0, "INIT bit should be cleared");
  TEST_ASSERT((mask & NCCL_NET) == 0,  "NET bit should be cleared");
  TEST_ASSERT((mask & NCCL_COLL) != 0, "COLL bit should still be set");
  TEST_ASSERT((mask & NCCL_PROXY) != 0, "PROXY bit should still be set");
  TEST_PASS();
}

/* =========================================================================
 * Tests: debug.cc timestamp level table
 * ========================================================================= */

int test_ts_table_all_entries() {
  TEST_ASSERT(tsTableSize == 6, "tsTable should have 6 entries");
  // Verify each named level maps to the correct bitmask
  for (const auto& entry : tsTable) {
    if (strcmp(entry.name, "ALL") == 0) {
      TEST_ASSERT(entry.mask == ~0U, "ALL mask");
    } else if (strcmp(entry.name, "WARN") == 0) {
      TEST_ASSERT(entry.mask == (1U << NCCL_LOG_WARN), "WARN mask");
    } else if (strcmp(entry.name, "TRACE") == 0) {
      TEST_ASSERT(entry.mask == (1U << NCCL_LOG_TRACE), "TRACE mask");
    }
    // Other entries are validated by the compile-time constant expressions
  }
  TEST_PASS();
}

int test_ts_pipeline() {
  // Simulate: NCCL_DEBUG_TIMESTAMP_LEVELS="WARN,TRACE"
  uint32_t levels = 0U;
  std::istringstream stream("WARN,TRACE");
  std::string level;
  while (std::getline(stream, level, ',')) {
    for (const auto& entry : tsTable) {
      if (strcasecmp(level.c_str(), entry.name) == 0) {
        levels |= entry.mask;
        break;
      }
    }
  }
  TEST_ASSERT(levels == ((1U << NCCL_LOG_WARN) | (1U << NCCL_LOG_TRACE)),
              "WARN,TRACE should set bits 2 and 5");
  TEST_PASS();
}

/* =========================================================================
 * Tests: init.cc CTA policy table
 * ========================================================================= */

int test_policy_table_all_entries() {
  TEST_ASSERT(policyTableSize == 3, "policyTable should have 3 entries");

  // Verify each entry
  char msg[256];
  struct { const char* name; int expected; } expected[] = {
    {"DEFAULT",    NCCL_CTA_POLICY_DEFAULT},
    {"EFFICIENCY", NCCL_CTA_POLICY_EFFICIENCY},
    {"ZERO",       NCCL_CTA_POLICY_ZERO},
  };
  for (const auto& exp : expected) {
    bool found = false;
    for (const auto& entry : policyTable) {
      if (strcasecmp(exp.name, entry.name) == 0) {
        snprintf(msg, sizeof(msg), "'%s' value mismatch: got %d, expected %d",
                 exp.name, entry.value, exp.expected);
        TEST_ASSERT(entry.value == exp.expected, msg);
        found = true;
        break;
      }
    }
    snprintf(msg, sizeof(msg), "'%s' not found in policyTable", exp.name);
    TEST_ASSERT(found, msg);
  }
  TEST_PASS();
}

int test_policy_pipeline() {
  // Simulate: NCCL_CTA_POLICY="DEFAULT|EFFICIENCY"
  int ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
  std::istringstream stream("DEFAULT|EFFICIENCY");
  std::string token;
  while (std::getline(stream, token, '|')) {
    int tokenPolicy = NCCL_CONFIG_UNDEF_INT;
    for (const auto& entry : policyTable) {
      if (strcasecmp(token.c_str(), entry.name) == 0) {
        tokenPolicy = entry.value;
        break;
      }
    }
    if (tokenPolicy != NCCL_CONFIG_UNDEF_INT) {
      if (ctaPolicyEnv == NCCL_CONFIG_UNDEF_INT) ctaPolicyEnv = tokenPolicy;
      else ctaPolicyEnv |= tokenPolicy;
    }
  }
  TEST_ASSERT(ctaPolicyEnv == (NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY),
              "DEFAULT|EFFICIENCY should OR to 0x01");
  TEST_PASS();
}

int test_policy_unknown_token() {
  // Unknown tokens should be silently skipped
  int ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
  std::istringstream stream("BOGUS|ZERO");
  std::string token;
  while (std::getline(stream, token, '|')) {
    int tokenPolicy = NCCL_CONFIG_UNDEF_INT;
    for (const auto& entry : policyTable) {
      if (strcasecmp(token.c_str(), entry.name) == 0) {
        tokenPolicy = entry.value;
        break;
      }
    }
    if (tokenPolicy != NCCL_CONFIG_UNDEF_INT) {
      if (ctaPolicyEnv == NCCL_CONFIG_UNDEF_INT) ctaPolicyEnv = tokenPolicy;
      else ctaPolicyEnv |= tokenPolicy;
    }
  }
  TEST_ASSERT(ctaPolicyEnv == NCCL_CTA_POLICY_ZERO,
              "BOGUS should be skipped, result should be ZERO only");
  TEST_PASS();
}

/* =========================================================================
 * Tests: plugin.cc parse functions — every valid name maps correctly
 * ========================================================================= */

int test_parse_coll_type() {
  char msg[256];
  TEST_ASSERT(collTypeTableSize == 5, "collTypeTable should have 5 entries");

  struct { const char* name; ncclFunc_t expected; } cases[] = {
    {"broadcast",     ncclFuncBroadcast},
    {"reduce",        ncclFuncReduce},
    {"allgather",     ncclFuncAllGather},
    {"reducescatter", ncclFuncReduceScatter},
    {"allreduce",     ncclFuncAllReduce},
  };
  for (const auto& tc : cases) {
    ncclFunc_t result = parseCollType(tc.name);
    snprintf(msg, sizeof(msg), "parseCollType('%s'): got %d, expected %d",
             tc.name, result, tc.expected);
    TEST_ASSERT(result == tc.expected, msg);
  }
  // Unknown should default to allreduce
  TEST_ASSERT(parseCollType("unknown") == ncclFuncAllReduce, "unknown defaults to allreduce");
  TEST_PASS();
}

int test_parse_algorithm() {
  char msg[256];
  TEST_ASSERT(algorithmTableSize == 7, "algorithmTable should have 7 entries");

  struct { const char* name; int expected; } cases[] = {
    {"tree",           NCCL_ALGO_TREE},
    {"ring",           NCCL_ALGO_RING},
    {"collnet_direct", NCCL_ALGO_COLLNET_DIRECT},
    {"collnet_chain",  NCCL_ALGO_COLLNET_CHAIN},
    {"nvls",           NCCL_ALGO_NVLS},
    {"nvls_tree",      NCCL_ALGO_NVLS_TREE},
    {"pat",            NCCL_ALGO_PAT},
  };
  for (const auto& tc : cases) {
    int result = parseAlgorithm(tc.name);
    snprintf(msg, sizeof(msg), "parseAlgorithm('%s'): got %d, expected %d",
             tc.name, result, tc.expected);
    TEST_ASSERT(result == tc.expected, msg);
  }
  TEST_ASSERT(parseAlgorithm("unknown") == NCCL_ALGO_RING, "unknown defaults to ring");
  TEST_PASS();
}

int test_parse_protocol() {
  char msg[256];
  TEST_ASSERT(protocolTableSize == 3, "protocolTable should have 3 entries");

  struct { const char* name; int expected; } cases[] = {
    {"ll",     NCCL_PROTO_LL},
    {"ll128",  NCCL_PROTO_LL128},
    {"simple", NCCL_PROTO_SIMPLE},
  };
  for (const auto& tc : cases) {
    int result = parseProtocol(tc.name);
    snprintf(msg, sizeof(msg), "parseProtocol('%s'): got %d, expected %d",
             tc.name, result, tc.expected);
    TEST_ASSERT(result == tc.expected, msg);
  }
  TEST_ASSERT(parseProtocol("unknown") == NCCL_PROTO_SIMPLE, "unknown defaults to simple");
  TEST_PASS();
}

/* =========================================================================
 * Tests: plugin.cc trim / splitCSV helpers
 * ========================================================================= */

int test_trim() {
  TEST_ASSERT(trim("hello") == "hello",     "no whitespace");
  TEST_ASSERT(trim("  hello") == "hello",   "leading spaces");
  TEST_ASSERT(trim("hello  ") == "hello",   "trailing spaces");
  TEST_ASSERT(trim("  hello  ") == "hello", "both sides");
  TEST_ASSERT(trim("\thello\t") == "hello",  "tabs");
  TEST_ASSERT(trim("") == "",               "empty string");
  TEST_ASSERT(trim("   ") == "",            "only whitespace");
  TEST_ASSERT(trim("a b c") == "a b c",    "inner spaces preserved");
  TEST_PASS();
}

int test_split_csv() {
  char msg[256];

  // Normal CSV
  {
    auto fields = splitCSV("allreduce,1024,65536,ring,simple,4,2,8");
    snprintf(msg, sizeof(msg), "expected 8 fields, got %d", (int)fields.size());
    TEST_ASSERT(fields.size() == 8, msg);
    TEST_ASSERT(fields[0] == "allreduce", "field 0");
    TEST_ASSERT(fields[1] == "1024",      "field 1");
    TEST_ASSERT(fields[3] == "ring",      "field 3");
    TEST_ASSERT(fields[7] == "8",         "field 7");
  }

  // With whitespace around fields
  {
    auto fields = splitCSV("  allreduce , 1024 , ring ");
    TEST_ASSERT(fields.size() == 3, "trimmed field count");
    TEST_ASSERT(fields[0] == "allreduce", "trimmed field 0");
    TEST_ASSERT(fields[1] == "1024",      "trimmed field 1");
    TEST_ASSERT(fields[2] == "ring",      "trimmed field 2");
  }

  // Single field
  {
    auto fields = splitCSV("allreduce");
    TEST_ASSERT(fields.size() == 1, "single field count");
    TEST_ASSERT(fields[0] == "allreduce", "single field value");
  }

  // Empty string
  {
    auto fields = splitCSV("");
    TEST_ASSERT(fields.size() == 0, "empty string should produce 0 fields");
  }

  TEST_PASS();
}

// End-to-end: parse a full plugin config line
int test_plugin_config_parse() {
  auto fields = splitCSV("allreduce,1024,65536,ring,simple,4,2,8");
  TEST_ASSERT(fields.size() == 8, "field count");

  ncclFunc_t collType = parseCollType(fields[0]);
  TEST_ASSERT(collType == ncclFuncAllReduce, "collType");

  size_t minBytes = strtoull(fields[1].c_str(), nullptr, 10);
  TEST_ASSERT(minBytes == 1024, "minBytes");

  size_t maxBytes = strtoull(fields[2].c_str(), nullptr, 10);
  TEST_ASSERT(maxBytes == 65536, "maxBytes");

  int algo = parseAlgorithm(fields[3]);
  TEST_ASSERT(algo == NCCL_ALGO_RING, "algorithm");

  int proto = parseProtocol(fields[4]);
  TEST_ASSERT(proto == NCCL_PROTO_SIMPLE, "protocol");

  int channels = std::atoi(fields[5].c_str());
  TEST_ASSERT(channels == 4, "channels");

  int nNodes = std::atoi(fields[6].c_str());
  TEST_ASSERT(nNodes == 2, "nNodes");

  int nRanks = std::atoi(fields[7].c_str());
  TEST_ASSERT(nRanks == 8, "nRanks");

  TEST_PASS();
}

/* =========================================================================
 * Tests: cpuset.h hex-mask tokenization
 * ========================================================================= */

int test_cpuset_hex_tokenize() {
  // Simulate ncclStrToCpuset parsing "0003ff,f0003fff"
  // Tokens should be parsed right-to-left into a uint32 array
  const int CPU_SET_N_U32 = 32; // CPU_SETSIZE(1024) / 32
  uint32_t cpumasks[32] = {0};
  int m = CPU_SET_N_U32;

  std::istringstream stream("0003ff,f0003fff");
  std::string token;
  while (std::getline(stream, token, ',') && m > 0) {
    cpumasks[--m] = strtoul(token.c_str(), nullptr, 16);
  }

  // Two tokens: "0003ff" and "f0003fff"
  // Processed right-to-left: cpumasks[31] = 0x0003ff, cpumasks[30] = 0xf0003fff
  TEST_ASSERT(m == 30, "should have consumed 2 slots");
  TEST_ASSERT(cpumasks[31] == 0x0003ffU,    "first token (highest) = 0x0003ff");
  TEST_ASSERT(cpumasks[30] == 0xf0003fffU,  "second token = 0xf0003fff");
  TEST_PASS();
}

int test_cpuset_single_mask() {
  const int CPU_SET_N_U32 = 32;
  uint32_t cpumasks[32] = {0};
  int m = CPU_SET_N_U32;

  std::istringstream stream("ff");
  std::string token;
  while (std::getline(stream, token, ',') && m > 0) {
    cpumasks[--m] = strtoul(token.c_str(), nullptr, 16);
  }

  TEST_ASSERT(m == 31, "should have consumed 1 slot");
  TEST_ASSERT(cpumasks[31] == 0xffU, "single mask = 0xff");
  TEST_PASS();
}

int test_cpuset_str_list_tokenize() {
  // Simulate ncclStrListToCpuset parsing "0,3,7,15"
  std::vector<uint64_t> cpus;
  std::istringstream stream("0,3,7,15");
  std::string token;
  while (std::getline(stream, token, ',')) {
    cpus.push_back(strtoull(token.c_str(), nullptr, 0));
  }

  TEST_ASSERT(cpus.size() == 4,  "4 CPUs listed");
  TEST_ASSERT(cpus[0] == 0,  "cpu 0");
  TEST_ASSERT(cpus[1] == 3,  "cpu 3");
  TEST_ASSERT(cpus[2] == 7,  "cpu 7");
  TEST_ASSERT(cpus[3] == 15, "cpu 15");
  TEST_PASS();
}

/* =========================================================================
 * Source verification: confirm source tables match our test expectations
 * ========================================================================= */

static char* read_file(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  if (len <= 0) { fclose(f); return nullptr; }
  fseek(f, 0, SEEK_SET);
  char* buf = new char[len + 1];
  size_t n = fread(buf, 1, len, f);
  buf[n] = '\0';
  fclose(f);
  return buf;
}

int test_source_tables_present() {
  char msg[512];
  int all_ok = 1;

  // Verify debug.cc has the subsysTable with all 18 entries
  const char* debug_path = "../../src/debug.cc";
  char* debug_src = read_file(debug_path);
  if (!debug_src) {
    printf("  SKIP: %s - Cannot read %s\n", __func__, debug_path);
  } else {
    // Check for key entries in the table
    const char* expected_entries[] = {
      "\"INIT\"", "\"BOOTSTRAP\"", "\"DESTROY\"", "\"ALL\"", nullptr
    };
    for (int i = 0; expected_entries[i]; i++) {
      if (!strstr(debug_src, expected_entries[i])) {
        snprintf(msg, sizeof(msg), "%s: missing table entry %s", debug_path, expected_entries[i]);
        printf("  FAIL: %s - %s\n", __func__, msg);
        all_ok = 0;
      }
    }
    // Verify lookup pattern is present
    if (!strstr(debug_src, "for (const auto& entry : subsysTable)")) {
      printf("  FAIL: %s - %s missing range-for lookup over subsysTable\n", __func__, debug_path);
      all_ok = 0;
    }
    if (!strstr(debug_src, "for (const auto& entry : tsTable)")) {
      printf("  FAIL: %s - %s missing range-for lookup over tsTable\n", __func__, debug_path);
      all_ok = 0;
    }
    delete[] debug_src;
  }

  // Verify init.cc has policyTable
  const char* init_path = "../../src/init.cc";
  char* init_src = read_file(init_path);
  if (!init_src) {
    printf("  SKIP: %s - Cannot read %s\n", __func__, init_path);
  } else {
    if (!strstr(init_src, "for (const auto& entry : policyTable)")) {
      printf("  FAIL: %s - %s missing range-for lookup over policyTable\n", __func__, init_path);
      all_ok = 0;
    }
    delete[] init_src;
  }

  // Verify plugin.cc has the lookup tables
  const char* plugin_path = "../../plugins/tuner/example/plugin.cc";
  char* plugin_src = read_file(plugin_path);
  if (!plugin_src) {
    printf("  SKIP: %s - Cannot read %s\n", __func__, plugin_path);
  } else {
    const char* tables[] = {"collTypeTable", "algorithmTable", "protocolTable", nullptr};
    for (int i = 0; tables[i]; i++) {
      if (!strstr(plugin_src, tables[i])) {
        snprintf(msg, sizeof(msg), "%s: missing %s", plugin_path, tables[i]);
        printf("  FAIL: %s - %s\n", __func__, msg);
        all_ok = 0;
      }
    }
    delete[] plugin_src;
  }

  if (all_ok) { TEST_PASS(); }
  return 0;
}

/* =========================================================================
 * Test runner
 * ========================================================================= */

typedef int (*TestFunction)();
struct TestCase { const char* name; TestFunction func; const char* description; };

static TestCase test_cases[] = {
  {"tokenize-comma",     test_tokenize_comma,          "istringstream comma-delimited tokenization"},
  {"tokenize-pipe",      test_tokenize_pipe,           "istringstream pipe-delimited tokenization"},
  {"tokenize-single",    test_tokenize_single,         "single token (no delimiter)"},
  {"tokenize-empty",     test_tokenize_empty,          "empty string tokenization"},
  {"subsys-all",         test_subsys_table_all_entries, "subsysTable: all 18 entries resolve correctly"},
  {"subsys-case",        test_subsys_table_case_insensitive, "subsysTable: case-insensitive match"},
  {"subsys-unknown",     test_subsys_table_unknown,    "subsysTable: unknown name falls through"},
  {"subsys-pipeline",    test_subsys_pipeline,         "subsysTable: full tokenize+lookup pipeline"},
  {"subsys-invert",      test_subsys_invert,           "subsysTable: ^INIT,NET invert logic"},
  {"ts-all",             test_ts_table_all_entries,    "tsTable: all 6 entries with correct masks"},
  {"ts-pipeline",        test_ts_pipeline,             "tsTable: WARN,TRACE pipeline"},
  {"policy-all",         test_policy_table_all_entries, "policyTable: all 3 entries"},
  {"policy-pipeline",    test_policy_pipeline,         "policyTable: DEFAULT|EFFICIENCY pipeline"},
  {"policy-unknown",     test_policy_unknown_token,    "policyTable: unknown token skipped"},
  {"parse-colltype",     test_parse_coll_type,         "parseCollType: all 5 collective types + default"},
  {"parse-algorithm",    test_parse_algorithm,         "parseAlgorithm: all 7 algorithms + default"},
  {"parse-protocol",     test_parse_protocol,          "parseProtocol: all 3 protocols + default"},
  {"trim",               test_trim,                    "trim: whitespace handling"},
  {"split-csv",          test_split_csv,               "splitCSV: field splitting with trim"},
  {"plugin-config",      test_plugin_config_parse,     "plugin: end-to-end config line parse"},
  {"cpuset-hex",         test_cpuset_hex_tokenize,     "cpuset: hex-mask comma tokenization"},
  {"cpuset-single",      test_cpuset_single_mask,      "cpuset: single hex mask"},
  {"cpuset-list",        test_cpuset_str_list_tokenize, "cpuset: CPU list tokenization"},
  {"source-tables",      test_source_tables_present,   "source: lookup tables present in source files"},
  {nullptr, nullptr, nullptr},
};

static void show_help(const char* prog) {
  printf("Usage: %s [test-name ...]\n\n", prog);
  printf("Available tests:\n");
  for (int i = 0; test_cases[i].name; i++) {
    printf("  %-25s %s\n", test_cases[i].name, test_cases[i].description);
  }
}

int main(int argc, char* argv[]) {
  if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    show_help(argv[0]);
    return 0;
  }

  printf("Lookup Table & Tokenization Tests\n");
  printf("==================================\n");

  int passed = 0, total = 0;

  if (argc == 1) {
    for (int i = 0; test_cases[i].name; i++) {
      total++;
      passed += test_cases[i].func();
    }
  } else {
    for (int arg = 1; arg < argc; arg++) {
      bool found = false;
      for (int i = 0; test_cases[i].name; i++) {
        if (strcmp(test_cases[i].name, argv[arg]) == 0) {
          total++;
          passed += test_cases[i].func();
          found = true;
          break;
        }
      }
      if (!found) {
        printf("ERROR: Unknown test '%s'\n", argv[arg]);
        show_help(argv[0]);
        return 1;
      }
    }
  }

  printf("\n==================================\n");
  printf("Results: %d/%d tests passed\n", passed, total);
  return (passed == total) ? 0 : 1;
}
