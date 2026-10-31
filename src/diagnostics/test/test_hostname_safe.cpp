// CPU-only tests for ncclDiagIsSafeHostname() in ../hostname.h.

#include "../hostname.h"
#include <gtest/gtest.h>

namespace {

struct Row {
  const char* description;
  char host[64];
  bool expected;
};

constexpr Row kRows[] = {
    {"short name", "node0", true},
    {"fqdn with dots", "node0.cluster.example.com", true},
    {"name with dash", "gpu-node-01", true},
    {"digits only", "10", true},
    {"upper and lower", "GPU-Node", true},

    {"single letter", "a", true},
    {"single digit", "0", true},
    {"single dot", ".", true},
    {"single dash", "-", false},

    {"empty string", "", false},
    {"leading dash", "-node", false},
    {"long option", "--version", false},
    {"short option", "-V", false},

    {"semicolon", "node;reboot", false},
    {"pipe", "node|cat", false},
    {"backtick", "node`id`", false},
    {"dollar paren", "node$(id)", false},
    {"ampersand", "node&", false},
    {"space", "node name", false},
    {"redirect", "node>f", false},
    {"quote", "node'x", false},
    {"newline", "node\nx", false},
    {"tab", "node\tx", false},
    {"carriage return", "node\rx", false},
    {"double quote", "node\"x", false},
    {"backslash", "node\\x", false},
    {"high byte", "node\xFF", false},
};

TEST(IsSafeHostname, Table) {
  for (const Row& r : kRows) {
    EXPECT_EQ(ncclDiagIsSafeHostname(r.host), r.expected) << "case: " << r.description;
  }
}

TEST(IsSafeHostname, BufferBoundaries) {
  EXPECT_FALSE(ncclDiagIsSafeHostname(""));
  EXPECT_TRUE(ncclDiagIsSafeHostname("a"));

  const char unterminated[] = {'a'};
  EXPECT_FALSE(ncclDiagIsSafeHostname(unterminated));

  char host[1024];
  std::fill(host, host + sizeof(host), 'a');
  EXPECT_FALSE(ncclDiagIsSafeHostname(host));
  host[sizeof(host) - 1] = '\0';
  EXPECT_TRUE(ncclDiagIsSafeHostname(host));
}

}  // namespace
