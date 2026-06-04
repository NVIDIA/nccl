/*
 * kv_sanity_test - KVStoreClient sanity check.
 *
 * Tests:
 *   1. set/get round-trip with a binary struct
 *
 * Usage:
 *   NCCL_CHECKPOINT_KVS_PATH=<path-to-kvs-file> \
 *   WORLD_RANK=<rank> WORLD_SIZE=<n> ./kv_sanity_test
 *
 * WORLD_SIZE/WORLD_RANK are only used to populate the test payload.
 */

#include "kv_store_client.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

using nccl_checkpoint::KVStoreClient;

struct BusinessCard {
  int rank;
  int world_size;
  char hostname[256];
  pid_t pid;
};

static void print_card(const BusinessCard& c) {
  printf("  rank=%d/%d host=%s pid=%d\n", c.rank, c.world_size, c.hostname, (int)c.pid);
}

static int run_set_get() {
  int world_rank = atoi(getenv("WORLD_RANK") ? getenv("WORLD_RANK") : "0");
  int world_size = atoi(getenv("WORLD_SIZE") ? getenv("WORLD_SIZE") : "1");

  KVStoreClient kv;
  if (!kv.connect_from_env()) {
    fprintf(stderr, "[rank %d] failed to connect - set NCCL_CHECKPOINT_KVS_PATH\n", world_rank);
    return 1;
  }
  printf("[rank %d] connected\n", world_rank);

  int failures = 0;

    /* ------------------------------------------------------------------
   * Test 1: set/get round-trip
   * ------------------------------------------------------------------ */
  BusinessCard my_card{};
  my_card.rank = world_rank;
  my_card.world_size = world_size;
  my_card.pid = getpid();
  gethostname(my_card.hostname, sizeof(my_card.hostname));

  char key[64];
  snprintf(key, sizeof(key), "rank/%d/card", world_rank);

  if (!kv.set(key, &my_card, sizeof(my_card))) {
    fprintf(stderr, "[rank %d] set failed\n", world_rank);
    kv.disconnect();
    return 1;
  }

  BusinessCard got{};
  size_t out_len = 0;
  if (!kv.get(key, &got, sizeof(got), &out_len) || out_len != sizeof(BusinessCard)) {
    fprintf(stderr, "[rank %d] get failed or wrong size (%zu vs %zu)\n", world_rank, out_len, sizeof(BusinessCard));
    failures++;
  } else if (memcmp(&my_card, &got, sizeof(got)) != 0) {
    fprintf(stderr, "[rank %d] get returned wrong data\n", world_rank);
    failures++;
  } else {
    printf("[rank %d] set/get OK: ", world_rank);
    print_card(got);
  }
  kv.del(key);

  kv.disconnect();
  return failures ? 1 : 0;
}

static int run_missing_key_timeout() {
  int world_rank = atoi(getenv("WORLD_RANK") ? getenv("WORLD_RANK") : "0");

  KVStoreClient kv;
  if (!kv.connect_from_env()) {
    fprintf(stderr, "[rank %d] failed to connect - set NCCL_CHECKPOINT_KVS_PATH\n", world_rank);
    return 1;
  }

  char key[64];
  snprintf(key, sizeof(key), "rank/%d/missing", world_rank);
  char value = 0;
  size_t out_len = 0;
  bool found = kv.get(key, &value, sizeof(value), &out_len);
  kv.disconnect();
  if (found) {
    fprintf(stderr, "[rank %d] missing key unexpectedly existed\n", world_rank);
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "set-get";
  if (strcmp(mode, "set-get") == 0) return run_set_get();
  if (strcmp(mode, "missing-key-timeout") == 0) return run_missing_key_timeout();

  fprintf(stderr, "unknown kv_sanity_test mode: %s\n", mode);
  return 2;
}
