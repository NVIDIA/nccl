#pragma once
/*
 * kv_store_client.h — Redis-backed key/value store client.
 *
 * Hides all Redis/hiredis details behind a plain interface.
 * Values are treated as opaque binary blobs (length-prefixed RESP bulk
 * strings), so ncclUniqueId and similar structs can be stored directly.
 *
 * Typical rendezvous pattern:
 *
 *   KVStoreClient kv;
 *   kv.connect(host, port);
 *   kv.set(my_key, &uid, sizeof(uid));          // publish
 *   kv.get(peer_key, &peer_uid, sizeof(peer_uid)); // fetch peer data
 *   kv.disconnect();
 */

#include <cstddef>

namespace nccl_checkpoint {

class KVStoreClient {
public:
  KVStoreClient();
  ~KVStoreClient();

  KVStoreClient(const KVStoreClient&) = delete;
  KVStoreClient& operator=(const KVStoreClient&) = delete;
  KVStoreClient(KVStoreClient&&) = delete;
  KVStoreClient& operator=(KVStoreClient&&) = delete;

  /* Connect to a running Redis server.  Retries until
   * NCCL_CHECKPOINT_KVS_TIMEOUT seconds elapse.  Returns false on failure. */
  bool connect(const char* host, int port);

  /* Connect using the file named by NCCL_CHECKPOINT_KVS_PATH.
   * The file must contain a single line: <host>:<port>[/<prefix>]
   * Returns false if the env var is unset, the file is missing, or the
   * connection fails. */
  bool connect_from_env();

  void disconnect();
  bool is_connected() const;

  /* Prepend prefix/ to all subsequent key operations.
   * Pass nullptr or "" to clear the prefix. */
  void set_prefix(const char* prefix);

  /* Store a binary value.  Returns false on failure. */
  bool set(const char* key, const void* data, size_t len);

  /* Retrieve a binary value into buf (up to buf_len bytes), waiting until
   * NCCL_CHECKPOINT_KVS_TIMEOUT seconds elapse if the key does not exist yet.
   * *out_len is set to the actual value size on success.
   * Returns false on timeout or error. */
  bool get(const char* key, void* buf, size_t buf_len, size_t* out_len);

  /* Delete a key.  Best-effort; errors are ignored. */
  void del(const char* key);

private:
  struct Impl;
  Impl* impl_;
};

}  // namespace nccl_checkpoint
