/*
 * kv_store_client.cc — Redis-backed KV store client via hiredis.
 *
 * All hiredis types are confined to this translation unit.
 * Link with -lhiredis.
 */

#include "kv_store_client.h"

#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>
#include <sys/time.h>
#include <hiredis/hiredis.h>

namespace {
constexpr int kDefaultTimeoutSec = 300;
constexpr int kConnectAttemptTimeoutMs = 1000;
constexpr int kConnectPollMs = 100;
constexpr int kGetPollMs = 10;
constexpr const char* kTimeoutEnvName = "NCCL_CHECKPOINT_KVS_TIMEOUT";

std::chrono::milliseconds kvs_timeout() {
  const char* env = getenv(kTimeoutEnvName);
  if (env == nullptr || *env == '\0') {
    return std::chrono::seconds(kDefaultTimeoutSec);
  }

  char* end = nullptr;
  double seconds = strtod(env, &end);
  if (end == env || seconds < 0) {
    return std::chrono::seconds(kDefaultTimeoutSec);
  }
  return std::chrono::milliseconds(static_cast<long long>(seconds * 1000));
}

} // namespace

/* -------------------------------------------------------------------------
 * Impl — holds the redisContext and helper utilities
 * ------------------------------------------------------------------------- */
namespace nccl_checkpoint {

struct KVStoreClient::Impl {
  redisContext* ctx = nullptr;
  std::string prefix;

  /* Return key with prefix prepended if set. */
  std::string key(const char* k) const {
    if (prefix.empty()) return k;
    return prefix + "/" + k;
  }

  /* Issue a command and return the reply.  Caller must freeReplyObject().
   * Returns nullptr on connection error. */
  redisReply* cmd(const char* fmt, ...) {
    if (!ok()) return nullptr;
    va_list ap;
    va_start(ap, fmt);
    auto* r = static_cast<redisReply*>(redisvCommand(ctx, fmt, ap));
    va_end(ap);
    return r;
  }

  /* Issue a command with a binary-safe value using redisCommandArgv. */
  redisReply* cmd_binary(const char* op, const char* key, const void* val, size_t val_len) {
    if (!ok()) return nullptr;
    const char* argv[3] = {op, key, static_cast<const char*>(val)};
    const size_t lens[3] = {strlen(op), strlen(key), val_len};
    return static_cast<redisReply*>(redisCommandArgv(ctx, 3, argv, lens));
  }

  bool ok() const {
    return ctx != nullptr && ctx->err == 0;
  }

  static const char* errstr(const redisContext* c) {
    if (c == nullptr) return "null context";
    if (c->errstr[0] == '\0') return "unknown error";
    return c->errstr;
  }

  static const char* reply_type_name(int t) {
    switch (t) {
    case REDIS_REPLY_STRING:
      return "STRING";
    case REDIS_REPLY_ARRAY:
      return "ARRAY";
    case REDIS_REPLY_INTEGER:
      return "INTEGER";
    case REDIS_REPLY_NIL:
      return "NIL";
    case REDIS_REPLY_STATUS:
      return "STATUS";
    case REDIS_REPLY_ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
    }
  }

  /* Log a descriptive error.  Call after any failed command. */
  void log_err(const char* op, const char* k, const redisReply* r) const {
    if (!ctx || ctx->err) {
      fprintf(stderr, "KVS [%s key=%s]: connection error %d: %s\n", op, k, ctx ? ctx->err : -1, errstr(ctx));
    } else if (!r) {
      fprintf(stderr, "KVS [%s key=%s]: null reply (connection lost?)\n", op, k);
    } else if (r->type == REDIS_REPLY_ERROR) {
      fprintf(stderr, "KVS [%s key=%s]: server error: %s\n", op, k, r->str);
    } else {
      fprintf(stderr, "KVS [%s key=%s]: unexpected reply type %s\n", op, k, reply_type_name(r->type));
    }
  }
};

/* -------------------------------------------------------------------------
 * KVStoreClient
 * ------------------------------------------------------------------------- */
KVStoreClient::KVStoreClient() : impl_(new Impl{}) {}

KVStoreClient::~KVStoreClient() {
  disconnect();
  delete impl_;
}

bool KVStoreClient::connect(const char* host, int port) {
  disconnect();

  const auto timeout = kvs_timeout();
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::string lastError = "unknown error";

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      fprintf(stderr, "KVS connect(%s:%d): timed out after %.3f seconds (%s=%s, last error: %s)\n", host, port,
              timeout.count() / 1000.0, kTimeoutEnvName, getenv(kTimeoutEnvName) ? getenv(kTimeoutEnvName) : "default",
              lastError.c_str());
      return false;
    }

    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    int attemptMs = static_cast<int>(remaining.count());
    if (attemptMs <= 0) attemptMs = 1;
    if (attemptMs > kConnectAttemptTimeoutMs) attemptMs = kConnectAttemptTimeoutMs;

    timeval tv;
    tv.tv_sec = attemptMs / 1000;
    tv.tv_usec = (attemptMs % 1000) * 1000;

    redisContext* ctx = redisConnectWithTimeout(host, port, tv);
    if (ctx != nullptr && ctx->err == 0) {
      impl_->ctx = ctx;
      return true;
    }

    lastError = Impl::errstr(ctx);
    if (ctx != nullptr) redisFree(ctx);

    std::this_thread::sleep_for(std::chrono::milliseconds(kConnectPollMs));
  }
}

/* Parse <host>:<port>[/<prefix>] and connect.  Returns false if the string
 * is malformed or the connection fails. */
static bool connect_from_string(KVStoreClient* kv, const std::string& s) {
  /* Format: <host>:<port>[/<prefix>] */
  std::string addr, prefix;
  auto slash = s.find('/');
  if (slash != std::string::npos) {
    addr = s.substr(0, slash);
    prefix = s.substr(slash + 1);
  } else {
    addr = s;
  }

  auto colon = addr.rfind(':');
  if (colon == std::string::npos) return false;
  std::string host = addr.substr(0, colon);
  int port = atoi(addr.substr(colon + 1).c_str());
  if (port <= 0) return false;

  if (!kv->connect(host.c_str(), port)) return false;
  if (!prefix.empty()) kv->set_prefix(prefix.c_str());
  return true;
}

bool KVStoreClient::connect_from_env() {
  /* Read connection string from the file named by NCCL_CHECKPOINT_KVS_PATH.
   * File contains a single line: <host>:<port>[/<prefix>]
   * Written by the checkpointer before CRIU restore so the path is stable
   * across checkpoint/restore (the env var is captured by CRIU). */
  const char* path = getenv("NCCL_CHECKPOINT_KVS_PATH");
  if (!path) {
    fprintf(stderr, "KVStoreClient: NCCL_CHECKPOINT_KVS_PATH not set\n");
    return false;
  }

  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "KVStoreClient: cannot open %s: %s\n", path, strerror(errno));
    return false;
  }

  char line[512];
  bool ok = false;
  if (fgets(line, sizeof(line), f)) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    ok = connect_from_string(this, s);
  }
  fclose(f);
  return ok;
}

void KVStoreClient::set_prefix(const char* prefix) {
  impl_->prefix = (prefix && *prefix) ? prefix : "";
}

void KVStoreClient::disconnect() {
  if (impl_->ctx) {
    redisFree(impl_->ctx);
    impl_->ctx = nullptr;
  }
}

bool KVStoreClient::is_connected() const {
  return impl_->ok();
}

bool KVStoreClient::set(const char* key, const void* data, size_t len) {
  const std::string full_key = impl_->key(key);
  auto* r = impl_->cmd_binary("SET", full_key.c_str(), data, len);
  if (!r || !(r->type == REDIS_REPLY_STATUS && strncmp(r->str, "OK", 2) == 0)) {
    impl_->log_err("SET", full_key.c_str(), r);
    if (r) freeReplyObject(r);
    return false;
  }
  freeReplyObject(r);
  return true;
}

bool KVStoreClient::get(const char* key, void* buf, size_t buf_len, size_t* out_len) {
  const std::string full_key = impl_->key(key);
  if (!impl_->ok()) {
    impl_->log_err("GET", full_key.c_str(), nullptr);
    return false;
  }
  const auto timeout = kvs_timeout();
  const auto deadline = std::chrono::steady_clock::now() + timeout;

  while (true) {
    auto* r = static_cast<redisReply*>(redisCommand(impl_->ctx, "GET %s", full_key.c_str()));
    if (!r) {
      impl_->log_err("GET", full_key.c_str(), r);
      return false;
    }

    if (r->type == REDIS_REPLY_STRING) {
      if (r->len > (long long)buf_len) {
        fprintf(stderr, "KVS [GET key=%s]: value too large: %zu > %zu bytes\n", full_key.c_str(), (size_t)r->len,
                buf_len);
        freeReplyObject(r);
        return false;
      }
      memcpy(buf, r->str, r->len);
      if (out_len) *out_len = r->len;
      freeReplyObject(r);
      return true;
    }

    if (r->type != REDIS_REPLY_NIL) {
      impl_->log_err("GET", full_key.c_str(), r);
      freeReplyObject(r);
      return false;
    }

    freeReplyObject(r);
    if (std::chrono::steady_clock::now() >= deadline) {
      fprintf(stderr, "KVS [GET key=%s]: timed out after %.3f seconds waiting for key (%s=%s)\n", full_key.c_str(),
              timeout.count() / 1000.0, kTimeoutEnvName, getenv(kTimeoutEnvName) ? getenv(kTimeoutEnvName) : "default");
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kGetPollMs));
  }
}

void KVStoreClient::del(const char* key) {
  if (!impl_->ok()) return;
  auto* r = static_cast<redisReply*>(redisCommand(impl_->ctx, "DEL %s", impl_->key(key).c_str()));
  if (r) freeReplyObject(r);
}

}  // namespace nccl_checkpoint
