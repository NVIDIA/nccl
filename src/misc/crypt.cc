/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "crypt.h"
#include "core.h"
#include "alloc.h"
#include "os.h"
#include "param/param.h"
#include "socket.h"
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/kdf.h>
#include <openssl/opensslv.h>
#include <openssl/params.h>
#include <openssl/ssl.h>

#if OPENSSL_VERSION_MAJOR != 3
#error "NCCL TLS support requires OpenSSL 3"
#endif

static constexpr const char* ncclCryptKdfParamDigest = "digest";
static constexpr const char* ncclCryptKdfParamKey = "key";
static constexpr const char* ncclCryptKdfParamInfo = "info";

#define NCCL_CRYPT_CRYPTO_SYMBOLS(SYMBOL) \
  SYMBOL(BIO_clear_flags) \
  SYMBOL(BIO_get_data) \
  SYMBOL(BIO_get_new_index) \
  SYMBOL(BIO_meth_free) \
  SYMBOL(BIO_meth_new) \
  SYMBOL(BIO_meth_set_create) \
  SYMBOL(BIO_meth_set_ctrl) \
  SYMBOL(BIO_meth_set_destroy) \
  SYMBOL(BIO_meth_set_read) \
  SYMBOL(BIO_meth_set_write) \
  SYMBOL(BIO_new) \
  SYMBOL(BIO_set_data) \
  SYMBOL(BIO_set_flags) \
  SYMBOL(BIO_set_init) \
  SYMBOL(CRYPTO_memcmp) \
  SYMBOL(ERR_clear_error) \
  SYMBOL(ERR_error_string_n) \
  SYMBOL(ERR_get_error) \
  SYMBOL(EVP_KDF_CTX_free) \
  SYMBOL(EVP_KDF_CTX_new) \
  SYMBOL(EVP_KDF_derive) \
  SYMBOL(EVP_KDF_fetch) \
  SYMBOL(EVP_KDF_free) \
  SYMBOL(OPENSSL_cleanse) \
  SYMBOL(OSSL_PARAM_construct_end) \
  SYMBOL(OSSL_PARAM_construct_octet_string) \
  SYMBOL(OSSL_PARAM_construct_utf8_string)

#define NCCL_CRYPT_SSL_SYMBOLS(SYMBOL) \
  SYMBOL(OPENSSL_init_ssl) \
  SYMBOL(SSL_CIPHER_find) \
  SYMBOL(SSL_CTX_clear_options) \
  SYMBOL(SSL_CTX_ctrl) \
  SYMBOL(SSL_CTX_free) \
  SYMBOL(SSL_CTX_new) \
  SYMBOL(SSL_CTX_set_ciphersuites) \
  SYMBOL(SSL_CTX_set_psk_find_session_callback) \
  SYMBOL(SSL_CTX_set_psk_use_session_callback) \
  SYMBOL(SSL_CTX_set_verify) \
  SYMBOL(SSL_do_handshake) \
  SYMBOL(SSL_free) \
  SYMBOL(SSL_get_error) \
  SYMBOL(SSL_new) \
  SYMBOL(SSL_read_ex) \
  SYMBOL(SSL_SESSION_free) \
  SYMBOL(SSL_SESSION_new) \
  SYMBOL(SSL_SESSION_set1_master_key) \
  SYMBOL(SSL_SESSION_set_cipher) \
  SYMBOL(SSL_SESSION_set_protocol_version) \
  SYMBOL(SSL_session_reused) \
  SYMBOL(SSL_set_accept_state) \
  SYMBOL(SSL_set_bio) \
  SYMBOL(SSL_set_connect_state) \
  SYMBOL(SSL_write_ex) \
  SYMBOL(TLS_method)

struct ncclCryptOpenSslSymbols {
#define NCCL_CRYPT_DECLARE_SYMBOL(name) decltype(&::name) pfn_##name;
  NCCL_CRYPT_CRYPTO_SYMBOLS(NCCL_CRYPT_DECLARE_SYMBOL)
  NCCL_CRYPT_SSL_SYMBOLS(NCCL_CRYPT_DECLARE_SYMBOL)
#undef NCCL_CRYPT_DECLARE_SYMBOL
};

static struct ncclCryptOpenSslSymbols ncclCryptOpenSsl;
static ncclOsLibraryHandle ncclCryptCryptoHandle;
static ncclOsLibraryHandle ncclCryptSslHandle;
static bool ncclCryptSymbolsReady;

#if defined(NCCL_OS_WINDOWS)
static constexpr const char* ncclCryptCryptoLibrary = "libcrypto-3-x64.dll";
static constexpr const char* ncclCryptSslLibrary = "libssl-3-x64.dll";
#else
static constexpr const char* ncclCryptCryptoLibrary = "libcrypto.so.3";
static constexpr const char* ncclCryptSslLibrary = "libssl.so.3";
#endif
#endif

static constexpr size_t ncclPskMinBytes = 32;
extern int64_t ncclParamPollTimeOut();

struct ncclSocketCrypto {
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  SSL* ssl;
  BIO* bio;
  ncclResult_t ioResult;
  bool ioClosed;
  int ioWantOp; // -1 if unset; otherwise NCCL_SOCKET_SEND or NCCL_SOCKET_RECV.
#endif
  bool handshakeDone; // TLS handshake complete.
};

static std::mutex ncclCryptLock; // owns all static-scoped variables for this translation unit
static bool ncclCryptHaveKey = false;
static unsigned char* ncclCryptConfigKey;
static size_t ncclCryptConfigKeyLen;

// PSK set through ncclSetEncryption().
static int ncclCryptApiMode = NCCL_ENCRYPTION_MODE_NONE;
static unsigned char* ncclCryptApiPsk;
static size_t ncclCryptApiPskLen;

static void cryptEraseAndFree(unsigned char* key, size_t keyLen) {
  volatile unsigned char* p = key;
  for (size_t i = 0; i < keyLen; i++) p[i] = 0;
  free(key);
}

static bool cryptConfigKeyMatches(const unsigned char* key, size_t keyLen) {
  return ncclCryptHaveKey && keyLen == ncclCryptConfigKeyLen &&
         (keyLen == 0 || memcmp(key, ncclCryptConfigKey, keyLen) == 0);
}

static void cryptPollSocket(struct ncclSocket* sock, int op) {
  if (sock->asyncFlag == 0 && ncclParamPollTimeOut()) ncclOsPollSocket(sock->socketDescriptor, op);
}

static int cryptPollOp(struct ncclSocket* sock, int fallbackOp) {
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  if (sock->crypto != nullptr && sock->crypto->ioWantOp != -1) return sock->crypto->ioWantOp;
#endif
  return fallbackOp;
}

static ncclResult_t cryptSocketProgressRaw(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  int closed = 0;
  NCCLCHECK(ncclOsSocketProgressOpt(op, sock, ptr, size, offset, /*block=*/0, &closed));
  if (closed) {
    char line[SOCKET_NAME_MAXLEN + 1];
    WARN("cryptSocketProgressRaw: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line));
    return ncclRemoteError;
  }
  return ncclSuccess;
}

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
static const unsigned char ncclCryptPskIdentity[] = "nccl-psk-v1";
static bool ncclCryptLibReady = false;
static BIO_METHOD* ncclCryptBioMethod;
static SSL_CTX* ncclCryptClientCtx;
static SSL_CTX* ncclCryptServerCtx;
static unsigned char ncclCryptPsk[NCCL_CRYPT_KEY_BYTES];

// Called with ncclCryptLock held.
static ncclResult_t cryptLoadOpenSslSymbols() {
  if (ncclCryptSymbolsReady) return ncclSuccess;
  void* tmp;
  void** cast;

  ncclCryptCryptoHandle = ncclOsDlopen(ncclCryptCryptoLibrary);
  if (ncclCryptCryptoHandle == nullptr) {
    WARN("ncclCrypt: failed to open %s: %s", ncclCryptCryptoLibrary, ncclOsDlerror());
    return ncclSystemError;
  }
  ncclCryptSslHandle = ncclOsDlopen(ncclCryptSslLibrary);
  if (ncclCryptSslHandle == nullptr) {
    WARN("ncclCrypt: failed to open %s: %s", ncclCryptSslLibrary, ncclOsDlerror());
    goto teardown;
  }

#define NCCL_CRYPT_LOAD_SYMBOL(handle, name) \
  do { \
    cast = (void**)&ncclCryptOpenSsl.pfn_##name; \
    tmp = ncclOsDlsym(handle, #name); \
    if (tmp == nullptr) { \
      WARN("ncclCrypt: ncclOsDlsym failed on %s in %s: %s", #name, \
           handle == ncclCryptCryptoHandle ? ncclCryptCryptoLibrary : ncclCryptSslLibrary, ncclOsDlerror()); \
      goto teardown; \
    } \
    *cast = tmp; \
  } while (0);

#define NCCL_CRYPT_LOAD_CRYPTO_SYMBOL(name) NCCL_CRYPT_LOAD_SYMBOL(ncclCryptCryptoHandle, name)
  NCCL_CRYPT_CRYPTO_SYMBOLS(NCCL_CRYPT_LOAD_CRYPTO_SYMBOL)
#undef NCCL_CRYPT_LOAD_CRYPTO_SYMBOL
#define NCCL_CRYPT_LOAD_SSL_SYMBOL(name) NCCL_CRYPT_LOAD_SYMBOL(ncclCryptSslHandle, name)
  NCCL_CRYPT_SSL_SYMBOLS(NCCL_CRYPT_LOAD_SSL_SYMBOL)
#undef NCCL_CRYPT_LOAD_SSL_SYMBOL
#undef NCCL_CRYPT_LOAD_SYMBOL

  // OpenSSL objects and function pointers remain in use until process exit.
  ncclCryptSymbolsReady = true;
  return ncclSuccess;

teardown:
  ncclCryptOpenSsl = {};
  ncclOsDlclose(ncclCryptSslHandle);
  ncclOsDlclose(ncclCryptCryptoHandle);
  ncclCryptSslHandle = nullptr;
  ncclCryptCryptoHandle = nullptr;
  return ncclSystemError;
}

static void cryptWarnOpenSsl(const char* what, const struct ncclSocket* sock, int sslError) {
  char err[256];
  char line[SOCKET_NAME_MAXLEN + 1];
  unsigned long rc = ncclCryptOpenSsl.pfn_ERR_get_error();
  if (rc != 0) ncclCryptOpenSsl.pfn_ERR_error_string_n(rc, err, sizeof(err));
  else snprintf(err, sizeof(err), "SSL error %d", sslError);
  WARN("%s with %s failed: %s", what, ncclSocketToString(&sock->addr, line), err);
}

static SSL_SESSION* cryptPskSession(SSL* ssl) {
  static const unsigned char cipherId[] = {0x13, 0x01};  // TLS_AES_128_GCM_SHA256
  const SSL_CIPHER* cipher = ncclCryptOpenSsl.pfn_SSL_CIPHER_find(ssl, cipherId);
  SSL_SESSION* session = ncclCryptOpenSsl.pfn_SSL_SESSION_new();
  if (session == nullptr || cipher == nullptr ||
      ncclCryptOpenSsl.pfn_SSL_SESSION_set_protocol_version(session, TLS1_3_VERSION) != 1 ||
      ncclCryptOpenSsl.pfn_SSL_SESSION_set_cipher(session, cipher) != 1 ||
      ncclCryptOpenSsl.pfn_SSL_SESSION_set1_master_key(session, ncclCryptPsk, sizeof(ncclCryptPsk)) != 1) {
    ncclCryptOpenSsl.pfn_SSL_SESSION_free(session);
    return nullptr;
  }
  return session;
}

static int cryptPskUseSession(SSL* ssl, const EVP_MD*, const unsigned char** id, size_t* idLen, SSL_SESSION** session) {
  *id = ncclCryptPskIdentity;
  *idLen = sizeof(ncclCryptPskIdentity) - 1;
  *session = cryptPskSession(ssl);
  return *session != nullptr;
}

static int cryptPskFindSession(SSL* ssl, const unsigned char* id, size_t idLen, SSL_SESSION** session) {
  *session = nullptr;
  if (idLen != sizeof(ncclCryptPskIdentity) - 1 ||
      ncclCryptOpenSsl.pfn_CRYPTO_memcmp(id, ncclCryptPskIdentity, idLen) != 0)
    return 0;
  *session = cryptPskSession(ssl);
  return *session != nullptr;
}

static int cryptBioCreate(BIO* bio) {
  ncclCryptOpenSsl.pfn_BIO_set_init(bio, /*init=*/1);
  ncclCryptOpenSsl.pfn_BIO_set_data(bio, nullptr);
  return 1;
}

static int cryptBioDestroy(BIO* bio) {
  if (bio == nullptr) return 0;
  ncclCryptOpenSsl.pfn_BIO_set_init(bio, /*init=*/0);
  ncclCryptOpenSsl.pfn_BIO_set_data(bio, nullptr);
  return 1;
}

static long cryptBioCtrl(BIO*, int cmd, long, void*) {
  return cmd == BIO_CTRL_FLUSH ? 1 : 0;
}

static int cryptBioWrite(BIO* bio, const char* buf, int len) {
  struct ncclSocket* sock = (struct ncclSocket*)ncclCryptOpenSsl.pfn_BIO_get_data(bio);
  int offset = 0;
  int closed = 0;
  ncclCryptOpenSsl.pfn_BIO_clear_flags(bio, BIO_FLAGS_RWS | BIO_FLAGS_SHOULD_RETRY);
  sock->crypto->ioResult =
    ncclOsSocketProgressOpt(NCCL_SOCKET_SEND, sock, (void*)buf, len, &offset, /*block=*/0, &closed);
  sock->crypto->ioClosed = closed != 0;
  if (sock->crypto->ioResult != ncclSuccess || closed) return -1;
  if (offset == 0) {
    ncclCryptOpenSsl.pfn_BIO_set_flags(bio, BIO_FLAGS_WRITE | BIO_FLAGS_SHOULD_RETRY);
    return -1;
  }
  return offset;
}

static int cryptBioRead(BIO* bio, char* buf, int len) {
  struct ncclSocket* sock = (struct ncclSocket*)ncclCryptOpenSsl.pfn_BIO_get_data(bio);
  int offset = 0;
  int closed = 0;
  ncclCryptOpenSsl.pfn_BIO_clear_flags(bio, BIO_FLAGS_RWS | BIO_FLAGS_SHOULD_RETRY);
  sock->crypto->ioResult = ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, sock, buf, len, &offset, /*block=*/0, &closed);
  sock->crypto->ioClosed = closed != 0;
  if (sock->crypto->ioResult != ncclSuccess) return -1;
  if (closed && offset == 0) return 0;
  if (offset == 0) {
    ncclCryptOpenSsl.pfn_BIO_set_flags(bio, BIO_FLAGS_READ | BIO_FLAGS_SHOULD_RETRY);
    return -1;
  }
  return offset;
}

static void cryptFreeLibObjects() {
  if (ncclCryptServerCtx != nullptr) ncclCryptOpenSsl.pfn_SSL_CTX_free(ncclCryptServerCtx);
  if (ncclCryptClientCtx != nullptr) ncclCryptOpenSsl.pfn_SSL_CTX_free(ncclCryptClientCtx);
  if (ncclCryptBioMethod != nullptr) ncclCryptOpenSsl.pfn_BIO_meth_free(ncclCryptBioMethod);
  ncclCryptServerCtx = nullptr;
  ncclCryptClientCtx = nullptr;
  ncclCryptBioMethod = nullptr;
  ncclCryptLibReady = false;
}

// Called with ncclCryptLock held.
static ncclResult_t cryptInitLib() {
  if (ncclCryptLibReady) return ncclSuccess;
  ncclResult_t ret = ncclSuccess;
  SSL_CTX* contexts[2] = {nullptr, nullptr};
  NCCLCHECKGOTO(cryptLoadOpenSslSymbols(), ret, fail);
  if (ncclCryptOpenSsl.pfn_OPENSSL_init_ssl(/*opts=*/0, /*settings=*/nullptr) != 1) {
    WARN("ncclCrypt: OPENSSL_init_ssl failed");
    ret = ncclSystemError;
    goto fail;
  }
  ncclCryptBioMethod =
    ncclCryptOpenSsl.pfn_BIO_meth_new(ncclCryptOpenSsl.pfn_BIO_get_new_index() | BIO_TYPE_SOURCE_SINK, "NCCL socket");
  if (ncclCryptBioMethod == nullptr ||
      ncclCryptOpenSsl.pfn_BIO_meth_set_write(ncclCryptBioMethod, cryptBioWrite) != 1 ||
      ncclCryptOpenSsl.pfn_BIO_meth_set_read(ncclCryptBioMethod, cryptBioRead) != 1 ||
      ncclCryptOpenSsl.pfn_BIO_meth_set_ctrl(ncclCryptBioMethod, cryptBioCtrl) != 1 ||
      ncclCryptOpenSsl.pfn_BIO_meth_set_create(ncclCryptBioMethod, cryptBioCreate) != 1 ||
      ncclCryptOpenSsl.pfn_BIO_meth_set_destroy(ncclCryptBioMethod, cryptBioDestroy) != 1) {
    WARN("ncclCrypt: OpenSSL BIO initialization failed");
    ret = ncclSystemError;
    goto fail;
  }
  ncclCryptClientCtx = ncclCryptOpenSsl.pfn_SSL_CTX_new(ncclCryptOpenSsl.pfn_TLS_method());
  ncclCryptServerCtx = ncclCryptOpenSsl.pfn_SSL_CTX_new(ncclCryptOpenSsl.pfn_TLS_method());
  contexts[0] = ncclCryptClientCtx;
  contexts[1] = ncclCryptServerCtx;
  for (int i = 0; i < 2; i++) {
    if (contexts[i] == nullptr ||
        ncclCryptOpenSsl.pfn_SSL_CTX_ctrl(contexts[i], SSL_CTRL_SET_MIN_PROTO_VERSION, TLS1_3_VERSION, nullptr) != 1 ||
        ncclCryptOpenSsl.pfn_SSL_CTX_ctrl(contexts[i], SSL_CTRL_SET_MAX_PROTO_VERSION, TLS1_3_VERSION, nullptr) != 1 ||
        ncclCryptOpenSsl.pfn_SSL_CTX_set_ciphersuites(contexts[i], "TLS_AES_128_GCM_SHA256") != 1) {
      WARN("ncclCrypt: OpenSSL TLS context initialization failed");
      ret = ncclSystemError;
      goto fail;
    }
    ncclCryptOpenSsl.pfn_SSL_CTX_clear_options(contexts[i], SSL_OP_ALLOW_NO_DHE_KEX);
  }
  ncclCryptOpenSsl.pfn_SSL_CTX_set_verify(ncclCryptClientCtx, SSL_VERIFY_PEER, nullptr);
  ncclCryptOpenSsl.pfn_SSL_CTX_set_verify(ncclCryptServerCtx, SSL_VERIFY_NONE, nullptr);
  ncclCryptOpenSsl.pfn_SSL_CTX_set_psk_use_session_callback(ncclCryptClientCtx, cryptPskUseSession);
  ncclCryptOpenSsl.pfn_SSL_CTX_set_psk_find_session_callback(ncclCryptServerCtx, cryptPskFindSession);
  ncclCryptLibReady = true;
  return ncclSuccess;
fail:
  cryptFreeLibObjects();
  return ret;
}

// Called with ncclCryptLock held.
static ncclResult_t cryptSetMasterKey(const unsigned char* key, size_t keyLen) {
  NCCLCHECK(cryptInitLib());
  ncclResult_t ret = ncclSuccess;
  unsigned char* keyCopy = nullptr;
  NCCLCHECK(ncclCalloc(&keyCopy, keyLen));
  memcpy(keyCopy, key, keyLen);
  static const char info[] = "NCCL PSK domain separation key";
  EVP_KDF* kdf = ncclCryptOpenSsl.pfn_EVP_KDF_fetch(nullptr, "HKDF", nullptr);
  EVP_KDF_CTX* ctx = kdf == nullptr ? nullptr : ncclCryptOpenSsl.pfn_EVP_KDF_CTX_new(kdf);
  OSSL_PARAM params[] = {
    ncclCryptOpenSsl.pfn_OSSL_PARAM_construct_utf8_string(ncclCryptKdfParamDigest, (char*)"SHA256", /*bsize=*/0),
    ncclCryptOpenSsl.pfn_OSSL_PARAM_construct_octet_string(ncclCryptKdfParamKey, (void*)key, keyLen),
    ncclCryptOpenSsl.pfn_OSSL_PARAM_construct_octet_string(ncclCryptKdfParamInfo, (void*)info, sizeof(info) - 1),
    ncclCryptOpenSsl.pfn_OSSL_PARAM_construct_end()
  };
  int rc = ctx == nullptr ? 0 : ncclCryptOpenSsl.pfn_EVP_KDF_derive(ctx, ncclCryptPsk, sizeof(ncclCryptPsk), params);
  if (rc != 1) {
    WARN("ncclCrypt: OpenSSL HKDF failed");
    ret = ncclSystemError;
    goto fail;
  }
  ncclCryptOpenSsl.pfn_EVP_KDF_CTX_free(ctx);
  ncclCryptOpenSsl.pfn_EVP_KDF_free(kdf);
  cryptEraseAndFree(ncclCryptConfigKey, ncclCryptConfigKeyLen);
  ncclCryptConfigKey = keyCopy;
  ncclCryptConfigKeyLen = keyLen;
  ncclCryptHaveKey = true;
  return ncclSuccess;
fail:
  ncclCryptOpenSsl.pfn_EVP_KDF_CTX_free(ctx);
  ncclCryptOpenSsl.pfn_EVP_KDF_free(kdf);
  ncclCryptOpenSsl.pfn_OPENSSL_cleanse(ncclCryptPsk, sizeof(ncclCryptPsk));
  cryptEraseAndFree(keyCopy, keyLen);
  ncclCryptHaveKey = false;
  return ret;
}
#else
static ncclResult_t cryptSetMasterKey(const unsigned char*, size_t) {
  WARN("NCCL was built without TLS support but a pre-shared key was configured");
  return ncclInvalidUsage;
}
#endif

// Called with ncclCryptLock held.
static void cryptClearMasterKey() {
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  ncclCryptOpenSsl.pfn_OPENSSL_cleanse(ncclCryptPsk, sizeof(ncclCryptPsk));
#endif
  cryptEraseAndFree(ncclCryptConfigKey, ncclCryptConfigKeyLen);
  ncclCryptConfigKey = nullptr;
  ncclCryptConfigKeyLen = 0;
  ncclCryptHaveKey = false;
}

ncclResult_t ncclGetCryptConnectionMode(bool* encrypted) {
  *encrypted = false;

  std::lock_guard<std::mutex> lock(ncclCryptLock);
  const unsigned char* key = nullptr;
  size_t keyLen = 0;

  switch (ncclCryptApiMode) {
  case NCCL_ENCRYPTION_MODE_PSK:
    key = ncclCryptApiPsk;
    keyLen = ncclCryptApiPskLen;
    break;
  case NCCL_ENCRYPTION_MODE_NONE:
    break;
  default:
    WARN("ncclGetCryptConnectionMode: unsupported encryption mode %d", ncclCryptApiMode);
    return ncclInvalidArgument;
  }

  if (keyLen == 0) {
    if (ncclCryptHaveKey || ncclCryptConfigKey != nullptr) cryptClearMasterKey();
  } else {
    if (!cryptConfigKeyMatches((const unsigned char*)key, keyLen)) {
      NCCLCHECK(cryptSetMasterKey((const unsigned char*)key, keyLen));
    }
    *encrypted = true;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclSetEncryption, const ncclEncryptionConfig_t* config);
ncclResult_t ncclSetEncryption(const ncclEncryptionConfig_t* config) {
  ncclEncryptionConfig_t internal = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  internal.magic = 0;      // like ncclConfig_t we use magic to enforce the init pattern
  if (config != nullptr) {
    if (config->magic != NCCL_API_MAGIC) {
      WARN("ncclSetEncryption: config not initialized via NCCL_ENCRYPTION_CONFIG_INITIALIZER");
      return ncclInvalidArgument;
    }
    size_t realSize = config->size > sizeof(internal) ? sizeof(internal) : config->size;
    memcpy(&internal, config, realSize);
  }
  // checks for config mode
  size_t pskLen = 0;
  switch (internal.mode) {
  case NCCL_ENCRYPTION_MODE_NONE:
    break;
  case NCCL_ENCRYPTION_MODE_PSK:
    if (internal.psk == NCCL_CONFIG_UNDEF_PTR) {
      WARN("ncclSetEncryption: NCCL_ENCRYPTION_MODE_PSK requires config->psk");
      return ncclInvalidArgument;
    }
    pskLen = strlen(internal.psk);
    if (pskLen < ncclPskMinBytes) {
      WARN("ncclSetEncryption: config->psk must be at least %zu bytes", ncclPskMinBytes);
      return ncclInvalidArgument;
    }
    break;
  default:
    WARN("ncclSetEncryption: config->mode=%d is not a supported encryption mode", internal.mode);
    return ncclInvalidArgument;
  }

  std::lock_guard<std::mutex> lock(ncclCryptLock);

  // Case 1: store the API PSK; ncclGetCryptConnectionMode() installs the derived key later
  if (internal.mode == NCCL_ENCRYPTION_MODE_PSK) {
    unsigned char* keyCopy = nullptr;
    NCCLCHECK(ncclCalloc(&keyCopy, pskLen));
    memcpy(keyCopy, internal.psk, pskLen);
    cryptEraseAndFree(ncclCryptApiPsk, ncclCryptApiPskLen);
    ncclCryptApiPsk = keyCopy;
    ncclCryptApiPskLen = pskLen;
  } else {
    // Case 2: default or non-PSK/plaintext mode, clear installed master key
    cryptEraseAndFree(ncclCryptApiPsk, ncclCryptApiPskLen);
    ncclCryptApiPsk = nullptr;
    ncclCryptApiPskLen = 0;
    if (ncclCryptHaveKey || ncclCryptConfigKey != nullptr) cryptClearMasterKey();
  }
  ncclCryptApiMode = internal.mode;
  return ncclSuccess;
}

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
static ncclResult_t cryptEnsure(struct ncclSocket* sock, bool isServer) {
  if (sock->crypto != nullptr) return ncclSuccess;
  ncclResult_t ret = ncclSuccess;
  struct ncclSocketCrypto* crypto = nullptr;
  BIO* bio = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&crypto, 1), ret, fail);
  crypto->ssl = ncclCryptOpenSsl.pfn_SSL_new(isServer ? ncclCryptServerCtx : ncclCryptClientCtx);
  bio = crypto->ssl == nullptr ? nullptr : ncclCryptOpenSsl.pfn_BIO_new(ncclCryptBioMethod);
  if (bio == nullptr) {
    WARN("ncclCrypt: OpenSSL connection initialization failed");
    ret = ncclSystemError;
    goto fail;
  }
  ncclCryptOpenSsl.pfn_BIO_set_data(bio, sock);
  ncclCryptOpenSsl.pfn_SSL_set_bio(crypto->ssl, bio, bio);
  if (isServer) ncclCryptOpenSsl.pfn_SSL_set_accept_state(crypto->ssl);
  else ncclCryptOpenSsl.pfn_SSL_set_connect_state(crypto->ssl);
  crypto->bio = bio;
  sock->crypto = crypto;
  return ncclSuccess;
fail:
  if (crypto != nullptr) {
    ncclCryptOpenSsl.pfn_SSL_free(crypto->ssl);
    free(crypto);
  }
  return ret;
}

void ncclCryptRebindSocket(struct ncclSocket* sock) {
  if (sock->crypto != nullptr) ncclCryptOpenSsl.pfn_BIO_set_data(sock->crypto->bio, sock);
}

static void cryptPrepareIo(struct ncclSocket* sock) {
  sock->crypto->ioResult = ncclSuccess;
  sock->crypto->ioClosed = false;
  sock->crypto->ioWantOp = -1;
}

static bool cryptClosedRc(struct ncclSocketCrypto* crypto, int rc, int sslError) {
  return crypto->ioClosed || sslError == SSL_ERROR_ZERO_RETURN || (sslError == SSL_ERROR_SYSCALL && rc == 0);
}

// One non-blocking TLS handshake step. *done is true once the handshake has completed.
// A failed handshake returns ncclRemoteError.
static ncclResult_t cryptHandshakeStep(struct ncclSocket* sock, bool* done) {
  *done = false;
  if (sock->crypto->handshakeDone) {
    *done = true;
    return ncclSuccess;
  }
  cryptPrepareIo(sock);
  ncclCryptOpenSsl.pfn_ERR_clear_error();
  int rc = ncclCryptOpenSsl.pfn_SSL_do_handshake(sock->crypto->ssl);
  if (rc == 1) {
    // The client also offers a certificate handshake, so a server that ignored the PSK could otherwise
    // complete one (the client's SSL_VERIFY_PEER, set in cryptInitLib, independently refuses any
    // certificate). On both roles, only a handshake that used the PSK proves the peer holds the key.
    if (ncclCryptOpenSsl.pfn_SSL_session_reused(sock->crypto->ssl) != 1) {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("ncclCrypt: TLS handshake with %s did not use the pre-shared key; rejecting peer",
           ncclSocketToString(&sock->addr, line));
      return ncclRemoteError;
    }
    sock->crypto->handshakeDone = true;
    *done = true;
    return ncclSuccess;
  }
  int sslError = ncclCryptOpenSsl.pfn_SSL_get_error(sock->crypto->ssl, rc);
  if (sock->crypto->ioResult != ncclSuccess) return sock->crypto->ioResult;
  if (sslError == SSL_ERROR_WANT_READ || sslError == SSL_ERROR_WANT_WRITE) {
    cryptPollSocket(sock, sslError == SSL_ERROR_WANT_READ ? NCCL_SOCKET_RECV : NCCL_SOCKET_SEND);
    return ncclSuccess;
  }
  cryptWarnOpenSsl("ncclCrypt: TLS handshake", sock, sslError);
  return ncclRemoteError;
}
#else
void ncclCryptRebindSocket(struct ncclSocket*) {}

static ncclResult_t cryptEnsure(struct ncclSocket*, bool) {
  WARN("ncclCrypt: TLS connection requested without TLS support");
  return ncclInternalError;
}

static ncclResult_t cryptHandshakeStep(struct ncclSocket*, bool*) {
  WARN("ncclCrypt: TLS handshake requested without TLS support");
  return ncclInternalError;
}
#endif

ncclResult_t ncclCryptStartConnect(struct ncclSocket* sock) {
  bool encrypted;
  NCCLCHECK(ncclGetCryptConnectionMode(&encrypted));
  if (encrypted) NCCLCHECK(cryptEnsure(sock, /*isServer=*/false));
  memcpy(sock->finalizeBuffer, &sock->magic, sizeof(sock->magic));
  memcpy(sock->finalizeBuffer + sizeof(sock->magic), &sock->type, sizeof(sock->type));
  return ncclSuccess;
}

ncclResult_t ncclCryptConnectHello(struct ncclSocket* sock) {
  if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
    if (sock->crypto != nullptr) {
      bool handshakeDone;
      NCCLCHECK(cryptHandshakeStep(sock, &handshakeDone));
      if (!handshakeDone) return ncclInProgress;
      NCCLCHECK(ncclCryptSocketSend(sock, sock->finalizeBuffer, NCCL_SOCKET_PLAIN_HELLO_BYTES, &sock->finalizeCounter,
                                    /*pclosed=*/nullptr));
    } else {
      NCCLCHECK(cryptSocketProgressRaw(NCCL_SOCKET_SEND, sock, sock->finalizeBuffer, NCCL_SOCKET_PLAIN_HELLO_BYTES,
                                       &sock->finalizeCounter));
    }
    if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      cryptPollSocket(sock, cryptPollOp(sock, NCCL_SOCKET_SEND));
      return ncclInProgress;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclCryptAcceptHello(struct ncclSocket* sock, enum ncclCryptHelloVerdict* verdict) {
  uint64_t magic;
  enum ncclSocketType type;
  char line[SOCKET_NAME_MAXLEN + 1];
  *verdict = NCCL_CRYPT_HELLO_VERDICT_AGAIN;
  // Resolve the current process-global PSK before interpreting the peer hello.
  bool keyed;
  NCCLCHECK(ncclGetCryptConnectionMode(&keyed));

  if (keyed) {
    NCCLCHECK(cryptEnsure(sock, /*isServer=*/true));
    if (!sock->crypto->handshakeDone) {
      bool handshakeDone;
      // Anything that cannot complete the PSK handshake (plaintext stock peers,
      // port scanners, a peer with the wrong key) is dropped so one bad
      // connection cannot error out an accept loop.
      ncclResult_t res = cryptHandshakeStep(sock, &handshakeDone);
      if (res != ncclSuccess) {
        ATTN("TLS handshake from %s failed, discarding peer connection", ncclSocketToString(&sock->addr, line));
        *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
        return ncclSuccess;
      }
      if (!handshakeDone) return ncclInProgress;
    }
    if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      bool closed = false;
      ncclResult_t res =
        ncclCryptSocketRecv(sock, sock->finalizeBuffer, NCCL_SOCKET_PLAIN_HELLO_BYTES, &sock->finalizeCounter, &closed);
      if (res == ncclRemoteError) {
        ATTN("Handshake receive from %s failed, discarding peer connection", ncclSocketToString(&sock->addr, line));
        *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
        return ncclSuccess;
      }
      NCCLCHECK(res);
      if (closed) {
        ATTN("Peer %s closed before socket hello, discarding", ncclSocketToString(&sock->addr, line));
        *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
        return ncclSuccess;
      }
      if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
        cryptPollSocket(sock, cryptPollOp(sock, NCCL_SOCKET_RECV));
        return ncclInProgress;
      }
    }
  } else {
    if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      int closed = 0;
      // A recv error during the hello resets like a close (stock's synchronous
      // path did the same): one flaky connection must not kill the accept loop.
      ncclResult_t res =
        ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, NCCL_SOCKET_PLAIN_HELLO_BYTES,
                                &sock->finalizeCounter, /*block=*/0, &closed);
      if (res == ncclRemoteError) {
        ATTN("Handshake receive from %s failed, discarding peer connection", ncclSocketToString(&sock->addr, line));
        *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
        return ncclSuccess;
      }
      NCCLCHECK(res);
      if (closed) {
        ATTN("Peer %s closed before socket hello, discarding", ncclSocketToString(&sock->addr, line));
        *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
        return ncclSuccess;
      }
      if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
        cryptPollSocket(sock, NCCL_SOCKET_RECV);
        return ncclInProgress;
      }
    }
  }

  memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
  if (magic != sock->magic) {
    // A TLS ClientHello arriving at a plaintext listener also lands here: the
    // record header never matches the socket magic, so mixed configurations
    // reset instead of erroring out the accept loop.
    ATTN("Socket magic mismatch from %s (peer 0x%016llx != expected 0x%016llx), discarding peer connection",
         ncclSocketToString(&sock->addr, line), (unsigned long long)magic, (unsigned long long)sock->magic);
    *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
    return ncclSuccess;
  }
  memcpy(&type, sock->finalizeBuffer + sizeof(magic), sizeof(type));
  if (type != sock->type) {
    ATTN("Wrong socket type from %s (peer %d != expected %d -- peer connected to wrong NCCL socket), discarding peer "
         "connection",
         ncclSocketToString(&sock->addr, line), (int)type, (int)sock->type);
    *verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
    return ncclSuccess;
  }
  *verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  return ncclSuccess;
}

ncclResult_t ncclCryptSocketSend(struct ncclSocket* sock, void* ptr, int size, int* offset, bool* pclosed) {
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  char line[SOCKET_NAME_MAXLEN + 1];
  cryptPrepareIo(sock);
  while (*offset < size) {
    size_t written = 0;
    ncclCryptOpenSsl.pfn_ERR_clear_error();
    int rc = ncclCryptOpenSsl.pfn_SSL_write_ex(sock->crypto->ssl, (const char*)ptr + *offset, size - *offset, &written);
    if (rc == 1) {
      *offset += written;
      continue;
    }
    int sslError = ncclCryptOpenSsl.pfn_SSL_get_error(sock->crypto->ssl, rc);
    if (sock->crypto->ioResult != ncclSuccess) return sock->crypto->ioResult;
    if (sslError == SSL_ERROR_WANT_READ || sslError == SSL_ERROR_WANT_WRITE) {
      sock->crypto->ioWantOp = sslError == SSL_ERROR_WANT_READ ? NCCL_SOCKET_RECV : NCCL_SOCKET_SEND;
      return ncclSuccess;
    }
    if (cryptClosedRc(sock->crypto, rc, sslError)) {
      if (pclosed) {
        *pclosed = true;
        return ncclSuccess;
      }
      WARN("ncclCryptSocketSend: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line));
      return ncclRemoteError;
    }
    cryptWarnOpenSsl("ncclCryptSocketSend", sock, sslError);
    return ncclRemoteError;
  }
  return ncclSuccess;
#else
  WARN("ncclCryptSocketSend called without TLS support");
  return ncclInternalError;
#endif
}

ncclResult_t ncclCryptSocketRecv(struct ncclSocket* sock, void* ptr, int size, int* offset, bool* pclosed) {
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  char line[SOCKET_NAME_MAXLEN + 1];
  cryptPrepareIo(sock);
  while (*offset < size) {
    size_t read = 0;
    ncclCryptOpenSsl.pfn_ERR_clear_error();
    int rc = ncclCryptOpenSsl.pfn_SSL_read_ex(sock->crypto->ssl, (char*)ptr + *offset, size - *offset, &read);
    if (rc == 1) {
      *offset += read;
      continue;
    }
    int sslError = ncclCryptOpenSsl.pfn_SSL_get_error(sock->crypto->ssl, rc);
    if (sock->crypto->ioResult != ncclSuccess) return sock->crypto->ioResult;
    if (sslError == SSL_ERROR_WANT_READ || sslError == SSL_ERROR_WANT_WRITE) {
      sock->crypto->ioWantOp = sslError == SSL_ERROR_WANT_READ ? NCCL_SOCKET_RECV : NCCL_SOCKET_SEND;
      return ncclSuccess;
    }
    if (cryptClosedRc(sock->crypto, rc, sslError)) {
      if (pclosed) {
        *pclosed = true;
        return ncclSuccess;
      }
      WARN("ncclCryptSocketRecv: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line));
      return ncclRemoteError;
    }
    cryptWarnOpenSsl("ncclCryptSocketRecv", sock, sslError);
    return ncclRemoteError;
  }
  return ncclSuccess;
#else
  WARN("ncclCryptSocketRecv called without TLS support");
  return ncclInternalError;
#endif
}

void ncclCryptFree(struct ncclSocketCrypto* crypto) {
  if (crypto == nullptr) return;
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
  ncclCryptOpenSsl.pfn_SSL_free(crypto->ssl);
#endif
  free(crypto);
}
