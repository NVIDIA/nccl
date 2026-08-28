#include "crypt.h"
#include "core.h"
#include "argcheck.h"
#include "alloc.h"
#include "param.h"
#include "socket.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mbedtls/version.h>
#include <mbedtls/ssl.h>
#include <mbedtls/ssl_ciphersuites.h>
#include <mbedtls/net_sockets.h>
#include <mbedtls/entropy.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/hkdf.h>
#include <mbedtls/md.h>
#include <mbedtls/error.h>
#include <mbedtls/platform_util.h>
#if MBEDTLS_VERSION_MAJOR >= 3
#include <psa/crypto.h>
#endif

static const int ncclCryptCiphersuites[] = {
#if MBEDTLS_VERSION_MAJOR >= 3
  MBEDTLS_TLS1_3_AES_128_GCM_SHA256,
#else
  MBEDTLS_TLS_PSK_WITH_AES_128_GCM_SHA256,
#endif
  0
};

static const char ncclCryptPskIdentity[] = "nccl-psk-v1";

struct ncclSocketCrypto {
  mbedtls_ssl_context ssl;
  int isServer;
  int handshakeDone;
  // The stock magic+type hello, exchanged as the first application data once the handshake
  // completes: built and sent on the connect side, received and validated on the accept side.
  int helloOffset;
  char helloBuf[NCCL_SOCKET_PLAIN_HELLO_BYTES];
};

static pthread_mutex_t ncclCryptLock = PTHREAD_MUTEX_INITIALIZER;  // owns all static-scoped variables for this translation unit
static bool ncclCryptLibReady = false;
static bool ncclCryptHaveKey = false;
static mbedtls_entropy_context ncclCryptEntropy;
static mbedtls_ctr_drbg_context ncclCryptDrbg;
static mbedtls_ssl_config ncclCryptClientConf;
static mbedtls_ssl_config ncclCryptServerConf;

static void cryptWarnMbedtls(const char* what, const struct ncclSocket* sock, int rc) {
  char err[128];
  char line[SOCKET_NAME_MAXLEN+1];
  mbedtls_strerror(rc, err, sizeof(err));
  WARN("%s with %s failed: -0x%04x (%s)", what, ncclSocketToString(&sock->addr, line), (unsigned int)-rc, err);
}

#ifndef MBEDTLS_THREADING_C
#error "mbed TLS must be built with MBEDTLS_THREADING_C"
#endif

// Called with ncclCryptLock held.
static ncclResult_t cryptInitLib() {
  if (ncclCryptLibReady) return ncclSuccess;
  static const char pers[] = "nccl-psk";
  mbedtls_entropy_init(&ncclCryptEntropy);
  mbedtls_ctr_drbg_init(&ncclCryptDrbg);
  // backed by getrandom(2)
  int rc = mbedtls_ctr_drbg_seed(&ncclCryptDrbg, mbedtls_entropy_func, &ncclCryptEntropy,
                                 (const unsigned char*)pers, sizeof(pers) - 1);
  if (rc != 0) {
    WARN("ncclCrypt: mbedtls_ctr_drbg_seed failed: -0x%04x", (unsigned int)-rc);
    return ncclSystemError;
  }
#if MBEDTLS_VERSION_MAJOR >= 3
  psa_status_t pstatus = psa_crypto_init();
  if (pstatus != PSA_SUCCESS) {
    WARN("ncclCrypt: psa_crypto_init failed: %d", (int)pstatus);
    return ncclSystemError;
  }
#endif
  mbedtls_ssl_config* confs[2] = { &ncclCryptClientConf, &ncclCryptServerConf };
  for (int i = 0; i < 2; i++) {
    mbedtls_ssl_config_init(confs[i]);
    rc = mbedtls_ssl_config_defaults(confs[i], i == 0 ? MBEDTLS_SSL_IS_CLIENT : MBEDTLS_SSL_IS_SERVER,
                                     MBEDTLS_SSL_TRANSPORT_STREAM, MBEDTLS_SSL_PRESET_DEFAULT);
    if (rc != 0) {
      WARN("ncclCrypt: mbedtls_ssl_config_defaults failed: -0x%04x", (unsigned int)-rc);
      return ncclSystemError;
    }
#if MBEDTLS_VERSION_MAJOR >= 3
    mbedtls_ssl_conf_min_tls_version(confs[i], MBEDTLS_SSL_VERSION_TLS1_3);
    mbedtls_ssl_conf_max_tls_version(confs[i], MBEDTLS_SSL_VERSION_TLS1_3);
    mbedtls_ssl_conf_tls13_key_exchange_modes(confs[i], MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL);
#else
    mbedtls_ssl_conf_min_version(confs[i], MBEDTLS_SSL_MAJOR_VERSION_3, MBEDTLS_SSL_MINOR_VERSION_3);
    mbedtls_ssl_conf_max_version(confs[i], MBEDTLS_SSL_MAJOR_VERSION_3, MBEDTLS_SSL_MINOR_VERSION_3);
#endif
    mbedtls_ssl_conf_ciphersuites(confs[i], ncclCryptCiphersuites);
    mbedtls_ssl_conf_authmode(confs[i], MBEDTLS_SSL_VERIFY_NONE);
    mbedtls_ssl_conf_renegotiation(confs[i], MBEDTLS_SSL_RENEGOTIATION_DISABLED);
    mbedtls_ssl_conf_rng(confs[i], mbedtls_ctr_drbg_random, &ncclCryptDrbg);
  }
  ncclCryptLibReady = true;
  return ncclSuccess;
}

// Called with ncclCryptLock held, after cryptInitLib.
static ncclResult_t cryptSetMasterKey(const unsigned char* key, size_t keyLen) {
  // Domain-separate the caller's key from any other use of the same secret
  // before handing it to TLS as the PSK.
  const char* info = "NCCL PSK domain separation key";
  unsigned char psk[NCCL_CRYPT_KEY_BYTES];
  const mbedtls_md_info_t* sha256 = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  int rc = mbedtls_hkdf(sha256, NULL, 0, key, keyLen,
                        (const unsigned char*)info, strlen(info), psk, sizeof(psk));
  if (rc != 0) {
    WARN("ncclCrypt: HKDF failed: -0x%04x", (unsigned int)-rc);
    return ncclSystemError;
  }
  mbedtls_ssl_config* confs[2] = { &ncclCryptClientConf, &ncclCryptServerConf };
  for (int i = 0; i < 2; i++) {
    rc = mbedtls_ssl_conf_psk(confs[i], psk, sizeof(psk),
                              (const unsigned char*)ncclCryptPskIdentity, strlen(ncclCryptPskIdentity));
    if (rc != 0) {
      WARN("ncclCrypt: mbedtls_ssl_conf_psk failed: -0x%04x", (unsigned int)-rc);
      mbedtls_platform_zeroize(psk, sizeof(psk));
      return ncclSystemError;
    }
  }
  mbedtls_platform_zeroize(psk, sizeof(psk));
  ncclCryptHaveKey = true;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclSetPSK, const void* key, size_t key_len);
ncclResult_t ncclSetPSK(const void* key, size_t key_len) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECK(PtrCheck((void*)key, "SetPSK", "key"));
  if (key_len < NCCL_PSK_MIN_BYTES) {
    WARN("ncclSetPSK : key length %zu is below the minimum of %d bytes", key_len, NCCL_PSK_MIN_BYTES);
    return ncclInvalidArgument;
  }
  pthread_mutex_lock(&ncclCryptLock);
  NCCLCHECKGOTO(cryptInitLib(), ret, exit);
  NCCLCHECKGOTO(cryptSetMasterKey((const unsigned char*)key, key_len), ret, exit);
exit:
  pthread_mutex_unlock(&ncclCryptLock);
  return ret;
}

ncclResult_t ncclGetCryptConnectionMode(int* encrypted) {
  ncclResult_t ret = ncclSuccess;
  const char* env = NULL;
  *encrypted = 1;
  pthread_mutex_lock(&ncclCryptLock);
  if (ncclCryptHaveKey) goto exit;
  NCCLCHECKGOTO(cryptInitLib(), ret, exit);
  env = ncclGetEnv("NCCL_PSK_FOR_TESTING_ONLY");
  if (env == NULL) {
    env = ncclGetEnv("NCCL_REQUIRE_TCP_ENCRYPTION");
    if (env != NULL && strcmp(env, "0") != 0) {
      WARN("NCCL_REQUIRE_TCP_ENCRYPTION is set but no pre-shared key is set; call ncclSetPSK() with a key of at least %d bytes before creating communicators.",
           NCCL_PSK_MIN_BYTES);
      ret = ncclInvalidUsage;
      goto exit;
    }
    *encrypted = 0;
    goto exit;
  }
  INFO(NCCL_ENV, "Using %zu-byte pre-shared key from NCCL_PSK_FOR_TESTING_ONLY", strlen(env));
  NCCLCHECKGOTO(cryptSetMasterKey((const unsigned char*)env, strlen(env)), ret, exit);
exit:
  pthread_mutex_unlock(&ncclCryptLock);
  if (ret == ncclSuccess && *encrypted) {
    env = ncclGetEnv("NCCL_OOB_NET_ENABLE");
    if (env != NULL && strtoll(env, NULL, 0) != 0) {  // same parsing as NCCL_PARAM
      WARN("NCCL_OOB_NET_ENABLE routes bootstrap traffic through the network plugin, bypassing TCP encryption; unset it or do not set a pre-shared key.");
      return ncclInvalidUsage;
    }
  }
  return ret;
}

// mbed TLS pulls and pushes handshake and record bytes through these callbacks,
// which route through socket.cc's raw progress helper so abort handling and the
// wire-byte counters keep working. The ncclSocket is passed per mbed TLS call
// (never captured at setup time) because callers copy ncclSocket structs by
// value; the heap crypto state must not point at a stale copy.
static int cryptBioSend(void* ctx, const unsigned char* buf, size_t len) {
  struct ncclSocket* sock = (struct ncclSocket*)ctx;
  int offset = 0;
  int closed = 0;
  if (len > 0x40000000) len = 0x40000000;  // limit writes to 1GiB to not overflow (int) cast
  ncclResult_t res = socketProgressOpt(NCCL_SOCKET_SEND, sock, (void*)buf, (int)len, &offset, 0, &closed);
  if (res != ncclSuccess) return MBEDTLS_ERR_NET_SEND_FAILED;
  if (closed) return MBEDTLS_ERR_NET_CONN_RESET;
  if (offset == 0) return MBEDTLS_ERR_SSL_WANT_WRITE;
  return offset;
}

static int cryptBioRecv(void* ctx, unsigned char* buf, size_t len) {
  struct ncclSocket* sock = (struct ncclSocket*)ctx;
  int offset = 0;
  int closed = 0;
  if (len > 0x40000000) len = 0x40000000;  // limit writes to 1GiB to not overflow (int) cast
  ncclResult_t res = socketProgressOpt(NCCL_SOCKET_RECV, sock, buf, (int)len, &offset, 0, &closed);
  if (res != ncclSuccess) return MBEDTLS_ERR_NET_RECV_FAILED;
  if (closed && offset == 0) return 0; // EOF
  if (offset == 0) return MBEDTLS_ERR_SSL_WANT_READ;
  return offset;
}

static ncclResult_t cryptEnsure(struct ncclSocket* sock, int isServer) {
  if (sock->crypto != NULL) return ncclSuccess;
  struct ncclSocketCrypto* c;
  NCCLCHECK(ncclCalloc(&c, 1));
  mbedtls_ssl_init(&c->ssl);
  c->isServer = isServer;
  int rc = mbedtls_ssl_setup(&c->ssl, isServer ? &ncclCryptServerConf : &ncclCryptClientConf);
  if (rc != 0) {
    WARN("ncclCrypt: mbedtls_ssl_setup failed: -0x%04x", (unsigned int)-rc);
    mbedtls_ssl_free(&c->ssl);
    free(c);
    return ncclSystemError;
  }
  sock->crypto = c;
  return ncclSuccess;
}

static int cryptClosedRc(int rc) {
  return rc == 0 || rc == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY || rc == MBEDTLS_ERR_SSL_CONN_EOF ||
         rc == MBEDTLS_ERR_NET_CONN_RESET;
}

static int cryptAgainRc(int rc) {
  return rc == MBEDTLS_ERR_SSL_WANT_READ || rc == MBEDTLS_ERR_SSL_WANT_WRITE;
}

// One non-blocking handshake step. *done is 1 once the handshake has completed,
// 0 if it still wants socket progress. A failed handshake returns ncclRemoteError.
static ncclResult_t cryptHandshakeStep(struct ncclSocket* sock, int* done) {
  struct ncclSocketCrypto* c = sock->crypto;
  *done = 0;
  if (c->handshakeDone) {
    *done = 1;
    return ncclSuccess;
  }
  mbedtls_ssl_set_bio(&c->ssl, sock, cryptBioSend, cryptBioRecv, NULL);
  int rc = mbedtls_ssl_handshake(&c->ssl);
  if (rc == 0) {
    c->handshakeDone = 1;
    *done = 1;
    return ncclSuccess;
  }
  if (cryptAgainRc(rc)) return ncclSuccess;
  cryptWarnMbedtls("ncclCrypt: TLS handshake", sock, rc);
  return ncclRemoteError;
}

ncclResult_t ncclCryptStartConnect(struct ncclSocket* sock) {
  int encrypted;
  NCCLCHECK(ncclGetCryptConnectionMode(&encrypted));
  if (encrypted) {
    NCCLCHECK(cryptEnsure(sock, 0 /*isServer*/));
    memcpy(sock->crypto->helloBuf, &sock->magic, sizeof(sock->magic));
    memcpy(sock->crypto->helloBuf + sizeof(sock->magic), &sock->type, sizeof(int));
  } else {
    memcpy(sock->finalizeBuffer, &sock->magic, sizeof(sock->magic));
    memcpy(sock->finalizeBuffer + sizeof(sock->magic), &sock->type, sizeof(int));
  }
  return ncclSuccess;
}

ncclResult_t ncclCryptConnectHello(struct ncclSocket* sock, int* done) {
  struct ncclSocketCrypto* c = sock->crypto;
  *done = 0;
  int hsDone;
  NCCLCHECK(cryptHandshakeStep(sock, &hsDone));
  if (!hsDone) return ncclSuccess;
  if (c->helloOffset < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
    NCCLCHECK(socketCryptoSend(sock, c->helloBuf, NCCL_SOCKET_PLAIN_HELLO_BYTES, &c->helloOffset, NULL));
    if (c->helloOffset < NCCL_SOCKET_PLAIN_HELLO_BYTES) return ncclSuccess;
  }
  *done = 1;
  return ncclSuccess;
}

ncclResult_t ncclCryptAcceptHello(struct ncclSocket* sock, int* verdict) {
  uint64_t magic;
  enum ncclSocketType type;
  char line[SOCKET_NAME_MAXLEN+1];
  *verdict = NCCL_CRYPT_HELLO_AGAIN;
  // Mode is resolved here, not at listen time, so creating a listener (ncclGetUniqueId) does not
  // require the key. But, the key must be set before any peer starts connecting. This is also where
  // NCCL_REQUIRE_TCP_ENCRYPTION without a key fails.
  int keyed;
  NCCLCHECK(ncclGetCryptConnectionMode(&keyed));

  char* hello;
  if (keyed) {
    NCCLCHECK(cryptEnsure(sock, 1 /*isServer*/));
    struct ncclSocketCrypto* c = sock->crypto;
    if (!c->handshakeDone) {
      int hsDone;
      // Anything that cannot complete the PSK handshake (plaintext stock peers,
      // port scanners, a peer with the wrong key) is dropped so one bad
      // connection cannot error out an accept loop.
      ncclResult_t res = cryptHandshakeStep(sock, &hsDone);
      if (res != ncclSuccess) {
        *verdict = NCCL_CRYPT_HELLO_RESET;
        return ncclSuccess;
      }
      if (!hsDone) return ncclSuccess;
    }
    if (c->helloOffset < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      int closed = 0;
      NCCLCHECK(socketCryptoRecv(sock, c->helloBuf, NCCL_SOCKET_PLAIN_HELLO_BYTES, &c->helloOffset, &closed));
      if (closed) {
        *verdict = NCCL_CRYPT_HELLO_RESET;
        return ncclSuccess;
      }
      if (c->helloOffset < NCCL_SOCKET_PLAIN_HELLO_BYTES) return ncclSuccess;
    }
    hello = c->helloBuf;
  } else {
    if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) {
      int closed = 0;
      // A recv error during the hello resets like a close (stock's synchronous
      // path did the same): one flaky connection must not kill the accept loop.
      ncclResult_t res = socketProgressOpt(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer,
                                           NCCL_SOCKET_PLAIN_HELLO_BYTES, &sock->finalizeCounter, 0, &closed);
      if (res == ncclRemoteError) closed = 1;
      else NCCLCHECK(res);
      if (closed) {
        *verdict = NCCL_CRYPT_HELLO_RESET;
        return ncclSuccess;
      }
      if (sock->finalizeCounter < NCCL_SOCKET_PLAIN_HELLO_BYTES) return ncclSuccess;
    }
    hello = sock->finalizeBuffer;
  }

  memcpy(&magic, hello, sizeof(magic));
  if (magic != sock->magic) {
    // A TLS ClientHello arriving at a plaintext listener also lands here: the
    // record header never matches the socket magic, so mixed configurations
    // reset instead of erroring out the accept loop.
    *verdict = NCCL_CRYPT_HELLO_RESET;
    return ncclSuccess;
  }
  memcpy(&type, hello + sizeof(magic), sizeof(int));
  if (type != sock->type) {
    INFO(NCCL_NET | NCCL_INIT,
         "ncclCryptAcceptHello from %s: wrong socket type (peer %d != expected %d -- peer connected to wrong NCCL "
         "socket), discarding peer connection",
         ncclSocketToString(&sock->addr, line), (int)type, (int)sock->type);
    *verdict = NCCL_CRYPT_HELLO_RESET;
    return ncclSuccess;
  }
  *verdict = NCCL_CRYPT_HELLO_READY;
  return ncclSuccess;
}

ncclResult_t socketCryptoSend(struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed) {
  struct ncclSocketCrypto* c = sock->crypto;
  char line[SOCKET_NAME_MAXLEN+1];
  mbedtls_ssl_set_bio(&c->ssl, sock, cryptBioSend, cryptBioRecv, NULL);
  while (*offset < size) {
    int rc = mbedtls_ssl_write(&c->ssl, (const unsigned char*)ptr + *offset, (size_t)(size - *offset));
    if (rc > 0) {
      *offset += rc;
      continue;
    }
    if (cryptAgainRc(rc)) return ncclSuccess;
    if (cryptClosedRc(rc)) {
      if (pclosed) {
        *pclosed = 1;
        return ncclSuccess;
      }
      WARN("socketCryptoSend: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line));
      return ncclRemoteError;
    }
    cryptWarnMbedtls("socketCryptoSend", sock, rc);
    return ncclRemoteError;
  }
  return ncclSuccess;
}

ncclResult_t socketCryptoRecv(struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed) {
  struct ncclSocketCrypto* c = sock->crypto;
  char line[SOCKET_NAME_MAXLEN+1];
  mbedtls_ssl_set_bio(&c->ssl, sock, cryptBioSend, cryptBioRecv, NULL);
  while (*offset < size) {
    int rc = mbedtls_ssl_read(&c->ssl, (unsigned char*)ptr + *offset, (size_t)(size - *offset));
    if (rc > 0) {
      *offset += rc;
      continue;
    }
    if (cryptAgainRc(rc)) return ncclSuccess;
    if (cryptClosedRc(rc)) {
      if (pclosed) {
        *pclosed = 1;
        return ncclSuccess;
      }
      WARN("socketCryptoRecv: Connection closed by remote peer %s", ncclSocketToString(&sock->addr, line));
      return ncclRemoteError;
    }
    cryptWarnMbedtls("socketCryptoRecv", sock, rc);
    return ncclRemoteError;
  }
  return ncclSuccess;
}

void ncclCryptFree(struct ncclSocketCrypto* crypto) {
  if (crypto == NULL) return;
  mbedtls_ssl_free(&crypto->ssl);
  free(crypto);
}
