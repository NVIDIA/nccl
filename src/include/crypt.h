#ifndef NCCL_CRYPT_H_
#define NCCL_CRYPT_H_

#include "nccl.h"
#include <stdint.h>

#define NCCL_CRYPT_KEY_BYTES 32

struct ncclSocket;
// Per-socket TLS state (mbed TLS contexts); defined in crypt.cc so mbedtls
// headers stay out of the socket layer.
struct ncclSocketCrypto;

// *encrypted is 1 if a pre-shared key is available (ncclSetPSK, or
// NCCL_PSK_FOR_TESTING_ONLY read on first call) and sockets must encrypt, 0 if no
// key is set and sockets run in plain NCCL mode. Returns ncclInvalidUsage instead
// when no key is set but NCCL_REQUIRE_TCP_ENCRYPTION is set to anything but "0",
// or when a key is in use while NCCL_OOB_NET_ENABLE would bypass it.
ncclResult_t ncclGetCryptConnectionMode(int* encrypted);

// Raw nonblocking socket I/O; defined in src/misc/socket.cc, non-static so the
// TLS BIO callbacks can use it and its byte counters.
ncclResult_t socketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed);

// Connect side. ncclCryptStartConnect picks the mode for this socket and queues
// the plaintext hello; ncclCryptConnectHello then runs after the TCP connect
// completes: it drives the TLS handshake (creating the client state on first
// call) and sends the hello through the channel. *done stays 0 while the
// handshake or the hello send still wants socket progress.
ncclResult_t ncclCryptStartConnect(struct ncclSocket* sock);
ncclResult_t ncclCryptConnectHello(struct ncclSocket* sock, int* done);

// Accept side: drives the server handshake (creating the server state on first
// call) and then reads and validates the connector's hello through the channel.
// In plaintext mode it reads and validates the stock hello instead.
#define NCCL_CRYPT_HELLO_AGAIN 0 // needs more socket progress
#define NCCL_CRYPT_HELLO_READY 1 // connection ready
#define NCCL_CRYPT_HELLO_RESET 2 // silently drop and re-accept
ncclResult_t ncclCryptAcceptHello(struct ncclSocket* sock, int* verdict);

// Encrypted-socket progress: TLS-record the outgoing bytes, unrecord the
// incoming ones. Same contract as socket.cc's socketProgress.
ncclResult_t socketCryptoSend(struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed);
ncclResult_t socketCryptoRecv(struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed);

void ncclCryptFree(struct ncclSocketCrypto* crypto);

#endif
