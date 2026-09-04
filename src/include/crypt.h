#ifndef NCCL_CRYPT_H_
#define NCCL_CRYPT_H_

#include "nccl.h"
#include "socket.h"
#include <stdint.h>

#define NCCL_CRYPT_KEY_BYTES 32

// *encrypted is true if ncclSetEncryption() configured a PSK; default is plaintext.
ncclResult_t ncclGetCryptConnectionMode(bool* encrypted);

// Rebind the opaque TLS state after its owning ncclSocket moves.
void ncclCryptRebindSocket(struct ncclSocket* sock);

// Select the connect-side crypt mode and prepare the NCCL magic/type hello.
ncclResult_t ncclCryptStartConnect(struct ncclSocket* sock);

// After the TCP connect completes, progress the TLS handshake when enabled and
// send the hello. Returns ncclInProgress while either operation still needs progress.
ncclResult_t ncclCryptConnectHello(struct ncclSocket* sock);

// Accept side: drives the server handshake (creating the server state on first
// call) and then reads and validates the connector's hello through the channel.
// In plaintext mode it reads and validates the stock hello instead.
enum ncclCryptHelloVerdict {
  NCCL_CRYPT_HELLO_VERDICT_AGAIN = 0, // needs more socket progress
  NCCL_CRYPT_HELLO_VERDICT_READY = 1, // connection ready
  NCCL_CRYPT_HELLO_VERDICT_RESET = 2  // silently drop and re-accept
};
ncclResult_t ncclCryptAcceptHello(struct ncclSocket* sock, enum ncclCryptHelloVerdict* verdict);

// Encrypted-socket progress: TLS-record the outgoing bytes, unrecord the
// incoming ones. Same contract as ncclSocketProgress.
ncclResult_t ncclCryptSocketSend(struct ncclSocket* sock, void* ptr, int size, int* offset, bool* pclosed);
ncclResult_t ncclCryptSocketRecv(struct ncclSocket* sock, void* ptr, int size, int* offset, bool* pclosed);

// Free the opaque state allocated by cryptEnsure() and stored in ncclSocket::crypto.
void ncclCryptFree(struct ncclSocketCrypto* crypto);

#endif
