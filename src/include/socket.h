/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include "nccl.h"
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include "os.h"

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum ncclSocketState {
  ncclSocketStateNone = 0,
  ncclSocketStateInitialized = 1,
  ncclSocketStateAccepting = 2,
  ncclSocketStateAccepted = 3,
  ncclSocketStateConnecting = 4,
  ncclSocketStateConnectPolling = 5,
  ncclSocketStateConnected = 6,
  ncclSocketStateReady = 7,
  ncclSocketStateTerminating = 8,
  ncclSocketStateClosed = 9,
  ncclSocketStateError = 10,
  ncclSocketStateNum = 11
};

enum ncclSocketType {
  ncclSocketTypeUnknown = 0,
  ncclSocketTypeBootstrap = 1,
  ncclSocketTypeProxy = 2,
  ncclSocketTypeNetSocket = 3,
  ncclSocketTypeNetIb = 4,
  ncclSocketTypeRasNetwork = 5
};

struct ncclSocket {
  ncclSocketDescriptor socketDescriptor;
  ncclSocketDescriptor acceptSocketDescriptor;
  int errorRetries;
  union ncclSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum ncclSocketState state;
  int salen;
  uint64_t magic;
  enum ncclSocketType type;
  int customRetry;
  int finalizeCounter; // Used to keep track of initial handshake for async sockets.
  char finalizeBuffer[sizeof(uint64_t)]; // Used to keep track of initial handshake for async sockets.
#ifdef NCCL_OS_WINDOWS
  int socketBlockingMode;  // 0 - blocking mode; 1 - non-blocking mode
#endif
};
struct ncclSocketOp {
  int op;                    // NCCL_SOCKET_SEND or NCCL_SOCKET_RECV
  struct ncclSocket* sock;   // Socket to operate on
  void* ptr;                 // Data pointer
  int size;                  // Size of data
  int offset;                // Current progress offset
};
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm = 1);
ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found);
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs);

// Initialize a socket
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr = NULL, uint64_t magic = NCCL_SOCKET_MAGIC, enum ncclSocketType type = ncclSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0, int customRetry = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
ncclResult_t ncclSocketListen(struct ncclSocket* sock);
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr);
// Connect to sock->addr. sock->socketDescriptor is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);
// Return socket connection state.
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running);
// Accept an incoming connection from listenSock->socketDescriptor and keep the file descriptor in sock->socketDescriptor, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock);
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* socketDescriptor);
ncclResult_t ncclSocketSetFd(ncclSocketDescriptor socketDescriptor, struct ncclSocket* sock);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

int ncclEnvSocketFamily(void);
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed = NULL);
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize);
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps);
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking);
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how);
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait = false);
uint16_t ncclSocketToPort(union ncclSocketAddress *addr);
#endif
