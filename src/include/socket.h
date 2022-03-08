/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include "nccl.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum ncclSocketState {
  ncclSocketConnecting = 0,
  ncclSocketConnected = 1,
  ncclSocketError = 2,
  ncclSocketStateNum = 3
} ;

struct ncclSocket {
  int fd;
  union ncclSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum ncclSocketState state;
};

const char *ncclSocketToString(union ncclSocketAddress *addr, char *buf, const int numericHostForm = 1);
ncclResult_t ncclGetSocketAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);
int ncclFindInterfaceMatchSubnet(char* ifNames, union ncclSocketAddress* localAddrs, union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs);
int ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
ncclResult_t ncclSocketListen(struct ncclSocket* sock);
// Connect to sock->addr. sock->fd is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);
// Return socket connection state.
ncclResult_t ncclGetSocketState(struct ncclSocket* sock, enum ncclSocketState* state);
// Accept an incoming connection from listenSocket->fd and keep the file descriptor in sock->fd, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSocket);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed);
/* initialize a socket. */
ncclResult_t ncclSocketInit(struct ncclSocket* sock, union ncclSocketAddress* addr = NULL, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0);
#endif
