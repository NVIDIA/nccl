/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket.h"
#include "utils.h"
#include "os.h"
#include <stdlib.h>

#include <unistd.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "param.h"
#include <time.h>
#include <atomic>

NCCL_PARAM(RetryCnt, "SOCKET_RETRY_CNT", 34);
NCCL_PARAM(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);
NCCL_PARAM(PollTimeOut, "SOCKET_POLL_TIMEOUT_MSEC", 0);
NCCL_PARAM(SocketMaxRecvBuff, "SOCKET_RCVBUF", -1);
NCCL_PARAM(SocketMaxSendBuff, "SOCKET_SNDBUF", -1);

static ncclResult_t socketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* pclosed = NULL) {
  int closed;
  NCCLCHECK(ncclOsSocketProgressOpt(op, sock, ptr, size, offset, 0, &closed));
  if (closed) {
    if (pclosed) {
      *pclosed = closed;
      return ncclSuccess;
    } else {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("socketProgress: Connection closed by remote peer %s",
           ncclSocketToString(&sock->addr, line, /*numericHostForm*/0));
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

static ncclResult_t socketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  while (*offset < size) {
    NCCLCHECK(socketProgress(op, sock, ptr, size, offset));
    // If we have more data to read or write, use the poll system call to wait
    // until the socket becomes readable or writable again.
    if ((*offset < size) && ncclParamPollTimeOut()) {
      ncclOsPollSocket(sock->socketDescriptor, op);
    }
  }
  return ncclSuccess;
}

uint16_t ncclSocketToPort(union ncclSocketAddress *addr) {
  return ntohs(addr->sa.sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how) {
  if (sock != NULL) {
    if (ncclOsSocketIsValid(sock)) {
      SYSCHECK(shutdown(sock->socketDescriptor, how), "shutdown");
    }
    sock->state = ncclSocketStateTerminating;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* socketDescriptor) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (socketDescriptor) *socketDescriptor = sock->socketDescriptor;
  return ncclSuccess;
}

ncclResult_t ncclSocketSetFd(ncclSocketDescriptor socketDescriptor, struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketSetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  sock->socketDescriptor = socketDescriptor;
  return ncclSuccess;
}


ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclSocketListen: socket is invalid");
    return ncclInvalidArgument;
  }

  if (ncclSocketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // addr port should be 0 (Any port)
  SYSCHECK(bind(sock->socketDescriptor, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->socketDescriptor, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  SYSCHECK(listen(sock->socketDescriptor, 16384), "listen");

  // Set acceptSocketDescriptor to the same value as socketDescriptor for listening sockets
  sock->acceptSocketDescriptor = sock->socketDescriptor;
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

/* Format a string representation of a (union ncclSocketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm /*= 1*/) {
  const struct sockaddr *saddr = &addr->sa;
  char host[NI_MAXHOST], service[NI_MAXSERV];
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  if (buf == NULL || addr == NULL) goto fail;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) goto fail;
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  if (getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag)) goto fail;
  sprintf(buf, "%s<%s>", host, service);
  return buf;
fail:
  if (buf)
    buf[0] = '\0';
  return buf;
}

/* Allow the user to force the IPv4/IPv6 interface selection */
int ncclEnvSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char* env = ncclGetEnv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  INFO(NCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs,
  int* nIfs) {
  static int shownIfName = 0;
  // Allow user to force the INET socket family selection
  int sock_family = ncclEnvSocketFamily();
  // User specified interface
  const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");
  *nIfs = 0;
  if (env && strlen(env) > 1) {
    INFO(NCCL_ENV, "NCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0) INFO(NCCL_NET, "NCCL_SOCKET_IFNAME set to %s", env);
    NCCLCHECK(ncclOsFindInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
  } else {
    // Try to automatically pick the right one
    // Start with IB
    NCCLCHECK(ncclOsFindInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    // else see if we can get some hint from COMM ID
    if (*nIfs == 0) {
      const char* commId = ncclGetEnv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union ncclSocketAddress idAddr;
        NCCLCHECK(ncclSocketGetAddrFromString(&idAddr, commId));
        NCCLCHECK(ncclFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr, ifNameMaxSize, nIfs));
      }
    }
    // Then look for anything else (but not docker,lo, or virtual)
    if (*nIfs == 0) NCCLCHECK(ncclOsFindInterfaces("^docker,lo,virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    // Finally look for docker, then lo.
    if (*nIfs == 0) NCCLCHECK(ncclOsFindInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) NCCLCHECK(ncclOsFindInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) NCCLCHECK(ncclOsFindInterfaces("virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      freeaddrinfo(p);
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = atoi(port_str);
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (sock == NULL) {
    WARN("ncclSocketGetAddr: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  memcpy(addr, &sock->addr, sizeof(union ncclSocketAddress));
  return ncclSuccess;
}

static ncclResult_t socketFinalizeAccept(struct ncclSocket* sock) {
  uint64_t magic;
  enum ncclSocketType type;
  int received;
  char line[SOCKET_NAME_MAXLEN+1];
  // once accepted, linux sockets do NOT inherit file status flags such as O_NONBLOCK (BSD ones do)
  NCCLCHECK(ncclOsSocketSetFlags(sock));

  if (sock->asyncFlag == 0 || sock->finalizeCounter < sizeof(magic)) {
    if (sock->asyncFlag == 0) {
      received = 0;
      if (socketWait(NCCL_SOCKET_RECV, sock, &magic, sizeof(magic), &received) != ncclSuccess) {
        ncclOsSocketResetAccept(sock);
        return ncclSuccess;
      }
    } else {
      int closed = 0;
      received = sock->finalizeCounter;
      NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(magic), &received, &closed));
      sock->finalizeCounter = received;
      if (received < sizeof(magic)) {
        if (closed) {
          ncclOsSocketResetAccept(sock);
        }
        return ncclSuccess;
      }
      memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
    }
    if (magic != sock->magic) {
      ncclOsSocketResetAccept(sock);
      return ncclSuccess;
    }
  }
  if (sock->asyncFlag == 0) {
    received = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, &type, sizeof(type), &received));
  } else {
    received = sock->finalizeCounter - sizeof(magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(type), &received));
    sock->finalizeCounter = received + sizeof(magic);
    if (received < sizeof(type)) return ncclSuccess;
    memcpy(&type, sock->finalizeBuffer, sizeof(type));
  }
  if (type != sock->type) {
    WARN("socketFinalizeAccept from %s: wrong type %d != %d", ncclSocketToString(&sock->addr, line), type, sock->type);
    (void) ncclSocketClose(sock);
    sock->state = ncclSocketStateError;
    return ncclInternalError;
  } else {
    sock->state = ncclSocketStateReady;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketPollConnect(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketPollConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclOsSocketPollConnect(sock));
  return ncclSuccess;
}

static ncclResult_t socketFinalizeConnect(struct ncclSocket* sock) {
  int sent;
  if (sock->asyncFlag == 0) {
    sent = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
    sent = 0;
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
  } else {
    if (sock->finalizeCounter < sizeof(sock->magic)) {
      sent = sock->finalizeCounter;
      NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
      sock->finalizeCounter = sent;
      if (sent < sizeof(sock->magic)) return ncclSuccess;
    }
    sent = sock->finalizeCounter - sizeof(sock->magic);
    NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
    sock->finalizeCounter = sent + sizeof(sock->magic);
    if (sent < sizeof(sock->type)) return ncclSuccess;
  }
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

static ncclResult_t socketProgressState(struct ncclSocket* sock) {
  if (sock->state == ncclSocketStateAccepting) {
    NCCLCHECK(ncclOsSocketTryAccept(sock));
  }
  if (sock->state == ncclSocketStateAccepted) {
    NCCLCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == ncclSocketStateConnecting) {
    NCCLCHECK(ncclOsSocketStartConnect(sock));
  }
  if (sock->state == ncclSocketStateConnectPolling) {
    NCCLCHECK(ncclOsSocketPollConnect(sock));
  }
  if (sock->state == ncclSocketStateConnected) {
    NCCLCHECK(socketFinalizeConnect(sock));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running) {
  if (sock == NULL) {
    *running = 0;
    return ncclSuccess;
  }
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) {
    WARN("ncclSocketReady: unexpected socket state %d", sock->state);
    return ncclRemoteError;
  }
  *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  if (*running == 0) {
    NCCLCHECK(socketProgressState(sock));
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
#endif

  if (sock == NULL) {
    WARN("ncclSocketConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclSocketConnect: socket is invalid");
    return ncclInvalidArgument;
  }

  if (sock->state != ncclSocketStateInitialized) {
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == ncclSocketStateError) return ncclRemoteError;
    return ncclInternalError;
  }
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", ncclSocketToString(&sock->addr, line));

  sock->state = ncclSocketStateConnecting;
  sock->finalizeCounter = 0;
  do {
    NCCLCHECK(socketProgressState(sock));
  } while (sock->asyncFlag == 0 &&
      (sock->abortFlag == NULL || COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire) == 0) &&
      (sock->state == ncclSocketStateConnecting ||
       sock->state == ncclSocketStateConnectPolling ||
       sock->state == ncclSocketStateConnected));

  if (sock->abortFlag && COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire)) return ncclInternalError;

  switch (sock->state) {
    case ncclSocketStateConnecting:
    case ncclSocketStateConnectPolling:
    case ncclSocketStateConnected:
    case ncclSocketStateReady:
      return ncclSuccess;
    case ncclSocketStateError:
      return ncclSystemError;
    default:
      WARN("ncclSocketConnect: wrong socket state %d", sock->state);
      return ncclInternalError;
  }
}

ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock) {
  ncclResult_t ret = ncclSuccess;

  if (listenSock == NULL || sock == NULL) {
    WARN("ncclSocketAccept: pass NULL socket");
    ret = ncclInvalidArgument;
    goto exit;
  }
  if (listenSock->state != ncclSocketStateReady) {
    WARN("ncclSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == ncclSocketStateError)
      ret = ncclSystemError;
    else
      ret = ncclInternalError;
    goto exit;
  }

  if (!ncclOsSocketDescriptorIsValid(sock->acceptSocketDescriptor)) {
    memcpy(sock, listenSock, sizeof(struct ncclSocket));
    sock->acceptSocketDescriptor = listenSock->acceptSocketDescriptor;
    sock->state = ncclSocketStateAccepting;
    sock->finalizeCounter = 0;
  }

  do {
    NCCLCHECKGOTO(socketProgressState(sock), ret, exit);
  } while (sock->asyncFlag == 0 &&
      (sock->abortFlag == NULL || COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire) == 0) &&
      (sock->state == ncclSocketStateAccepting ||
       sock->state == ncclSocketStateAccepted));

  if (sock->abortFlag && COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire)) return ncclInternalError;

  switch (sock->state) {
    case ncclSocketStateAccepting:
    case ncclSocketStateAccepted:
    case ncclSocketStateReady:
      ret = ncclSuccess;
      break;
    case ncclSocketStateError:
      ret = ncclSystemError;
      break;
    default:
      WARN("ncclSocketAccept: wrong socket state %d", sock->state);
      ret = ncclInternalError;
      break;
  }

exit:
  return ret;
}

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic, enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ncclResult_t ret = ncclSuccess;

  if (sock == NULL) goto exit;
  sock->errorRetries = 0;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = ncclSocketStateInitialized;
  sock->magic = magic;
  sock->type = type;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  sock->customRetry = customRetry;
#ifdef NCCL_OS_WINDOWS
  sock->socketBlockingMode = 1;
#endif

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&sock->addr, addr, sizeof(union ncclSocketAddress));
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("ncclSocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)",
          ncclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = ncclInternalError;
      goto exit;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    // in case of error, we close the descriptor before returning as it's unclear if the caller has to use ncclSocketClose for cleanup
    NCCLCHECKGOTO(ncclOsSocketResetFd(sock), ret, fail);
  } else {
    memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
  }
exit:
  return ret;
fail:
  (void) ncclSocketClose(sock);
  goto exit;
}

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  if (sock == NULL) {
    WARN("ncclSocketProgress: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketProgress(op, sock, ptr, size, offset, closed));
  return ncclSuccess;
}

ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  if (sock == NULL) {
    WARN("ncclSocketWait: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketWait(op, sock, ptr, size, offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketSend: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) {
    WARN("ncclSocketSend: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady && sock->state != ncclSocketStateTerminating) {
    WARN("ncclSocketRecv: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock, void* recvPtr, int recvSize) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    WARN("ncclSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return ncclInternalError;
  }
  if (sendSock->state != ncclSocketStateReady ||
      (recvSock->state != ncclSocketStateReady && recvSock->state != ncclSocketStateTerminating)) {
    WARN("ncclSocketSendRecv: socket state (%d/%d) is not ready", sendSock->state, recvSock->state);
    return ncclInternalError;
  }
  while (sendOffset < sendSize || recvOffset < recvSize) {
    if (sendOffset < sendSize) NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sendSock, sendPtr, sendSize, &sendOffset));
    if (recvOffset < recvSize) NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, recvSock, recvPtr, recvSize, &recvOffset));
  }
  return ncclSuccess;
}


ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps) {
  if (ops == NULL || numOps <= 0) {
    WARN("ncclSocketMultiOp: invalid arguments ops=%p numOps=%d", ops, numOps);
    return ncclInvalidArgument;
  }

  for (int i = 0; i < numOps; i++) {
    if (ops[i].sock == NULL) {
      WARN("ncclSocketMultiOp: invalid socket at index %d", i);
      return ncclInvalidArgument;
    }
    ops[i].offset = 0;
  }
  int completedOps=0, i=0;
  while(completedOps < numOps){
    if (ops[i].offset < ops[i].size){
      NCCLCHECK(socketProgress(ops[i].op, ops[i].sock, ops[i].ptr, ops[i].size, &ops[i].offset));
      if(ops[i].offset >= ops[i].size) completedOps++;
    }
    i=(i+1)%numOps;
  }
  return ncclSuccess;
}
// Receive or detect connection closed
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketTryRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  *closed = 0;
  // Block until connection closes or nbytes received
  if (blocking) {
    while (offset < size) {
      NCCLCHECK(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
      if (*closed) return ncclSuccess;
    }
  } else {
    NCCLCHECK(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
    if (*closed) return ncclSuccess;

    // If any bytes were received, block waiting for the rest
    if (offset > 0) {
      while (offset < size) {
        NCCLCHECK(ncclOsSocketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, 0, closed));
        if (*closed) return ncclSuccess;
      }
    // No bytes were received, return ncclInProgress
    } else {
      return ncclInProgress;
    }
  }
  return ncclSuccess;
}
