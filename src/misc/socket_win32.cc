/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Windows socket implementation using Winsock2.
 *
 * This provides socket functionality for NCCL on Windows, enabling
 * multi-process communication. Uses Winsock2 TCP/IP sockets.
 */

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS

#include "socket.h"
#include "checks.h"
#include "debug.h"
#include "param.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <mstcpip.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

NCCL_PARAM(SocketRetryCnt, "SOCKET_RETRY_CNT", 34);
NCCL_PARAM(SocketRetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);

// Initialize/cleanup Winsock
static bool wsaInitialized = false;
static CRITICAL_SECTION wsaInitLock;
static bool wsaLockInitialized = false;

static void ensureLockInitialized()
{
    if (!wsaLockInitialized)
    {
        InitializeCriticalSection(&wsaInitLock);
        wsaLockInitialized = true;
    }
}

static ncclResult_t ncclSocketInitWinsock()
{
    ensureLockInitialized();
    EnterCriticalSection(&wsaInitLock);
    if (!wsaInitialized)
    {
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0)
        {
            LeaveCriticalSection(&wsaInitLock);
            WARN("WSAStartup failed with error: %d", result);
            return ncclSystemError;
        }
        wsaInitialized = true;
    }
    LeaveCriticalSection(&wsaInitLock);
    return ncclSuccess;
}

void ncclSocketFinalize()
{
    if (wsaInitialized)
    {
        WSACleanup();
        wsaInitialized = false;
    }
}

// Thread-local buffer for ncclSocketToString
#define NCCL_SOCKET_MAX_STR_LEN 1064
static thread_local char socketToStringBuf[NCCL_SOCKET_MAX_STR_LEN];

const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm)
{
    if (buf == NULL)
        buf = socketToStringBuf;
    if (addr == NULL)
    {
        snprintf(buf, NCCL_SOCKET_MAX_STR_LEN, "NULL");
        return buf;
    }

    char host[NI_MAXHOST], service[NI_MAXSERV];
    int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);

    if (addr->sa.sa_family != AF_INET && addr->sa.sa_family != AF_INET6)
    {
        snprintf(buf, NCCL_SOCKET_MAX_STR_LEN, "unknown-af(%d)", addr->sa.sa_family);
        return buf;
    }

    if (getnameinfo(&addr->sa, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag) != 0)
    {
        snprintf(buf, NCCL_SOCKET_MAX_STR_LEN, "unknown");
        return buf;
    }

    snprintf(buf, NCCL_SOCKET_MAX_STR_LEN, "%s<%s>", host, service);
    return buf;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress *ua, const char *ip_port_pair)
{
    if (ua == NULL || ip_port_pair == NULL)
        return ncclInvalidArgument;

    NCCLCHECK(ncclSocketInitWinsock());

    memset(ua, 0, sizeof(*ua));

    // Parse "host:port" or "[host]:port" for IPv6
    char host[NI_MAXHOST];
    char port[NI_MAXSERV];
    const char *portStart = NULL;

    if (ip_port_pair[0] == '[')
    {
        // IPv6 format: [host]:port
        const char *bracket = strchr(ip_port_pair, ']');
        if (bracket == NULL)
            return ncclInvalidArgument;
        size_t hostLen = bracket - ip_port_pair - 1;
        if (hostLen >= sizeof(host))
            return ncclInvalidArgument;
        strncpy(host, ip_port_pair + 1, hostLen);
        host[hostLen] = '\0';
        if (bracket[1] == ':')
            portStart = bracket + 2;
    }
    else
    {
        // IPv4 format: host:port
        const char *colon = strrchr(ip_port_pair, ':');
        if (colon == NULL)
        {
            strncpy(host, ip_port_pair, sizeof(host) - 1);
            host[sizeof(host) - 1] = '\0';
        }
        else
        {
            size_t hostLen = colon - ip_port_pair;
            if (hostLen >= sizeof(host))
                return ncclInvalidArgument;
            strncpy(host, ip_port_pair, hostLen);
            host[hostLen] = '\0';
            portStart = colon + 1;
        }
    }

    if (portStart)
    {
        strncpy(port, portStart, sizeof(port) - 1);
        port[sizeof(port) - 1] = '\0';
    }
    else
    {
        port[0] = '0';
        port[1] = '\0';
    }

    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int ret = getaddrinfo(host, port, &hints, &res);
    if (ret != 0)
    {
        WARN("getaddrinfo(%s, %s) failed: %d", host, port, ret);
        return ncclInvalidArgument;
    }

    if (res->ai_family == AF_INET)
    {
        memcpy(&ua->sin, res->ai_addr, sizeof(struct sockaddr_in));
    }
    else if (res->ai_family == AF_INET6)
    {
        memcpy(&ua->sin6, res->ai_addr, sizeof(struct sockaddr_in6));
    }
    else
    {
        freeaddrinfo(res);
        return ncclInvalidArgument;
    }

    freeaddrinfo(res);
    return ncclSuccess;
}

int ncclSocketGetPort(union ncclSocketAddress *addr)
{
    if (addr == NULL)
        return 0;
    if (addr->sa.sa_family == AF_INET)
        return ntohs(addr->sin.sin_port);
    if (addr->sa.sa_family == AF_INET6)
        return ntohs(addr->sin6.sin6_port);
    return 0;
}

static ncclResult_t setSocketNonBlocking(SOCKET fd, bool nonBlocking)
{
    u_long mode = nonBlocking ? 1 : 0;
    if (ioctlsocket(fd, FIONBIO, &mode) != 0)
    {
        WARN("ioctlsocket(FIONBIO) failed: %d", WSAGetLastError());
        return ncclSystemError;
    }
    return ncclSuccess;
}

static ncclResult_t setSocketOptions(SOCKET fd)
{
    // Enable TCP_NODELAY
    int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(one)) != 0)
    {
        WARN("setsockopt(TCP_NODELAY) failed: %d", WSAGetLastError());
    }

    // Enable SO_REUSEADDR
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char *)&one, sizeof(one)) != 0)
    {
        WARN("setsockopt(SO_REUSEADDR) failed: %d", WSAGetLastError());
    }

    return ncclSuccess;
}

ncclResult_t ncclSocketInit(struct ncclSocket *sock, const union ncclSocketAddress *addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t *abortFlag, int asyncFlag, int customRetry)
{
    NCCLCHECK(ncclSocketInitWinsock());

    if (sock == NULL)
        return ncclInvalidArgument;

    memset(sock, 0, sizeof(*sock));
    sock->fd = INVALID_SOCKET;
    sock->acceptFd = INVALID_SOCKET;
    sock->magic = magic;
    sock->type = type;
    sock->abortFlag = abortFlag;
    sock->asyncFlag = asyncFlag;
    sock->customRetry = customRetry;
    sock->state = ncclSocketStateInitialized;

    if (addr)
    {
        memcpy(&sock->addr, addr, sizeof(sock->addr));
        sock->salen = (addr->sa.sa_family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    }

    return ncclSuccess;
}

ncclResult_t ncclSocketListen(struct ncclSocket *sock)
{
    if (sock == NULL)
        return ncclInvalidArgument;

    NCCLCHECK(ncclSocketInitWinsock());

    int family = sock->addr.sa.sa_family;
    if (family == 0)
        family = AF_INET;

    sock->fd = socket(family, SOCK_STREAM, IPPROTO_TCP);
    if (sock->fd == INVALID_SOCKET)
    {
        WARN("socket() failed: %d", WSAGetLastError());
        return ncclSystemError;
    }

    NCCLCHECK(setSocketOptions(sock->fd));

    if (bind(sock->fd, &sock->addr.sa, sock->salen) != 0)
    {
        WARN("bind() failed: %d", WSAGetLastError());
        closesocket(sock->fd);
        sock->fd = INVALID_SOCKET;
        return ncclSystemError;
    }

    // Get assigned port if port was 0
    if (ncclSocketGetPort(&sock->addr) == 0)
    {
        int addrLen = sizeof(sock->addr);
        getsockname(sock->fd, &sock->addr.sa, &addrLen);
    }

    if (listen(sock->fd, SOMAXCONN) != 0)
    {
        WARN("listen() failed: %d", WSAGetLastError());
        closesocket(sock->fd);
        sock->fd = INVALID_SOCKET;
        return ncclSystemError;
    }

    sock->state = ncclSocketStateReady;

    char line[NCCL_SOCKET_MAX_STR_LEN];
    TRACE(NCCL_INIT | NCCL_NET, "Listening on socket %s", ncclSocketToString(&sock->addr, line));

    return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket *sock)
{
    if (sock == NULL)
        return ncclInvalidArgument;

    NCCLCHECK(ncclSocketInitWinsock());

    int family = sock->addr.sa.sa_family;
    if (family == 0)
    {
        WARN("Socket address family not set");
        return ncclInvalidArgument;
    }

    sock->fd = socket(family, SOCK_STREAM, IPPROTO_TCP);
    if (sock->fd == INVALID_SOCKET)
    {
        WARN("socket() failed: %d", WSAGetLastError());
        return ncclSystemError;
    }

    NCCLCHECK(setSocketOptions(sock->fd));

    if (sock->asyncFlag)
    {
        NCCLCHECK(setSocketNonBlocking(sock->fd, true));
    }

    char line[NCCL_SOCKET_MAX_STR_LEN];
    TRACE(NCCL_INIT | NCCL_NET, "Connecting to socket %s", ncclSocketToString(&sock->addr, line));

    sock->state = ncclSocketStateConnecting;

    int ret = connect(sock->fd, &sock->addr.sa, sock->salen);
    if (ret != 0)
    {
        int wsaErr = WSAGetLastError();
        if (wsaErr == WSAEWOULDBLOCK || wsaErr == WSAEINPROGRESS)
        {
            sock->state = ncclSocketStateConnectPolling;
            return ncclSuccess;
        }
        WARN("connect() failed: %d", wsaErr);
        closesocket(sock->fd);
        sock->fd = INVALID_SOCKET;
        return ncclSystemError;
    }

    sock->state = ncclSocketStateConnected;
    return ncclSuccess;
}

ncclResult_t ncclSocketAccept(struct ncclSocket *sock, struct ncclSocket *listenSock)
{
    if (sock == NULL || listenSock == NULL)
        return ncclInvalidArgument;

    memset(sock, 0, sizeof(*sock));
    sock->magic = listenSock->magic;
    sock->type = listenSock->type;
    sock->abortFlag = listenSock->abortFlag;
    sock->asyncFlag = listenSock->asyncFlag;
    sock->salen = sizeof(sock->addr);

    sock->fd = accept(listenSock->fd, &sock->addr.sa, &sock->salen);
    if (sock->fd == INVALID_SOCKET)
    {
        int wsaErr = WSAGetLastError();
        if (wsaErr == WSAEWOULDBLOCK)
        {
            sock->state = ncclSocketStateAccepting;
            return ncclSuccess;
        }
        WARN("accept() failed: %d", wsaErr);
        return ncclSystemError;
    }

    NCCLCHECK(setSocketOptions(sock->fd));
    sock->state = ncclSocketStateAccepted;

    char line[NCCL_SOCKET_MAX_STR_LEN];
    TRACE(NCCL_INIT | NCCL_NET, "Accepted connection from %s", ncclSocketToString(&sock->addr, line));

    return ncclSuccess;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket *sock, union ncclSocketAddress *addr)
{
    if (sock == NULL || addr == NULL)
        return ncclInvalidArgument;
    memcpy(addr, &sock->addr, sizeof(*addr));
    return ncclSuccess;
}

// Forward declarations for socket progress functions
static ncclResult_t socketProgressOpt(int op, struct ncclSocket *sock, void *ptr, int size, int *offset, int block, int *closed);
ncclResult_t ncclSocketWait(int op, struct ncclSocket *sock, void *ptr, int size, int *offset);

// Send magic and type bytes to finalize client-side connection
// Only for transport sockets, not proxy sockets which use their own protocol
static ncclResult_t socketFinalizeConnect(struct ncclSocket *sock)
{
    // Proxy sockets don't use the magic/type handshake - they have their own message protocol
    // Only transport sockets (NetSocket, NetIb, etc.) use the finalization handshake
    if (sock->type == ncclSocketTypeProxy || sock->type == ncclSocketTypeBootstrap)
    {
        sock->state = ncclSocketStateReady;
        return ncclSuccess;
    }


    int sent;
    if (sock->asyncFlag == 0)
    {
        // Blocking mode: wait for full send
        sent = 0;
        NCCLCHECK(ncclSocketWait(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent));
        sent = 0;
        NCCLCHECK(ncclSocketWait(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent));
    }
    else
    {
        // Non-blocking mode: use progress with counter
        if (sock->finalizeCounter < sizeof(sock->magic))
        {
            sent = sock->finalizeCounter;
            int closed = 0;
            NCCLCHECK(socketProgressOpt(NCCL_SOCKET_SEND, sock, &sock->magic, sizeof(sock->magic), &sent, 0, &closed));
            sock->finalizeCounter = sent;
            if (sent < sizeof(sock->magic))
                return ncclSuccess; // Not ready yet
        }
        sent = sock->finalizeCounter - sizeof(sock->magic);
        int closed = 0;
        NCCLCHECK(socketProgressOpt(NCCL_SOCKET_SEND, sock, &sock->type, sizeof(sock->type), &sent, 0, &closed));
        sock->finalizeCounter = sent + sizeof(sock->magic);
        if (sent < sizeof(sock->type))
            return ncclSuccess; // Not ready yet
    }
    sock->state = ncclSocketStateReady;
    return ncclSuccess;
}

// Receive magic and type bytes to finalize server-side accept
// Only for transport sockets, not proxy sockets which use their own protocol
static ncclResult_t socketFinalizeAccept(struct ncclSocket *sock)
{
    // Proxy sockets don't use the magic/type handshake - they have their own message protocol
    if (sock->type == ncclSocketTypeProxy || sock->type == ncclSocketTypeBootstrap)
    {
        sock->state = ncclSocketStateReady;
        return ncclSuccess;
    }

    {
        sock->state = ncclSocketStateReady;
        return ncclSuccess;
    }

    uint64_t magic;
    enum ncclSocketType type;
    int received;

    // Set non-blocking mode on accepted socket if async
    if (sock->asyncFlag)
    {
        NCCLCHECK(setSocketNonBlocking(sock->fd, true));
    }

    if (sock->asyncFlag == 0 || sock->finalizeCounter < sizeof(magic))
    {
        if (sock->asyncFlag == 0)
        {
            // Blocking mode
            received = 0;
            NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, sock, &magic, sizeof(magic), &received));
        }
        else
        {
            // Non-blocking mode
            int closed = 0;
            received = sock->finalizeCounter;
            NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(magic), &received, 0, &closed));
            sock->finalizeCounter = received;
            if (received < sizeof(magic))
            {
                if (closed)
                {
                    sock->state = ncclSocketStateError;
                }
                return ncclSuccess; // Not ready yet
            }
            memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
        }
        if (magic != sock->magic)
        {
            WARN("socketFinalizeAccept: invalid magic received");
            sock->state = ncclSocketStateError;
            return ncclSuccess;
        }
    }

    if (sock->asyncFlag == 0)
    {
        received = 0;
        NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, sock, &type, sizeof(type), &received));
    }
    else
    {
        received = sock->finalizeCounter - sizeof(magic);
        int closed = 0;
        NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, sock->finalizeBuffer, sizeof(type), &received, 0, &closed));
        sock->finalizeCounter = received + sizeof(magic);
        if (received < sizeof(type))
            return ncclSuccess; // Not ready yet
        memcpy(&type, sock->finalizeBuffer, sizeof(type));
    }

    if (type != sock->type)
    {
        char line[NCCL_SOCKET_MAX_STR_LEN];
        WARN("socketFinalizeAccept from %s: wrong type %d != %d", ncclSocketToString(&sock->addr, line), type, sock->type);
        sock->state = ncclSocketStateError;
        closesocket(sock->fd);
        sock->fd = INVALID_SOCKET;
        return ncclInternalError;
    }

    sock->state = ncclSocketStateReady;
    return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket *sock, int *running)
{
    if (sock == NULL || running == NULL)
        return ncclInvalidArgument;

    *running = 0;

    // Handle error/closed states
    if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed)
    {
        WARN("ncclSocketReady: unexpected socket state %d", sock->state);
        return ncclRemoteError;
    }

    // Check if already ready
    if (sock->state == ncclSocketStateReady)
    {
        *running = 1;
        return ncclSuccess;
    }

    // Progress connect polling to connected state
    if (sock->state == ncclSocketStateConnectPolling)
    {
        fd_set writefds, exceptfds;
        FD_ZERO(&writefds);
        FD_ZERO(&exceptfds);
        FD_SET(sock->fd, &writefds);
        FD_SET(sock->fd, &exceptfds);

        struct timeval tv = {0, 0};
        int ret = select(0, NULL, &writefds, &exceptfds, &tv);
        if (ret > 0)
        {
            if (FD_ISSET(sock->fd, &exceptfds))
            {
                sock->state = ncclSocketStateError;
                return ncclSuccess;
            }
            if (FD_ISSET(sock->fd, &writefds))
            {
                sock->state = ncclSocketStateConnected;
            }
        }
    }

    // Finalize connect (send magic/type) to reach ready state
    if (sock->state == ncclSocketStateConnected)
    {
        NCCLCHECK(socketFinalizeConnect(sock));
    }

    // Finalize accept (receive magic/type) to reach ready state
    if (sock->state == ncclSocketStateAccepted)
    {
        NCCLCHECK(socketFinalizeAccept(sock));
    }

    // Set running=1 if ready, otherwise still in progress
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;

    return ncclSuccess;
}

ncclResult_t ncclSocketClose(struct ncclSocket *sock, bool wait)
{
    (void)wait;
    if (sock == NULL)
        return ncclSuccess;

    if (sock->fd != INVALID_SOCKET)
    {
        closesocket(sock->fd);
        sock->fd = INVALID_SOCKET;
    }
    if (sock->acceptFd != INVALID_SOCKET)
    {
        closesocket(sock->acceptFd);
        sock->acceptFd = INVALID_SOCKET;
    }
    sock->state = ncclSocketStateClosed;
    return ncclSuccess;
}

ncclResult_t ncclSocketShutdown(struct ncclSocket *sock, int how)
{
    if (sock == NULL)
        return ncclSuccess;
    if (sock->fd != INVALID_SOCKET)
    {
        shutdown(sock->fd, how);
    }
    return ncclSuccess;
}

static ncclResult_t socketProgressOpt(int op, struct ncclSocket *sock, void *ptr, int size, int *offset, int block, int *closed)
{
    int bytes = 0;
    *closed = 0;
    char *data = (char *)ptr;
    char line[NCCL_SOCKET_MAX_STR_LEN];

    do
    {
        if (op == NCCL_SOCKET_RECV)
            bytes = recv(sock->fd, data + (*offset), size - (*offset), 0);
        else if (op == NCCL_SOCKET_SEND)
            bytes = send(sock->fd, data + (*offset), size - (*offset), 0);

        if (op == NCCL_SOCKET_RECV && bytes == 0)
        {
            *closed = 1;
            return ncclSuccess;
        }

        if (bytes == SOCKET_ERROR)
        {
            int wsaErr = WSAGetLastError();
            if ((op == NCCL_SOCKET_SEND && wsaErr == WSAENOTCONN) ||
                (op == NCCL_SOCKET_RECV && wsaErr == WSAECONNRESET))
            {
                *closed = 1;
                return ncclSuccess;
            }
            if (wsaErr != WSAEINTR && wsaErr != WSAEWOULDBLOCK)
            {
                WARN("socketProgressOpt: Call to %s %s failed: error %d",
                     (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
                     ncclSocketToString(&sock->addr, line), wsaErr);
                return ncclRemoteError;
            }
            bytes = 0;
        }

        (*offset) += bytes;
        if (sock->abortFlag && *sock->abortFlag)
            return ncclInternalError;
    } while (bytes > 0 && (*offset) < size && block);

    return ncclSuccess;
}

ncclResult_t ncclSocketProgress(int op, struct ncclSocket *sock, void *ptr, int size, int *offset, int *closed)
{
    int tmpClosed = 0;
    if (closed == NULL)
        closed = &tmpClosed;
    return socketProgressOpt(op, sock, ptr, size, offset, 0, closed);
}

ncclResult_t ncclSocketWait(int op, struct ncclSocket *sock, void *ptr, int size, int *offset)
{
    int closed = 0;
    while (*offset < size && !closed)
    {
        NCCLCHECK(socketProgressOpt(op, sock, ptr, size, offset, 1, &closed));
        if (closed)
            return ncclSystemError;
    }
    return ncclSuccess;
}

ncclResult_t ncclSocketSend(struct ncclSocket *sock, void *ptr, int size)
{
    int offset = 0;
    NCCLCHECK(ncclSocketWait(NCCL_SOCKET_SEND, sock, ptr, size, &offset));
    return ncclSuccess;
}

ncclResult_t ncclSocketRecv(struct ncclSocket *sock, void *ptr, int size)
{
    int offset = 0;
    NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, sock, ptr, size, &offset));
    return ncclSuccess;
}

ncclResult_t ncclSocketSendRecv(struct ncclSocket *sendSock, void *sendPtr, int sendSize,
                                struct ncclSocket *recvSock, void *recvPtr, int recvSize)
{
    int sendOffset = 0, recvOffset = 0;
    int sendClosed = 0, recvClosed = 0;

    while ((sendOffset < sendSize || recvOffset < recvSize) && !sendClosed && !recvClosed)
    {
        if (sendOffset < sendSize)
        {
            NCCLCHECK(socketProgressOpt(NCCL_SOCKET_SEND, sendSock, sendPtr, sendSize, &sendOffset, 0, &sendClosed));
        }
        if (recvOffset < recvSize)
        {
            NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, recvSock, recvPtr, recvSize, &recvOffset, 0, &recvClosed));
        }
    }

    if (sendClosed || recvClosed)
        return ncclSystemError;
    return ncclSuccess;
}

ncclResult_t ncclSocketMultiOp(struct ncclSocketOp *ops, int numOps)
{
    if (ops == NULL || numOps <= 0)
        return ncclInvalidArgument;

    bool allComplete = false;
    while (!allComplete)
    {
        allComplete = true;
        for (int i = 0; i < numOps; i++)
        {
            if (ops[i].offset < ops[i].size)
            {
                int closed = 0;
                NCCLCHECK(socketProgressOpt(ops[i].op, ops[i].sock, ops[i].ptr, ops[i].size, &ops[i].offset, 0, &closed));
                if (closed)
                    return ncclSystemError;
                if (ops[i].offset < ops[i].size)
                    allComplete = false;
            }
        }
    }
    return ncclSuccess;
}

ncclResult_t ncclSocketTryRecv(struct ncclSocket *sock, void *ptr, int size, int *closed, bool blocking)
{
    if (closed)
        *closed = 0;
    int offset = 0;
    int tmpClosed = 0;
    NCCLCHECK(socketProgressOpt(NCCL_SOCKET_RECV, sock, ptr, size, &offset, blocking ? 1 : 0, &tmpClosed));
    if (closed)
        *closed = tmpClosed;
    return ncclSuccess;
}

ncclResult_t ncclSocketGetFd(struct ncclSocket *sock, ncclSocketFd_t *fd)
{
    if (sock == NULL || fd == NULL)
        return ncclInvalidArgument;
    *fd = sock->fd;
    return ncclSuccess;
}

ncclResult_t ncclSocketSetFd(ncclSocketFd_t fd, struct ncclSocket *sock)
{
    if (sock == NULL)
        return ncclInvalidArgument;
    sock->fd = fd;
    return ncclSuccess;
}

// Network interface discovery using Windows IP Helper API
ncclResult_t ncclFindInterfaces(char *ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs, int *nIfs)
{
    if (nIfs == NULL)
        return ncclInvalidArgument;
    *nIfs = 0;

    NCCLCHECK(ncclSocketInitWinsock());

    // Get adapter addresses
    ULONG bufLen = 15000;
    PIP_ADAPTER_ADDRESSES addresses = NULL;
    ULONG ret;

    do
    {
        addresses = (PIP_ADAPTER_ADDRESSES)malloc(bufLen);
        if (addresses == NULL)
            return ncclSystemError;

        ret = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST,
                                   NULL, addresses, &bufLen);
        if (ret == ERROR_BUFFER_OVERFLOW)
        {
            free(addresses);
            addresses = NULL;
        }
    } while (ret == ERROR_BUFFER_OVERFLOW);

    if (ret != NO_ERROR)
    {
        if (addresses)
            free(addresses);
        WARN("GetAdaptersAddresses failed: %lu", ret);
        return ncclSystemError;
    }

    int count = 0;
    for (PIP_ADAPTER_ADDRESSES adapter = addresses; adapter != NULL && count < maxIfs; adapter = adapter->Next)
    {
        // Skip loopback and non-operational adapters
        if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
            continue;
        if (adapter->OperStatus != IfOperStatusUp)
            continue;

        for (PIP_ADAPTER_UNICAST_ADDRESS ua = adapter->FirstUnicastAddress; ua != NULL && count < maxIfs; ua = ua->Next)
        {
            struct sockaddr *sa = ua->Address.lpSockaddr;
            if (sa->sa_family != AF_INET && sa->sa_family != AF_INET6)
                continue;

            // Skip link-local IPv6 addresses
            if (sa->sa_family == AF_INET6)
            {
                struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)sa;
                if (IN6_IS_ADDR_LINKLOCAL(&sin6->sin6_addr))
                    continue;
            }

            // Copy interface name (convert from wide string)
            if (ifNames)
            {
                char *name = ifNames + count * ifNameMaxSize;
                WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1, name, ifNameMaxSize - 1, NULL, NULL);
                name[ifNameMaxSize - 1] = '\0';
            }

            // Copy address
            if (ifAddrs)
            {
                if (sa->sa_family == AF_INET)
                {
                    memcpy(&ifAddrs[count].sin, sa, sizeof(struct sockaddr_in));
                }
                else
                {
                    memcpy(&ifAddrs[count].sin6, sa, sizeof(struct sockaddr_in6));
                }
            }

            count++;
        }
    }

    free(addresses);
    *nIfs = count;

    TRACE(NCCL_INIT | NCCL_NET, "Found %d network interfaces", count);
    return ncclSuccess;
}

ncclResult_t ncclFindInterfaceMatchSubnet(char *ifName, union ncclSocketAddress *localAddr,
                                          union ncclSocketAddress *remoteAddr, int ifNameMaxSize, int *found)
{
    if (found == NULL)
        return ncclInvalidArgument;
    *found = 0;

    if (remoteAddr == NULL)
        return ncclSuccess;

    // Get all interfaces
    char ifNames[MAX_IFS * MAX_IF_NAME_SIZE];
    union ncclSocketAddress ifAddrs[MAX_IFS];
    int nIfs = 0;

    NCCLCHECK(ncclFindInterfaces(ifNames, ifAddrs, MAX_IF_NAME_SIZE, MAX_IFS, &nIfs));

    for (int i = 0; i < nIfs; i++)
    {
        if (ifAddrs[i].sa.sa_family != remoteAddr->sa.sa_family)
            continue;

        // For simplicity, return the first matching address family
        if (ifName)
        {
            strncpy(ifName, ifNames + i * MAX_IF_NAME_SIZE, ifNameMaxSize - 1);
            ifName[ifNameMaxSize - 1] = '\0';
        }
        if (localAddr)
        {
            memcpy(localAddr, &ifAddrs[i], sizeof(*localAddr));
        }
        *found = 1;
        return ncclSuccess;
    }

    return ncclSuccess;
}

#endif // NCCL_PLATFORM_WINDOWS
