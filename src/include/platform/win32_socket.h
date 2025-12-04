/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_SOCKET_H_
#define NCCL_WIN32_SOCKET_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <iphlpapi.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "iphlpapi.lib")

/* Interface flags (POSIX compatibility) */
#ifndef IFF_UP
#define IFF_UP 0x0001
#endif
#ifndef IFF_BROADCAST
#define IFF_BROADCAST 0x0002
#endif
#ifndef IFF_LOOPBACK
#define IFF_LOOPBACK 0x0008
#endif
#ifndef IFF_RUNNING
#define IFF_RUNNING 0x0040
#endif
#ifndef IFF_MULTICAST
#define IFF_MULTICAST 0x1000
#endif

/* ========================================================================== */
/*                    Network Interface Enumeration                           */
/* ========================================================================== */

struct ncclIfaddrs
{
    struct ncclIfaddrs *ifa_next;
    char *ifa_name;
    unsigned int ifa_flags;
    struct sockaddr *ifa_addr;
    struct sockaddr *ifa_netmask;
    struct sockaddr *ifa_broadaddr;
    void *ifa_data;
    struct sockaddr_storage _addr_storage;
    struct sockaddr_storage _netmask_storage;
    char _name_storage[256];
};

static inline void ncclFreeIfaddrs(struct ncclIfaddrs *ifa);

static inline int ncclGetIfaddrs(struct ncclIfaddrs **ifap)
{
    PIP_ADAPTER_ADDRESSES pAddresses = NULL;
    PIP_ADAPTER_ADDRESSES pCurrAddresses;
    PIP_ADAPTER_UNICAST_ADDRESS pUnicast;
    ULONG outBufLen = 15000;
    DWORD dwRetVal;
    struct ncclIfaddrs *head = NULL;
    struct ncclIfaddrs *tail = NULL;
    struct ncclIfaddrs *ifa;
    int family = AF_UNSPEC;
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_SKIP_ANYCAST |
                  GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
    int attempt = 0;

    if (ifap == NULL)
        return -1;
    *ifap = NULL;

    do
    {
        pAddresses = (IP_ADAPTER_ADDRESSES *)malloc(outBufLen);
        if (pAddresses == NULL)
            return -1;

        dwRetVal = GetAdaptersAddresses(family, flags, NULL, pAddresses, &outBufLen);
        if (dwRetVal == ERROR_BUFFER_OVERFLOW)
        {
            free(pAddresses);
            pAddresses = NULL;
            attempt++;
        }
        else
        {
            break;
        }
    } while (attempt < 3);

    if (dwRetVal != NO_ERROR)
    {
        if (pAddresses)
            free(pAddresses);
        return -1;
    }

    for (pCurrAddresses = pAddresses; pCurrAddresses != NULL;
         pCurrAddresses = pCurrAddresses->Next)
    {
        if (pCurrAddresses->OperStatus != IfOperStatusUp)
            continue;

        for (pUnicast = pCurrAddresses->FirstUnicastAddress; pUnicast != NULL;
             pUnicast = pUnicast->Next)
        {
            struct sockaddr *sa = pUnicast->Address.lpSockaddr;

            if (sa->sa_family != AF_INET && sa->sa_family != AF_INET6)
                continue;

            ifa = (struct ncclIfaddrs *)calloc(1, sizeof(struct ncclIfaddrs));
            if (ifa == NULL)
            {
                ncclFreeIfaddrs(head);
                free(pAddresses);
                return -1;
            }

            WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->FriendlyName, -1,
                                ifa->_name_storage, sizeof(ifa->_name_storage), NULL, NULL);
            ifa->ifa_name = ifa->_name_storage;

            ifa->ifa_flags = 0;
            if (pCurrAddresses->OperStatus == IfOperStatusUp)
            {
                ifa->ifa_flags |= IFF_UP | IFF_RUNNING;
            }
            if (pCurrAddresses->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
            {
                ifa->ifa_flags |= IFF_LOOPBACK;
            }

            memcpy(&ifa->_addr_storage, sa, pUnicast->Address.iSockaddrLength);
            ifa->ifa_addr = (struct sockaddr *)&ifa->_addr_storage;

            if (sa->sa_family == AF_INET)
            {
                struct sockaddr_in *sin = (struct sockaddr_in *)&ifa->_netmask_storage;
                sin->sin_family = AF_INET;
                if (pUnicast->OnLinkPrefixLength <= 32)
                {
                    sin->sin_addr.s_addr = htonl(~((1UL << (32 - pUnicast->OnLinkPrefixLength)) - 1));
                }
                else
                {
                    sin->sin_addr.s_addr = 0xFFFFFFFF;
                }
                ifa->ifa_netmask = (struct sockaddr *)&ifa->_netmask_storage;
            }
            else if (sa->sa_family == AF_INET6)
            {
                struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)&ifa->_netmask_storage;
                int bits = pUnicast->OnLinkPrefixLength;
                int i;
                sin6->sin6_family = AF_INET6;
                memset(&sin6->sin6_addr, 0, sizeof(sin6->sin6_addr));
                for (i = 0; i < 16 && bits > 0; i++)
                {
                    if (bits >= 8)
                    {
                        sin6->sin6_addr.s6_addr[i] = 0xFF;
                        bits -= 8;
                    }
                    else
                    {
                        sin6->sin6_addr.s6_addr[i] = (unsigned char)(0xFF << (8 - bits));
                        bits = 0;
                    }
                }
                ifa->ifa_netmask = (struct sockaddr *)&ifa->_netmask_storage;
            }

            ifa->ifa_next = NULL;
            if (tail == NULL)
            {
                head = tail = ifa;
            }
            else
            {
                tail->ifa_next = ifa;
                tail = ifa;
            }
        }
    }

    free(pAddresses);
    *ifap = head;
    return 0;
}

static inline void ncclFreeIfaddrs(struct ncclIfaddrs *ifa)
{
    struct ncclIfaddrs *next;
    while (ifa != NULL)
    {
        next = ifa->ifa_next;
        free(ifa);
        ifa = next;
    }
}

#define ifaddrs ncclIfaddrs
#define getifaddrs ncclGetIfaddrs
#define freeifaddrs ncclFreeIfaddrs

/* ========================================================================== */
/*                        Socket Helper Functions                             */
/* ========================================================================== */

static inline int ncclSocketSetNonBlocking(SOCKET sock, int nonblocking)
{
    u_long mode = nonblocking ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode);
}

static inline int ncclSocketSetNoDelay(SOCKET sock, int nodelay)
{
    return setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&nodelay, sizeof(nodelay));
}

static inline int ncclSocketSetReuseAddr(SOCKET sock, int reuse)
{
    return setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse, sizeof(reuse));
}

static inline int ncclSocketOptimize(SOCKET sock)
{
    int result = 0;
    int optval;

    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;

    optval = 4 * 1024 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;

    optval = 1;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char *)&optval, sizeof(optval));

    return result;
}

static inline int ncclSocketOptimizeLowLatency(SOCKET sock)
{
    int result = 0;
    int optval;

    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;

    optval = 256 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;

    return result;
}

/* ========================================================================== */
/*                        Overlapped I/O Support                              */
/* ========================================================================== */

struct ncclSocketOverlapped
{
    WSAOVERLAPPED overlapped;
    WSABUF wsaBuf;
    DWORD flags;
    DWORD bytesTransferred;
    int completed;
};

static inline int ncclSocketOverlappedInit(struct ncclSocketOverlapped *ov,
                                           void *buffer, size_t size)
{
    if (ov == NULL)
        return -1;

    memset(ov, 0, sizeof(*ov));
    ov->overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (ov->overlapped.hEvent == NULL)
        return -1;

    ov->wsaBuf.buf = (char *)buffer;
    ov->wsaBuf.len = (ULONG)size;
    ov->flags = 0;
    ov->completed = 0;

    return 0;
}

static inline int ncclSocketSendAsync(SOCKET sock, struct ncclSocketOverlapped *ov)
{
    int result;

    ResetEvent(ov->overlapped.hEvent);
    ov->completed = 0;

    result = WSASend(sock, &ov->wsaBuf, 1, NULL, 0, &ov->overlapped, NULL);

    if (result == SOCKET_ERROR)
    {
        int err = WSAGetLastError();
        if (err == WSA_IO_PENDING)
            return 0;
        return -1;
    }
    return 0;
}

static inline int ncclSocketRecvAsync(SOCKET sock, struct ncclSocketOverlapped *ov)
{
    int result;

    ResetEvent(ov->overlapped.hEvent);
    ov->completed = 0;
    ov->flags = 0;

    result = WSARecv(sock, &ov->wsaBuf, 1, NULL, &ov->flags, &ov->overlapped, NULL);

    if (result == SOCKET_ERROR)
    {
        int err = WSAGetLastError();
        if (err == WSA_IO_PENDING)
            return 0;
        return -1;
    }
    return 0;
}

static inline int ncclSocketOverlappedWait(SOCKET sock, struct ncclSocketOverlapped *ov,
                                           DWORD timeoutMs)
{
    DWORD waitResult;
    DWORD bytesTransferred;
    DWORD flags;

    waitResult = WaitForSingleObject(ov->overlapped.hEvent, timeoutMs);

    if (waitResult == WAIT_TIMEOUT)
        return 0;
    if (waitResult != WAIT_OBJECT_0)
        return -1;

    if (!WSAGetOverlappedResult(sock, &ov->overlapped, &bytesTransferred, FALSE, &flags))
        return -1;

    ov->bytesTransferred = bytesTransferred;
    ov->completed = 1;
    return 1;
}

static inline void ncclSocketOverlappedFree(struct ncclSocketOverlapped *ov)
{
    if (ov && ov->overlapped.hEvent != NULL)
    {
        CloseHandle(ov->overlapped.hEvent);
        ov->overlapped.hEvent = NULL;
    }
}

/* ========================================================================== */
/*                        Socket Error Handling                               */
/* ========================================================================== */

static inline int ncclSocketGetError(SOCKET sock)
{
    int error = 0;
    int len = sizeof(error);
    getsockopt(sock, SOL_SOCKET, SO_ERROR, (char *)&error, &len);
    return error;
}

static inline int ncclSocketWouldBlock(int error)
{
    return (error == WSAEWOULDBLOCK || error == WSAEINPROGRESS);
}

static inline int ncclSocketConnReset(int error)
{
    return (error == WSAECONNRESET || error == WSAECONNABORTED);
}

static inline const char *ncclSocketStrerror(int error)
{
    static char buffer[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   buffer, sizeof(buffer), NULL);
    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r'))
    {
        buffer[--len] = '\0';
    }
    return buffer;
}

#define SOCKET_ERRNO WSAGetLastError()
#define socket_strerror(e) ncclSocketStrerror(e)

/* ========================================================================== */
/*                      Interface Speed and Vendor                            */
/* ========================================================================== */

static inline int ncclGetInterfaceSpeed(const char *ifName, int *speedMbps)
{
    PIP_ADAPTER_ADDRESSES pAddresses = NULL;
    PIP_ADAPTER_ADDRESSES pCurrAddresses;
    ULONG outBufLen = 15000;
    DWORD dwRetVal;
    int found = 0;
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX;

    *speedMbps = 0;

    pAddresses = (IP_ADAPTER_ADDRESSES *)malloc(outBufLen);
    if (pAddresses == NULL)
        return -1;

    dwRetVal = GetAdaptersAddresses(AF_UNSPEC, flags, NULL, pAddresses, &outBufLen);
    if (dwRetVal == ERROR_BUFFER_OVERFLOW)
    {
        free(pAddresses);
        pAddresses = (IP_ADAPTER_ADDRESSES *)malloc(outBufLen);
        if (pAddresses == NULL)
            return -1;
        dwRetVal = GetAdaptersAddresses(AF_UNSPEC, flags, NULL, pAddresses, &outBufLen);
    }

    if (dwRetVal != NO_ERROR)
    {
        free(pAddresses);
        return -1;
    }

    for (pCurrAddresses = pAddresses; pCurrAddresses != NULL;
         pCurrAddresses = pCurrAddresses->Next)
    {
        char friendlyName[256];
        WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->FriendlyName, -1,
                            friendlyName, sizeof(friendlyName), NULL, NULL);

        if (strcmp(friendlyName, ifName) == 0 ||
            strcmp(pCurrAddresses->AdapterName, ifName) == 0)
        {
            ULONG64 linkSpeed = pCurrAddresses->ReceiveLinkSpeed;
            if (linkSpeed > 0)
            {
                *speedMbps = (int)(linkSpeed / 1000000ULL);
                found = 1;
            }
            break;
        }
    }

    free(pAddresses);

    if (!found || *speedMbps <= 0)
    {
        *speedMbps = 10000;
        return 0;
    }
    return 0;
}

static inline int ncclGetInterfaceVendor(const char *ifName, char *vendor, size_t vendorLen)
{
    PIP_ADAPTER_ADDRESSES pAddresses = NULL;
    PIP_ADAPTER_ADDRESSES pCurrAddresses;
    ULONG outBufLen = 15000;
    DWORD dwRetVal;
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX;

    if (vendor == NULL || vendorLen == 0)
        return -1;
    vendor[0] = '\0';

    pAddresses = (IP_ADAPTER_ADDRESSES *)malloc(outBufLen);
    if (pAddresses == NULL)
        return -1;

    dwRetVal = GetAdaptersAddresses(AF_UNSPEC, flags, NULL, pAddresses, &outBufLen);
    if (dwRetVal == ERROR_BUFFER_OVERFLOW)
    {
        free(pAddresses);
        pAddresses = (IP_ADAPTER_ADDRESSES *)malloc(outBufLen);
        if (pAddresses == NULL)
            return -1;
        dwRetVal = GetAdaptersAddresses(AF_UNSPEC, flags, NULL, pAddresses, &outBufLen);
    }

    if (dwRetVal != NO_ERROR)
    {
        free(pAddresses);
        return -1;
    }

    for (pCurrAddresses = pAddresses; pCurrAddresses != NULL;
         pCurrAddresses = pCurrAddresses->Next)
    {
        char friendlyName[256];
        WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->FriendlyName, -1,
                            friendlyName, sizeof(friendlyName), NULL, NULL);

        if (strcmp(friendlyName, ifName) == 0 ||
            strcmp(pCurrAddresses->AdapterName, ifName) == 0)
        {
            char description[256];
            WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->Description, -1,
                                description, sizeof(description), NULL, NULL);

            if (strstr(description, "Amazon") || strstr(description, "ENA"))
            {
                strncpy(vendor, "0x1d0f", vendorLen);
            }
            else if (strstr(description, "Google") || strstr(description, "gVNIC"))
            {
                strncpy(vendor, "0x1ae0", vendorLen);
            }
            else if (strstr(description, "Mellanox") || strstr(description, "ConnectX"))
            {
                strncpy(vendor, "0x15b3", vendorLen);
            }
            else if (strstr(description, "Intel"))
            {
                strncpy(vendor, "0x8086", vendorLen);
            }
            else if (strstr(description, "Broadcom"))
            {
                strncpy(vendor, "0x14e4", vendorLen);
            }
            else
            {
                strncpy(vendor, description, vendorLen);
            }
            vendor[vendorLen - 1] = '\0';
            break;
        }
    }

    free(pAddresses);
    return 0;
}

/* ========================================================================== */
/*                       Advanced Optimizations                               */
/* ========================================================================== */

#ifndef SIO_LOOPBACK_FAST_PATH
#define SIO_LOOPBACK_FAST_PATH _WSAIOW(IOC_VENDOR, 16)
#endif

static inline int ncclSocketEnableLoopbackFastPath(SOCKET sock)
{
    int optval = 1;
    DWORD bytesReturned = 0;
    return WSAIoctl(sock, SIO_LOOPBACK_FAST_PATH, &optval, sizeof(optval),
                    NULL, 0, &bytesReturned, NULL, NULL);
}

#ifndef TCP_FASTOPEN
#define TCP_FASTOPEN 15
#endif

static inline int ncclSocketEnableFastOpen(SOCKET sock)
{
    int optval = 1;
    return setsockopt(sock, IPPROTO_TCP, TCP_FASTOPEN, (const char *)&optval, sizeof(optval));
}

static inline int ncclSocketSetPriority(SOCKET sock, int priority)
{
    int tos;
    switch (priority)
    {
    case 7:
        tos = 0xB8;
        break;
    case 6:
        tos = 0x80;
        break;
    case 5:
        tos = 0x60;
        break;
    case 4:
        tos = 0x40;
        break;
    default:
        tos = 0x00;
        break;
    }
    return setsockopt(sock, IPPROTO_IP, IP_TOS, (const char *)&tos, sizeof(tos));
}

static inline int ncclSocketOptimizeUltraLowLatency(SOCKET sock)
{
    int result = 0;
    int optval;

    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;

    optval = 64 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char *)&optval, sizeof(optval));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char *)&optval, sizeof(optval));

    ncclSocketEnableLoopbackFastPath(sock);
    ncclSocketSetPriority(sock, 6);

    return result;
}

static inline int ncclSocketOptimizeMaxThroughput(SOCKET sock)
{
    int result = 0;
    int optval;

    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&optval, sizeof(optval)) != 0)
        result = -1;

    optval = 8 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char *)&optval, sizeof(optval));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char *)&optval, sizeof(optval));

    ncclSocketEnableLoopbackFastPath(sock);

    optval = 1;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char *)&optval, sizeof(optval));

    return result;
}

/* ========================================================================== */
/*                          IOCP Support                                      */
/* ========================================================================== */

struct ncclSocketIOCP
{
    HANDLE hCompletionPort;
    int refCount;
};

static inline int ncclSocketIOCPCreate(struct ncclSocketIOCP *iocp, int concurrency)
{
    if (iocp == NULL)
        return -1;

    iocp->hCompletionPort = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0,
                                                   (DWORD)concurrency);
    if (iocp->hCompletionPort == NULL)
        return -1;

    iocp->refCount = 0;
    return 0;
}

static inline int ncclSocketIOCPAssociate(struct ncclSocketIOCP *iocp, SOCKET sock,
                                          ULONG_PTR completionKey)
{
    if (iocp == NULL || iocp->hCompletionPort == NULL)
        return -1;

    HANDLE h = CreateIoCompletionPort((HANDLE)sock, iocp->hCompletionPort, completionKey, 0);
    if (h == NULL)
        return -1;

    iocp->refCount++;
    return 0;
}

static inline int ncclSocketIOCPWait(struct ncclSocketIOCP *iocp, DWORD timeout,
                                     DWORD *bytesTransferred, ULONG_PTR *completionKey,
                                     LPOVERLAPPED *overlapped)
{
    if (iocp == NULL || iocp->hCompletionPort == NULL)
        return -1;

    BOOL result = GetQueuedCompletionStatus(iocp->hCompletionPort,
                                            bytesTransferred, completionKey,
                                            overlapped, timeout);

    if (!result)
    {
        if (*overlapped == NULL)
            return -1;
        return 1;
    }
    return 0;
}

static inline void ncclSocketIOCPDestroy(struct ncclSocketIOCP *iocp)
{
    if (iocp != NULL && iocp->hCompletionPort != NULL)
    {
        CloseHandle(iocp->hCompletionPort);
        iocp->hCompletionPort = NULL;
        iocp->refCount = 0;
    }
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_SOCKET_H_ */
