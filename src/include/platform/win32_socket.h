/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_SOCKET_H_
#define NCCL_WIN32_SOCKET_H_

#ifdef _WIN32

#include "win32_defs.h"
#include <stdlib.h>
#include <string.h>

/*
 * Windows implementation of network interface enumeration
 * Equivalent to Linux getifaddrs()/freeifaddrs()
 */

/* Interface address structure (POSIX ifaddrs equivalent) */
struct ncclIfaddrs
{
    struct ncclIfaddrs *ifa_next;
    char *ifa_name;
    unsigned int ifa_flags;
    struct sockaddr *ifa_addr;
    struct sockaddr *ifa_netmask;
    struct sockaddr *ifa_broadaddr;
    void *ifa_data;
    /* Storage for address data */
    struct sockaddr_storage _addr_storage;
    struct sockaddr_storage _netmask_storage;
    char _name_storage[256];
};

/* Forward declaration */
static inline void ncclFreeIfaddrs(struct ncclIfaddrs *ifa);

/*
 * Get list of network interfaces on Windows
 * Uses IP Helper API (GetAdaptersAddresses)
 */
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
    int family = AF_UNSPEC; /* Get both IPv4 and IPv6 */
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX | GAA_FLAG_SKIP_ANYCAST |
                  GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
    int attempt = 0;

    if (ifap == NULL)
        return -1;
    *ifap = NULL;

    /* Allocate initial buffer */
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

    /* Iterate through adapters and their addresses */
    for (pCurrAddresses = pAddresses; pCurrAddresses != NULL; pCurrAddresses = pCurrAddresses->Next)
    {
        /* Skip adapters that are not operational */
        if (pCurrAddresses->OperStatus != IfOperStatusUp)
            continue;

        for (pUnicast = pCurrAddresses->FirstUnicastAddress; pUnicast != NULL; pUnicast = pUnicast->Next)
        {
            struct sockaddr *sa = pUnicast->Address.lpSockaddr;

            /* Only support IPv4 and IPv6 */
            if (sa->sa_family != AF_INET && sa->sa_family != AF_INET6)
                continue;

            /* Allocate new interface entry */
            ifa = (struct ncclIfaddrs *)calloc(1, sizeof(struct ncclIfaddrs));
            if (ifa == NULL)
            {
                ncclFreeIfaddrs(head);
                free(pAddresses);
                return -1;
            }

            /* Set interface name */
            WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->FriendlyName, -1,
                                ifa->_name_storage, sizeof(ifa->_name_storage), NULL, NULL);
            ifa->ifa_name = ifa->_name_storage;

            /* Set flags */
            ifa->ifa_flags = 0;
            if (pCurrAddresses->OperStatus == IfOperStatusUp)
            {
                ifa->ifa_flags |= IFF_UP | IFF_RUNNING;
            }
            if (pCurrAddresses->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
            {
                ifa->ifa_flags |= IFF_LOOPBACK;
            }

            /* Copy address */
            memcpy(&ifa->_addr_storage, sa, pUnicast->Address.iSockaddrLength);
            ifa->ifa_addr = (struct sockaddr *)&ifa->_addr_storage;

            /* Get netmask (calculate from prefix length) */
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

            /* Link to list */
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

/*
 * Free interface list allocated by ncclGetIfaddrs
 */
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

/* Compatibility macros for Linux code */
#define ifaddrs ncclIfaddrs
#define getifaddrs ncclGetIfaddrs
#define freeifaddrs ncclFreeIfaddrs

/*
 * Socket helper functions for Windows
 */

/* Set socket to non-blocking mode */
static inline int ncclSocketSetNonBlocking(SOCKET sock, int nonblocking)
{
    u_long mode = nonblocking ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode);
}

/* Set TCP_NODELAY option */
static inline int ncclSocketSetNoDelay(SOCKET sock, int nodelay)
{
    return setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&nodelay, sizeof(nodelay));
}

/* Set SO_REUSEADDR option */
static inline int ncclSocketSetReuseAddr(SOCKET sock, int reuse)
{
    return setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse, sizeof(reuse));
}

/* Get socket error */
static inline int ncclSocketGetError(SOCKET sock)
{
    int error = 0;
    int len = sizeof(error);
    getsockopt(sock, SOL_SOCKET, SO_ERROR, (char *)&error, &len);
    return error;
}

/* Check if socket error is "would block" */
static inline int ncclSocketWouldBlock(int error)
{
    return (error == WSAEWOULDBLOCK || error == WSAEINPROGRESS);
}

/* Check if socket error is connection reset */
static inline int ncclSocketConnReset(int error)
{
    return (error == WSAECONNRESET || error == WSAECONNABORTED);
}

/* Windows socket error to string */
static inline const char *ncclSocketStrerror(int error)
{
    static char buffer[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   buffer, sizeof(buffer), NULL);
    /* Remove trailing newline */
    size_t len = strlen(buffer);
    while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r'))
    {
        buffer[--len] = '\0';
    }
    return buffer;
}

/* Replace errno-based socket error handling */
#define SOCKET_ERRNO WSAGetLastError()
#define socket_strerror(e) ncclSocketStrerror(e)

/*
 * poll() support is defined in win32_defs.h
 * Using WSAPOLLFD from winsock2.h and WSAPoll
 */

/*
 * Get network interface speed in Mbps using Windows API
 * Returns the link speed of the specified interface
 */
static inline int ncclGetInterfaceSpeed(const char *ifName, int *speedMbps)
{
    PIP_ADAPTER_ADDRESSES pAddresses = NULL;
    PIP_ADAPTER_ADDRESSES pCurrAddresses;
    ULONG outBufLen = 15000;
    DWORD dwRetVal;
    int found = 0;
    ULONG flags = GAA_FLAG_INCLUDE_PREFIX;

    *speedMbps = 0;

    /* Allocate buffer */
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

    /* Find the matching interface */
    for (pCurrAddresses = pAddresses; pCurrAddresses != NULL; pCurrAddresses = pCurrAddresses->Next)
    {
        char friendlyName[256];
        WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->FriendlyName, -1,
                            friendlyName, sizeof(friendlyName), NULL, NULL);

        /* Match by friendly name or adapter name */
        if (strcmp(friendlyName, ifName) == 0 ||
            strcmp(pCurrAddresses->AdapterName, ifName) == 0)
        {
            /* ReceiveLinkSpeed and TransmitLinkSpeed are in bits per second */
            /* Convert to Mbps (divide by 1,000,000) */
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

    /* If not found or speed is 0, return default */
    if (!found || *speedMbps <= 0)
    {
        *speedMbps = 10000; /* Default to 10 Gbps */
        return 0;
    }

    return 0;
}

/*
 * Get network interface vendor ID
 * Returns vendor string (e.g., "Intel", "Mellanox", "AWS", "GCP")
 */
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

    /* Allocate buffer */
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

    /* Find the matching interface */
    for (pCurrAddresses = pAddresses; pCurrAddresses != NULL; pCurrAddresses = pCurrAddresses->Next)
    {
        char friendlyName[256];
        WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->FriendlyName, -1,
                            friendlyName, sizeof(friendlyName), NULL, NULL);

        if (strcmp(friendlyName, ifName) == 0 ||
            strcmp(pCurrAddresses->AdapterName, ifName) == 0)
        {
            /* Extract vendor from description */
            char description[256];
            WideCharToMultiByte(CP_UTF8, 0, pCurrAddresses->Description, -1,
                                description, sizeof(description), NULL, NULL);

            /* Try to identify known vendors */
            if (strstr(description, "Amazon") || strstr(description, "ENA"))
            {
                strncpy(vendor, "0x1d0f", vendorLen); /* AWS vendor ID */
            }
            else if (strstr(description, "Google") || strstr(description, "gVNIC"))
            {
                strncpy(vendor, "0x1ae0", vendorLen); /* GCP vendor ID */
            }
            else if (strstr(description, "Mellanox") || strstr(description, "ConnectX"))
            {
                strncpy(vendor, "0x15b3", vendorLen); /* Mellanox vendor ID */
            }
            else if (strstr(description, "Intel"))
            {
                strncpy(vendor, "0x8086", vendorLen); /* Intel vendor ID */
            }
            else if (strstr(description, "Broadcom"))
            {
                strncpy(vendor, "0x14e4", vendorLen); /* Broadcom vendor ID */
            }
            else
            {
                /* Unknown vendor, use description */
                strncpy(vendor, description, vendorLen);
            }
            vendor[vendorLen - 1] = '\0';
            break;
        }
    }

    free(pAddresses);
    return 0;
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_SOCKET_H_ */
