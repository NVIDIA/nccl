/*************************************************************************
 * NCCL Platform Abstraction Tests - Socket Abstraction
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#include "test_framework.h"
#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_socket.h"
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#endif

static int init_sockets(void)
{
#if NCCL_PLATFORM_WINDOWS
    WSADATA wsaData;
    return WSAStartup(MAKEWORD(2, 2), &wsaData);
#else
    return 0;
#endif
}

static void cleanup_sockets(void)
{
#if NCCL_PLATFORM_WINDOWS
    WSACleanup();
#endif
}

static void close_socket(ncclSocketHandle_t sock)
{
#if NCCL_PLATFORM_WINDOWS
    closesocket(sock);
#else
    close(sock);
#endif
}

void test_socket_abstraction(void)
{
    TEST_SECTION("Socket Abstraction Tests");

    /* Initialize sockets */
    int init_ret = init_sockets();
    TEST_ASSERT_EQ(0, init_ret, "Socket initialization should succeed");

    /* Test socket creation */
    ncclSocketHandle_t sock = socket(AF_INET, SOCK_STREAM, 0);
    TEST_ASSERT_NE(NCCL_INVALID_SOCKET, sock, "Socket creation should succeed");

    /* Test setting socket options (non-blocking) */
#if NCCL_PLATFORM_WINDOWS
    u_long mode = 1;
    int ioctlret = ioctlsocket(sock, FIONBIO, &mode);
    TEST_ASSERT_EQ(0, ioctlret, "ioctlsocket for non-blocking should succeed");
#else
    int flags = fcntl(sock, F_GETFL, 0);
    int fcntlret = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    TEST_ASSERT_NE(-1, fcntlret, "fcntl for non-blocking should succeed");
#endif

    /* Test bind to any available port */
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; /* Let OS choose port */

    int bindret = bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    TEST_ASSERT_EQ(0, bindret, "Bind to loopback should succeed");

    /* Get assigned port */
    socklen_t addrlen = sizeof(addr);
    int getsockret = getsockname(sock, (struct sockaddr *)&addr, &addrlen);
    TEST_ASSERT_EQ(0, getsockret, "getsockname should succeed");
    TEST_ASSERT_NE(0, ntohs(addr.sin_port), "Assigned port should be non-zero");
    printf("  [INFO] Bound to port %d\n", ntohs(addr.sin_port));

    /* Test listen */
    int listenret = listen(sock, 5);
    TEST_ASSERT_EQ(0, listenret, "Listen should succeed");

    /* Test poll (with timeout, expecting no connections) */
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;
    pfd.revents = 0;

    int pollret = poll(&pfd, 1, 10); /* 10ms timeout */
    TEST_ASSERT_EQ(0, pollret, "Poll should timeout with no connections");

    /* Cleanup */
    close_socket(sock);

    /* Test NCCL socket error codes */
    TEST_ASSERT_NE(0, NCCL_ERRNO_EINTR, "EINTR should be defined");
    TEST_ASSERT_NE(0, NCCL_ERRNO_EWOULDBLOCK, "EWOULDBLOCK should be defined");
    TEST_ASSERT_NE(0, NCCL_ERRNO_EINPROGRESS, "EINPROGRESS should be defined");

#if NCCL_PLATFORM_WINDOWS
    /* Test Windows-specific network speed detection */
    int speedMbps = 0;
    int speedret = ncclGetInterfaceSpeed("Ethernet", &speedMbps);
    /* This may fail if no Ethernet interface exists, but shouldn't crash */
    if (speedret == 0)
    {
        TEST_ASSERT_GT(speedMbps, 0, "Interface speed should be positive");
        printf("  [INFO] Ethernet speed: %d Mbps\n", speedMbps);
    }
    else
    {
        printf("  [INFO] Could not get Ethernet speed (interface may not exist)\n");
    }
#endif

    cleanup_sockets();
}
