/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PLATFORM_LINUX_SOCKET_H_
#define NCCL_PLATFORM_LINUX_SOCKET_H_

/*
 * Linux Socket Optimizations for NCCL
 *
 * Based on findings from "Demystifying NCCL" (arXiv:2507.04786v2)
 * Provides protocol-aware socket tuning for optimal collective communication.
 */

#ifndef NCCL_PLATFORM_LINUX
#error "This header is for Linux only"
#endif

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>

/* Socket options that may not be available on all systems */
#ifndef TCP_QUICKACK
#define TCP_QUICKACK 12
#endif

#ifndef TCP_FASTOPEN
#define TCP_FASTOPEN 23
#endif

#ifndef TCP_NOTSENT_LOWAT
#define TCP_NOTSENT_LOWAT 25
#endif

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif

#ifndef SO_INCOMING_CPU
#define SO_INCOMING_CPU 49
#endif

#ifndef SO_ZEROCOPY
#define SO_ZEROCOPY 60
#endif

#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY 0x4000000
#endif

/* ========================================================================== */
/*                    Socket Optimization Functions                           */
/* ========================================================================== */

/*
 * ncclSocketOptimize - Optimize socket for NCCL Simple protocol (large messages)
 *
 * Configures socket with:
 * - TCP_NODELAY: Disable Nagle's algorithm for low latency
 * - Large send/receive buffers (4MB): Optimal for Simple protocol
 * - SO_KEEPALIVE: Connection health monitoring
 * - TCP_QUICKACK: Disable delayed ACKs for faster response
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketOptimize(int sock)
{
    int result = 0;
    int optval;

    /* TCP_NODELAY - Disable Nagle's algorithm */
    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Large buffers for Simple protocol (4MB) */
    optval = 4 * 1024 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optval, sizeof(optval)) != 0)
        result = -1;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Keep-alive for connection monitoring */
    optval = 1;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(optval));

    /* TCP_QUICKACK - Disable delayed ACKs */
    optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval));

    return result;
}

/*
 * ncclSocketOptimizeLowLatency - Optimize for LL/LL128 protocols (small messages)
 *
 * Configures socket with smaller buffers optimal for low-latency protocols.
 * Buffer size matches NCCL's LL128 buffer requirements.
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketOptimizeLowLatency(int sock)
{
    int result = 0;
    int optval;

    /* TCP_NODELAY - Critical for low latency */
    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Smaller buffers for LL protocols (256KB) */
    optval = 256 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optval, sizeof(optval)) != 0)
        result = -1;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optval, sizeof(optval)) != 0)
        result = -1;

    /* TCP_QUICKACK - Immediate ACKs */
    optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval));

    return result;
}

/*
 * ncclSocketOptimizeUltraLowLatency - Optimize for ultra-low latency
 *
 * Aggressive optimizations for the smallest possible latency:
 * - Minimum practical buffer sizes
 * - Busy polling for reduced syscall overhead
 * - TCP_NOTSENT_LOWAT for write-side notification
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketOptimizeUltraLowLatency(int sock)
{
    int result = 0;
    int optval;

    /* TCP_NODELAY - Essential */
    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Smaller buffers (64KB) for minimum latency */
    optval = 64 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optval, sizeof(optval));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optval, sizeof(optval));

    /* TCP_QUICKACK - Immediate ACKs */
    optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval));

    /* SO_BUSY_POLL - Enable busy polling (50 microseconds) */
    optval = 50;
    setsockopt(sock, SOL_SOCKET, SO_BUSY_POLL, &optval, sizeof(optval));

    /* TCP_NOTSENT_LOWAT - Notify when send buffer is nearly empty (16KB) */
    optval = 16 * 1024;
    setsockopt(sock, IPPROTO_TCP, TCP_NOTSENT_LOWAT, &optval, sizeof(optval));

    return result;
}

/*
 * ncclSocketOptimizeMaxThroughput - Optimize for maximum throughput
 *
 * Configures socket for bulk data transfer:
 * - Very large buffers (8MB)
 * - Zero-copy send where available
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketOptimizeMaxThroughput(int sock)
{
    int result = 0;
    int optval;

    /* TCP_NODELAY - Still needed to avoid small-packet delays */
    optval = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)) != 0)
        result = -1;

    /* Very large buffers (8MB) for max throughput */
    optval = 8 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &optval, sizeof(optval));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &optval, sizeof(optval));

    /* TCP_QUICKACK - Faster ACKs */
    optval = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval));

    /* Enable zero-copy if available (kernel 4.14+) */
    optval = 1;
    setsockopt(sock, SOL_SOCKET, SO_ZEROCOPY, &optval, sizeof(optval));

    return result;
}

/*
 * ncclSocketEnableFastOpen - Enable TCP Fast Open
 *
 * TCP Fast Open allows data to be sent in the SYN packet,
 * reducing connection establishment latency.
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketEnableFastOpen(int sock)
{
    int optval = 5; /* Queue length for TFO */
    return setsockopt(sock, IPPROTO_TCP, TCP_FASTOPEN, &optval, sizeof(optval));
}

/*
 * ncclSocketSetPriority - Set socket priority using IP TOS
 *
 * Maps priority levels to DSCP values:
 * - 7: Highest (Network Control) - EF/DSCP 46 (0xB8)
 * - 6: Voice - AF41/DSCP 34 (0x80)
 * - 5: Video - AF31/DSCP 26 (0x60)
 * - 4: Controlled Load - AF21/DSCP 18 (0x40)
 * - 0-3: Best Effort (0x00)
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketSetPriority(int sock, int priority)
{
    int tos;
    switch (priority)
    {
    case 7:
        tos = 0xB8;
        break; /* EF - Expedited Forwarding */
    case 6:
        tos = 0x80;
        break; /* AF41 */
    case 5:
        tos = 0x60;
        break; /* AF31 */
    case 4:
        tos = 0x40;
        break; /* AF21 */
    default:
        tos = 0x00;
        break; /* Best Effort */
    }
    return setsockopt(sock, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
}

/*
 * ncclSocketSetCpuAffinity - Bind socket to specific CPU for NUMA optimization
 *
 * SO_INCOMING_CPU hints the kernel to process packets on the specified CPU,
 * which can improve cache locality on NUMA systems.
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketSetCpuAffinity(int sock, int cpu)
{
    return setsockopt(sock, SOL_SOCKET, SO_INCOMING_CPU, &cpu, sizeof(cpu));
}

/*
 * ncclSocketSetNonBlocking - Set socket to non-blocking mode
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketSetNonBlocking(int sock)
{
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1)
        return -1;
    return fcntl(sock, F_SETFL, flags | O_NONBLOCK);
}

/*
 * ncclSocketSetBlocking - Set socket to blocking mode
 *
 * Returns: 0 on success, -1 on failure
 */
static inline int ncclSocketSetBlocking(int sock)
{
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1)
        return -1;
    return fcntl(sock, F_SETFL, flags & ~O_NONBLOCK);
}

/* ========================================================================== */
/*                    Socket Information Functions                            */
/* ========================================================================== */

/*
 * ncclSocketGetBufferSizes - Get current socket buffer sizes
 */
static inline int ncclSocketGetBufferSizes(int sock, int *sendBuf, int *recvBuf)
{
    socklen_t len = sizeof(int);
    int result = 0;

    if (sendBuf)
    {
        if (getsockopt(sock, SOL_SOCKET, SO_SNDBUF, sendBuf, &len) != 0)
            result = -1;
    }
    if (recvBuf)
    {
        if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF, recvBuf, &len) != 0)
            result = -1;
    }
    return result;
}

#endif /* NCCL_PLATFORM_LINUX_SOCKET_H_ */
