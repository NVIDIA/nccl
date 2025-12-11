/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Windows Socket Network Plugin
 *
 * A socket-based network plugin for Windows that enables multi-GPU NCCL
 * communication using TCP/IP sockets. This allows NCCL to work on Windows
 * systems where P2P GPU access is not available.
 *
 * Features:
 * - Multi-GPU support via TCP sockets
 * - Compatible with NCCL's plugin interface (v11)
 * - Uses Windows Winsock2 API
 */

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS

#include "nccl_net.h"
#include "debug.h"
#include "checks.h"
#include "socket.h"
#include "param.h"

#include <string.h>
#include <stdlib.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <mutex>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

// Plugin metadata
#define NCCL_NET_SOCKET_WIN_NAME "Socket"
#define MAX_IFS 16
#define MAX_REQUESTS 32

// Use existing definitions from socket.h for MAX_IF_NAME_SIZE, SOCKET_NAME_MAXLEN, NCCL_SOCKET_MAGIC

// Connection state machine
enum ncclSocketCommState
{
    ncclSocketCommStateStart = 0,
    ncclSocketCommStateConnect = 1,
    ncclSocketCommStateAccept = 2,
    ncclSocketCommStateReady = 3
};

// Forward declaration
struct ncclSocketComm;

// Staging structure for async connect/accept
struct ncclSocketCommStage
{
    enum ncclSocketCommState state;
    struct ncclSocketComm *comm;
    SOCKET sock;
};

// Socket handle structure for connection info - passed via bootstrap
struct ncclSocketHandle
{
    uint64_t magic;
    struct sockaddr_in connectAddr;
    struct ncclSocketCommStage stage; // Staging for async connect
};

// Listen comm structure
struct ncclSocketListenComm
{
    SOCKET sock;
    int dev;
    struct ncclSocketCommStage stage; // Staging for async accept
};

// Send/Recv comm structure
struct ncclSocketComm
{
    SOCKET sock;
    int dev;
};

// Request structure for async operations
#define SOCKET_CTRL_SIZE 4 // Size header: 4 bytes for message size
struct ncclSocketRequest
{
    int used;
    int done;
    void *data;
    size_t size;       // Expected max size (recv) or actual size (send)
    size_t actualSize; // Actual message size (received from header)
    size_t offset;
    int op;    // 0 = send, 1 = recv
    int phase; // 0 = header, 1 = data
    struct ncclSocketComm *comm;
    char sizeHeader[SOCKET_CTRL_SIZE]; // Buffer for size header
};

// Device info
struct ncclSocketDevice
{
    char name[MAX_IF_NAME_SIZE];
    struct sockaddr_in addr;
    char *pciPath;
    int speed;
};

static struct ncclSocketDevice socketDevices[MAX_IFS];
static int numSocketDevices = -1;
static std::mutex socketMutex;
static int wsaInitialized = 0;

// Initialize Winsock
static ncclResult_t initWinsock()
{
    if (!wsaInitialized)
    {
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0)
        {
            WARN("NET/Socket: WSAStartup failed with error: %d", result);
            return ncclSystemError;
        }
        wsaInitialized = 1;
    }
    return ncclSuccess;
}

// Find network interfaces
static ncclResult_t findInterfaces()
{
    if (numSocketDevices >= 0)
        return ncclSuccess;

    std::lock_guard<std::mutex> lock(socketMutex);
    if (numSocketDevices >= 0)
        return ncclSuccess;

    NCCLCHECK(initWinsock());

    numSocketDevices = 0;

    // Get adapter info
    ULONG bufLen = 15000;
    PIP_ADAPTER_ADDRESSES addresses = NULL;
    DWORD ret;

    do
    {
        addresses = (PIP_ADAPTER_ADDRESSES)malloc(bufLen);
        if (!addresses)
            return ncclSystemError;

        ret = GetAdaptersAddresses(AF_INET, GAA_FLAG_INCLUDE_PREFIX, NULL, addresses, &bufLen);
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
        WARN("NET/Socket: GetAdaptersAddresses failed");
        return ncclSystemError;
    }

    // Iterate through adapters
    for (PIP_ADAPTER_ADDRESSES adapter = addresses; adapter && numSocketDevices < MAX_IFS; adapter = adapter->Next)
    {
        // Skip loopback and non-operational adapters
        if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
            continue;
        if (adapter->OperStatus != IfOperStatusUp)
            continue;

        // Get first IPv4 address
        for (PIP_ADAPTER_UNICAST_ADDRESS unicast = adapter->FirstUnicastAddress; unicast; unicast = unicast->Next)
        {
            if (unicast->Address.lpSockaddr->sa_family == AF_INET)
            {
                struct sockaddr_in *addr = (struct sockaddr_in *)unicast->Address.lpSockaddr;

                // Skip link-local addresses
                if ((ntohl(addr->sin_addr.s_addr) & 0xFFFF0000) == 0xA9FE0000)
                    continue;

                // Store device info
                WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1,
                                    socketDevices[numSocketDevices].name, MAX_IF_NAME_SIZE, NULL, NULL);
                memcpy(&socketDevices[numSocketDevices].addr, addr, sizeof(struct sockaddr_in));
                socketDevices[numSocketDevices].pciPath = NULL;
                socketDevices[numSocketDevices].speed = 10000; // Default 10 Gbps

                // Try to get actual speed
                if (adapter->TransmitLinkSpeed != (ULONG64)-1)
                {
                    socketDevices[numSocketDevices].speed = (int)(adapter->TransmitLinkSpeed / 1000000);
                }

                char addrStr[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &addr->sin_addr, addrStr, sizeof(addrStr));
                INFO(NCCL_INIT | NCCL_NET, "NET/Socket: Found interface [%d] %s: %s (%d Mbps)",
                     numSocketDevices, socketDevices[numSocketDevices].name, addrStr,
                     socketDevices[numSocketDevices].speed);

                numSocketDevices++;
                break;
            }
        }
    }

    free(addresses);

    if (numSocketDevices == 0)
    {
        WARN("NET/Socket: No network interfaces found");
        return ncclSystemError;
    }

    INFO(NCCL_INIT | NCCL_NET, "NET/Socket: Using %d network interface(s)", numSocketDevices);
    return ncclSuccess;
}

// Plugin interface implementation

static ncclResult_t socketInit(void **ctx, uint64_t commId, ncclNetCommConfig_v11_t *config,
                               ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction)
{
    (void)commId;
    (void)config;
    (void)logFunction;
    (void)profFunction;


    NCCLCHECK(findInterfaces());
    *ctx = NULL;


    return ncclSuccess;
}

static ncclResult_t socketDevices_fn(int *ndev)
{
    if (numSocketDevices < 0)
    {
        NCCLCHECK(findInterfaces());
    }
    *ndev = numSocketDevices > 0 ? numSocketDevices : 0;
    return ncclSuccess;
}

static ncclResult_t socketGetProperties(int dev, ncclNetProperties_v11_t *props)
{

    if (dev < 0 || dev >= numSocketDevices || props == NULL)
        return ncclInvalidArgument;

    memset(props, 0, sizeof(*props));
    props->name = socketDevices[dev].name;
    props->pciPath = socketDevices[dev].pciPath;
    props->guid = dev;
    props->ptrSupport = NCCL_PTR_HOST;
    props->regIsGlobal = 0;
    props->forceFlush = 0;
    props->speed = socketDevices[dev].speed;
    props->port = 0;
    props->maxComms = 65536;
    props->maxRecvs = 1;
    props->latency = 0;
    props->netDeviceType = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    props->maxP2pBytes = MAX_NET_SIZE;
    props->maxCollBytes = MAX_COLLNET_SIZE;
    props->maxMultiRequestSize = 1;

    return ncclSuccess;
}

static ncclResult_t socketListen(void *ctx, int dev, void *opaqueHandle, void **listenComm)
{
    (void)ctx;
    static_assert(sizeof(struct ncclSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");


    if (dev < 0 || dev >= numSocketDevices)
        return ncclInvalidArgument;

    struct ncclSocketHandle *handle = (struct ncclSocketHandle *)opaqueHandle;
    memset(handle, 0, sizeof(struct ncclSocketHandle));

    struct ncclSocketListenComm *comm = (struct ncclSocketListenComm *)calloc(1, sizeof(struct ncclSocketListenComm));
    if (!comm)
        return ncclSystemError;

    // Create listening socket
    comm->sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (comm->sock == INVALID_SOCKET)
    {
        WARN("NET/Socket: socket() failed with error: %d", WSAGetLastError());
        free(comm);
        return ncclSystemError;
    }

    // Allow address reuse
    int yes = 1;
    setsockopt(comm->sock, SOL_SOCKET, SO_REUSEADDR, (char *)&yes, sizeof(yes));

    // Set non-blocking for accept later
    u_long mode = 1;
    ioctlsocket(comm->sock, FIONBIO, &mode);

    // Bind to device address with port 0 (let OS assign)
    struct sockaddr_in bindAddr = socketDevices[dev].addr;
    bindAddr.sin_port = 0;

    if (bind(comm->sock, (struct sockaddr *)&bindAddr, sizeof(bindAddr)) == SOCKET_ERROR)
    {
        WARN("NET/Socket: bind() failed with error: %d", WSAGetLastError());
        closesocket(comm->sock);
        free(comm);
        return ncclSystemError;
    }

    // Start listening
    if (listen(comm->sock, 128) == SOCKET_ERROR)
    {
        WARN("NET/Socket: listen() failed with error: %d", WSAGetLastError());
        closesocket(comm->sock);
        free(comm);
        return ncclSystemError;
    }

    // Get assigned port
    struct sockaddr_in listenAddr;
    int addrLen = sizeof(listenAddr);
    if (getsockname(comm->sock, (struct sockaddr *)&listenAddr, &addrLen) == SOCKET_ERROR)
    {
        WARN("NET/Socket: getsockname() failed with error: %d", WSAGetLastError());
        closesocket(comm->sock);
        free(comm);
        return ncclSystemError;
    }

    // Fill handle with connection info - this gets sent to the connecting peer
    handle->magic = NCCL_SOCKET_MAGIC;
    memcpy(&handle->connectAddr, &listenAddr, sizeof(listenAddr));
    handle->connectAddr.sin_addr = socketDevices[dev].addr.sin_addr;
    // stage is initialized to zeros by memset above

    comm->dev = dev;
    // stage is initialized to zeros by calloc above
    *listenComm = comm;

    char addrStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &handle->connectAddr.sin_addr, addrStr, sizeof(addrStr));
    INFO(NCCL_NET, "NET/Socket: Listening on %s:%d", addrStr, ntohs(handle->connectAddr.sin_port));

    return ncclSuccess;
}

static ncclResult_t socketConnect(void *ctx, int dev, void *opaqueHandle, void **sendComm,
                                  ncclNetDeviceHandle_v11_t **sendDevComm)
{
    (void)ctx;
    if (sendDevComm)
        *sendDevComm = NULL;
    *sendComm = NULL;

    struct ncclSocketHandle *handle = (struct ncclSocketHandle *)opaqueHandle;
    struct ncclSocketCommStage *stage = &handle->stage;
    struct ncclSocketComm *comm = stage->comm;

    TRACE(NCCL_NET, "NET/Socket: socketConnect dev=%d stage->state=%d handle=%p", dev, stage->state, handle);

    // Check if we're in the middle of connecting
    if (stage->state == ncclSocketCommStateConnect)
    {
        // Check if the non-blocking connect completed
        fd_set writefds;
        FD_ZERO(&writefds);
        FD_SET(stage->sock, &writefds);
        struct timeval tv = {0, 0};

        int result = select(0, NULL, &writefds, NULL, &tv);
        if (result > 0)
        {
            // Check if connect succeeded
            int err = 0;
            int len = sizeof(err);
            getsockopt(stage->sock, SOL_SOCKET, SO_ERROR, (char *)&err, &len);
            if (err == 0)
            {
                // Connect succeeded - keep socket non-blocking for async progress (isend/test pattern)
                stage->state = ncclSocketCommStateReady;
                *sendComm = comm;
                INFO(NCCL_NET, "NET/Socket: Connected successfully");
                return ncclSuccess;
            }
            else
            {
                WARN("NET/Socket: connect() failed with error: %d", err);
                closesocket(stage->sock);
                free(comm);
                stage->comm = NULL;
                stage->state = ncclSocketCommStateStart;
                return ncclSystemError;
            }
        }
        else if (result == 0)
        {
            // Still connecting - return with NULL comm
            return ncclSuccess;
        }
        else
        {
            WARN("NET/Socket: select() failed with error: %d", WSAGetLastError());
            closesocket(stage->sock);
            free(comm);
            stage->comm = NULL;
            stage->state = ncclSocketCommStateStart;
            return ncclSystemError;
        }
    }

    // First call - validate handle and start connect
    if (handle->magic != NCCL_SOCKET_MAGIC)
    {
        WARN("NET/Socket: Invalid handle magic 0x%llx (expected 0x%llx)", (unsigned long long)handle->magic, (unsigned long long)NCCL_SOCKET_MAGIC);
        return ncclInvalidArgument;
    }

    INFO(NCCL_NET, "NET/Socket: socketConnect starting connection, dev=%d", dev);

    comm = (struct ncclSocketComm *)calloc(1, sizeof(struct ncclSocketComm));
    if (!comm)
        return ncclSystemError;

    stage->comm = comm;
    comm->dev = dev;

    // Create socket
    comm->sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (comm->sock == INVALID_SOCKET)
    {
        WARN("NET/Socket: socket() failed with error: %d", WSAGetLastError());
        free(comm);
        stage->comm = NULL;
        return ncclSystemError;
    }

    {
        // Disable Nagle's algorithm for lower latency
        int yes = 1;
        setsockopt(comm->sock, IPPROTO_TCP, TCP_NODELAY, (char *)&yes, sizeof(yes));

        // Set non-blocking mode for async connect
        u_long mode = 1;
        ioctlsocket(comm->sock, FIONBIO, &mode);

        // Start non-blocking connect
        char addrStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &handle->connectAddr.sin_addr, addrStr, sizeof(addrStr));
        INFO(NCCL_NET, "NET/Socket: Connecting to %s:%d", addrStr, ntohs(handle->connectAddr.sin_port));

        int result = connect(comm->sock, (struct sockaddr *)&handle->connectAddr, sizeof(handle->connectAddr));
        if (result == 0)
        {
            // Connected immediately (unlikely but possible)
            // Keep socket non-blocking for async progress (isend/test pattern)
            stage->state = ncclSocketCommStateReady;
            *sendComm = comm;
            INFO(NCCL_NET, "NET/Socket: Connected immediately");
            return ncclSuccess;
        }

        int err = WSAGetLastError();
        if (err != WSAEWOULDBLOCK)
        {
            WARN("NET/Socket: connect() failed with error: %d", err);
            closesocket(comm->sock);
            free(comm);
            stage->comm = NULL;
            return ncclSystemError;
        }
    }

    // Connection in progress
    stage->state = ncclSocketCommStateConnect;
    stage->sock = comm->sock;
    return ncclSuccess; // Return with *sendComm = NULL, caller will poll
}

static ncclResult_t socketAccept(void *listenComm, void **recvComm, ncclNetDeviceHandle_v11_t **recvDevComm)
{
    if (recvDevComm)
        *recvDevComm = NULL;
    *recvComm = NULL;

    struct ncclSocketListenComm *lcomm = (struct ncclSocketListenComm *)listenComm;
    if (!lcomm)
        return ncclInvalidArgument;

    struct ncclSocketCommStage *stage = &lcomm->stage;
    struct ncclSocketComm *comm = stage->comm;

    TRACE(NCCL_NET, "NET/Socket: socketAccept listenComm=%p sock=%lld", lcomm, (long long)lcomm->sock);

    // Try to accept (non-blocking since we set it in listen)
    struct sockaddr_in clientAddr;
    int addrLen = sizeof(clientAddr);

    SOCKET clientSock = accept(lcomm->sock, (struct sockaddr *)&clientAddr, &addrLen);
    if (clientSock == INVALID_SOCKET)
    {
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK)
        {
            // No connection yet - return with NULL comm
            return ncclSuccess;
        }
        WARN("NET/Socket: accept() failed with error: %d", err);
        return ncclSystemError;
    }

    // Got a connection!
    comm = (struct ncclSocketComm *)calloc(1, sizeof(struct ncclSocketComm));
    if (!comm)
    {
        closesocket(clientSock);
        return ncclSystemError;
    }

    comm->sock = clientSock;
    comm->dev = lcomm->dev;

    // Disable Nagle's algorithm
    int yes = 1;
    setsockopt(comm->sock, IPPROTO_TCP, TCP_NODELAY, (char *)&yes, sizeof(yes));

    // Keep socket non-blocking for async progress (irecv/test pattern)
    u_long mode = 1;
    ioctlsocket(comm->sock, FIONBIO, &mode);

    *recvComm = comm;

    char addrStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &clientAddr.sin_addr, addrStr, sizeof(addrStr));
    INFO(NCCL_NET, "NET/Socket: Accepted connection from %s:%d", addrStr, ntohs(clientAddr.sin_port));

    return ncclSuccess;
}

static ncclResult_t socketRegMr(void *comm, void *data, size_t size, int type, void **mhandle)
{
    (void)comm;
    (void)size;
    (void)type;
    *mhandle = data;
    return ncclSuccess;
}

static ncclResult_t socketRegMrDmaBuf(void *comm, void *data, size_t size, int type,
                                      uint64_t offset, int fd, void **mhandle)
{
    (void)offset;
    (void)fd;
    return socketRegMr(comm, data, size, type, mhandle);
}

static ncclResult_t socketDeregMr(void *comm, void *mhandle)
{
    (void)comm;
    (void)mhandle;
    return ncclSuccess;
}

static ncclResult_t socketIsend(void *sendComm, void *data, size_t size, int tag,
                                void *mhandle, void *phandle, void **request)
{
    (void)tag;
    (void)mhandle;
    (void)phandle;


    struct ncclSocketComm *comm = (struct ncclSocketComm *)sendComm;

    struct ncclSocketRequest *req = (struct ncclSocketRequest *)calloc(1, sizeof(struct ncclSocketRequest));
    if (!req)
        return ncclSystemError;

    req->used = 1;
    req->data = data;
    req->size = size;
    req->actualSize = size;
    req->offset = 0;
    req->op = 0;    // send
    req->phase = 0; // Start with header phase
    req->comm = comm;
    req->done = 0;

    // Prepare size header
    memcpy(req->sizeHeader, &size, SOCKET_CTRL_SIZE);

    *request = req;
    return ncclSuccess;
}

static ncclResult_t socketIrecv(void *recvComm, int n, void **data, size_t *sizes, int *tags,
                                void **mhandles, void **phandles, void **request)
{
    (void)tags;
    (void)mhandles;
    (void)phandles;

    if (n != 1)
    {
        WARN("NET/Socket: Multiple receives not supported");
        return ncclInvalidArgument;
    }

    struct ncclSocketComm *comm = (struct ncclSocketComm *)recvComm;

    struct ncclSocketRequest *req = (struct ncclSocketRequest *)calloc(1, sizeof(struct ncclSocketRequest));
    if (!req)
        return ncclSystemError;

    req->used = 1;
    req->data = data[0];
    req->size = sizes[0]; // Max buffer size
    req->actualSize = 0;  // Will be filled from header
    req->offset = 0;
    req->op = 1;    // recv
    req->phase = 0; // Start with header phase
    req->comm = comm;
    req->done = 0;

    *request = req;
    return ncclSuccess;
}

static ncclResult_t socketIflush(void *recvComm, int n, void **data, int *sizes,
                                 void **mhandles, void **request)
{
    (void)recvComm;
    (void)n;
    (void)data;
    (void)sizes;
    (void)mhandles;
    *request = NULL;
    return ncclSuccess;
}

static ncclResult_t socketTest(void *request, int *done, int *sizes)
{
    struct ncclSocketRequest *req = (struct ncclSocketRequest *)request;

    if (!req)
    {
        *done = 1;
        if (sizes)
            *sizes = 0;
        return ncclSuccess;
    }

    if (req->done)
    {
        *done = 1;
        if (sizes)
            *sizes = (int)req->actualSize;
        return ncclSuccess;
    }

    struct ncclSocketComm *comm = req->comm;

    // Phase 0: Send/receive size header
    if (req->phase == 0)
    {
        char *ptr = req->sizeHeader + req->offset;
        int remaining = SOCKET_CTRL_SIZE - (int)req->offset;

        if (remaining > 0)
        {
            int result;
            if (req->op == 0)
            {
                // Send header
                result = send(comm->sock, ptr, remaining, 0);
            }
            else
            {
                // Recv header
                result = recv(comm->sock, ptr, remaining, 0);
            }

            if (result == SOCKET_ERROR)
            {
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK)
                {
                    WARN("NET/Socket: %s header failed with error: %d", req->op == 0 ? "send" : "recv", err);
                    return ncclSystemError;
                }
                *done = 0;
                return ncclSuccess;
            }
            else if (result == 0 && req->op == 1)
            {
                WARN("NET/Socket: Connection closed during recv header");
                return ncclSystemError;
            }

            req->offset += result;
        }

        if (req->offset >= SOCKET_CTRL_SIZE)
        {
            // Header complete, move to data phase
            req->phase = 1;
            req->offset = 0;

            if (req->op == 1)
            {
                // Recv: extract actual size from header
                memcpy(&req->actualSize, req->sizeHeader, SOCKET_CTRL_SIZE);
                if (req->actualSize > req->size)
                {
                    WARN("NET/Socket: Received size %zu exceeds buffer %zu", req->actualSize, req->size);
                    return ncclSystemError;
                }
            }
        }
        else
        {
            *done = 0;
            return ncclSuccess;
        }
    }

    // Phase 1: Send/receive actual data
    if (req->phase == 1)
    {
        size_t dataSize = (req->op == 0) ? req->size : req->actualSize;
        char *ptr = (char *)req->data + req->offset;
        int remaining = (int)(dataSize - req->offset);

        if (remaining > 0)
        {
            int result;
            if (req->op == 0)
            {
                result = send(comm->sock, ptr, remaining, 0);
            }
            else
            {
                result = recv(comm->sock, ptr, remaining, 0);
            }

            if (result == SOCKET_ERROR)
            {
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK)
                {
                    WARN("NET/Socket: %s data failed with error: %d", req->op == 0 ? "send" : "recv", err);
                    return ncclSystemError;
                }
                *done = 0;
                return ncclSuccess;
            }
            else if (result == 0 && req->op == 1)
            {
                WARN("NET/Socket: Connection closed during recv data");
                return ncclSystemError;
            }

            req->offset += result;
        }

        if (req->offset >= dataSize)
        {
            req->done = 1;
            *done = 1;
            if (sizes)
                *sizes = (int)dataSize;
        }
        else
        {
            *done = 0;
        }
    }

    return ncclSuccess;
}

static ncclResult_t socketCloseSend(void *sendComm)
{
    struct ncclSocketComm *comm = (struct ncclSocketComm *)sendComm;
    if (comm)
    {
        if (comm->sock != INVALID_SOCKET)
        {
            closesocket(comm->sock);
        }
        free(comm);
    }
    return ncclSuccess;
}

static ncclResult_t socketCloseRecv(void *recvComm)
{
    return socketCloseSend(recvComm);
}

static ncclResult_t socketCloseListen(void *listenComm)
{
    struct ncclSocketListenComm *comm = (struct ncclSocketListenComm *)listenComm;
    if (comm)
    {
        if (comm->sock != INVALID_SOCKET)
        {
            closesocket(comm->sock);
        }
        free(comm);
    }
    return ncclSuccess;
}

static ncclResult_t socketGetDeviceMr(void *comm, void *mhandle, void **dptr_mhandle)
{
    (void)comm;
    (void)mhandle;
    *dptr_mhandle = NULL;
    return ncclSuccess;
}

static ncclResult_t socketIrecvConsumed(void *recvComm, int n, void *request)
{
    (void)recvComm;
    (void)n;
    if (request)
        free(request);
    return ncclSuccess;
}

static ncclResult_t socketMakeVDevice(int *d, ncclNetVDeviceProps_v11_t *props)
{
    (void)props;
    *d = 0;
    return ncclSuccess;
}

static ncclResult_t socketFinalize(void *ctx)
{
    (void)ctx;
    if (wsaInitialized)
    {
        WSACleanup();
        wsaInitialized = 0;
    }
    numSocketDevices = -1;
    return ncclSuccess;
}

static ncclResult_t socketSetNetAttr(void *ctx, ncclNetAttr_v11_t *netAttr)
{
    (void)ctx;
    (void)netAttr;
    return ncclSuccess;
}

// NCCL Net plugin structure (v11)
ncclNet_t ncclNetSocket = {
    .name = NCCL_NET_SOCKET_WIN_NAME,
    .init = socketInit,
    .devices = socketDevices_fn,
    .getProperties = socketGetProperties,
    .listen = socketListen,
    .connect = socketConnect,
    .accept = socketAccept,
    .regMr = socketRegMr,
    .regMrDmaBuf = socketRegMrDmaBuf,
    .deregMr = socketDeregMr,
    .isend = socketIsend,
    .irecv = socketIrecv,
    .iflush = socketIflush,
    .test = socketTest,
    .closeSend = socketCloseSend,
    .closeRecv = socketCloseRecv,
    .closeListen = socketCloseListen,
    .getDeviceMr = socketGetDeviceMr,
    .irecvConsumed = socketIrecvConsumed,
    .makeVDevice = socketMakeVDevice,
    .finalize = socketFinalize,
    .setNetAttr = socketSetNetAttr,
};

#endif // NCCL_PLATFORM_WINDOWS
