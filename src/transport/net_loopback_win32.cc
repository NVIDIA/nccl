/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Windows Loopback Network Plugin
 *
 * A minimal network plugin for Windows that enables single-node NCCL operation.
 * This plugin provides a loopback transport for communication within a single
 * process, allowing NCCL to work on Windows without InfiniBand or external
 * network plugins.
 *
 * Features:
 * - Single-process multi-GPU support
 * - Loopback communication via shared memory
 * - Compatible with NCCL's plugin interface (v11)
 */

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS

#include "nccl_net.h"
#include "debug.h"
#include "checks.h"
#include "socket.h"

#include <string.h>
#include <stdlib.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// Plugin metadata
#define NCCL_NET_LOOPBACK_NAME "Loopback"
#define NCCL_NET_LOOPBACK_MAX_REQUESTS 32

// Loopback handle - shared memory based communication
struct ncclLoopbackHandle
{
    uint64_t magic;
    int rank;
    char addr[128]; // Address string for identification
};

// Connection state for loopback transport
struct ncclLoopbackConnection
{
    int connected;
    void *sendBuf;
    void *recvBuf;
    size_t bufSize;
    HANDLE sendEvent;
    HANDLE recvEvent;
};

// Request tracking
struct ncclLoopbackRequest
{
    int used;
    int done;
    void *data;
    size_t size;
};

// Device state
struct ncclLoopbackDevice
{
    int initialized;
    char name[64];
};

static struct ncclLoopbackDevice loopbackDevices[1] = {0};
static int loopbackInitialized = 0;
static void *loopbackContext = NULL;

// Plugin interface implementation (v11 signatures)

static ncclResult_t loopbackInit(void **ctx, uint64_t commId, ncclNetCommConfig_v11_t *config,
                                 ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction)
{
    (void)commId;
    (void)config;
    (void)logFunction;
    (void)profFunction;

    if (!loopbackInitialized)
    {
        // Initialize single loopback device
        loopbackDevices[0].initialized = 1;
        snprintf(loopbackDevices[0].name, sizeof(loopbackDevices[0].name), "Loopback0");
        loopbackInitialized = 1;
        INFO(NCCL_INIT | NCCL_NET, "Loopback network plugin initialized");
    }
    *ctx = &loopbackContext;
    return ncclSuccess;
}

static ncclResult_t loopbackDevices_fn(int *ndev)
{
    *ndev = 1; // Single loopback device
    return ncclSuccess;
}

static ncclResult_t loopbackGetProperties(int dev, ncclNetProperties_v11_t *props)
{
    if (dev != 0 || props == NULL)
        return ncclInvalidArgument;

    memset(props, 0, sizeof(*props));
    props->name = loopbackDevices[0].name;
    props->pciPath = NULL; // No PCI device
    props->guid = 0;
    props->ptrSupport = NCCL_PTR_HOST; // Host pointers only
    props->regIsGlobal = 0;
    props->forceFlush = 0;
    props->speed = 100000; // 100 Gbps virtual speed
    props->port = 0;
    props->maxComms = 65536;
    props->maxRecvs = 1;
    props->latency = 0.001f; // 1 microsecond latency
    props->netDeviceType = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    props->maxP2pBytes = MAX_NET_SIZE;
    props->maxCollBytes = MAX_COLLNET_SIZE;
    props->maxMultiRequestSize = 1;

    return ncclSuccess;
}

static ncclResult_t loopbackListen(void *ctx, int dev, void *opaqueHandle, void **listenComm)
{
    (void)ctx;
    if (dev != 0)
        return ncclInvalidArgument;

    struct ncclLoopbackHandle *handle = (struct ncclLoopbackHandle *)opaqueHandle;
    handle->magic = 0x4C4F4F5042414B; // "LOOPBAK"
    handle->rank = 0;
    snprintf(handle->addr, sizeof(handle->addr), "loopback://localhost");

    // Create listening "socket" (placeholder)
    *listenComm = malloc(sizeof(struct ncclLoopbackConnection));
    if (*listenComm == NULL)
        return ncclSystemError;
    memset(*listenComm, 0, sizeof(struct ncclLoopbackConnection));

    return ncclSuccess;
}

static ncclResult_t loopbackConnect(void *ctx, int dev, void *opaqueHandle, void **sendComm, ncclNetDeviceHandle_v11_t **sendDevComm)
{
    (void)ctx;
    (void)dev;
    (void)sendDevComm;

    struct ncclLoopbackHandle *handle = (struct ncclLoopbackHandle *)opaqueHandle;
    if (handle->magic != 0x4C4F4F5042414B)
    {
        WARN("Invalid loopback handle");
        return ncclInvalidArgument;
    }

    struct ncclLoopbackConnection *conn = (struct ncclLoopbackConnection *)malloc(sizeof(struct ncclLoopbackConnection));
    if (conn == NULL)
        return ncclSystemError;

    memset(conn, 0, sizeof(*conn));
    conn->connected = 1;

    *sendComm = conn;
    if (sendDevComm)
        *sendDevComm = NULL;
    return ncclSuccess;
}

static ncclResult_t loopbackAccept(void *listenComm, void **recvComm, ncclNetDeviceHandle_v11_t **recvDevComm)
{
    (void)recvDevComm;

    if (listenComm == NULL)
        return ncclInvalidArgument;

    struct ncclLoopbackConnection *conn = (struct ncclLoopbackConnection *)malloc(sizeof(struct ncclLoopbackConnection));
    if (conn == NULL)
        return ncclSystemError;

    memset(conn, 0, sizeof(*conn));
    conn->connected = 1;

    *recvComm = conn;
    if (recvDevComm)
        *recvDevComm = NULL;
    return ncclSuccess;
}

static ncclResult_t loopbackRegMr(void *comm, void *data, size_t size, int type, void **mhandle)
{
    (void)comm;
    (void)data;
    (void)size;
    (void)type;
    // For loopback, memory registration is a no-op
    *mhandle = data; // Just return the data pointer as handle
    return ncclSuccess;
}

static ncclResult_t loopbackRegMrDmaBuf(void *comm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle)
{
    (void)offset;
    (void)fd;
    return loopbackRegMr(comm, data, size, type, mhandle);
}

static ncclResult_t loopbackDeregMr(void *comm, void *mhandle)
{
    (void)comm;
    (void)mhandle;
    // No-op for loopback
    return ncclSuccess;
}

static ncclResult_t loopbackIsend(void *sendComm, void *data, size_t size, int tag, void *mhandle, void *phandle, void **request)
{
    (void)sendComm;
    (void)tag;
    (void)mhandle;
    (void)phandle;

    struct ncclLoopbackRequest *req = (struct ncclLoopbackRequest *)malloc(sizeof(struct ncclLoopbackRequest));
    if (req == NULL)
        return ncclSystemError;

    req->used = 1;
    req->done = 1; // Immediately complete for loopback
    req->data = data;
    req->size = size;

    *request = req;
    return ncclSuccess;
}

static ncclResult_t loopbackIrecv(void *recvComm, int n, void **data, size_t *sizes, int *tags, void **mhandles, void **phandles, void **request)
{
    (void)recvComm;
    (void)n;
    (void)data;
    (void)sizes;
    (void)tags;
    (void)mhandles;
    (void)phandles;

    struct ncclLoopbackRequest *req = (struct ncclLoopbackRequest *)malloc(sizeof(struct ncclLoopbackRequest));
    if (req == NULL)
        return ncclSystemError;

    req->used = 1;
    req->done = 1; // Immediately complete for loopback
    req->data = data ? data[0] : NULL;
    req->size = sizes ? sizes[0] : 0;

    *request = req;
    return ncclSuccess;
}

static ncclResult_t loopbackIflush(void *recvComm, int n, void **data, int *sizes, void **mhandles, void **request)
{
    (void)recvComm;
    (void)n;
    (void)data;
    (void)sizes;
    (void)mhandles;

    *request = NULL;
    return ncclSuccess;
}

static ncclResult_t loopbackTest(void *request, int *done, int *sizes)
{
    struct ncclLoopbackRequest *req = (struct ncclLoopbackRequest *)request;

    if (req == NULL)
    {
        *done = 1;
        if (sizes)
            *sizes = 0;
        return ncclSuccess;
    }

    *done = req->done;
    if (sizes)
        *sizes = (int)req->size;

    return ncclSuccess;
}

static ncclResult_t loopbackClose(void *comm)
{
    if (comm)
        free(comm);
    return ncclSuccess;
}

static ncclResult_t loopbackCloseListen(void *listenComm)
{
    if (listenComm)
        free(listenComm);
    return ncclSuccess;
}

static ncclResult_t loopbackCloseSend(void *sendComm)
{
    return loopbackClose(sendComm);
}

static ncclResult_t loopbackCloseRecv(void *recvComm)
{
    return loopbackClose(recvComm);
}

static ncclResult_t loopbackGetDeviceMr(void *comm, void *mhandle, void **dptr_mhandle)
{
    (void)comm;
    (void)mhandle;
    *dptr_mhandle = NULL;
    return ncclSuccess;
}

static ncclResult_t loopbackIrecvConsumed(void *recvComm, int n, void *request)
{
    (void)recvComm;
    (void)n;
    if (request)
        free(request);
    return ncclSuccess;
}

static ncclResult_t loopbackMakeVDevice(int *d, ncclNetVDeviceProps_v11_t *props)
{
    (void)props;
    *d = 0;
    return ncclSuccess;
}

static ncclResult_t loopbackFinalize(void *ctx)
{
    (void)ctx;
    loopbackInitialized = 0;
    return ncclSuccess;
}

static ncclResult_t loopbackSetNetAttr(void *ctx, ncclNetAttr_v11_t *netAttr)
{
    (void)ctx;
    (void)netAttr;
    return ncclSuccess;
}

// NCCL Net plugin structure (v11)
ncclNet_t ncclNetLoopback = {
    .name = NCCL_NET_LOOPBACK_NAME,
    .init = loopbackInit,
    .devices = loopbackDevices_fn,
    .getProperties = loopbackGetProperties,
    .listen = loopbackListen,
    .connect = loopbackConnect,
    .accept = loopbackAccept,
    .regMr = loopbackRegMr,
    .regMrDmaBuf = loopbackRegMrDmaBuf,
    .deregMr = loopbackDeregMr,
    .isend = loopbackIsend,
    .irecv = loopbackIrecv,
    .iflush = loopbackIflush,
    .test = loopbackTest,
    .closeSend = loopbackCloseSend,
    .closeRecv = loopbackCloseRecv,
    .closeListen = loopbackCloseListen,
    .getDeviceMr = loopbackGetDeviceMr,
    .irecvConsumed = loopbackIrecvConsumed,
    .makeVDevice = loopbackMakeVDevice,
    .finalize = loopbackFinalize,
    .setNetAttr = loopbackSetNetAttr,
};

#endif // NCCL_PLATFORM_WINDOWS
