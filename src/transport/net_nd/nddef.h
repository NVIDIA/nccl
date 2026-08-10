//
// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//
//
// Module Name:
//
//    nddef.h
//
// Abstract:
//
//    NetworkDirect Service Provider structure definitions
//
// Environment:
//
//    User mode and kernel mode
//


#ifndef _NDDEF_H_
#define _NDDEF_H_

#pragma once

#define ND_VERSION_1    0x1
#define ND_VERSION_2    0x20000

#ifndef NDVER
#define NDVER      ND_VERSION_2
#endif

#define ND_ADAPTER_FLAG_IN_ORDER_DMA_SUPPORTED              0x00000001
#define ND_ADAPTER_FLAG_CQ_INTERRUPT_MODERATION_SUPPORTED   0x00000004
#define ND_ADAPTER_FLAG_MULTI_ENGINE_SUPPORTED              0x00000008
#define ND_ADAPTER_FLAG_CQ_RESIZE_SUPPORTED                 0x00000100
#define ND_ADAPTER_FLAG_LOOPBACK_CONNECTIONS_SUPPORTED      0x00010000

#define ND_CQ_NOTIFY_ERRORS                                 0
#define ND_CQ_NOTIFY_ANY                                    1
#define ND_CQ_NOTIFY_SOLICITED                              2

#define ND_MR_FLAG_ALLOW_LOCAL_WRITE                        0x00000001
#define ND_MR_FLAG_ALLOW_REMOTE_READ                        0x00000002
#define ND_MR_FLAG_ALLOW_REMOTE_WRITE                       0x00000005
#define ND_MR_FLAG_RDMA_READ_SINK                           0x00000008
#define ND_MR_FLAG_DO_NOT_SECURE_VM                         0x80000000

#define ND_OP_FLAG_SILENT_SUCCESS                           0x00000001
#define ND_OP_FLAG_READ_FENCE                               0x00000002
#define ND_OP_FLAG_SEND_AND_SOLICIT_EVENT                   0x00000004
#define ND_OP_FLAG_ALLOW_READ                               0x00000008
#define ND_OP_FLAG_ALLOW_WRITE                              0x00000010
#if NDVER >= ND_VERSION_2
#define ND_OP_FLAG_INLINE                                   0x00000020
#endif

typedef struct _ND2_ADAPTER_INFO {
    ULONG   InfoVersion;
    UINT16  VendorId;
    UINT16  DeviceId;
    UINT64  AdapterId;
    SIZE_T  MaxRegistrationSize;
    SIZE_T  MaxWindowSize;
    ULONG   MaxInitiatorSge;
    ULONG   MaxReceiveSge;
    ULONG   MaxReadSge;
    ULONG   MaxTransferLength;
    ULONG   MaxInlineDataSize;
    ULONG   MaxInboundReadLimit;
    ULONG   MaxOutboundReadLimit;
    ULONG   MaxReceiveQueueDepth;
    ULONG   MaxInitiatorQueueDepth;
    ULONG   MaxSharedReceiveQueueDepth;
    ULONG   MaxCompletionQueueDepth;
    ULONG   InlineRequestThreshold;
    ULONG   LargeRequestThreshold;
    ULONG   MaxCallerData;
    ULONG   MaxCalleeData;
    ULONG   AdapterFlags;
} ND2_ADAPTER_INFO;

typedef struct _ND2_ADAPTER_INFO32 {
    ULONG   InfoVersion;
    UINT16  VendorId;
    UINT16  DeviceId;
    UINT64  AdapterId;
    ULONG   MaxRegistrationSize;
    ULONG   MaxWindowSize;
    ULONG   MaxInitiatorSge;
    ULONG   MaxReceiveSge;
    ULONG   MaxReadSge;
    ULONG   MaxTransferLength;
    ULONG   MaxInlineDataSize;
    ULONG   MaxInboundReadLimit;
    ULONG   MaxOutboundReadLimit;
    ULONG   MaxReceiveQueueDepth;
    ULONG   MaxInitiatorQueueDepth;
    ULONG   MaxSharedReceiveQueueDepth;
    ULONG   MaxCompletionQueueDepth;
    ULONG   InlineRequestThreshold;
    ULONG   LargeRequestThreshold;
    ULONG   MaxCallerData;
    ULONG   MaxCalleeData;
    ULONG   AdapterFlags;
} ND2_ADAPTER_INFO32;

typedef enum _ND2_REQUEST_TYPE {
    Nd2RequestTypeReceive,
    Nd2RequestTypeSend,
    Nd2RequestTypeBind,
    Nd2RequestTypeInvalidate,
    Nd2RequestTypeRead,
    Nd2RequestTypeWrite
} ND2_REQUEST_TYPE;

typedef struct _ND2_RESULT {
    HRESULT             Status;
    ULONG               BytesTransferred;
    VOID*               QueuePairContext;
    VOID*               RequestContext;
    ND2_REQUEST_TYPE    RequestType;
} ND2_RESULT;

typedef struct _ND2_SGE {
    VOID*   Buffer;
    ULONG   BufferLength;
    UINT32  MemoryRegionToken;
} ND2_SGE;

#endif // _NDDEF_H_
