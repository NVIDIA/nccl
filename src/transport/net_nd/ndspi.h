//
// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//
// Module Name:
//
//    ndspi.h
// 
// Abstract:
//
//    NetworkDirect Service Provider Interfaces
//
// Environment:
//
//    User mode
//


#pragma once

#ifndef _NDSPI_H_
#define _NDSPI_H_

#include <winsock2.h>
#include <unknwn.h>
#include "ndstatus.h"
#include "nddef.h"


//
// Overlapped object
//
#undef INTERFACE
#define INTERFACE IND2Overlapped

// {ABF72719-B016-4a40-A6F7-622791A7044C}
DEFINE_GUID(IID_IND2Overlapped,
    0xabf72719, 0xb016, 0x4a40, 0xa6, 0xf7, 0x62, 0x27, 0x91, 0xa7, 0x4, 0x4c);

DECLARE_INTERFACE_(IND2Overlapped, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Overlapped methods ***
    STDMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    STDMETHOD(GetOverlappedResult)(
        THIS_
        __in OVERLAPPED* pOverlapped,
        BOOL wait
        ) PURE;
};


//
// Completion Queue
//
#undef INTERFACE
#define INTERFACE IND2CompletionQueue

// {20CC445E-64A0-4cbb-AA75-F6A7251FDA9E}
DEFINE_GUID(IID_IND2CompletionQueue,
    0x20cc445e, 0x64a0, 0x4cbb, 0xaa, 0x75, 0xf6, 0xa7, 0x25, 0x1f, 0xda, 0x9e);

DECLARE_INTERFACE_(IND2CompletionQueue, IND2Overlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Overlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __in OVERLAPPED* pOverlapped,
        BOOL wait
        ) PURE;

    // *** IND2CompletionQueue methods ***
    STDMETHOD(GetNotifyAffinity)(
        THIS_
        __out USHORT* pGroup,
        __out KAFFINITY* pAffinity
        ) PURE;

    STDMETHOD(Resize)(
        THIS_
        ULONG queueDepth
        ) PURE;

    STDMETHOD(Notify)(
        THIS_
        ULONG type,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD_(ULONG, GetResults)(
        THIS_
        __out_ecount_part(nResults, return) ND2_RESULT results[],
        ULONG nResults
        ) PURE;
};


//
// Shared Receive Queue
//
#undef INTERFACE
#define INTERFACE IND2SharedReceiveQueue

// {AABD67DC-459A-4db1-826B-56CFCC278883}
DEFINE_GUID(IID_IND2SharedReceiveQueue,
    0xaabd67dc, 0x459a, 0x4db1, 0x82, 0x6b, 0x56, 0xcf, 0xcc, 0x27, 0x88, 0x83);

DECLARE_INTERFACE_(IND2SharedReceiveQueue, IND2Overlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Overlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __in OVERLAPPED* pOverlapped,
        BOOL wait
        ) PURE;

    // *** IND2SharedReceiveQueue methods ***
    STDMETHOD(GetNotifyAffinity)(
        THIS_
        __out USHORT* pGroup,
        __out KAFFINITY* pAffinity
        ) PURE;

    STDMETHOD(Modify)(
        THIS_
        ULONG queueDepth,
        ULONG notifyThreshold
        ) PURE;

    STDMETHOD(Notify)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Receive)(
        THIS_
        __in VOID* requestContext,
        __in_ecount_opt(nSge) const ND2_SGE sge[],
        ULONG nSge
        ) PURE;
};


//
// Memory Window
//
#undef INTERFACE
#define INTERFACE IND2MemoryWindow

// {070FE1F5-0AB5-4361-88DB-974BA704D4B9}
DEFINE_GUID(IID_IND2MemoryWindow,
    0x70fe1f5, 0xab5, 0x4361, 0x88, 0xdb, 0x97, 0x4b, 0xa7, 0x4, 0xd4, 0xb9);

DECLARE_INTERFACE_(IND2MemoryWindow, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2MemoryWindow methods ***
    STDMETHOD_(UINT32, GetRemoteToken)(
        THIS
        ) PURE;
};


//
// Memory Region
//
#undef INTERFACE
#define INTERFACE IND2MemoryRegion

// {55DFEA2F-BC56-4982-8A45-0301BE46C413}
DEFINE_GUID(IID_IND2MemoryRegion,
    0x55dfea2f, 0xbc56, 0x4982, 0x8a, 0x45, 0x3, 0x1, 0xbe, 0x46, 0xc4, 0x13);

DECLARE_INTERFACE_(IND2MemoryRegion, IND2Overlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Overlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __in OVERLAPPED* pOverlapped,
        BOOL wait
        ) PURE;

    // *** IND2MemoryRegion methods ***

    //////////////////////////////////
    // flags - Combination of ND_MR_FLAG_ALLOW_XXX.  Note remote flags imply local.
    STDMETHOD(Register)(
        THIS_
        __in_bcount(cbBuffer) const VOID* pBuffer,
        SIZE_T cbBuffer,
        ULONG flags,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Deregister)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD_(UINT32, GetLocalToken)(
        THIS
        ) PURE;

    STDMETHOD_(UINT32, GetRemoteToken)(
        THIS
        ) PURE;
};


//
// QueuePair
//
#undef INTERFACE
#define INTERFACE IND2QueuePair

// {EEF2F332-B75D-4063-BCE3-3A0BAD2D02CE}
DEFINE_GUID(IID_IND2QueuePair,
    0xeef2f332, 0xb75d, 0x4063, 0xbc, 0xe3, 0x3a, 0xb, 0xad, 0x2d, 0x2, 0xce);

DECLARE_INTERFACE_(IND2QueuePair, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2QueuePair methods ***
    STDMETHOD(Flush)(
        THIS
        ) PURE;

    STDMETHOD(Send)(
        THIS_
        __in_opt VOID* requestContext,
        __in_ecount_opt(nSge) const ND2_SGE sge[],
        ULONG nSge,
        ULONG flags
        ) PURE;

    STDMETHOD(Receive)(
        THIS_
        __in_opt VOID* requestContext,
        __in_ecount_opt(nSge) const ND2_SGE sge[],
        ULONG nSge
        ) PURE;

    // RemoteToken available thorugh IND2Mw::GetRemoteToken.
    STDMETHOD(Bind)(
        THIS_
        __in_opt VOID* requestContext,
        __in IUnknown* pMemoryRegion,
        __inout IUnknown* pMemoryWindow,
        __in_bcount(cbBuffer) const VOID* pBuffer,
        SIZE_T cbBuffer,
        ULONG flags
        ) PURE;

    STDMETHOD(Invalidate)(
        THIS_
        __in_opt VOID* requestContext,
        __in IUnknown* pMemoryWindow,
        ULONG flags
        ) PURE;

    STDMETHOD(Read)(
        THIS_
        __in_opt VOID* requestContext,
        __in_ecount_opt(nSge) const ND2_SGE sge[],
        ULONG nSge,
        UINT64 remoteAddress,
        UINT32 remoteToken,
        ULONG flags
        ) PURE;

    STDMETHOD(Write)(
        THIS_
        __in_opt VOID* requestContext,
        __in_ecount_opt(nSge) const ND2_SGE sge[],
        ULONG nSge,
        UINT64 remoteAddress,
        UINT32 remoteToken,
        ULONG flags
        ) PURE;
};


//
// Connector
//
#undef INTERFACE
#define INTERFACE IND2Connector

// {7DD615C4-6B4C-4866-950C-F3B1D25A5302}
DEFINE_GUID(IID_IND2Connector,
    0x7dd615c4, 0x6b4c, 0x4866, 0x95, 0xc, 0xf3, 0xb1, 0xd2, 0x5a, 0x53, 0x2);

DECLARE_INTERFACE_(IND2Connector, IND2Overlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Overlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __in OVERLAPPED* pOverlapped,
        BOOL wait
        ) PURE;

    // *** IND2Connector methods ***
    STDMETHOD(Bind)(
        THIS_
        __in_bcount(cbAddress) const struct sockaddr* pAddress,
        ULONG cbAddress
        ) PURE;

    STDMETHOD(Connect)(
        THIS_
        __in IUnknown* pQueuePair,
        __in_bcount(cbDestAddress) const struct sockaddr* pDestAddress,
        ULONG cbDestAddress,
        ULONG inboundReadLimit,
        ULONG outboundReadLimit,
        __in_bcount_opt(cbPrivateData) const VOID* pPrivateData,
        ULONG cbPrivateData,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(CompleteConnect)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Accept)(
        THIS_
        __in IUnknown* pQueuePair,
        ULONG inboundReadLimit,
        ULONG outboundReadLimit,
        __in_bcount_opt(cbPrivateData) const VOID* pPrivateData,
        ULONG cbPrivateData,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Reject)(
        THIS_
        __in_bcount_opt(cbPrivateData) const VOID* pPrivateData,
        ULONG cbPrivateData
        ) PURE;

    STDMETHOD(GetReadLimits)(
        THIS_
        __out_opt ULONG* pInboundReadLimit,
        __out_opt ULONG* pOutboundReadLimit
        ) PURE;

    STDMETHOD(GetPrivateData)(
        THIS_
        __out_bcount_opt(*pcbPrivateData) VOID* pPrivateData,
        __inout ULONG* pcbPrivateData
        ) PURE;

    STDMETHOD(GetLocalAddress)(
        THIS_
        __out_bcount_part_opt(*pcbAddress, *pcbAddress) struct sockaddr* pAddress,
        __inout ULONG* pcbAddress
        ) PURE;

    STDMETHOD(GetPeerAddress)(
        THIS_
        __out_bcount_part_opt(*pcbAddress, *pcbAddress) struct sockaddr* pAddress,
        __inout ULONG* pcbAddress
        ) PURE;

    STDMETHOD(NotifyDisconnect)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Disconnect)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;
};


//
// Listener
//
#undef INTERFACE
#define INTERFACE IND2Listener

// {65D23D83-3A57-4E02-86A4-350165C2D130}
DEFINE_GUID(IID_IND2Listener,
    0x65d23d83, 0x3a57, 0x4e02, 0x86, 0xa4, 0x35, 0x1, 0x65, 0xc2, 0xd1, 0x30);

DECLARE_INTERFACE_(IND2Listener, IND2Overlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Overlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __in OVERLAPPED* pOverlapped,
        BOOL wait
        ) PURE;

    // *** IND2Listen methods ***
    STDMETHOD(Bind)(
        THIS_
        __in_bcount(cbAddress) const struct sockaddr* pAddress,
        ULONG cbAddress
        ) PURE;

    STDMETHOD(Listen)(
        THIS_
        ULONG backlog
        ) PURE;

    STDMETHOD(GetLocalAddress)(
        THIS_
        __out_bcount_part_opt(*pcbAddress, *pcbAddress) struct sockaddr* pAddress,
        __inout ULONG* pcbAddress
        ) PURE;

    STDMETHOD(GetConnectionRequest)(
        THIS_
        __inout IUnknown* pConnector,
        __inout OVERLAPPED* pOverlapped
        ) PURE;
};


//
// Adapter
//
#undef INTERFACE
#define INTERFACE IND2Adapter

// {D89C798C-4823-4D69-846C-DFDCCFF9E5F3}
DEFINE_GUID(IID_IND2Adapter,
    0xd89c798c, 0x4823, 0x4d69, 0x84, 0x6c, 0xdf, 0xdc, 0xcf, 0xf9, 0xe5, 0xf3);

DECLARE_INTERFACE_(IND2Adapter, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Adapter methods ***
    STDMETHOD(CreateOverlappedFile)(
        THIS_
        __deref_out HANDLE* phOverlappedFile
        ) PURE;

    STDMETHOD(Query)(
        THIS_
        __inout_bcount_opt(*pcbInfo) ND2_ADAPTER_INFO* pInfo,
        __inout ULONG* pcbInfo
        ) PURE;

    STDMETHOD(QueryAddressList)(
        THIS_
        __out_bcount_part_opt(*pcbAddressList, *pcbAddressList) SOCKET_ADDRESS_LIST* pAddressList,
        __inout ULONG* pcbAddressList
        ) PURE;

    STDMETHOD(CreateCompletionQueue)(
        THIS_
        __in REFIID iid,
        __in HANDLE hOverlappedFile,
        ULONG queueDepth,
        USHORT group,
        KAFFINITY affinity,
        __deref_out VOID** ppCompletionQueue
        ) PURE;

    STDMETHOD(CreateMemoryRegion)(
        THIS_
        __in REFIID iid,
        __in HANDLE hOverlappedFile,
        __deref_out VOID** ppMemoryRegion
        ) PURE;

    STDMETHOD(CreateMemoryWindow)(
        THIS_
        __in REFIID iid,
        __deref_out VOID** ppMemoryWindow
        ) PURE;

    STDMETHOD(CreateSharedReceiveQueue)(
        THIS_
        __in REFIID iid,
        __in HANDLE hOverlappedFile,
        ULONG queueDepth,
        ULONG maxRequestSge,
        ULONG notifyThreshold,
        USHORT group,
        KAFFINITY affinity,
        __deref_out VOID** ppSharedReceiveQueue
        ) PURE;

    STDMETHOD(CreateQueuePair)(
        THIS_
        __in REFIID iid,
        __in IUnknown* pReceiveCompletionQueue,
        __in IUnknown* pInitiatorCompletionQueue,
        __in_opt VOID* context,
        ULONG receiveQueueDepth,
        ULONG initiatorQueueDepth,
        ULONG maxReceiveRequestSge,
        ULONG maxInitiatorRequestSge,
        ULONG inlineDataSize,
        __deref_out VOID** ppQueuePair
        ) PURE;

    STDMETHOD(CreateQueuePairWithSrq)(
        THIS_
        __in REFIID iid,
        __in IUnknown* pReceiveCompletionQueue,
        __in IUnknown* pInitiatorCompletionQueue,
        __in IUnknown* pSharedReceiveQueue,
        __in_opt VOID* context,
        ULONG initiatorQueueDepth,
        ULONG maxInitiatorRequestSge,
        ULONG inlineDataSize,
        __deref_out VOID** ppQueuePair
        ) PURE;

    STDMETHOD(CreateConnector)(
        THIS_
        __in REFIID iid,
        __in HANDLE hOverlappedFile,
        __deref_out VOID** ppConnector
        ) PURE;

    STDMETHOD(CreateListener)(
        THIS_
        __in REFIID iid,
        __in HANDLE hOverlappedFile,
        __deref_out VOID** ppListener
        ) PURE;
};


//
// Provider
//
#undef INTERFACE
#define INTERFACE IND2Provider

// {49EAE6C1-76C4-46D0-8003-5C2EAA2C9A8E}
DEFINE_GUID(IID_IND2Provider,
    0x49eae6c1, 0x76c4, 0x46d0, 0x80, 0x3, 0x5c, 0x2e, 0xaa, 0x2c, 0x9a, 0x8e);

DECLARE_INTERFACE_(IND2Provider, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** IND2Provider methods ***
    STDMETHOD(QueryAddressList)(
        THIS_
        __out_bcount_part_opt(*pcbAddressList, *pcbAddressList) SOCKET_ADDRESS_LIST* pAddressList,
        __inout ULONG* pcbAddressList
        ) PURE;

    STDMETHOD(ResolveAddress)(
        THIS_
        __in_bcount(cbAddress) const struct sockaddr* pAddress,
        ULONG cbAddress,
        __out UINT64* pAdapterId
        ) PURE;

    STDMETHOD(OpenAdapter)(
        THIS_
        __in REFIID iid,
        UINT64 adapterId,
        __deref_out VOID** ppAdapter
        ) PURE;
};


///////////////////////////////////////////////////////////////////////////////
//
// HPC Pack 2008 SDK interface
//
///////////////////////////////////////////////////////////////////////////////

DECLARE_HANDLE(ND_MR_HANDLE);


typedef struct _ND_ADAPTER_INFO1
{
    UINT32 VendorId;
    UINT32 DeviceId;
    SIZE_T MaxInboundSge;
    SIZE_T MaxInboundRequests;
    SIZE_T MaxInboundLength;
    SIZE_T MaxOutboundSge;
    SIZE_T MaxOutboundRequests;
    SIZE_T MaxOutboundLength;
    SIZE_T MaxInlineData;
    SIZE_T MaxInboundReadLimit;
    SIZE_T MaxOutboundReadLimit;
    SIZE_T MaxCqEntries;
    SIZE_T MaxRegistrationSize;
    SIZE_T MaxWindowSize;
    SIZE_T LargeRequestThreshold;
    SIZE_T MaxCallerData;
    SIZE_T MaxCalleeData;

} ND_ADAPTER_INFO1;
#define ND_ADAPTER_INFO ND_ADAPTER_INFO1

typedef struct _ND_RESULT
{
    HRESULT Status;
    SIZE_T BytesTransferred;

} ND_RESULT;

#pragma pack( push, 1 )
typedef struct _ND_MW_DESCRIPTOR
{
    UINT64 Base;    // Network byte order
    UINT64 Length;  // Network byte order
    UINT32 Token;   // Network byte order

} ND_MW_DESCRIPTOR;
#pragma pack( pop )

typedef struct _ND_SGE
{
    VOID* pAddr;
    SIZE_T Length;
    ND_MR_HANDLE hMr;

} ND_SGE;

//
// Overlapped object
//
#undef INTERFACE
#define INTERFACE INDOverlapped

// {C859E15E-75E2-4fe3-8D6D-0DFF36F02442}
DEFINE_GUID(IID_INDOverlapped,
    0xc859e15e, 0x75e2, 0x4fe3, 0x8d, 0x6d, 0xd, 0xff, 0x36, 0xf0, 0x24, 0x42);

DECLARE_INTERFACE_(INDOverlapped, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDOverlapped methods ***
    STDMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    STDMETHOD(GetOverlappedResult)(
        THIS_
        __inout OVERLAPPED *pOverlapped,
        __out SIZE_T *pNumberOfBytesTransferred,
        BOOL bWait
        ) PURE;
};


//
// Completion Queue
//
#undef INTERFACE
#define INTERFACE INDCompletionQueue

// {1245A633-2A32-473a-830C-E05D1F869D02}
DEFINE_GUID(IID_INDCompletionQueue,
    0x1245a633, 0x2a32, 0x473a, 0x83, 0xc, 0xe0, 0x5d, 0x1f, 0x86, 0x9d, 0x2);

DECLARE_INTERFACE_(INDCompletionQueue, INDOverlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDOverlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __inout OVERLAPPED *pOverlapped,
        __out SIZE_T *pNumberOfBytesTransferred,
        BOOL bWait
        ) PURE;

    // *** INDCompletionQueue methods ***
    STDMETHOD(Resize)(
        THIS_
        SIZE_T nEntries
        ) PURE;

    STDMETHOD(Notify)(
        THIS_
        DWORD Type,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD_(SIZE_T, GetResults)(
        THIS_
        __out_ecount_part_opt(nResults, return) ND_RESULT* pResults[],
        SIZE_T nResults
        ) PURE;
};


//
// Remove View
//
#undef INTERFACE
#define INTERFACE INDMemoryWindow

// {070FE1F5-0AB5-4361-88DB-974BA704D4B9}
DEFINE_GUID(IID_INDMemoryWindow,
    0x70fe1f5, 0xab5, 0x4361, 0x88, 0xdb, 0x97, 0x4b, 0xa7, 0x4, 0xd4, 0xb9);

DECLARE_INTERFACE_(INDMemoryWindow, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;
};


//
// Endpoint
//
#undef INTERFACE
#define INTERFACE INDEndpoint

// {DBD00EAB-B679-44a9-BD65-E82F3DE12D1A}
DEFINE_GUID(IID_INDEndpoint,
    0xdbd00eab, 0xb679, 0x44a9, 0xbd, 0x65, 0xe8, 0x2f, 0x3d, 0xe1, 0x2d, 0x1a);

DECLARE_INTERFACE_(INDEndpoint, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDEndpoint methods ***
    STDMETHOD(Flush)(
        THIS
        ) PURE;

    STDMETHOD_(void, StartRequestBatch)(
        THIS
        ) PURE;

    STDMETHOD_(void, SubmitRequestBatch)(
        THIS
        ) PURE;

    STDMETHOD(Send)(
        THIS_
        __out ND_RESULT* pResult,
        __in_ecount_opt(nSge) const ND_SGE* pSgl,
        SIZE_T nSge,
        DWORD Flags
        ) PURE;

    STDMETHOD(SendAndInvalidate)(
        THIS_
        __out ND_RESULT* pResult,
        __in_ecount_opt(nSge) const ND_SGE* pSgl,
        SIZE_T nSge,
        __in const ND_MW_DESCRIPTOR* pRemoteMwDescriptor,
        DWORD Flags
        ) PURE;

    STDMETHOD(Receive)(
        THIS_
        __out ND_RESULT* pResult,
        __in_ecount_opt(nSge) const ND_SGE* pSgl,
        SIZE_T nSge
        ) PURE;

    STDMETHOD(Bind)(
        THIS_
        __out ND_RESULT* pResult,
        __in ND_MR_HANDLE hMr,
        __in INDMemoryWindow* pMw,
        __in_bcount(BufferSize) const void* pBuffer,
        SIZE_T BufferSize,
        DWORD Flags,
        __out ND_MW_DESCRIPTOR* pMwDescriptor
        ) PURE;

    STDMETHOD(Invalidate)(
        THIS_
        __out ND_RESULT* pResult,
        __in INDMemoryWindow* pMw,
        DWORD Flags
        ) PURE;

    STDMETHOD(Read)(
        THIS_
        __out ND_RESULT* pResult,
        __in_ecount_opt(nSge) const ND_SGE* pSgl,
        SIZE_T nSge,
        __in const ND_MW_DESCRIPTOR* pRemoteMwDescriptor,
        ULONGLONG Offset,
        DWORD Flags
        ) PURE;

    STDMETHOD(Write)(
        THIS_
        __out ND_RESULT* pResult,
        __in_ecount_opt(nSge) const ND_SGE* pSgl,
        SIZE_T nSge,
        __in const ND_MW_DESCRIPTOR* pRemoteMwDescriptor,
        ULONGLONG Offset,
        DWORD Flags
        ) PURE;
};


//
// Connector
//
#undef INTERFACE
#define INTERFACE INDConnector

// {1BCAF2D1-E274-4aeb-AC57-CD5D4376E0B7}
DEFINE_GUID(IID_INDConnector,
    0x1bcaf2d1, 0xe274, 0x4aeb, 0xac, 0x57, 0xcd, 0x5d, 0x43, 0x76, 0xe0, 0xb7);

DECLARE_INTERFACE_(INDConnector, INDOverlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDOverlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __inout OVERLAPPED *pOverlapped,
        __out SIZE_T *pNumberOfBytesTransferred,
        BOOL bWait
        ) PURE;

    // *** INDConnector methods ***
    STDMETHOD(CreateEndpoint)(
        THIS_
        __in INDCompletionQueue* pInboundCq,
        __in INDCompletionQueue* pOutboundCq,
        SIZE_T nInboundEntries,
        SIZE_T nOutboundEntries,
        SIZE_T nInboundSge,
        SIZE_T nOutboundSge,
        SIZE_T InboundReadLimit,
        SIZE_T OutboundReadLimit,
        __out_opt SIZE_T* pMaxInlineData,
        __deref_out INDEndpoint** ppEndpoint
        ) PURE;

    STDMETHOD(Connect)(
        THIS_
        __in INDEndpoint* pEndpoint,
        __in_bcount(AddressLength) const struct sockaddr* pAddress,
        SIZE_T AddressLength,
        INT Protocol,
        USHORT LocalPort,
        __in_bcount_opt(PrivateDataLength) const void* pPrivateData,
        SIZE_T PrivateDataLength,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(CompleteConnect)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Accept)(
        THIS_
        __in INDEndpoint* pEndpoint,
        __in_bcount_opt(PrivateDataLength) const void* pPrivateData,
        SIZE_T PrivateDataLength,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Reject)(
        THIS_
        __in_bcount_opt(PrivateDataLength) const void* pPrivateData,
        SIZE_T PrivateDataLength
        ) PURE;

    STDMETHOD(GetConnectionData)(
        THIS_
        __out_opt SIZE_T* pInboundReadLimit,
        __out_opt SIZE_T* pOutboundReadLimit,
        __out_bcount_part_opt(*pPrivateDataLength, *pPrivateDataLength) void* pPrivateData,
        __inout_opt SIZE_T* pPrivateDataLength
        ) PURE;

    STDMETHOD(GetLocalAddress)(
        THIS_
        __out_bcount_part_opt(*pAddressLength, *pAddressLength) struct sockaddr* pAddress,
        __inout SIZE_T* pAddressLength
        ) PURE;

    STDMETHOD(GetPeerAddress)(
        THIS_
        __out_bcount_part_opt(*pAddressLength, *pAddressLength) struct sockaddr* pAddress,
        __inout SIZE_T* pAddressLength
        ) PURE;

    STDMETHOD(NotifyDisconnect)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(Disconnect)(
        THIS_
        __inout OVERLAPPED* pOverlapped
        ) PURE;
};


//
// Listen
//
#undef INTERFACE
#define INTERFACE INDListen

// {BB902588-BA3F-4441-9FE1-3B6795E4E668}
DEFINE_GUID(IID_INDListen,
    0xbb902588, 0xba3f, 0x4441, 0x9f, 0xe1, 0x3b, 0x67, 0x95, 0xe4, 0xe6, 0x68);

DECLARE_INTERFACE_(INDListen, INDOverlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDOverlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __inout OVERLAPPED *pOverlapped,
        __out SIZE_T *pNumberOfBytesTransferred,
        BOOL bWait
        ) PURE;

    // *** INDListen methods ***
    STDMETHOD(GetConnectionRequest)(
        THIS_
        __inout INDConnector* pConnector,
        __inout OVERLAPPED* pOverlapped
        ) PURE;
};


//
// INDAdapter
//
#undef INTERFACE
#define INTERFACE INDAdapter

// {A023C5A0-5B73-43bc-8D20-33AA07E9510F}
DEFINE_GUID(IID_INDAdapter,
    0xa023c5a0, 0x5b73, 0x43bc, 0x8d, 0x20, 0x33, 0xaa, 0x7, 0xe9, 0x51, 0xf);

DECLARE_INTERFACE_(INDAdapter, INDOverlapped)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDOverlapped methods ***
    IFACEMETHOD(CancelOverlappedRequests)(
        THIS
        ) PURE;

    IFACEMETHOD(GetOverlappedResult)(
        THIS_
        __inout OVERLAPPED *pOverlapped,
        __out SIZE_T *pNumberOfBytesTransferred,
        BOOL bWait
        ) PURE;

    // *** INDAdapter methods ***
    STDMETHOD_(HANDLE, GetFileHandle)(
        THIS
        ) PURE;

    STDMETHOD(Query)(
        THIS_
        DWORD VersionRequested,
        __out_bcount_part_opt(*pBufferSize, *pBufferSize) ND_ADAPTER_INFO* pInfo,
        __inout_opt SIZE_T* pBufferSize
        ) PURE;

    STDMETHOD(Control)(
        THIS_
        DWORD IoControlCode,
        __in_bcount_opt(InBufferSize) const void* pInBuffer,
        SIZE_T InBufferSize,
        __out_bcount_opt(OutBufferSize) void* pOutBuffer,
        SIZE_T OutBufferSize,
        __out SIZE_T* pBytesReturned,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(CreateCompletionQueue)(
        THIS_
        SIZE_T nEntries,
        __deref_out INDCompletionQueue** ppCq
        ) PURE;

    STDMETHOD(RegisterMemory)(
        THIS_
        __in_bcount(BufferSize) const void* pBuffer,
        SIZE_T BufferSize,
        __inout OVERLAPPED* pOverlapped,
        __deref_out ND_MR_HANDLE* phMr
        ) PURE;

    STDMETHOD(DeregisterMemory)(
        THIS_
        __in ND_MR_HANDLE hMr,
        __inout OVERLAPPED* pOverlapped
        ) PURE;

    STDMETHOD(CreateMemoryWindow)(
        THIS_
        __out ND_RESULT* pInvalidateResult,
        __deref_out INDMemoryWindow** ppMw
        ) PURE;

    STDMETHOD(CreateConnector)(
        THIS_
        __deref_out INDConnector** ppConnector
        ) PURE;

    STDMETHOD(Listen)(
        THIS_
        SIZE_T Backlog,
        INT Protocol,
        USHORT Port,
        __out_opt USHORT* pAssignedPort,
        __deref_out INDListen** ppListen
        ) PURE;
};


//
// INDProvider
//
#undef INTERFACE
#define INTERFACE INDProvider

// {0C5DD316-5FDF-47e6-B2D0-2A6EDA8D39DD}
DEFINE_GUID(IID_INDProvider,
    0xc5dd316, 0x5fdf, 0x47e6, 0xb2, 0xd0, 0x2a, 0x6e, 0xda, 0x8d, 0x39, 0xdd);

DECLARE_INTERFACE_(INDProvider, IUnknown)
{
    // *** IUnknown methods ***
    IFACEMETHOD(QueryInterface)(
        THIS_
        REFIID riid,
        __deref_out LPVOID* ppvObj
        ) PURE;

    IFACEMETHOD_(ULONG, AddRef)(
        THIS
        ) PURE;

    IFACEMETHOD_(ULONG, Release)(
        THIS
        ) PURE;

    // *** INDProvider methods ***
    STDMETHOD(QueryAddressList)(
        THIS_
        __out_bcount_part_opt(*pBufferSize, *pBufferSize) SOCKET_ADDRESS_LIST* pAddressList,
        __inout SIZE_T* pBufferSize
        ) PURE;

    STDMETHOD(OpenAdapter)(
        THIS_
        __in_bcount(AddressLength) const struct sockaddr* pAddress,
        SIZE_T AddressLength,
        __deref_out INDAdapter** ppAdapter
        ) PURE;
};

//
// Map version 1 error values to version 2.
//
#define ND_LOCAL_LENGTH         ND_DATA_OVERRUN
#define ND_INVALIDATION_ERROR   ND_INVALID_DEVICE_REQUEST

#endif // _NDSPI_H_
