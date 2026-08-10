/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NDWRAP_H_
#define NCCL_NDWRAP_H_

#include "core.h"
// Official NetworkDirect v2 SPI interfaces and types
#include "ndspi.h"

// Wrapper function declarations
ncclResult_t wrap_nd_symbols(void);

// Provider functions
ncclResult_t wrap_nd_query_address_list(struct IND2Provider* provider, SOCKET_ADDRESS_LIST* pAddressList,
                                        ULONG* pBufferSize);
ncclResult_t wrap_nd_open_adapter(struct IND2Provider* provider, UINT64 adapterId, struct IND2Adapter** ppAdapter);

// Adapter functions
ncclResult_t wrap_nd_query_adapter(struct IND2Adapter* adapter, ND2_ADAPTER_INFO* pInfo, ULONG* pBufferSize);
ncclResult_t wrap_nd_create_overlapped_file(struct IND2Adapter* adapter, HANDLE* phOverlappedFile);
ncclResult_t wrap_nd_create_completion_queue(struct IND2Adapter* adapter, HANDLE hOverlappedFile, ULONG queueDepth,
                                             struct IND2CompletionQueue** ppCq);
ncclResult_t wrap_nd_create_memory_region(struct IND2Adapter* adapter, HANDLE hOverlappedFile,
                                          struct IND2MemoryRegion** ppMr);
ncclResult_t wrap_nd_create_queue_pair(struct IND2Adapter* adapter, struct IND2CompletionQueue* pReceiveCq,
                                       struct IND2CompletionQueue* pInitiatorCq, void* pContext,
                                       ULONG receiveQueueDepth, ULONG initiatorQueueDepth, ULONG maxReceiveRequestSge,
                                       ULONG maxInitiatorRequestSge, ULONG inlineDataSize, struct IND2QueuePair** ppQp);
ncclResult_t wrap_nd_create_connector(struct IND2Adapter* adapter, HANDLE hOverlappedFile,
                                      struct IND2Connector** ppConnector);
ncclResult_t wrap_nd_create_listener(struct IND2Adapter* adapter, HANDLE hOverlappedFile,
                                     struct IND2Listener** ppListener);

// Memory Region functions
ncclResult_t wrap_nd_register_memory(struct IND2MemoryRegion* mr, const void* pBuffer, SIZE_T bufferSize, ULONG flags,
                                     OVERLAPPED* pOverlapped);
ncclResult_t wrap_nd_deregister_memory(struct IND2MemoryRegion* mr, OVERLAPPED* pOverlapped);
static inline UINT32 wrap_nd_get_local_token(struct IND2MemoryRegion* mr) {
  return mr->GetLocalToken();
}
static inline UINT32 wrap_nd_get_remote_token(struct IND2MemoryRegion* mr) {
  return mr->GetRemoteToken();
}

// Completion Queue functions
static inline ULONG wrap_nd_get_results(struct IND2CompletionQueue* cq, ND2_RESULT results[], ULONG nResults) {
  return cq->GetResults(results, nResults);
}
ncclResult_t wrap_nd_notify(struct IND2CompletionQueue* cq, ULONG type, OVERLAPPED* pOverlapped);
ncclResult_t wrap_nd_cq_get_status(struct IND2CompletionQueue* cq, OVERLAPPED* pOverlapped, int* done, HRESULT* status);
ncclResult_t wrap_nd_cq_cancel_notifications(struct IND2CompletionQueue* cq);

// Queue Pair functions
ncclResult_t wrap_nd_write(struct IND2QueuePair* qp, void* requestContext, const ND2_SGE sge[], ULONG nSge,
                           UINT64 remoteAddress, UINT32 remoteToken, ULONG flags, HRESULT* status = NULL);

// Connector functions
ncclResult_t wrap_nd_connect(struct IND2Connector* connector, struct IND2QueuePair* qp, const struct sockaddr* pAddress,
                             ULONG cbAddress, ULONG inboundReadLimit, ULONG outboundReadLimit, OVERLAPPED* pOverlapped,
                             int* done);
ncclResult_t wrap_nd_connector_bind(struct IND2Connector* connector, const struct sockaddr* pAddress, ULONG cbAddress);
ncclResult_t wrap_nd_complete_connect(struct IND2Connector* connector, OVERLAPPED* pOverlapped, int* done);
ncclResult_t wrap_nd_accept(struct IND2Connector* connector, struct IND2QueuePair* qp, ULONG inboundReadLimit,
                            ULONG outboundReadLimit, OVERLAPPED* pOverlapped, int* done);

// Listener functions
ncclResult_t wrap_nd_bind(struct IND2Listener* listener, const struct sockaddr* pAddress, ULONG cbAddress);
ncclResult_t wrap_nd_listen(struct IND2Listener* listener, ULONG backlog);
ncclResult_t wrap_nd_get_listener_address(struct IND2Listener* listener, struct sockaddr* pAddress,
                                          ULONG* pAddressLength);
ncclResult_t wrap_nd_get_connection_request(struct IND2Listener* listener, struct IND2Connector* pConnector,
                                            OVERLAPPED* pOverlapped, int* done);

// Overlapped status helpers
ncclResult_t wrap_nd_listener_get_request_status(struct IND2Listener* listener, OVERLAPPED* pOverlapped, int* done);
ncclResult_t wrap_nd_connector_get_status(struct IND2Connector* connector, OVERLAPPED* pOverlapped, int* done);
ncclResult_t wrap_nd_prepare_overlapped(OVERLAPPED* pOverlapped);
void wrap_nd_close_overlapped(OVERLAPPED* pOverlapped);

// Release functions (IUnknown::Release wrapper)
static inline ULONG wrap_nd_release(void* pInterface) {
  if (pInterface) {
    IUnknown* unk = reinterpret_cast<IUnknown*>(pInterface);
    return unk->Release();
  }
  return 0;
}

// Provider accessors
struct IND2Provider* wrap_nd_get_provider(void);

// Resolve address → adapterId
ncclResult_t wrap_nd_resolve_address(struct IND2Provider* provider, const struct sockaddr* pAddress, ULONG cbAddress,
                                     UINT64* pAdapterId);

// Unload provider DLL and cleanup winsock when no longer needed
void wrap_nd_unload(void);

#endif // NCCL_NDWRAP_H_
