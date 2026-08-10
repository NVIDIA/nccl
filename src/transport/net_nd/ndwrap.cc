/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <initguid.h>
#include "ndwrap.h"
#include "core.h"
#include <ws2spi.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global provider handle
static struct IND2Provider* g_ndProvider = NULL;
static int g_ndInitialized = 0;
static HMODULE g_ndProviderModule = NULL;

// Return the process-wide NDv2 provider selected during initialization.
struct IND2Provider* wrap_nd_get_provider(void) {
  return g_ndProvider;
}

// Find and load the first usable NDv2 provider in the Winsock catalog.
static ncclResult_t ncclNdEnumerateProviders(struct IND2Provider** ppProvider) {
  WSAPROTOCOL_INFOW* protocolInfo = NULL;
  DWORD bufferSize = 0;
  int numProtocols;
  int err = 0;

  // First call to get required buffer size for all providers.
  // Passing NULL protocol list avoids filtering out providers unexpectedly.
  numProtocols = WSCEnumProtocols(NULL, NULL, &bufferSize, &err);
  if (numProtocols != SOCKET_ERROR || err != WSAENOBUFS || bufferSize == 0) {
    WARN("NET/ND : Failed to query Winsock provider buffer size: %d (numProtocols=%d, bufferSize=%lu)", err,
         numProtocols, (unsigned long)bufferSize);
    return ncclInternalError;
  }

  protocolInfo = (WSAPROTOCOL_INFOW*)malloc(bufferSize);
  if (!protocolInfo) {
    WARN("NET/ND : Failed to allocate buffer for Winsock providers");
    return ncclSystemError;
  }

  // Enumerate the full catalog after the sizing query.
  err = 0;
  numProtocols = WSCEnumProtocols(NULL, protocolInfo, &bufferSize, &err);
  if (numProtocols == SOCKET_ERROR) {
    WARN("NET/ND : Failed to enumerate Winsock providers: %d", err);
    free(protocolInfo);
    return ncclInternalError;
  }
  if (numProtocols == 0) {
    INFO(NCCL_NET, "NET/ND : Winsock provider catalog is empty");
    free(protocolInfo);
    return ncclInternalError;
  }

  INFO(NCCL_NET, "NET/ND : Found %d Winsock protocols, searching for ND provider", numProtocols);

  // Inspect each catalog entry until one exposes a working IND2Provider.
  for (int i = 0; i < numProtocols; i++) {
    // The WinOF2 catalog exposes NDv1 before NDv2. The v1 compatibility DLL
    // can return an IND2Provider interface but does not implement the v2
    // connection sequence correctly, leaving CompleteConnect pending forever.
    if (protocolInfo[i].iVersion < 2) continue;

    GUID providerId = protocolInfo[i].ProviderId;

    // Resolve and expand the provider DLL path.
    INT pathLen = 0;
    INT err = 0;
    WCHAR pathBuf[MAX_PATH];
    pathLen = MAX_PATH;
    INT ret = WSCGetProviderPath(&providerId, pathBuf, &pathLen, &err);
    WCHAR* path = NULL;
    if (ret == 0) {
      DWORD expandedLen = ExpandEnvironmentStringsW(pathBuf, NULL, 0);
      if (expandedLen == 0) {
        continue;
      }
      path = (WCHAR*)malloc(expandedLen * sizeof(WCHAR));
      if (!path) continue;
      if (ExpandEnvironmentStringsW(pathBuf, path, expandedLen) != expandedLen) {
        free(path);
        continue;
      }
    } else if (err == WSAEFAULT && pathLen > 0) {
      // Retry with the provider-reported path size.
      WCHAR* tmp = (WCHAR*)malloc(pathLen * sizeof(WCHAR));
      if (!tmp) continue;
      if (WSCGetProviderPath(&providerId, tmp, &pathLen, &err) != 0) {
        free(tmp);
        continue;
      }
      DWORD expandedLen = ExpandEnvironmentStringsW(tmp, NULL, 0);
      if (expandedLen == 0) {
        free(tmp);
        continue;
      }
      path = (WCHAR*)malloc(expandedLen * sizeof(WCHAR));
      if (!path) {
        free(tmp);
        continue;
      }
      if (ExpandEnvironmentStringsW(tmp, path, expandedLen) != expandedLen) {
        free(tmp);
        free(path);
        continue;
      }
      free(tmp);
    } else {
      // Could not get provider path
      continue;
    }

    // Load the candidate module only after its path is fully expanded.
    HMODULE hProvider = LoadLibraryW(path);
    if (!hProvider) {
      free(path);
      continue;
    }

    // Resolve the COM entry point used to request IND2Provider.
    typedef HRESULT(STDAPICALLTYPE * PFNDllGetClassObject)(REFCLSID, REFIID, LPVOID*);
    PFNDllGetClassObject pfn = (PFNDllGetClassObject)GetProcAddress(hProvider, "DllGetClassObject");
    if (!pfn) {
      FreeLibrary(hProvider);
      free(path);
      continue;
    }

    // Request IND2Provider using the catalog ProviderId as the CLSID.
    IND2Provider* provider = NULL;
    HRESULT hr = pfn(providerId, IID_IND2Provider, (void**)&provider);
    if (SUCCEEDED(hr) && provider != NULL) {
      *ppProvider = provider;
      // Keep the module loaded for the lifetime of the provider. It will be
      // released explicitly during plugin teardown.
      g_ndProviderModule = hProvider;
      free(path);
      free(protocolInfo);
      return ncclSuccess;
    }

    // Unload candidates that do not expose a usable NDv2 interface.
    FreeLibrary(hProvider);
    free(path);
  }

  free(protocolInfo);
  INFO(NCCL_NET, "NET/ND : No NetworkDirect v2 provider found");
  return ncclInternalError;
}

// Initialize Winsock and load the process-wide NDv2 provider once.
ncclResult_t wrap_nd_symbols(void) {
  if (g_ndInitialized) {
    return ncclSuccess;
  }

  // Initialize Winsock
  WSADATA wsaData;
  int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (result != 0) {
    WARN("NET/ND : WSAStartup failed: %d", result);
    return ncclSystemError;
  }

  // Enumerate and load ND provider
  ncclResult_t ret = ncclNdEnumerateProviders(&g_ndProvider);
  if (ret != ncclSuccess || g_ndProvider == NULL) {
    INFO(NCCL_NET, "NET/ND : No NetworkDirect provider found");
    WSACleanup();
    return ncclInternalError;
  }

  g_ndInitialized = 1;
  INFO(NCCL_NET, "NET/ND : Successfully loaded NetworkDirect provider");
  return ncclSuccess;
}

// Release the provider, its module, and Winsock in reverse order.
void wrap_nd_unload(void) {
  if (g_ndProvider) {
    wrap_nd_release(g_ndProvider);
    g_ndProvider = NULL;
  }
  if (g_ndProviderModule) {
    FreeLibrary(g_ndProviderModule);
    g_ndProviderModule = NULL;
  }
  if (g_ndInitialized) {
    WSACleanup();
    g_ndInitialized = 0;
  }
}

// Resolve a local network address to its ND adapter identifier.
ncclResult_t wrap_nd_resolve_address(struct IND2Provider* provider, const struct sockaddr* pAddress, ULONG cbAddress,
                                     UINT64* pAdapterId) {
  HRESULT hr = provider->ResolveAddress(pAddress, cbAddress, pAdapterId);
  if (FAILED(hr)) {
    WARN("NET/ND : ResolveAddress failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Enumerate addresses published by the selected ND provider.
ncclResult_t wrap_nd_query_address_list(struct IND2Provider* provider, SOCKET_ADDRESS_LIST* pAddressList,
                                        ULONG* pBufferSize) {
  HRESULT hr = provider->QueryAddressList(pAddressList, pBufferSize);
  if (FAILED(hr) && hr != ND_BUFFER_OVERFLOW && hr != HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)) {
    WARN("NET/ND : QueryAddressList failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Open one ND adapter by provider identifier.
ncclResult_t wrap_nd_open_adapter(struct IND2Provider* provider, UINT64 adapterId, struct IND2Adapter** ppAdapter) {
  HRESULT hr = provider->OpenAdapter(IID_IND2Adapter, adapterId, (void**)ppAdapter);
  if (FAILED(hr)) {
    WARN("NET/ND : OpenAdapter failed for adapter ID 0x%llx: 0x%08x", adapterId, hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Query NDv2 capabilities for an open adapter.
ncclResult_t wrap_nd_query_adapter(struct IND2Adapter* adapter, ND2_ADAPTER_INFO* pInfo, ULONG* pBufferSize) {
  if (pInfo) memset(pInfo, 0, *pBufferSize);
  if (pInfo && *pBufferSize >= sizeof(pInfo->InfoVersion)) pInfo->InfoVersion = ND_VERSION_2;
  HRESULT hr = adapter->Query(pInfo, pBufferSize);
  if (FAILED(hr)) {
    WARN("NET/ND : Query adapter failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Create the file handle used for asynchronous adapter operations.
ncclResult_t wrap_nd_create_overlapped_file(struct IND2Adapter* adapter, HANDLE* phOverlappedFile) {
  HRESULT hr = adapter->CreateOverlappedFile(phOverlappedFile);
  if (FAILED(hr)) {
    WARN("NET/ND : CreateOverlappedFile failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Create a completion queue with the requested depth.
ncclResult_t wrap_nd_create_completion_queue(struct IND2Adapter* adapter, HANDLE hOverlappedFile, ULONG queueDepth,
                                             struct IND2CompletionQueue** ppCq) {
  HRESULT hr = adapter->CreateCompletionQueue(IID_IND2CompletionQueue, hOverlappedFile, queueDepth, 0, 0, (void**)ppCq);
  if (FAILED(hr)) {
    WARN("NET/ND : CreateCompletionQueue failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Create an unregistered memory-region object.
ncclResult_t wrap_nd_create_memory_region(struct IND2Adapter* adapter, HANDLE hOverlappedFile,
                                          struct IND2MemoryRegion** ppMr) {
  HRESULT hr = adapter->CreateMemoryRegion(IID_IND2MemoryRegion, hOverlappedFile, (void**)ppMr);
  if (FAILED(hr)) {
    WARN("NET/ND : CreateMemoryRegion failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Create a queue pair with explicit queue, SGE, and inline limits.
ncclResult_t wrap_nd_create_queue_pair(struct IND2Adapter* adapter, struct IND2CompletionQueue* pReceiveCq,
                                       struct IND2CompletionQueue* pInitiatorCq, void* pContext,
                                       ULONG receiveQueueDepth, ULONG initiatorQueueDepth, ULONG maxReceiveRequestSge,
                                       ULONG maxInitiatorRequestSge, ULONG inlineDataSize,
                                       struct IND2QueuePair** ppQp) {
  HRESULT hr = adapter->CreateQueuePair(IID_IND2QueuePair, pReceiveCq, pInitiatorCq, pContext, receiveQueueDepth,
                                        initiatorQueueDepth, maxReceiveRequestSge, maxInitiatorRequestSge,
                                        inlineDataSize, (void**)ppQp);
  if (FAILED(hr)) {
    WARN("NET/ND : CreateQueuePair failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Create a connector for one asynchronous connection.
ncclResult_t wrap_nd_create_connector(struct IND2Adapter* adapter, HANDLE hOverlappedFile,
                                      struct IND2Connector** ppConnector) {
  HRESULT hr = adapter->CreateConnector(IID_IND2Connector, hOverlappedFile, (void**)ppConnector);
  if (FAILED(hr)) {
    WARN("NET/ND : CreateConnector failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Create a listener for one physical adapter.
ncclResult_t wrap_nd_create_listener(struct IND2Adapter* adapter, HANDLE hOverlappedFile,
                                     struct IND2Listener** ppListener) {
  HRESULT hr = adapter->CreateListener(IID_IND2Listener, hOverlappedFile, (void**)ppListener);
  if (FAILED(hr)) {
    WARN("NET/ND : CreateListener failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Register memory and wait if the provider completes asynchronously.
ncclResult_t wrap_nd_register_memory(struct IND2MemoryRegion* mr, const void* pBuffer, SIZE_T bufferSize, ULONG flags,
                                     OVERLAPPED* pOverlapped) {
  HRESULT hr = mr->Register(pBuffer, bufferSize, flags, pOverlapped);
  if (hr == ND_PENDING) {
    hr = static_cast<IND2Overlapped*>(mr)->GetOverlappedResult(pOverlapped, TRUE);
  }
  if (hr != ND_SUCCESS) {
    WARN("NET/ND : Register memory failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Deregister memory and wait if the provider completes asynchronously.
ncclResult_t wrap_nd_deregister_memory(struct IND2MemoryRegion* mr, OVERLAPPED* pOverlapped) {
  HRESULT hr = mr->Deregister(pOverlapped);
  if (hr == ND_PENDING) {
    hr = static_cast<IND2Overlapped*>(mr)->GetOverlappedResult(pOverlapped, TRUE);
  }
  if (hr != ND_SUCCESS) {
    WARN("NET/ND : Deregister memory failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Arm a completion-queue notification.
ncclResult_t wrap_nd_notify(struct IND2CompletionQueue* cq, ULONG type, OVERLAPPED* pOverlapped) {
  HRESULT hr = cq->Notify(type, pOverlapped);
  if (hr != ND_SUCCESS && hr != ND_PENDING) {
    WARN("NET/ND : CQ Notify failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Post one RDMA write and return the provider status to the caller.
ncclResult_t wrap_nd_write(struct IND2QueuePair* qp, void* requestContext, const ND2_SGE sge[], ULONG nSge,
                           UINT64 remoteAddress, UINT32 remoteToken, ULONG flags, HRESULT* status) {
  HRESULT hr = qp->Write(requestContext, sge, nSge, remoteAddress, remoteToken, flags);
  if (status != NULL) *status = hr;
  if (hr != ND_SUCCESS && hr != ND_PENDING) {
    WARN("NET/ND : Write failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Bind a connector to its local adapter address.
ncclResult_t wrap_nd_connector_bind(struct IND2Connector* connector, const struct sockaddr* pAddress, ULONG cbAddress) {
  HRESULT hr = connector->Bind(pAddress, cbAddress);
  if (hr != ND_SUCCESS) {
    WARN("NET/ND : Connector Bind failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Start an asynchronous connector operation and report immediate completion.
ncclResult_t wrap_nd_connect(struct IND2Connector* connector, struct IND2QueuePair* qp, const struct sockaddr* pAddress,
                             ULONG cbAddress, ULONG inboundReadLimit, ULONG outboundReadLimit, OVERLAPPED* pOverlapped,
                             int* done) {
  HRESULT hr = connector->Connect(qp, pAddress, cbAddress, inboundReadLimit, outboundReadLimit, NULL, 0, pOverlapped);
  if (hr != ND_SUCCESS && hr != ND_PENDING) {
    WARN("NET/ND : Connect failed: 0x%08x", hr);
    return ncclSystemError;
  }
  if (done) *done = (hr == ND_SUCCESS);
  return ncclSuccess;
}

// Poll an overlapped completion-queue operation without blocking.
ncclResult_t wrap_nd_cq_get_status(struct IND2CompletionQueue* cq, OVERLAPPED* pOverlapped, int* done,
                                   HRESULT* status) {
  if (done) *done = 0;
  HRESULT hr = static_cast<IND2Overlapped*>(cq)->GetOverlappedResult(pOverlapped, FALSE);
  if (status) *status = hr;
  if (hr == ND_PENDING) return ncclSuccess;
  if (done) *done = 1;
  return ncclSuccess;
}

// Cancel outstanding completion-queue notifications.
ncclResult_t wrap_nd_cq_cancel_notifications(struct IND2CompletionQueue* cq) {
  HRESULT hr = static_cast<IND2Overlapped*>(cq)->CancelOverlappedRequests();
  if (FAILED(hr)) {
    WARN("NET/ND : Cancel CQ notifications failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Complete connector negotiation and report immediate completion.
ncclResult_t wrap_nd_complete_connect(struct IND2Connector* connector, OVERLAPPED* pOverlapped, int* done) {
  HRESULT hr = connector->CompleteConnect(pOverlapped);
  if (hr != ND_SUCCESS && hr != ND_PENDING) {
    WARN("NET/ND : CompleteConnect failed: 0x%08x", hr);
    return ncclSystemError;
  }
  if (done) *done = (hr == ND_SUCCESS);
  return ncclSuccess;
}

// Start an asynchronous accept and report immediate completion.
ncclResult_t wrap_nd_accept(struct IND2Connector* connector, struct IND2QueuePair* qp, ULONG inboundReadLimit,
                            ULONG outboundReadLimit, OVERLAPPED* pOverlapped, int* done) {
  HRESULT hr = connector->Accept(qp, inboundReadLimit, outboundReadLimit, NULL, 0, pOverlapped);
  if (hr != ND_SUCCESS && hr != ND_PENDING) {
    WARN("NET/ND : Accept failed: 0x%08x", hr);
    return ncclSystemError;
  }
  if (done) *done = (hr == ND_SUCCESS);
  return ncclSuccess;
}

// Bind a listener to its local adapter address.
ncclResult_t wrap_nd_bind(struct IND2Listener* listener, const struct sockaddr* pAddress, ULONG cbAddress) {
  HRESULT hr = listener->Bind(pAddress, cbAddress);
  if (FAILED(hr)) {
    WARN("NET/ND : Listener Bind failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Start listening with the requested connection backlog.
ncclResult_t wrap_nd_listen(struct IND2Listener* listener, ULONG backlog) {
  HRESULT hr = listener->Listen(backlog);
  if (FAILED(hr)) {
    WARN("NET/ND : Listen failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Read the provider-assigned listener address.
ncclResult_t wrap_nd_get_listener_address(struct IND2Listener* listener, struct sockaddr* pAddress,
                                          ULONG* pAddressLength) {
  HRESULT hr = listener->GetLocalAddress(pAddress, pAddressLength);
  if (FAILED(hr)) {
    WARN("NET/ND : Listener GetLocalAddress failed: 0x%08x", hr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// Start receiving one connection request on a listener.
ncclResult_t wrap_nd_get_connection_request(struct IND2Listener* listener, struct IND2Connector* pConnector,
                                            OVERLAPPED* pOverlapped, int* done) {
  HRESULT hr = listener->GetConnectionRequest(pConnector, pOverlapped);
  if (hr != ND_SUCCESS && hr != ND_PENDING) {
    WARN("NET/ND : GetConnectionRequest failed: 0x%08x", hr);
    return ncclSystemError;
  }
  if (done) *done = (hr == ND_SUCCESS);
  return ncclSuccess;
}

// Poll a listener request without blocking.
ncclResult_t wrap_nd_listener_get_request_status(struct IND2Listener* listener, OVERLAPPED* pOverlapped, int* done) {
  if (done) *done = 0;
  // IND2Listener exposes the request status through IND2Overlapped.
  IND2Overlapped* ov = static_cast<IND2Overlapped*>(listener);
  HRESULT hr = ov->GetOverlappedResult(pOverlapped, FALSE);
  if (hr == ND_SUCCESS) {
    if (done) *done = 1;
    return ncclSuccess;
  }
  if (hr == ND_PENDING) {
    if (done) *done = 0;
    return ncclSuccess;
  }
  WARN("NET/ND : GetOverlappedResult (listener) failed: 0x%08x", hr);
  return ncclSystemError;
}

// Poll a connector operation without blocking.
ncclResult_t wrap_nd_connector_get_status(struct IND2Connector* connector, OVERLAPPED* pOverlapped, int* done) {
  if (done) *done = 0;
  // IND2Connector exposes operation status through IND2Overlapped.
  IND2Overlapped* ov = static_cast<IND2Overlapped*>(connector);
  HRESULT hr = ov->GetOverlappedResult(pOverlapped, FALSE);
  if (hr == ND_SUCCESS) {
    if (done) *done = 1;
    return ncclSuccess;
  }
  if (hr == ND_PENDING) {
    if (done) *done = 0;
    // Back off while the non-blocking state machine polls so provider progress
    // is not starved by a tight completion loop.
    Sleep(1);
    return ncclSuccess;
  }
  WARN("NET/ND : GetOverlappedResult (connector) failed: 0x%08x", hr);
  return ncclSystemError;
}

// Create or reset an event and clear its overlapped operation state.
ncclResult_t wrap_nd_prepare_overlapped(OVERLAPPED* pOverlapped) {
  HANDLE event = pOverlapped->hEvent;
  if (event == NULL) {
    event = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (event == NULL) {
      WARN("NET/ND : CreateEvent for overlapped operation failed: %lu", GetLastError());
      return ncclSystemError;
    }
  } else if (!ResetEvent(event)) {
    WARN("NET/ND : ResetEvent for overlapped operation failed: %lu", GetLastError());
    return ncclSystemError;
  }
  ZeroMemory(pOverlapped, sizeof(*pOverlapped));
  pOverlapped->hEvent = event;
  return ncclSuccess;
}

// Close an overlapped event and clear its state.
void wrap_nd_close_overlapped(OVERLAPPED* pOverlapped) {
  if (pOverlapped->hEvent != NULL) CloseHandle(pOverlapped->hEvent);
  ZeroMemory(pOverlapped, sizeof(*pOverlapped));
}
