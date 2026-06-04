/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os_socket_pair.h"
#include "checks.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstring>
#include <climits>

// Internal helper to create a connected socket pair (like Unix socketpair)
// This is needed because Windows pipes cannot be used with WSAPoll()
static ncclResult_t createSocketPair(SOCKET fds[2], int* wsaError) {
  SOCKET listener = INVALID_SOCKET;
  SOCKET client = INVALID_SOCKET;
  SOCKET server = INVALID_SOCKET;
  struct sockaddr_in addr;
  int addrlen = sizeof(addr);

  fds[0] = fds[1] = INVALID_SOCKET;
  *wsaError = 0;

  // Create listener socket
  listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listener == INVALID_SOCKET) goto fail;

  // Bind to localhost on any available port
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;  // Let OS choose port

  if (bind(listener, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) goto fail;
  if (listen(listener, 1) == SOCKET_ERROR) goto fail;
  if (getsockname(listener, (struct sockaddr*)&addr, &addrlen) == SOCKET_ERROR) goto fail;

  // Create client socket and connect
  client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (client == INVALID_SOCKET) goto fail;
  if (connect(client, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) goto fail;

  // Accept the connection
  server = accept(listener, NULL, NULL);
  if (server == INVALID_SOCKET) goto fail;

  // Close the listener - we don't need it anymore
  closesocket(listener);

  fds[0] = server;  // Read end (like pipe[0])
  fds[1] = client;  // Write end (like pipe[1])
  return ncclSuccess;

fail:
  *wsaError = WSAGetLastError();
  if (listener != INVALID_SOCKET) closesocket(listener);
  if (client != INVALID_SOCKET) closesocket(client);
  if (server != INVALID_SOCKET) closesocket(server);
  return ncclSystemError;
}

ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]) {
  SOCKET sockets[2];
  int wsaError = 0;
  ncclResult_t ret = createSocketPair(sockets, &wsaError);
  if (ret != ncclSuccess) {
    WARN("Failed to create socket pair: WSA error %d", wsaError);
    return ret;
  }
  pair[0] = sockets[0];
  pair[1] = sockets[1];
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]) {
  ncclResult_t firstError = ncclSuccess;
  for (int i = 0; i < 2; i++) {
    if (pair[i] != NCCL_SOCKET_PAIR_INVALID) {
      if (closesocket(pair[i]) == SOCKET_ERROR && firstError == ncclSuccess) {
        WARN("Failed to close socket: %d", WSAGetLastError());
        firstError = ncclSystemError;
      }
      pair[i] = NCCL_SOCKET_PAIR_INVALID;
    }
  }
  return firstError;
}

ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor descriptor, const void* buf, size_t len, size_t* written) {
  // Clamp to INT_MAX since send() takes int length
  // Callers will loop to send remaining data
  int sendLen = (len > INT_MAX) ? INT_MAX : (int)len;
  int n = send(descriptor, (const char*)buf, sendLen, 0);
  if (n == SOCKET_ERROR) {
    WARN("Failed to send: %d", WSAGetLastError());
    return ncclSystemError;
  }
  *written = (size_t)n;
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor descriptor, void* buf, size_t len, size_t* nread) {
  // Clamp to INT_MAX since recv() takes int length
  // Callers will loop to read remaining data
  int recvLen = (len > INT_MAX) ? INT_MAX : (int)len;
  int n = recv(descriptor, (char*)buf, recvLen, 0);
  if (n == SOCKET_ERROR) {
    WARN("Failed to recv: %d", WSAGetLastError());
    return ncclSystemError;
  }
  *nread = (size_t)n;
  return ncclSuccess;
}
