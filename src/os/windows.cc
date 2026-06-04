/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Force Win7+ so IP Helper and NUMA APIs are declared (project may set lower _WIN32_WINNT) */
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0601

/* GetAdaptersAddresses and IP_ADAPTER_ADDRESSES require winsock2 before iphlpapi */
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#include <winsock2.h>
#include <ws2ipdef.h>
#include <Iptypes.h>
#include <Iphlpapi.h>
#pragma comment(lib, "iphlpapi.lib")

#include <windows.h>
#include "os.h"
#include <cstring>
#include <cstdbool>
#include "socket.h"
#include "utils.h"
#include "checks.h"
#include "param.h"
#include "core.h"
#include "nvmlwrap.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <nmmintrin.h>
#include <cstdint>
#include <setupapi.h>
#include <cfgmgr32.h>
#include <devguid.h>
#include <initguid.h>
#include <devpkey.h>
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "cfgmgr32.lib")

// WSAEAGAIN not in winsock2.h; use WSAEWOULDBLOCK equivalent
#ifndef WSAEAGAIN
#define WSAEAGAIN WSAEWOULDBLOCK
#endif

// Windows-specific definitions for constants not available in Windows
#ifndef IFNAMSIZ
#define IFNAMSIZ 16
#endif

static thread_local char ncclDlErrorBuf[256] = {0};

static void saveDlError() {
  DWORD err = GetLastError();
  if (err != 0) {
    DWORD len = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, err,
                               MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), ncclDlErrorBuf, sizeof(ncclDlErrorBuf), NULL);
    if (len == 0) {
      snprintf(ncclDlErrorBuf, sizeof(ncclDlErrorBuf), "GetLastError=%lu", err);
    }
  } else {
    ncclDlErrorBuf[0] = '\0';
  }
}

ncclOsLibraryHandle ncclOsDlopen(const char* filename) {
  ncclOsLibraryHandle handle = (ncclOsLibraryHandle)LoadLibraryA(filename);
  if (handle == NULL) {
    saveDlError();
    INFO(NCCL_INIT, "ncclOsDlopen(%s) failed: %s", filename, ncclDlErrorBuf);
  }
  return handle;
}

void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  void* ptr = (void*)GetProcAddress((HMODULE)handle, symbol);
  if (ptr == NULL) {
    saveDlError();
    INFO(NCCL_INIT, "ncclOsDlsym(%s) failed: %s", symbol, ncclDlErrorBuf);
  }
  return ptr;
}

const char* ncclOsDlerror() {
  return ncclDlErrorBuf;
}

ncclOsLibraryHandle ncclOsDlopen(const char* path, int mode) {
  (void)mode;
  return (ncclOsLibraryHandle)LoadLibraryA(path);
}

void ncclOsDlclose(ncclOsLibraryHandle handle) {
  if (handle) FreeLibrary((HMODULE)handle);
}

uint64_t ncclOsGetPid() {
  return (uint64_t)GetCurrentProcessId();
}

std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf) {
  return localtime_s(buf, timer) == 0 ? buf : nullptr;
}

uint64_t ncclOsGetTid() {
  return (uint64_t)GetCurrentThreadId();
}

size_t ncclOsGetPageSize() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (size_t)si.dwPageSize;
}

void* ncclOsAlignedAlloc(size_t alignment, size_t size) {
  return _aligned_malloc(size, alignment);
}

void ncclOsAlignedFree(void* ptr) {
  _aligned_free(ptr);
}

void ncclOsSetEnv(const char* name, const char* value) {
  // Check if the environment variable already has the desired value before overriding.
  // MSDN documents the maximum environment variable size as 32767 characters
  // https://learn.microsoft.com/en-us/windows/win32/procthread/environment-variables
  char existingValue[32767];
  DWORD result = GetEnvironmentVariableA(name, existingValue, sizeof(existingValue));
  if (result == 0) {
    BOOL res = SetEnvironmentVariableA(name, value);
    if (!res) {
      WARN("Failed to set environment variable %s to %s: error %lu", name, value, GetLastError());
    }
  }
}

char* ncclOsStrSep(char** stringp, const char* delim) {
  if (*stringp == NULL) return NULL;
  char* start = *stringp;
  char* found = strpbrk(start, delim);
  if (found) {
    *found = '\0';
    *stringp = found + 1;
  } else {
    *stringp = NULL;
  }
  return start;
}

ncclResult_t ncclOsInitialize() {
  // Windows Winsock initialization
  WSADATA wsaData;
  int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (result != 0) {
    WARN("WSAStartup failed with error: %d", result);
    return ncclSystemError;
  }
  INFO(NCCL_INIT | NCCL_NET, "WSAStartup succeeded, Winsock version %d.%d", LOBYTE(wsaData.wVersion),
       HIBYTE(wsaData.wVersion));
  return ncclSuccess;
}

ncclResult_t ncclOsSetFilesLimit() {
  return ncclSuccess;
}

bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor sockDescriptor) {
  return sockDescriptor != INVALID_SOCKET;
}

bool ncclOsSocketIsValid(struct ncclSocket* sock) {
  return ncclOsSocketDescriptorIsValid(sock->socketDescriptor);
}

extern int ncclParamPollTimeOut();
extern int ncclParamRetryTimeOut();

void ncclOsPollSocket(SOCKET sock, int op) {
  WSAPOLLFD pfd;
  memset(&pfd, 0, sizeof(WSAPOLLFD));
  pfd.fd = sock;
  pfd.events = (op == NCCL_SOCKET_RECV) ? POLLIN : POLLOUT;
  WSAPoll(&pfd, 1, ncclParamPollTimeOut());
}

static const char* getWSAErrorMessage(int error) {
  static char errorMsg[256];
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, error,
                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), errorMsg, sizeof(errorMsg), NULL);
  return errorMsg;
}

extern long int ncclParamRetryCnt();

ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock) {
  int socklen = sizeof(union ncclSocketAddress);
  sock->socketDescriptor = accept(sock->acceptSocketDescriptor, (struct sockaddr*)&sock->addr, &socklen);
  if (ncclOsSocketIsValid(sock)) {
    sock->state = ncclSocketStateAccepted;
  } else {
    int wsaError = WSAGetLastError();
    if (wsaError == WSAEINPROGRESS) {
      // Connection in progress, retry with backoff
      if (++sock->errorRetries == ncclParamRetryCnt()) {
        WARN("ncclOsSocketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries,
             getWSAErrorMessage(wsaError));
        return ncclSystemError;
      }
      INFO(NCCL_NET | NCCL_INIT, "Call to accept returned %s, retrying", getWSAErrorMessage(wsaError));
    } else if (wsaError != WSAEINTR && wsaError != WSAEWOULDBLOCK) {
      // WSAEWOULDBLOCK (10035) is expected for non-blocking accept - means no pending connection yet
      // WSAEINTR means interrupted, both are normal and we just return success to try again later
      WARN("ncclOsSocketTryAccept: Accept failed: %s", getWSAErrorMessage(wsaError));
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

extern int ncclParamSocketMaxRecvBuff();
extern int ncclParamSocketMaxSendBuff();

ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock) {
  const int one = 1;
  ncclResult_t ret = ncclSuccess;
  int sndBuf, rcvBuf;
  /* Set socket as non-blocking if async or if we need to be able to abort */
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclOsSocketSetFlags: invalid socket");
    ret = ncclInvalidArgument;
    goto fail;
  }
  if (sock->asyncFlag || sock->abortFlag) {
    u_long mode = 1;
    int iResult = ioctlsocket(sock->socketDescriptor, FIONBIO, &mode);
    if (iResult != NO_ERROR) {
      WARN("ncclOsSocketSetFlags: ioctlsocket failed with error: %ld", iResult);
      ret = ncclSystemError;
      goto fail;
    }
    sock->socketBlockingMode = 0;
  }
  SYSCHECKGOTO(setsockopt(sock->socketDescriptor, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)),
               "setsockopt TCP NODELAY", ret, fail);
  // setsockopt should not fail even if the sizes are too large, do not change the default if unset by the user (=-1)
  rcvBuf = ncclParamSocketMaxRecvBuff();
  sndBuf = ncclParamSocketMaxSendBuff();
  if (sndBuf > 0) {
    SYSCHECKGOTO(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_SNDBUF, (char*)&sndBuf, sizeof(int)),
                 "setsockopt SO_SNDBUF", ret, fail);
  }
  if (rcvBuf > 0) {
    SYSCHECKGOTO(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBuf, sizeof(int)),
                 "setsockopt SO_RCVBUF", ret, fail);
  }
exit:
  return ret;
fail:
  goto exit;
}

void ncclOsSocketResetAccept(struct ncclSocket* sock) {
  // Close the accepted peer and return to listening for another connection (see socketFinalizeAccept logging).
  (void)closesocket(sock->socketDescriptor);
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateBadHandshake;
  sock->finalizeCounter = 0;
}

ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  SOCKET newSocket = INVALID_SOCKET;

  newSocket = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0);
  if (newSocket == INVALID_SOCKET) {
    int wsaError = WSAGetLastError();
    WARN("ncclOsSocketResetFd: socket() failed with error %d: %s", wsaError, getWSAErrorMessage(wsaError));
    ret = ncclSystemError;
    goto cleanup;
  }

  // if socket is valid, close it and replace with new socket
  if (ncclOsSocketIsValid(sock)) {
    (void)closesocket(sock->socketDescriptor);
  }
  sock->socketDescriptor = newSocket;
  NCCLCHECKGOTO(ncclOsSocketSetFlags(sock), ret, exit);
exit:
  return ret;
cleanup:
  // cleanup socket, leave sock->socketDescriptor untouched
  if (newSocket != INVALID_SOCKET) {
    (void)closesocket(newSocket);
  }
  goto exit;
}

static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode, const char funcName[]) {
  char line[SOCKET_NAME_MAXLEN + 1];
  if (errCode == 0) {
    sock->state = ncclSocketStateConnected;
  } else if (errCode == WSAEINPROGRESS || errCode == WSAEWOULDBLOCK) {
    // WSAEWOULDBLOCK (10035) on Windows for non-blocking connect() is equivalent to
    // EINPROGRESS on Linux - it means connection is in progress, poll for completion
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == WSAEINTR || errCode == WSAEAGAIN || errCode == WSAETIMEDOUT || errCode == WSAEHOSTUNREACH ||
             errCode == WSAECONNREFUSED) {
    if (sock->customRetry == 0) {
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts", funcName,
             ncclSocketToString(&sock->addr, line), getWSAErrorMessage(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET | NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec", funcName,
           ncclSocketToString(&sock->addr, line), getWSAErrorMessage(errCode), sock->errorRetries, ncclParamRetryCnt(),
           sleepTime);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
    }
    NCCLCHECK(ncclOsSocketResetFd(sock)); /* in case of failure in connect, socket state is unspecified */
    sock->state = ncclSocketStateConnecting;
  } else {
    sock->state = ncclSocketStateError;
    WARN("%s: connect to %s failed : %s", funcName, ncclSocketToString(&sock->addr, line), getWSAErrorMessage(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->socketDescriptor, &sock->addr.sa, sock->salen);
  return socketConnectCheck(sock, (ret == -1) ? WSAGetLastError() : 0, __func__);
}

ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock) {
  WSAPOLLFD pfd;
  int timeout = 1, ret;
  int optlen = sizeof(int);  /* Windows getsockopt expects int* for optlen */
  char line[SOCKET_NAME_MAXLEN + 1];

  memset(&pfd, 0, sizeof(WSAPOLLFD));
  pfd.fd = sock->socketDescriptor;
  pfd.events = POLLOUT;
  ret = WSAPoll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && WSAGetLastError() == WSAEINTR)) {
    return ncclSuccess;
  } else if (ret < 0) {
    int wsaError = WSAGetLastError();
    WARN("ncclOsSocketPollConnect to %s failed with error %s", ncclSocketToString(&sock->addr, line),
         getWSAErrorMessage(wsaError));
    return ncclSystemError;
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->socketDescriptor, SOL_SOCKET, SO_ERROR, (char*)&ret, &optlen), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block,
                                     int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;
  char line[SOCKET_NAME_MAXLEN + 1];
  if (sock->asyncFlag || sock->abortFlag) block = 0;
  if (block != sock->socketBlockingMode) {
    u_long mode = !block;
    int iResult = ioctlsocket(sock->socketDescriptor, FIONBIO, &mode);
    if (iResult != NO_ERROR) {
      WARN("ncclOsSocketProgressOpt: ioctlsocket failed with error: %ld", iResult);
      return ncclSystemError;
    }
    sock->socketBlockingMode = block;
  }
  do {
    if (op == NCCL_SOCKET_RECV) bytes = recv(sock->socketDescriptor, data + (*offset), size - (*offset), 0);
    if (op == NCCL_SOCKET_SEND) bytes = send(sock->socketDescriptor, data + (*offset), size - (*offset), 0);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == SOCKET_ERROR) {
      const int wsaError = WSAGetLastError();
      if (wsaError == WSAECONNRESET) {
        *closed = 1;
        return ncclSuccess;
      }
      // WSAEWOULDBLOCK (10035) is expected for non-blocking sockets - means "try again later"
      // WSAEINPROGRESS (10036) means operation is in progress
      // WSAEINTR means interrupted by signal
      if (wsaError != WSAEWOULDBLOCK && wsaError != WSAEINPROGRESS && wsaError != WSAEINTR) {
        WARN("ncclOsSocketProgressOpt: Call to %s %s failed : %d (%s)",
             (op == NCCL_SOCKET_RECV ? "recv from" : "send to"), ncclSocketToString(&sock->addr, line), wsaError,
             getWSAErrorMessage(wsaError));
        return ncclRemoteError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (sock->abortFlag &&
        std::atomic_load_explicit((std::atomic<uint32_t>*)sock->abortFlag, std::memory_order_acquire)) {
      INFO(NCCL_NET, "ncclOsSocketProgressOpt: abort called");
      return ncclInternalError;
    }
  } while (sock->asyncFlag == 0 && bytes > 0 && (*offset) < size);
  return ncclSuccess;
}

ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress* addrs, int sock_family,
                                  int maxIfNameSize, int maxIfs, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);
  ncclResult_t ret = ncclSuccess;
  *found = 0;

  // Get adapter addresses using Windows API
  ULONG bufferSize = 0;
  DWORD result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, NULL, &bufferSize);
  if (result != ERROR_BUFFER_OVERFLOW) {
    WARN("GetAdaptersAddresses failed with error: %ld", result);
    ret = ncclSystemError;
    goto exit;
  }

  IP_ADAPTER_ADDRESSES* adapterAddresses = (IP_ADAPTER_ADDRESSES*)malloc(bufferSize);
  if (adapterAddresses == NULL) {
    WARN("Failed to allocate memory for adapter addresses");
    ret = ncclSystemError;
    goto exit;
  }

  result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, adapterAddresses, &bufferSize);
  if (result != NO_ERROR) {
    WARN("GetAdaptersAddresses failed with error: %ld", result);
    ret = ncclSystemError;
    goto exit;
  }

  // Iterate through adapters
  for (IP_ADAPTER_ADDRESSES* adapter = adapterAddresses; adapter && *found < maxIfs; adapter = adapter->Next) {
    // Skip adapters that are not operational
    if (adapter->OperStatus != IfOperStatusUp) continue;

    // Skip loopback adapters
    if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;

    // Iterate through unicast addresses for this adapter
    for (IP_ADAPTER_UNICAST_ADDRESS* unicast = adapter->FirstUnicastAddress; unicast && *found < maxIfs;
         unicast = unicast->Next) {
      if (unicast->Address.lpSockaddr == NULL) continue;

      // Get address family
      int family = unicast->Address.lpSockaddr->sa_family;
      if (family != AF_INET && family != AF_INET6) continue;

      // Allow the caller to force the socket family type
      if (sock_family != -1 && family != sock_family) continue;

      // Skip IPv6 loopback addresses
      if (family == AF_INET6) {
        struct sockaddr_in6* sa = (struct sockaddr_in6*)(unicast->Address.lpSockaddr);
        if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
      }

      // Convert adapter name to char* for matching
      char adapterName[MAX_IF_NAME_SIZE];
      WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1, adapterName, MAX_IF_NAME_SIZE, NULL, NULL);

      TRACE(NCCL_INIT | NCCL_NET, "Found interface %s:%s", adapterName,
            ncclSocketToString((union ncclSocketAddress*)unicast->Address.lpSockaddr, line));

      // Check against user specified interfaces
      if (!(matchIfList(adapterName, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
        continue;
      }

      // Check that this interface has not already been saved
      bool duplicate = false;
      for (int i = 0; i < *found; i++) {
        if (strcmp(adapterName, names + i * maxIfNameSize) == 0) {
          duplicate = true;
          break;
        }
      }

      if (!duplicate) {
        // Store the interface name
        strncpy(names + (*found) * maxIfNameSize, adapterName, maxIfNameSize);
        names[(*found) * maxIfNameSize + maxIfNameSize - 1] = '\0';
        // Store the IP address
        int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
        memset(addrs + *found, '\0', sizeof(*addrs));
        memcpy(addrs + *found, unicast->Address.lpSockaddr, salen);
        (*found)++;
      }
    }
  }
exit:
  free(adapterAddresses);
  return ret;
}

static bool matchSubnet(IP_ADAPTER_UNICAST_ADDRESS* local_addr, union ncclSocketAddress* remote) {
  /* Check family first */
  int family = local_addr->Address.lpSockaddr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_sa = (struct sockaddr_in*)(local_addr->Address.lpSockaddr);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;

    // Create netmask from prefix length
    uint32_t prefix_len = local_addr->OnLinkPrefixLength;
    if (prefix_len > 32) prefix_len = 32; // Sanity check

    uint32_t mask;
    if (prefix_len == 0) {
      mask = 0;
    } else {
      mask = 0xFFFFFFFF << (32 - prefix_len);
    }

    local_subnet.s_addr = local_sa->sin_addr.s_addr & htonl(mask);
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & htonl(mask);

    return (local_subnet.s_addr == remote_subnet.s_addr);
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_sa = (struct sockaddr_in6*)(local_addr->Address.lpSockaddr);
    struct sockaddr_in6& remote_addr = remote->sin6;

    // Create netmask from prefix length
    uint32_t prefix_len = local_addr->OnLinkPrefixLength;
    if (prefix_len > 128) prefix_len = 128; // Sanity check

    struct in6_addr& local_in6 = local_sa->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;

    // Calculate how many full bytes to compare
    int full_bytes = prefix_len / 8;
    int remaining_bits = prefix_len % 8;

    // Compare full bytes
    for (int c = 0; c < full_bytes; c++) {
      if (local_in6.s6_addr[c] != remote_in6.s6_addr[c]) {
        same = false;
        break;
      }
    }

    // Compare remaining bits in the partial byte
    if (same && remaining_bits > 0 && full_bytes < 16) {
      unsigned char mask = 0xFF << (8 - remaining_bits);
      if ((local_in6.s6_addr[full_bytes] & mask) != (remote_in6.s6_addr[full_bytes] & mask)) {
        same = false;
      }
    }

    // Compare scope id for link-local addresses
    same &= (local_sa->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    INFO(NCCL_NET, "Net : Unsupported address family type");
    return false;
  }
}

ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  char line_a[SOCKET_NAME_MAXLEN + 1];
#endif
  ncclResult_t ret = ncclSuccess;
  *found = 0;

  // Get adapter addresses using Windows API
  ULONG bufferSize = 0;
  DWORD result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, NULL, &bufferSize);
  if (result != ERROR_BUFFER_OVERFLOW) {
    WARN("GetAdaptersAddresses failed with error: %ld", result);
    ret = ncclSystemError;
    goto exit;
  }

  IP_ADAPTER_ADDRESSES* adapterAddresses = (IP_ADAPTER_ADDRESSES*)malloc(bufferSize);
  if (adapterAddresses == NULL) {
    WARN("Failed to allocate memory for adapter addresses");
    ret = ncclSystemError;
    goto exit;
  }

  result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, adapterAddresses, &bufferSize);
  if (result != NO_ERROR) {
    WARN("GetAdaptersAddresses failed with error: %ld", result);
    free(adapterAddresses);
    ret = ncclSystemError;
    goto exit;
  }

  // Iterate through adapters
  for (IP_ADAPTER_ADDRESSES* adapter = adapterAddresses; adapter && !*found; adapter = adapter->Next) {
    // Skip adapters that are not operational
    if (adapter->OperStatus != IfOperStatusUp) continue;

    // Skip loopback adapters
    if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;

    // Iterate through unicast addresses for this adapter
    for (IP_ADAPTER_UNICAST_ADDRESS* unicast = adapter->FirstUnicastAddress; unicast && !*found;
         unicast = unicast->Next) {
      if (unicast->Address.lpSockaddr == NULL) continue;

      // Get address family
      int family = unicast->Address.lpSockaddr->sa_family;
      if (family != AF_INET && family != AF_INET6) continue;

      // Skip IPv6 loopback addresses
      if (family == AF_INET6) {
        struct sockaddr_in6* sa = (struct sockaddr_in6*)(unicast->Address.lpSockaddr);
        if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
      }

      // Check if this interface is in the same subnet as the remote address
      if (!matchSubnet(unicast, remoteAddr)) {
        continue;
      }

      // Store the local IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memcpy(localAddr, unicast->Address.lpSockaddr, salen);

      // Convert adapter name to char* for storage
      WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1, ifName, ifNameMaxSize, NULL, NULL);

      TRACE(NCCL_INIT | NCCL_NET, "NET : Found interface %s:%s in the same subnet as remote address %s", ifName,
            ncclSocketToString(localAddr, line), ncclSocketToString(remoteAddr, line_a));
      *found = 1;
    }
  }
exit:
  free(adapterAddresses);
  return ret;
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) {
  if (sock != NULL) {
    if (sock->state > ncclSocketStateNone && sock->state < ncclSocketStateNum && ncclOsSocketIsValid(sock)) {
      if (wait) {
        char data;
        int closed = 0;
        do {
          int offset = 0;
          if (ncclSocketProgress(NCCL_SOCKET_RECV, sock, &data, sizeof(char), &offset, &closed) != ncclSuccess) break;
        } while (closed == 0);
      }
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is not affected
       * by refcount of fd, but close() is. close() won't close a fd and send FIN packet if
       * the fd is duplicated (e.g. fork()). So shutdown() guarantees the correct and graceful
       * connection close here. */
      (void)shutdown(sock->socketDescriptor, SD_BOTH);
      (void)closesocket(sock->socketDescriptor);
    }
    sock->state = ncclSocketStateClosed;
    sock->socketDescriptor = NCCL_INVALID_SOCKET;
  }
  return ncclSuccess;
}

void ncclOsSetMutexCondShared(std::mutex& mutex, std::condition_variable& cond) {
  // Not implemented on Windows
}

void ncclOsCpuZero(ncclAffinity& affinity) {
  affinity = 0;
}

int ncclOsCpuCount(const ncclAffinity& affinity) {
  return _mm_popcnt_u64(affinity);
}

void ncclOsCpuSet(ncclAffinity& affinity, int cpu) {
  affinity |= (1ULL << cpu);
}

bool ncclOsCpuIsSet(const ncclAffinity& affinity, int cpu) {
  return (affinity & (1ULL << cpu)) != 0;
}

ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b) {
  return a & b;
}

ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) {
  DWORD_PTR processAffinityMask, systemAffinityMask;
  BOOL result = GetProcessAffinityMask(GetCurrentProcess(), &processAffinityMask, &systemAffinityMask);
  if (result == FALSE) {
    WARN("GetProcessAffinityMask failed with error: %ld", GetLastError());
    return ncclSystemError;
  }
  *affinity = processAffinityMask;
  return ncclSuccess;
}

ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) {
  BOOL result = SetProcessAffinityMask(GetCurrentProcess(), affinity);
  if (result == FALSE) {
    WARN("SetProcessAffinityMask failed with error: %ld", GetLastError());
    return ncclSystemError;
  }
  return ncclSuccess;
}

int ncclOsGetCpu() {
  return GetCurrentProcessorNumber();
}

ncclResult_t ncclOsNvmlOpen(ncclOsLibraryHandle* handle) {
  *handle = nullptr;

  // On Windows, try multiple possible locations for nvml.dll
  const char* nvmlPaths[] = {"nvml.dll",  // System PATH or current directory
                             "C:\\Windows\\System32\\nvml.dll",  // Common system location
                             nullptr};

  for (int i = 0; nvmlPaths[i] != nullptr && *handle == nullptr; i++) {
    *handle = ncclOsDlopen(nvmlPaths[i]);
    if (*handle != nullptr) {
      INFO(NCCL_INIT, "Loaded NVML from %s", nvmlPaths[i]);
    }
  }

  if (*handle == nullptr) {
    DWORD err = GetLastError();
    WARN("Failed to load nvml.dll, error code: %lu", err);
    return ncclSystemError;
  }

  return ncclSuccess;
}

char* ncclOsRealpath(const char* path, char* resolved_path) {
  if (path == NULL) {
    errno = EINVAL;
    return NULL;
  }

  // If resolved_path is provided, use it. Otherwise allocate a buffer.
  char* buffer = resolved_path;
  if (buffer == NULL) {
    buffer = (char*)malloc(PATH_MAX);
    if (buffer == NULL) {
      errno = ENOMEM;
      return NULL;
    }
  }

  // Use Windows _fullpath to resolve the path
  char* result = _fullpath(buffer, path, PATH_MAX);
  if (result == NULL) {
    // _fullpath failed
    if (resolved_path == NULL) {
      free(buffer);
    }
    // errno is already set by _fullpath
    return NULL;
  }

  return buffer;
}

// Shared memory implementation for Windows
#include "comm.h"
#include <string.h>
#include <stdlib.h>

void ncclOsShmHandleInit(ncclShmDescriptor shmDesc, char* shmPath, size_t shmSize, size_t realShmSize, char* hptr,
                         void* dptr, bool create, struct ncclShmHandleInternal* handle) {
  handle->shmDesc = shmDesc;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
  if (create) {
    int slen = strlen(shmPath);
    handle->shmPath = (char*)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    if (hptr) memset(hptr, 0, shmSize);
  } else {
    handle->shmPath = NULL;
  }
}

ncclResult_t ncclOsShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void** devShmPtr,
                           int refcount, struct ncclShmHandleInternal** handle) {
  HANDLE hMapFile = NULL;
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  bool captureModeSet = false;
  bool registered = false;
  struct ncclShmHandleInternal* tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(uint64_t);
  const size_t realShmSize = shmSize + refSize;

  *handle = NULL;
  *shmPtr = NULL;
  EQCHECKGOTO(tmphandle = (struct ncclShmHandleInternal*)calloc(1, sizeof(struct ncclShmHandleInternal)), NULL, ret,
              fail);

  if (create) {
    if (shmPath[0] == '\0') {
      // Generate unique shared memory name using process ID and timestamp
      uint64_t timestamp = clockNano();
      snprintf(shmPath, shmPathSize, "Local\\nccl-shm-%llu-%llu", (unsigned long long)GetCurrentProcessId(),
               (unsigned long long)timestamp);
    }

    // Create file mapping object
    hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE,    // use paging file
                                  NULL,                    // default security
                                  PAGE_READWRITE,          // read/write access
                                  (DWORD)((realShmSize >> 32) & 0xFFFFFFFF),  // high-order DWORD of size
                                  (DWORD)(realShmSize & 0xFFFFFFFF),          // low-order DWORD of size
                                  shmPath);                // name of mapping object

    if (hMapFile == NULL) {
      WARN("Error: failed to create shared memory mapping %s, error code: %lu", shmPath, GetLastError());
      ret = ncclSystemError;
      goto fail;
    }

    INFO(NCCL_ALLOC, "Created shared memory mapping %s with %ld bytes", shmPath, realShmSize);
  } else {
    // Open existing file mapping object
    hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS,   // read/write access
                                FALSE,                 // do not inherit the name
                                shmPath);              // name of mapping object

    if (hMapFile == NULL) {
      WARN("Error: failed to open shared memory mapping %s, error code: %lu", shmPath, GetLastError());
      ret = ncclSystemError;
      goto fail;
    }
  }

  // Map view of the file mapping into address space
  hptr = (char*)MapViewOfFile(hMapFile,            // handle to map object
                              FILE_MAP_ALL_ACCESS, // read/write permission
                              0,                   // high-order DWORD of offset
                              0,                   // low-order DWORD of offset
                              realShmSize);        // number of bytes to map

  if (hptr == NULL) {
    WARN("Error: Could not map view of file %s size %zu, error code: %lu", shmPath, realShmSize, GetLastError());
    ret = ncclSystemError;
    goto fail;
  }

  if (create) {
    *(int*)(hptr + shmSize) = refcount;
  } else {
    int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
    if (remref == 0) {
      INFO(NCCL_ALLOC, "Last reference to shared memory %s released", shmPath);
    }
  }

  if (devShmPtr) {
    INFO(NCCL_ALLOC,
         "SHM legacy: sharing buffer with GPU via cudaHostRegister + cudaHostGetDevicePointer (host %p size %ld)",
         (void*)hptr, (long)realShmSize);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_cuda);
    captureModeSet = true;
    CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret,
                  fail_cuda);
    registered = true;
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail_cuda);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail_cuda);
    captureModeSet = false;
  }

  ncclOsShmHandleInit(hMapFile, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  *handle = tmphandle;
  return ret;
fail_cuda:
  if (registered) {
    cudaError_t unregRes = cudaHostUnregister((void*)hptr);
    if (unregRes != cudaSuccess) {
      WARN("SHM legacy: cudaHostUnregister after setup failure failed: %s", cudaGetErrorString(unregRes));
    }
  }
  if (captureModeSet) {
    cudaError_t modeRes = cudaThreadExchangeStreamCaptureMode(&mode);
    if (modeRes != cudaSuccess) {
      WARN("SHM legacy: failed to restore CUDA stream capture mode: %s", cudaGetErrorString(modeRes));
    }
  }
  dptr = NULL;
fail:
  WARN("Error while %s shared memory segment %s (size %ld)", create ? "creating" : "attaching to", shmPath, shmSize);
  if (tmphandle) {
    ncclOsShmHandleInit(hMapFile, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    (void)ncclOsShmClose(tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

ncclResult_t ncclOsShmClose(struct ncclShmHandleInternal* handle) {
  ncclResult_t ret = ncclSuccess;
  if (handle) {
    if (handle->shmPtr) {
      if (handle->devShmPtr) CUDACHECK(cudaHostUnregister(handle->shmPtr));
      if (!UnmapViewOfFile(handle->shmPtr)) {
        WARN("UnmapViewOfFile of shared memory %p size %ld failed, error code: %lu", handle->shmPtr,
             handle->realShmSize, GetLastError());
        ret = ncclSystemError;
      }
    }

    if (handle->shmDesc != NULL) {
      if (!CloseHandle(handle->shmDesc)) {
        WARN("CloseHandle for shared memory %s failed, error code: %lu", handle->shmPath ? handle->shmPath : "(null)",
             GetLastError());
        ret = ncclSystemError;
      }
      free(handle->shmPath);
    }

    free(handle);
  }
  return ret;
}

ncclResult_t ncclOsShmUnlink(struct ncclShmHandleInternal* handle) {
  ncclResult_t ret = ncclSuccess;
  if (handle) {
    // On Windows, shared memory is automatically cleaned up when all handles are closed
    if (handle->shmPath != NULL && handle->refcount != NULL && *handle->refcount > 0) {
      INFO(NCCL_ALLOC, "Unlinking shared memory %s (Windows will clean up automatically)", handle->shmPath);
      free(handle->shmPath);
      handle->shmPath = NULL;
    }
  }
  return ret;
}

/* gettimeofday() replacement for Windows (struct timeval is in winsock2.h via os.h) */
int gettimeofday(struct timeval* tv, void* tz) {
  (void)tz;
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  ULARGE_INTEGER uli;
  uli.LowPart = ft.dwLowDateTime;
  uli.HighPart = ft.dwHighDateTime;
  ULONGLONG ns100 = uli.QuadPart;
  ns100 -= 116444736000000000ULL; /* 1601 to 1970 in 100ns units */
  tv->tv_sec = (long)(ns100 / 10000000ULL);
  tv->tv_usec = (int)((ns100 % 10000000ULL) / 10ULL);
  return 0;
}

/* Topology/PCI detection functions */
#define BUSID_SIZE 13 // "0000:00:00.0" + null terminator

// Helper function to parse PCI bus ID format and convert to Windows format
static bool parsePciBusId(const char* busId, DWORD* bus, DWORD* device, DWORD* function) {
  // Expected format: DDDD:BB:DD.F (domain:bus:device.function)
  unsigned int domain, b, d, f;
  if (sscanf(busId, "%x:%x:%x.%x", &domain, &b, &d, &f) == 4) {
    *bus = b;
    *device = d;
    *function = f;
    return true;
  }
  return false;
}

// Structure to cache device information
struct DeviceInfo {
  char deviceInstanceId[MAX_PATH];
  DEVINST devInst;
  char hwId[MAX_PATH];
};

// Helper to get device info from SetupAPI
static ncclResult_t getDeviceInfo(DWORD bus, DWORD device, DWORD function, DeviceInfo* devInfo) {
  HDEVINFO deviceInfoSet = SetupDiGetClassDevs(NULL, "PCI", NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
  if (deviceInfoSet == INVALID_HANDLE_VALUE) {
    return ncclSystemError;
  }

  SP_DEVINFO_DATA deviceInfoData;
  deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

  bool found = false;
  for (DWORD i = 0; SetupDiEnumDeviceInfo(deviceInfoSet, i, &deviceInfoData); i++) {
    // Get device instance ID
    if (CM_Get_Device_ID(deviceInfoData.DevInst, devInfo->deviceInstanceId, MAX_PATH, 0) != CR_SUCCESS) {
      continue;
    }

    // Get hardware ID which contains VEN/DEV/SUBSYS/REV
    if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, NULL, (PBYTE)devInfo->hwId,
                                          MAX_PATH, NULL)) {
      // Use CM_Get_DevNode_Registry_Property to get bus number
      ULONG busNumber = 0, addressNumber = 0;
      ULONG bufferSize = sizeof(ULONG);

      // Get bus number from device node
      if (CM_Get_DevNode_Registry_Property(deviceInfoData.DevInst, CM_DRP_BUSNUMBER, NULL, &busNumber, &bufferSize,
                                           0) == CR_SUCCESS) {
        bufferSize = sizeof(ULONG);
        // Get device address (device << 16 | function)
        if (CM_Get_DevNode_Registry_Property(deviceInfoData.DevInst, CM_DRP_ADDRESS, NULL, &addressNumber, &bufferSize,
                                             0) == CR_SUCCESS) {
          unsigned int dev = (addressNumber >> 16) & 0xFFFF;
          unsigned int func = addressNumber & 0xFFFF;

          if (busNumber == bus && dev == device && func == function) {
            devInfo->devInst = deviceInfoData.DevInst;
            found = true;
            break;
          }
        }
      }
    }
  }

  SetupDiDestroyDeviceInfoList(deviceInfoSet);
  return found ? ncclSuccess : ncclSystemError;
}

// Helper to get PCI bus ID string from DEVINST
static bool getPciBusIdFromDevInst(DEVINST devInst, char* busIdOut, size_t bufSize) {
  ULONG busNumber = 0, addressNumber = 0;
  ULONG bufferSize = sizeof(ULONG);

  // Get bus number from device node
  if (CM_Get_DevNode_Registry_Property(devInst, CM_DRP_BUSNUMBER, NULL, (PVOID)&busNumber, &bufferSize, 0) !=
      CR_SUCCESS) {
    return false;
  }

  // Get device address (device << 16 | function)
  bufferSize = sizeof(ULONG);
  if (CM_Get_DevNode_Registry_Property(devInst, CM_DRP_ADDRESS, NULL, (PVOID)&addressNumber, &bufferSize, 0) !=
      CR_SUCCESS) {
    return false;
  }

  DWORD device = (addressNumber >> 16) & 0xFFFF;
  DWORD function = addressNumber & 0xFFFF;

  // Format as standard PCI bus ID (e.g., "0000:3b:00.0")
  // Domain is always 0 on Windows
  snprintf(busIdOut, bufSize, "%04x:%02x:%02x.%x", 0, (unsigned int)busNumber, device, function);
  return true;
}

// BCM switch link detection is not supported on Windows.
ncclResult_t ncclOsGetBcmLinks(const char* busId, int* nlinks, char** peers) {
  *nlinks = 0;
  *peers = NULL;
  return ncclSuccess;
}

ncclResult_t ncclOsGetPciPath(const char* busId, char** path) {
  DWORD bus, device, function;
  if (!parsePciBusId(busId, &bus, &device, &function)) {
    WARN("Invalid PCI bus ID format: %s", busId);
    *path = NULL;
    return ncclSystemError;
  }

  DeviceInfo devInfo;
  if (getDeviceInfo(bus, device, function, &devInfo) != ncclSuccess) {
    *path = NULL;
    INFO(NCCL_GRAPH, "Could not find PCI device with bus ID %s", busId);
    return ncclSystemError;
  }

  // Return device instance ID as the "path"
  *path = _strdup(devInfo.deviceInstanceId);
  return *path ? ncclSuccess : ncclSystemError;
}

// Helper to extract hex value from hardware ID string (e.g., "VEN_10DE" -> "0x10de")
static bool extractHexFromHwId(const char* hwId, const char* prefix, char* output, int maxLen) {
  const char* pos = strstr(hwId, prefix);
  if (pos != NULL) {
    pos += strlen(prefix);
    // Extract hex digits until we hit & or end
    char hexStr[16];
    int i = 0;
    while (i < 15 && pos[i] && pos[i] != '&' && pos[i] != '\\') {
      hexStr[i] = pos[i];
      i++;
    }
    hexStr[i] = '\0';
    snprintf(output, maxLen, "0x%s", hexStr);
    return true;
  }
  return false;
}

// Helper to get device registry key path
static bool getDeviceRegPath(const char* deviceInstanceId, char* regPath, int maxLen) {
  // Convert device instance ID to registry path
  // Device instance: PCI\VEN_10DE&DEV_1234...\4&1234abcd&0&00
  // Registry path: SYSTEM\CurrentControlSet\Enum\PCI\VEN_10DE&DEV_1234...\4&1234abcd&0&00
  snprintf(regPath, maxLen, "SYSTEM\\CurrentControlSet\\Enum\\%s", deviceInstanceId);
  return true;
}

// Helper to read PCI configuration space from registry
static bool readPciConfigSpace(DEVINST devInst, BYTE* configData, DWORD maxSize, DWORD* actualSize) {
  char deviceInstanceId[MAX_PATH];
  if (CM_Get_Device_ID(devInst, deviceInstanceId, MAX_PATH, 0) != CR_SUCCESS) {
    return false;
  }

  char regPath[PATH_MAX];
  getDeviceRegPath(deviceInstanceId, regPath, PATH_MAX);

  HKEY hKey;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, regPath, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
    DWORD configSize = maxSize;
    LONG result = RegQueryValueExA(hKey, "ConfigData", NULL, NULL, configData, &configSize);
    RegCloseKey(hKey);

    if (result == ERROR_SUCCESS && configSize > 0) {
      if (actualSize) *actualSize = configSize;
      return true;
    }
  }
  return false;
}

// Helper to find PCIe capability in config space and read max link width
// Reads from Link Capabilities Register (PCIe Cap + 0x0C), bits 9:4
static bool getPcieMaxLinkWidth(DEVINST devInst, char* strValue, int maxLen) {
  BYTE configData[256];
  DWORD configSize = 0;

  if (!readPciConfigSpace(devInst, configData, sizeof(configData), &configSize)) {
    return false;
  }

  // Verify we have enough data for capability pointer
  if (configSize < 0x40) {
    return false;
  }

  // Get capabilities pointer from PCI header (offset 0x34)
  BYTE capPtr = configData[0x34];

  // Walk the capability list to find PCIe capability (ID = 0x10)
  while (capPtr != 0 && capPtr < 0xFC && capPtr < configSize - 16) {
    BYTE capId = configData[capPtr];

    if (capId == 0x10) {
      // PCIe capability found
      // Verify we have enough space to read Link Capabilities Register
      if (capPtr + 0x0F < configSize) {
        // Read Link Capabilities Register (4 bytes at offset cap_base + 0x0C)
        DWORD linkCap = *(DWORD*)(configData + capPtr + 0x0C);

        // Extract Maximum Link Width from bits 9:4
        BYTE maxWidth = (linkCap >> 4) & 0x3F;

        snprintf(strValue, maxLen, "%u", maxWidth);
        return true;
      }
      break;
    }

    // Move to next capability (next pointer is at offset +1)
    if (capPtr + 1 < configSize) {
      capPtr = configData[capPtr + 1];
    } else {
      break;
    }
  }

  return false;
}

constexpr const char* pcieGenSpeedStr[] = {"",          "2.5 GT/s",  "5.0 GT/s", "8.0 GT/s",
                                           "16.0 GT/s", "32.0 GT/s", "64.0 GT/s"};
constexpr size_t pcieGenSpeedsCount = sizeof(pcieGenSpeedStr) / sizeof(pcieGenSpeedStr[0]);

// Helper to find PCIe capability and read current link speed
// Reads from Link Capabilities Register (PCIe Cap + 0x0C), bits 3:0
static bool getPcieMaxLinkSpeed(DEVINST devInst, char* strValue, int maxLen) {
  BYTE configData[256];
  DWORD configSize = 0;

  if (!readPciConfigSpace(devInst, configData, sizeof(configData), &configSize)) {
    return false;
  }

  if (configSize < 0x40) {
    return false;
  }

  BYTE capPtr = configData[0x34];

  while (capPtr != 0 && capPtr < 0xFC && capPtr < configSize - 16) {
    BYTE capId = configData[capPtr];

    if (capId == 0x10) {
      // PCIe capability found
      if (capPtr + 0x0F < configSize) {
        // Read Link Capabilities Register
        DWORD linkCap = *(DWORD*)(configData + capPtr + 0x0C);

        // Extract Max Link Speed from bits 3:0
        BYTE maxSpeed = linkCap & 0x0F;
        const char* speedStr = maxSpeed < 1 || maxSpeed >= pcieGenSpeedsCount ? "Unknown" : pcieGenSpeedStr[maxSpeed];
        snprintf(strValue, maxLen, "%s", speedStr);
        return true;
      }
      break;
    }

    if (capPtr + 1 < configSize) {
      capPtr = configData[capPtr + 1];
    } else {
      break;
    }
  }

  return false;
}

ncclResult_t ncclOsTopoGetStrFromSys(const char* path, const char* fileName, char* strValue, int maxLen) {
  // Initialize to empty
  strValue[0] = '\0';

  // Open device info set to get device properties
  HDEVINFO deviceInfoSet = SetupDiGetClassDevs(NULL, "PCI", NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
  if (deviceInfoSet == INVALID_HANDLE_VALUE) {
    WARN("SetupDiGetClassDevs failed: %lu", GetLastError());
    return ncclSystemError;
  }

  SP_DEVINFO_DATA deviceInfoData;
  deviceInfoData.cbSize = sizeof(SP_DEVINFO_DATA);

  // Find the device with matching instance ID
  bool deviceFound = false;
  for (DWORD i = 0; SetupDiEnumDeviceInfo(deviceInfoSet, i, &deviceInfoData); i++) {
    char deviceInstanceId[MAX_PATH];
    if (CM_Get_Device_ID(deviceInfoData.DevInst, deviceInstanceId, MAX_PATH, 0) == CR_SUCCESS) {
      if (strcmp(deviceInstanceId, path) == 0) {
        deviceFound = true;
        break;
      }
    }
  }

  if (!deviceFound) {
    INFO(NCCL_GRAPH, "ncclOsTopoGetStrFromSys: device %s not found in SetupAPI enumeration", path);
    SetupDiDestroyDeviceInfoList(deviceInfoSet);
    return ncclSystemError;
  }

  // Handle different property queries
  if (strcmp(fileName, "class") == 0) {
    // Get PCI class code from Configuration Space (offset 0x08-0x0B)
    // Read directly from device registry
    ULONG bufferSize = sizeof(ULONG);

    // Read PCI configuration space to get class code
    BYTE configData[256];
    DWORD configSize = 0;
    if (readPciConfigSpace(deviceInfoData.DevInst, configData, sizeof(configData), &configSize)) {
      // PCI class code is at offset 0x09-0x0B in config space
      if (configSize >= 12) {
        // Extract class code: Base (0x0B), Sub (0x0A), Prog IF (0x09)
        DWORD baseClass = configData[11];
        DWORD subClass = configData[10];
        DWORD progIF = configData[9];
        snprintf(strValue, maxLen, "0x%02x%02x%02x", (unsigned int)baseClass, (unsigned int)subClass,
                 (unsigned int)progIF);
      }
    }
  } else if (strcmp(fileName, "vendor") == 0) {
    char hwId[MAX_PATH];
    if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, NULL, (PBYTE)hwId, MAX_PATH,
                                          NULL)) {
      extractHexFromHwId(hwId, "VEN_", strValue, maxLen);
    }
  } else if (strcmp(fileName, "device") == 0) {
    char hwId[MAX_PATH];
    if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, NULL, (PBYTE)hwId, MAX_PATH,
                                          NULL)) {
      extractHexFromHwId(hwId, "DEV_", strValue, maxLen);
    }
  } else if (strcmp(fileName, "subsystem_vendor") == 0) {
    char hwId[MAX_PATH];
    if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, NULL, (PBYTE)hwId, MAX_PATH,
                                          NULL)) {
      // SUBSYS format: SUBSYS_12345678 where first 4 hex = device, next 4 = vendor
      const char* subsys = strstr(hwId, "SUBSYS_");
      if (subsys != NULL && strlen(subsys) >= 15) {
        char subsysVendor[16];
        // Extract last 4 hex digits (vendor)
        snprintf(subsysVendor, sizeof(subsysVendor), "0x%.4s", subsys + 11);
        snprintf(strValue, maxLen, "%s", subsysVendor);
      }
    }
  } else if (strcmp(fileName, "subsystem_device") == 0) {
    char hwId[MAX_PATH];
    if (SetupDiGetDeviceRegistryPropertyA(deviceInfoSet, &deviceInfoData, SPDRP_HARDWAREID, NULL, (PBYTE)hwId, MAX_PATH,
                                          NULL)) {
      // SUBSYS format: SUBSYS_12345678 where first 4 hex = device, next 4 = vendor
      const char* subsys = strstr(hwId, "SUBSYS_");
      if (subsys != NULL && strlen(subsys) >= 15) {
        char subsysDevice[16];
        // Extract first 4 hex digits (device)
        snprintf(subsysDevice, sizeof(subsysDevice), "0x%.4s", subsys + 7);
        snprintf(strValue, maxLen, "%s", subsysDevice);
      }
    }
  } else if (strcmp(fileName, "max_link_speed") == 0 || strcmp(fileName, "../max_link_speed") == 0) {
    // Determine target device (self or parent)
    DEVINST targetDevInst = deviceInfoData.DevInst;

    // Check if we need to query parent device (upstream port)
    if (strncmp(fileName, "../", 3) == 0) {
      DEVINST parentDevInst;
      if (CM_Get_Parent(&parentDevInst, deviceInfoData.DevInst, 0) == CR_SUCCESS) {
        targetDevInst = parentDevInst;
      } else {
        INFO(NCCL_GRAPH, "ncclOsTopoGetStrFromSys: could not get parent device for %s", path);
        SetupDiDestroyDeviceInfoList(deviceInfoSet);
        return ncclSystemError;
      }
    }

    // Read from PCIe capability structure
    getPcieMaxLinkSpeed(targetDevInst, strValue, maxLen);
  } else if (strcmp(fileName, "max_link_width") == 0 || strcmp(fileName, "../max_link_width") == 0) {
    // Determine target device (self or parent)
    DEVINST targetDevInst = deviceInfoData.DevInst;

    // Check if we need to query parent device (upstream port)
    if (strncmp(fileName, "../", 3) == 0) {
      DEVINST parentDevInst;
      if (CM_Get_Parent(&parentDevInst, deviceInfoData.DevInst, 0) == CR_SUCCESS) {
        targetDevInst = parentDevInst;
      } else {
        INFO(NCCL_GRAPH, "ncclOsTopoGetStrFromSys: could not get parent device for %s", path);
        SetupDiDestroyDeviceInfoList(deviceInfoSet);
        return ncclSystemError;
      }
    }

    // Read from PCIe capability structure
    getPcieMaxLinkWidth(targetDevInst, strValue, maxLen);
  } else if (strcmp(fileName, "numa_node") == 0) {
    ULONG nodeNumber = 0;
    DEVPROPTYPE propertyType;
    DWORD propertySize = sizeof(nodeNumber);
    if (SetupDiGetDevicePropertyW(deviceInfoSet, &deviceInfoData, &DEVPKEY_Device_Numa_Node, &propertyType,
                                  (PBYTE)&nodeNumber, propertySize, &propertySize, 0)) {
      snprintf(strValue, maxLen, "%lu", nodeNumber);
    } else {
      // Default: Assume node 0 (safest default for single-socket or non-NUMA systems)
      snprintf(strValue, maxLen, "0");
    }
  }

  SetupDiDestroyDeviceInfoList(deviceInfoSet);
  return ncclSuccess;
}

/* NUMA and PCI device class functions */
ncclResult_t ncclOsGetNumaNodeAffinity(unsigned int numaId, char* affinityStr, size_t maxLen) {
  GROUP_AFFINITY groupAffinity = {};
  if (GetNumaNodeProcessorMaskEx((USHORT)numaId, &groupAffinity)) {
    KAFFINITY mask = groupAffinity.Mask;
    uint32_t hi = (uint32_t)((uint64_t)mask >> 32);
    uint32_t lo = (uint32_t)((uint64_t)mask & 0xFFFFFFFF);
    if (hi) snprintf(affinityStr, maxLen, "%08x,%08x", hi, lo);
    else snprintf(affinityStr, maxLen, "%08x", lo);
  } else {
    // Fallback: all CPUs set (64-bit mask as two 32-bit hex chunks)
    snprintf(affinityStr, maxLen, "ffffffff,ffffffff");
  }
  return ncclSuccess;
}

// Get device class by busId (works for any PCI device - GPUs, switches, bridges, etc.)
ncclResult_t ncclOsGetPciDeviceClassByBusId(const char* busId, char* deviceClass, size_t maxLen) {
  // Parse the bus ID to extract bus, device, and function numbers
  DWORD bus, dev, func;
  if (!parsePciBusId(busId, &bus, &dev, &func)) {
    WARN("ncclOsGetPciDeviceClassByBusId: Failed to parse PCI bus ID: %s", busId);
    deviceClass[0] = '\0';
    return ncclSystemError;
  }

  // Get device info using SetupAPI
  DeviceInfo devInfo;
  ncclResult_t ret = getDeviceInfo(bus, dev, func, &devInfo);
  if (ret != ncclSuccess) {
    // Device not found, return empty class
    deviceClass[0] = '\0';
    return ncclSystemError;
  }

  // Try to get class code from Configuration Manager registry.
  // The PCI class code is stored in the device's registry under the Enum key.
  char registryPath[512];
  snprintf(registryPath, sizeof(registryPath), "SYSTEM\\CurrentControlSet\\Enum\\%s", devInfo.deviceInstanceId);

  HKEY hKey;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, registryPath, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
    // Try to read ClassGUID first to determine device class
    char classGuid[MAX_PATH];
    DWORD dataSize = sizeof(classGuid);
    DWORD dataType;

    if (RegQueryValueExA(hKey, "ClassGUID", NULL, &dataType, (LPBYTE)classGuid, &dataSize) == ERROR_SUCCESS) {
      INFO(NCCL_INIT, "ncclOsGetPciDeviceClassByBusId: ClassGUID: %s", classGuid);
    }

    // Try to read CompatibleIDs which might have CC_ field
    char compatIds[1024];
    dataSize = sizeof(compatIds);
    if (RegQueryValueExA(hKey, "CompatibleIDs", NULL, &dataType, (LPBYTE)compatIds, &dataSize) == ERROR_SUCCESS) {
      // CompatibleIDs is a multi-string, search for CC_ in any of them
      for (DWORD i = 0; i < dataSize - 3; i++) {
        if (compatIds[i] == 'C' && compatIds[i + 1] == 'C' && compatIds[i + 2] == '_') {
          if (i + 9 < dataSize && strlen(&compatIds[i]) >= 9) {
            char classStr[16];
            snprintf(classStr, sizeof(classStr), "0x%.2s", &compatIds[i + 3]);
            snprintf(deviceClass, maxLen, "%s", classStr);
            INFO(NCCL_INIT, "ncclOsGetPciDeviceClassByBusId: Extracted class %s for %s", deviceClass, busId);
            RegCloseKey(hKey);
            return ncclSuccess;
          }
        }
      }
    }

    RegCloseKey(hKey);
  }

  // Fallback: use the hardware ID already cached by getDeviceInfo
  const char* classCode = strstr(devInfo.hwId, "CC_");
  if (classCode != NULL && strlen(classCode) >= 9) {
    char classStr[16];
    snprintf(classStr, sizeof(classStr), "0x%.2s", classCode + 3);
    snprintf(deviceClass, maxLen, "%s", classStr);
    INFO(NCCL_INIT, "ncclOsGetPciDeviceClassByBusId: Extracted class %s for %s", deviceClass, busId);
    return ncclSuccess;
  }

  // If still no class, try to infer from vendor/device ID
  // NVIDIA GPUs (VEN_10DE) -> class 0x03 (display)
  // Mellanox switches (VEN_15B3) -> class 0x06 (bridge) for DEV_1979 and similar
  if (strstr(devInfo.deviceInstanceId, "VEN_10DE") != NULL) {
    snprintf(deviceClass, maxLen, "0x03");
  } else if (strstr(devInfo.deviceInstanceId, "VEN_15B3") != NULL &&
             strstr(devInfo.deviceInstanceId, "DEV_1979") != NULL) {
    snprintf(deviceClass, maxLen, "0x06");
  } else {
    deviceClass[0] = '\0';
  }
  return ncclSuccess;
}

ncclResult_t ncclOsGetPciDeviceClass(nvmlDevice_t device, char* deviceClass, size_t maxLen) {
  // Get PCI bus ID from NVML device
  nvmlPciInfo_t pciInfo;
  ncclResult_t ret = ncclNvmlDeviceGetPciInfo(device, &pciInfo);
  if (ret != ncclSuccess) {
    WARN("Failed to get PCI info from NVML device");
    return ret;
  }

  // Use the helper function with the busId
  ncclResult_t classRet = ncclOsGetPciDeviceClassByBusId(pciInfo.busId, deviceClass, maxLen);
  if (classRet != ncclSuccess) return classRet;
  return ncclSuccess;
}

ncclResult_t ncclOsGetPciDeviceParent(nvmlDevice_t device, char** parentBusId) {
  *parentBusId = NULL;

  // Get PCI bus ID from NVML device
  nvmlPciInfo_t pciInfo;
  ncclResult_t ret = ncclNvmlDeviceGetPciInfo(device, &pciInfo);
  if (ret != ncclSuccess) {
    INFO(NCCL_INIT, "ncclOsGetPciDeviceParent: Failed to get PCI info from NVML device");
    return ret;
  }

  INFO(NCCL_INIT, "ncclOsGetPciDeviceParent: Getting parent for device %s", pciInfo.busId);

  // Parse the bus ID to extract bus, device, and function numbers
  DWORD bus, dev, func;
  if (!parsePciBusId(pciInfo.busId, &bus, &dev, &func)) {
    WARN("ncclOsGetPciDeviceParent: Failed to parse PCI bus ID: %s", pciInfo.busId);
    return ncclSystemError;
  }

  // Get device info using SetupAPI
  DeviceInfo devInfo;
  ret = getDeviceInfo(bus, dev, func, &devInfo);
  if (ret != ncclSuccess) {
    INFO(NCCL_INIT, "ncclOsGetPciDeviceParent: Could not find device %s", pciInfo.busId);
    return ncclSystemError;
  }

  // Get parent device instance
  DEVINST parentDevInst;
  if (CM_Get_Parent(&parentDevInst, devInfo.devInst, 0) != CR_SUCCESS) {
    INFO(NCCL_INIT, "ncclOsGetPciDeviceParent: No parent found for device %s", pciInfo.busId);
    return ncclSystemError;
  }

  // Convert parent device instance to PCI bus ID
  char parentBusIdBuf[BUSID_SIZE];
  if (!getPciBusIdFromDevInst(parentDevInst, parentBusIdBuf, sizeof(parentBusIdBuf))) {
    INFO(NCCL_INIT, "ncclOsGetPciDeviceParent: Could not get bus ID for parent device");
    return ncclSystemError;
  }

  // Allocate and copy the parent bus ID
  *parentBusId = _strdup(parentBusIdBuf);
  if (*parentBusId == NULL) {
    WARN("ncclOsGetPciDeviceParent: Failed to allocate memory for parent bus ID");
    return ncclSystemError;
  }

  INFO(NCCL_INIT, "ncclOsGetPciDeviceParent: Device %s has parent %s", pciInfo.busId, *parentBusId);
  return ncclSuccess;
}
