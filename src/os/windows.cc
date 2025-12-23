/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os.h"
#include <cstring>
#include <cstdbool>
#include "socket.h"
#include "utils.h"
#include "os.h"
#include "checks.h"
#include "param.h"
#include <atomic>
#include <nmmintrin.h>

// Windows-specific definitions for constants not available in Windows
#ifndef IFNAMSIZ
#define IFNAMSIZ 16
#endif

uint64_t ncclOsGetpid() {
  return (uint64_t)GetCurrentProcessId();
}

ncclResult_t ncclOsSetCpuStackSize() {
  // Not implemented on Windows
  return ncclSuccess;
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

void ncclOsSleep(unsigned int time_msec) {
  Sleep((DWORD)time_msec);
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
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                 NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                 errorMsg, sizeof(errorMsg), NULL);
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
      if (++sock->errorRetries == ncclParamRetryCnt()) {
        WARN("ncclOsSocketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries, getWSAErrorMessage(wsaError));
        return ncclSystemError;
      }
      INFO(NCCL_NET|NCCL_INIT, "Call to accept returned %s, retrying", getWSAErrorMessage(wsaError));
    } else {
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
  SYSCHECKGOTO(setsockopt(sock->socketDescriptor, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt TCP NODELAY", ret, fail);
  // setsockopt should not fail even if the sizes are too large, do not change the default if unset by the user (=-1)
  rcvBuf = ncclParamSocketMaxRecvBuff();
  sndBuf = ncclParamSocketMaxSendBuff();
  if (sndBuf > 0) SYSCHECKGOTO(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_SNDBUF, (char*)&sndBuf, sizeof(int)), "setsockopt SO_SNDBUF", ret, fail);
  if (rcvBuf > 0) SYSCHECKGOTO(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBuf, sizeof(int)), "setsockopt SO_RCVBUF", ret, fail);
exit:
  return ret;
fail:
  goto exit;
}

void ncclOsSocketResetAccept(struct ncclSocket* sock) {
  char line[SOCKET_NAME_MAXLEN+1];
  INFO(NCCL_NET|NCCL_INIT, "socketFinalizeAccept: didn't receive a valid magic from %s",
       ncclSocketToString(&sock->addr, line));
  // Ignore spurious connection and accept again
  (void)closesocket(sock->socketDescriptor);
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateAccepting;
  sock->finalizeCounter = 0;
}

ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  SOCKET newSocket = INVALID_SOCKET;
  SYSCHECKGOTO(newSocket = socket(sock->sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
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
  char line[SOCKET_NAME_MAXLEN+1];
  if (errCode == 0) {
    sock->state = ncclSocketStateConnected;
  } else if (errCode == WSAEINPROGRESS) {
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == WSAEINTR || errCode == WSAEWOULDBLOCK || errCode == WSAEAGAIN ||
             errCode == WSAETIMEDOUT || errCode == WSAEHOSTUNREACH || errCode == WSAECONNREFUSED) {
    if (sock->customRetry == 0) {
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts",
             funcName, ncclSocketToString(&sock->addr, line), getWSAErrorMessage(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET|NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec",
           funcName, ncclSocketToString(&sock->addr, line), getWSAErrorMessage(errCode),
           sock->errorRetries, ncclParamRetryCnt(), sleepTime);
      ncclOsSleep(sleepTime);
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
  socklen_t rlen = sizeof(int);
  char line[SOCKET_NAME_MAXLEN+1];

  memset(&pfd, 0, sizeof(WSAPOLLFD));
  pfd.fd = sock->socketDescriptor;
  pfd.events = POLLOUT;
  ret = WSAPoll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && WSAGetLastError() == WSAEINTR)) {
    return ncclSuccess;
  } else if (ret < 0) {
    int wsaError = WSAGetLastError();
    WARN("ncclOsSocketPollConnect to %s failed with error %s", ncclSocketToString(&sock->addr, line), getWSAErrorMessage(wsaError));
    return ncclSystemError;
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->socketDescriptor, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;
  char line[SOCKET_NAME_MAXLEN+1];
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
    if (op == NCCL_SOCKET_RECV) bytes = recv(sock->socketDescriptor, data+(*offset), size-(*offset), 0);
    if (op == NCCL_SOCKET_SEND) bytes = send(sock->socketDescriptor, data+(*offset), size-(*offset), 0);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == SOCKET_ERROR) {
      if (WSAGetLastError() == WSAECONNRESET) {
        *closed = 1;
        return ncclSuccess;
      }
      if (WSAGetLastError() != WSAEINPROGRESS) {
        WARN("ncclOsSocketProgressOpt: Call to %s %s failed : %u", (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
              ncclSocketToString(&sock->addr, line), WSAGetLastError());
        return ncclRemoteError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (sock->abortFlag && std::atomic_load_explicit((std::atomic<uint32_t>*)sock->abortFlag, std::memory_order_acquire)) {
      INFO(NCCL_NET, "ncclOsSocketProgressOpt: abort called");
      return ncclInternalError;
    }
  } while (sock->asyncFlag == 0 && bytes > 0 && (*offset) < size);
  return ncclSuccess;
}

ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress *addrs, int sock_family,
  int maxIfNameSize, int maxIfs, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
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
    for (IP_ADAPTER_UNICAST_ADDRESS* unicast = adapter->FirstUnicastAddress;
         unicast && *found < maxIfs; unicast = unicast->Next) {

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

      TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", adapterName, ncclSocketToString((union ncclSocketAddress *) unicast->Address.lpSockaddr, line));

      // Check against user specified interfaces
      if (!(matchIfList(adapterName, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
        continue;
      }

      // Check that this interface has not already been saved
      bool duplicate = false;
      for (int i = 0; i < *found; i++) {
        if (strcmp(adapterName, names+i*maxIfNameSize) == 0) {
          duplicate = true;
          break;
        }
      }

      if (!duplicate) {
        // Store the interface name
        strncpy(names + (*found)*maxIfNameSize, adapterName, maxIfNameSize);
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
  char line[SOCKET_NAME_MAXLEN+1];
  char line_a[SOCKET_NAME_MAXLEN+1];
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
    for (IP_ADAPTER_UNICAST_ADDRESS* unicast = adapter->FirstUnicastAddress;
         unicast && !*found; unicast = unicast->Next) {

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

      TRACE(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s",
            ifName, ncclSocketToString(localAddr, line), ncclSocketToString(remoteAddr, line_a));
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

void ncclOsSetMutexCondShared(std::mutex &mutex, std::condition_variable &cond) {
  // Not implemented on Windows
}

void ncclOsCpuZero(ncclAffinity& affinity) {
  affinity = 0;
}

int ncclOsCpuCount(const ncclAffinity affinity) {
  return _mm_popcnt_u64(affinity);
}

void ncclOsCpuSet(ncclAffinity& affinity, int cpu) {
  affinity |= (1ULL << cpu);
}

bool ncclOsCpuIsSet(const ncclAffinity affinity, int cpu) {
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

ncclResult_t ncclOsSetAffinity(const ncclAffinity affinity) {
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
