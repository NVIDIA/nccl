/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "os.h"

#include "checks.h"
#include "utils.h"

#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cstring>
#include <cstdbool>
#include "socket.h"
#include "utils.h"
#include "checks.h"
#include "param.h"
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <atomic>

static thread_local char ncclDlErrorBuf[256] = {0};

static void saveDlError() {
  const char* err = dlerror();
  if (err) {
    snprintf(ncclDlErrorBuf, sizeof(ncclDlErrorBuf), "%s", err);
  } else {
    ncclDlErrorBuf[0] = '\0';
  }
}

ncclOsLibraryHandle ncclOsDlopen(const char* filename) {
  ncclOsLibraryHandle handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    saveDlError();
    INFO(NCCL_INIT, "ncclOsDlopen(%s) failed: %s", filename, ncclDlErrorBuf);
  }
  return handle;
}

void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  void* ptr = dlsym(handle, symbol);
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
  return (ncclOsLibraryHandle)dlopen(path, (mode == NCCL_OS_DL_NOW) ? RTLD_NOW : RTLD_LAZY);
}

void ncclOsDlclose(ncclOsLibraryHandle handle) {
  if (handle) dlclose(handle);
}

// Process Management
uint64_t ncclOsGetPid() {
  return (uint64_t)getpid();
}

std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf) {
  return localtime_r(timer, buf);
}

uint64_t ncclOsGetTid() {
  return (uint64_t)syscall(SYS_gettid);
}

size_t ncclOsGetPageSize() {
  return (size_t)sysconf(_SC_PAGESIZE);
}

void* ncclOsAlignedAlloc(size_t alignment, size_t size) {
    return aligned_alloc(alignment, size);
}

void ncclOsAlignedFree(void* ptr) {
    free(ptr);
}

void ncclOsSetEnv(const char* name, const char* value) {
  setenv(name, value, 0);
}

char* ncclOsRealpath(const char* path, char* resolved_path) {
  return realpath(path, resolved_path);
}

// The default Linux stack size (8MB) is safe.
#define SAFE_STACK_SIZE (8192*1024)

static ncclResult_t setCpuStackSize() {
  // Query the stack size used for newly launched threads.
  pthread_attr_t attr;
  size_t stackSize;
  PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");
  PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");

  if (stackSize < SAFE_STACK_SIZE) {
    // GNU libc normally uses RLIMIT_STACK as the default pthread stack size, unless it's set to "unlimited" --
    // in that case a fallback value of 2MB (!) is used.

    // Query the actual resource limit so that we can distinguish between the settings of 2MB and unlimited.
    struct rlimit stackLimit;
    char buf[30];
    SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");
    if (stackLimit.rlim_cur == RLIM_INFINITY)
      strcpy(buf, "unlimited");
    else
      snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur / 1024);
    INFO(NCCL_INIT | NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads",
         buf, SAFE_STACK_SIZE / 1024);

    // Change the default pthread stack size (via a nonportable API as the feature is not available in std::thread)
    PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");
    PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");
  }

  PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");
  return ncclSuccess;
}

extern int ncclParamSetCpuStackSize();

ncclResult_t ncclOsInitialize() {
  if (ncclParamSetCpuStackSize() != 0) {
    NCCLCHECK(setCpuStackSize());
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSetFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor sockDescriptor) {
  return sockDescriptor >= 0;
}

bool ncclOsSocketIsValid(struct ncclSocket* sock) {
  return ncclOsSocketDescriptorIsValid(sock->socketDescriptor);
}

extern int ncclParamPollTimeOut();
extern int ncclParamRetryTimeOut();

void ncclOsPollSocket(int socketDescriptor, int op) {
  struct pollfd pfd;
  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = socketDescriptor;
  pfd.events = (op == NCCL_SOCKET_RECV) ? POLLIN : POLLOUT;
  (void) poll(&pfd, 1, ncclParamPollTimeOut());
}

extern long int ncclParamRetryCnt();

ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock) {
  socklen_t socklen = sizeof(union ncclSocketAddress);
  sock->socketDescriptor = accept(sock->acceptSocketDescriptor, (struct sockaddr*)&sock->addr, &socklen);
  if (ncclOsSocketIsValid(sock)) {
    sock->state = ncclSocketStateAccepted;
  } else if (errno == ENETDOWN || errno == EPROTO || errno == ENOPROTOOPT || errno == EHOSTDOWN ||
             errno == ENONET || errno == EHOSTUNREACH || errno == EOPNOTSUPP || errno == ENETUNREACH ||
             errno == EINTR) {
    /* per accept's man page, for linux sockets, the following errors might be already pending errors
     * and should be considered as EAGAIN. To avoid infinite loop in case of errors, we use the retry count*/
    if (++sock->errorRetries == ncclParamRetryCnt()) {
      WARN("ncclOsSocketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries, strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET|NCCL_INIT, "Call to accept returned %s, retrying", strerror(errno));
  } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
    WARN("ncclOsSocketTryAccept: Accept failed: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

extern int ncclParamSocketMaxRecvBuff();
extern int ncclParamSocketMaxSendBuff();

ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  const int one = 1;
  int flags;
  int rcvBuf, sndBuf;
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclOsSocketSetFlags: invalid socket");
    ret = ncclInvalidArgument;
    goto fail;
  }
  /* Set socket as non-blocking if async or if we need to be able to abort */
  if (sock->asyncFlag || sock->abortFlag) {
    SYSCHECKGOTO(flags = fcntl(sock->socketDescriptor, F_GETFL), "fcntl", ret, fail);
    SYSCHECKGOTO(fcntl(sock->socketDescriptor, F_SETFL, flags | O_NONBLOCK), "fcntl", ret, fail);
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
  (void)close(sock->socketDescriptor);
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateAccepting;
  sock->finalizeCounter = 0;
}

ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  int socketDescriptor = NCCL_INVALID_SOCKET;
  SYSCHECKGOTO(socketDescriptor = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
  // if sock->socketDescriptor is valid, reuse its file descriptor number
  if (ncclOsSocketIsValid(sock)) {
    SYSCHECKGOTO(dup2(socketDescriptor, sock->socketDescriptor), "dup2", ret, cleanup);
    SYSCHECKGOTO(close(socketDescriptor), "close", ret, cleanup);
  } else {
    sock->socketDescriptor = socketDescriptor;
  }
  NCCLCHECKGOTO(ncclOsSocketSetFlags(sock), ret, exit);
exit:
  return ret;
cleanup:
  // cleanup socketDescriptor, leave sock->socketDescriptor untouched
  if (socketDescriptor != NCCL_INVALID_SOCKET) {
    (void)close(socketDescriptor);
  }
  goto exit;
}

static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode, const char funcName[]) {
  char line[SOCKET_NAME_MAXLEN+1];
  if (errCode == 0) {
    sock->state = ncclSocketStateConnected;
  } else if (errCode == EINPROGRESS) {
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == EINTR || errCode == EWOULDBLOCK || errCode == EAGAIN || errCode == ETIMEDOUT ||
             errCode == EHOSTUNREACH || errCode == ECONNREFUSED) {
    if (sock->customRetry == 0) {
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts",
             funcName, ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET|NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec",
           funcName, ncclSocketToString(&sock->addr, line), strerror(errCode),
           sock->errorRetries, ncclParamRetryCnt(), sleepTime);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
    }
    NCCLCHECK(ncclOsSocketResetFd(sock)); /* in case of failure in connect, socket state is unspecified */
    sock->state = ncclSocketStateConnecting;
  } else {
    sock->state = ncclSocketStateError;
    WARN("%s: connect to %s failed : %s", funcName, ncclSocketToString(&sock->addr, line), strerror(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->socketDescriptor, &sock->addr.sa, sock->salen);
  return socketConnectCheck(sock, (ret == -1) ? errno : 0, __func__);
}

ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);
  char line[SOCKET_NAME_MAXLEN+1];

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->socketDescriptor;
  pfd.events = POLLOUT;
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    return ncclSuccess;
  } else if (ret < 0) {
    WARN("ncclOsSocketPollConnect to %s failed with error %s", ncclSocketToString(&sock->addr, line), strerror(errno));
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
  do {
    if (op == NCCL_SOCKET_RECV) bytes = recv(sock->socketDescriptor, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    if (op == NCCL_SOCKET_SEND) bytes = send(sock->socketDescriptor, data+(*offset), size-(*offset), block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == -1) {
      if ((op == NCCL_SOCKET_SEND && errno == EPIPE) || (op == NCCL_SOCKET_RECV && errno == ECONNRESET)) {
        *closed = 1;
        return ncclSuccess;
      }
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("ncclOsSocketProgressOpt: Call to %s %s failed : %s", (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
              ncclSocketToString(&sock->addr, line), strerror(errno));
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

  *found = 0;
  struct ifaddrs *interfaces, *interface;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  for (interface = interfaces; interface && *found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    /* Only consider running interfaces, i.e. UP and physically attached. */
    if (!(interface->ifa_flags & IFF_RUNNING)) continue;

    TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, ncclSocketToString((union ncclSocketAddress *) interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < *found; i++) {
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = true; break; }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + (*found)*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memset(addrs + *found, '\0', sizeof(*addrs));
      memcpy(addrs + *found, interface->ifa_addr, salen);
      (*found)++;
    }
  }
  freeifaddrs(interfaces);
  return ncclSuccess;
}

static bool matchSubnet(struct ifaddrs local_if, union ncclSocketAddress* remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr == remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  //IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  //Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
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
  *found = 0;
  struct ifaddrs *interfaces, *interface;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  for (interface = interfaces; interface && !*found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memcpy(localAddr, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifName, interface->ifa_name, ifNameMaxSize);

    TRACE(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, ncclSocketToString(localAddr, line), ncclSocketToString(remoteAddr, line_a));
    *found = 1;
  }

  freeifaddrs(interfaces);
  return ncclSuccess;
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
      (void)shutdown(sock->socketDescriptor, SHUT_RDWR);
      (void)close(sock->socketDescriptor);
    }
    sock->state = ncclSocketStateClosed;
    sock->socketDescriptor = NCCL_INVALID_SOCKET;
  }
  return ncclSuccess;
}

void ncclOsSetMutexCondShared(std::mutex &mutex, std::condition_variable &cond) {
  pthread_mutexattr_t mutexAttr;
  pthread_mutexattr_init(&mutexAttr);
  pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_t* mutexHandle = mutex.native_handle();
  pthread_mutex_init(mutexHandle, &mutexAttr);
  pthread_mutexattr_destroy(&mutexAttr);

  pthread_condattr_t condAttr;
  pthread_condattr_init(&condAttr);
  pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
  pthread_cond_t* condHandle = cond.native_handle();
  pthread_cond_init(condHandle, &condAttr);
  pthread_condattr_destroy(&condAttr);
}

void ncclOsCpuZero(ncclAffinity& affinity) {
  CPU_ZERO(&affinity);
}

int ncclOsCpuCount(const ncclAffinity& affinity) {
  return CPU_COUNT(&affinity);
}

void ncclOsCpuSet(ncclAffinity& affinity, int cpu) {
  CPU_SET(cpu, &affinity);
}

bool ncclOsCpuIsSet(const ncclAffinity& affinity, int cpu) {
  return CPU_ISSET(cpu, &affinity);
}

ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b) {
  ncclAffinity result;
  CPU_AND(&result, &a, &b);
  return result;
}

ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) {
  int result = sched_getaffinity(0, sizeof(ncclAffinity), affinity);
  if (result == -1) {
    WARN("sched_getaffinity failed with error: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) {
  int result = sched_setaffinity(0, sizeof(ncclAffinity), &affinity);
  if (result == -1) {
    WARN("sched_setaffinity failed with error: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

int ncclOsGetCpu() {
  return sched_getcpu();
}

ncclResult_t ncclOsNvmlOpen(ncclOsLibraryHandle* handle) {
  *handle = nullptr;

  *handle = ncclOsDlopen("libnvidia-ml.so.1");
  if (*handle == nullptr) {
    WARN("Failed to open libnvidia-ml.so.1: %s", ncclOsDlerror());
    return ncclSystemError;
  }

  INFO(NCCL_INIT, "Loaded NVML from libnvidia-ml.so.1");
  return ncclSuccess;
}


// Shared memory implementation for Linux
#include "comm.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <stdlib.h>

void ncclOsShmHandleInit(ncclShmDescriptor shmDesc, char* shmPath, size_t shmSize, size_t realShmSize,
                         char* hptr, void* dptr, bool create,
                         struct ncclShmHandleInternal* handle) {
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

ncclResult_t ncclOsShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize,
                           void** shmPtr, void** devShmPtr, int refcount,
                           struct ncclShmHandleInternal** handle) {
  int fd = -1;
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct ncclShmHandleInternal* tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int);
  const size_t realShmSize = shmSize + refSize;

  *handle = NULL;
  *shmPtr = NULL;
  EQCHECKGOTO(tmphandle = (struct ncclShmHandleInternal*)calloc(1, sizeof(struct ncclShmHandleInternal)), NULL, ret, fail);

  if (create) {
    if (shmPath[0] == '\0') {
      snprintf(shmPath, shmPathSize, "/dev/shm/nccl-XXXXXX");
    retry_mkstemp:
      fd = mkstemp(shmPath);
      if (fd < 0) {
        if (errno == EINTR) {
          INFO(NCCL_ALL, "mkstemp: Failed to create %s, error: %s (%d) - retrying", shmPath, strerror(errno), errno);
          goto retry_mkstemp;
        }
        WARN("Error: failed to create shared memory file %s, error %s (%d)", shmPath, strerror(errno), errno);
        ret = ncclSystemError;
        goto fail;
      }
    } else {
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
    }

  retry_fallocate:
    if (fallocate(fd, 0, 0, realShmSize) != 0) {
      if (errno == EINTR) {
        INFO(NCCL_ALL, "fallocate: Failed to extend %s to %ld bytes, error: %s (%d) - retrying", shmPath, realShmSize, strerror(errno), errno);
        goto retry_fallocate;
      }
      WARN("Error: failed to extend %s to %ld bytes, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
      ret = ncclSystemError;
      goto fail;
    }
    INFO(NCCL_ALLOC, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
  } else {
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
  }

  hptr = (char*)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hptr == MAP_FAILED) {
    WARN("Error: Could not map %s size %zu, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
    ret = ncclSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create) {
    *(int*)(hptr + shmSize) = refcount;
  } else {
    int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
    if (remref == 0) {
      if (unlink(shmPath) != 0) {
        INFO(NCCL_ALLOC, "unlink shared memory %s failed, error: %s (%d)", shmPath, strerror(errno), errno);
      }
    }
  }

  if (devShmPtr) {
    INFO(NCCL_ALLOC, "SHM legacy: sharing buffer with GPU via cudaHostRegister + cudaHostGetDevicePointer (host %p size %ld)", (void*)hptr, (long)realShmSize);
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
    CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret, fail);
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
  }

  ncclOsShmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  *handle = tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %ld), error: %s (%d)", create ? "creating" : "attaching to",
       shmPath, shmSize, strerror(errno), errno);
  if (tmphandle) {
    ncclOsShmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
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
    if (handle->shmDesc >= 0) {
      close(handle->shmDesc);
      if (handle->shmPath != NULL && handle->refcount != NULL && *handle->refcount > 0) {
        if (unlink(handle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s (%d)", handle->shmPath, strerror(errno), errno);
          ret = ncclSystemError;
        }
      }
      free(handle->shmPath);
    }

    if (handle->shmPtr) {
      if (handle->devShmPtr) CUDACHECK(cudaHostUnregister(handle->shmPtr));
      if (munmap(handle->shmPtr, handle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s (%d)", handle->shmPtr, handle->realShmSize, strerror(errno), errno);
        ret = ncclSystemError;
      }
    }
    free(handle);
  }
  return ret;
}

ncclResult_t ncclOsShmUnlink(struct ncclShmHandleInternal* handle) {
  ncclResult_t ret = ncclSuccess;
  if (handle) {
    if (handle->shmPath != NULL && handle->refcount != NULL && *handle->refcount > 0) {
      if (unlink(handle->shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s (%d)", handle->shmPath, strerror(errno), errno);
        ret = ncclSystemError;
      }
      free(handle->shmPath);
      handle->shmPath = NULL;
    }
  }
  return ret;
}
