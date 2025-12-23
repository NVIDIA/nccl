/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os.h"

#include "checks.h"
#include "utils.h"

#include <cstdint>
#include <unistd.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstring>
#include <cstdbool>
#include "socket.h"
#include "utils.h"
#include "os.h"
#include "checks.h"
#include "param.h"
#include <pthread.h>
#include <sys/resource.h>
#include <atomic>

// Process Management
uint64_t ncclOsGetpid() {
  return (uint64_t)getpid();
}

// The default Linux stack size (8MB) is safe.
#define SAFE_STACK_SIZE (8192*1024)

ncclResult_t ncclOsSetCpuStackSize() {
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

void ncclOsSetEnv(const char* name, const char* value) {
  setenv(name, value, 0);
}

void ncclOsSleep(unsigned int time_msec) {
  const long c_1e6 = 1e6;
  struct timespec tv = (struct timespec){
    .tv_sec = time_msec / 1000,
    .tv_nsec = (time_msec % 1000) * c_1e6,
  };
  nanosleep(&tv, NULL);
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
      ncclOsSleep(sleepTime);
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

int ncclOsCpuCount(const ncclAffinity affinity) {
  return CPU_COUNT(&affinity);
}

void ncclOsCpuSet(ncclAffinity& affinity, int cpu) {
  CPU_SET(cpu, &affinity);
}

bool ncclOsCpuIsSet(const ncclAffinity affinity, int cpu) {
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

ncclResult_t ncclOsSetAffinity(const ncclAffinity affinity) {
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
