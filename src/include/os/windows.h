#ifndef NCCL_WINDOWS_H_
#define NCCL_WINDOWS_H_

/* Require Vista+ so GetAdaptersAddresses and IP_ADAPTER_* types are declared in iphlpapi.h */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

/* WIN32_LEAN_AND_MEAN prevents windows.h from pulling in winsock.h, so
 * _WINSOCKAPI_ is only defined once (by winsock2.h) and C4005 is avoided.
 * For device (.cu) builds this is also set via target_compile_definitions. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Include standard C headers first
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* POSIX strcasecmp/strncasecmp are not available on MSVC; use ISO C _stricmp/_strnicmp */
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#ifndef strncasecmp
#define strncasecmp _strnicmp
#endif
#include <windows.h>
/* Winsock2 must be included before iphlpapi.h so IP_ADAPTER_ADDRESSES and GetAdaptersAddresses are declared */
#pragma warning(push)
#pragma warning(disable: 4005) /* _WINSOCKAPI_ redefinition when we already defined it to avoid winsock.h */
#include <winsock2.h>
#pragma warning(pop)
#include <ws2ipdef.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#include <processthreadsapi.h>
#include <io.h>
#include <process.h>
#include <time.h>
#include <profileapi.h>
#include <libloaderapi.h>
#include <string.h>

/* POSIX shutdown() how argument; Winsock uses SD_SEND/SD_RECEIVE/SD_BOTH */
#ifndef SHUT_WR
#define SHUT_WR SD_SEND
#endif

#define NCCL_INVALID_SOCKET INVALID_SOCKET
typedef SOCKET ncclSocketDescriptor;

typedef DWORD_PTR ncclAffinity;

typedef unsigned long ncclPid_t;

/* gettimeofday() replacement for Windows (struct timeval is in winsock2.h via os.h) */
int gettimeofday(struct timeval* tv, void* tz);

/* WSAPoll requires POLLRDNORM instead of POLLIN for listen/accept sockets */
#define NCCL_POLLIN POLLRDNORM
#define NCCL_POLLERR (POLLHUP | POLLERR | POLLNVAL)

#endif
