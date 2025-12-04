/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_DEFS_H_
#define NCCL_WIN32_DEFS_H_

#ifdef _WIN32

/*
 * Windows POSIX compatibility definitions
 * This header provides minimal POSIX compatibility for Windows.
 * More complete implementations are in other win32_*.h headers.
 */

/* Windows headers must be included before any POSIX compatibility definitions */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <io.h>
#include <process.h>
#include <direct.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdarg.h>

/* Link with Windows socket library */
#pragma comment(lib, "ws2_32.lib")

/*
 * Basic type definitions for POSIX compatibility
 * Note: ssize_t is defined in platform.h, not here, to avoid redefinition
 */
typedef int pid_t;
typedef int uid_t;
typedef int gid_t;
typedef int mode_t;
typedef int socklen_t;

/* Socket type compatibility */
#ifndef SOCKET
typedef UINT_PTR SOCKET;
#endif

/*
 * POSIX I/O function compatibility
 * Use inline functions instead of macros to avoid conflicts with C++ STL
 * (e.g., std::basic_istream::read would conflict with a read macro)
 */
#define close(fd) _close(fd)

static inline int nccl_read(int fd, void *buf, size_t len)
{
    return _read(fd, buf, (unsigned int)(len));
}

static inline int nccl_write(int fd, const void *buf, size_t len)
{
    return _write(fd, buf, (unsigned int)(len));
}

/* Socket-specific close (different from file close) */
static inline int closesocket_compat(SOCKET s)
{
    return closesocket(s);
}

/* File descriptor operations */
#define open _open
#define O_RDONLY _O_RDONLY
#define O_WRONLY _O_WRONLY
#define O_RDWR _O_RDWR
#define O_CREAT _O_CREAT
#define O_TRUNC _O_TRUNC
#define O_APPEND _O_APPEND
#define O_BINARY _O_BINARY
#define O_TEXT _O_TEXT

/* File stat compatibility */
#define stat _stat64
#define fstat _fstat64
#define lstat _stat64 /* Windows doesn't have symlinks in the same way */

/* Directory operations */
#define mkdir(path, mode) _mkdir(path)
#define rmdir _rmdir
#define getcwd _getcwd
#define chdir _chdir

/* Process operations */
#define getpid _getpid
static inline uid_t getuid(void) { return 0; }
static inline gid_t getgid(void) { return 0; }
static inline uid_t geteuid(void) { return 0; }
static inline gid_t getegid(void) { return 0; }

/* String operations */
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define strdup _strdup

/* Access function modes */
#ifndef F_OK
#define F_OK 0
#endif
#ifndef R_OK
#define R_OK 4
#endif
#ifndef W_OK
#define W_OK 2
#endif
#ifndef X_OK
#define X_OK 1
#endif
#define access _access

/* Pipe operations */
#define pipe(fds) _pipe(fds, 4096, _O_BINARY)

/* Memory mapping placeholders (actual implementations in win32_shm.h) */
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4
#define PROT_NONE 0x0

#define MAP_PRIVATE 0x02
#define MAP_SHARED 0x01
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_FAILED ((void *)-1)

/* Shared memory constants */
#define SHM_RDONLY 010000

/* Unix domain socket path max length */
#ifndef UNIX_PATH_MAX
#define UNIX_PATH_MAX 108
#endif

/* Fcntl compatibility */
#define F_GETFL 3
#define F_SETFL 4
#define O_NONBLOCK 0x0004

/* For Windows, fcntl is emulated using ioctlsocket for sockets */
static inline int fcntl(SOCKET fd, int cmd, ...)
{
    if (cmd == F_GETFL)
    {
        return 0;
    }
    else if (cmd == F_SETFL)
    {
        /* Handle non-blocking mode via ioctlsocket */
        va_list args;
        va_start(args, cmd);
        int flags = va_arg(args, int);
        va_end(args);

        if (flags & O_NONBLOCK)
        {
            u_long mode = 1; /* Non-blocking mode */
            if (ioctlsocket(fd, FIONBIO, &mode) != 0)
            {
                return -1;
            }
        }
        return 0;
    }
    return -1;
}

/* Poll compatibility (basic version, full implementation in win32_socket.h) */
#ifndef POLLIN
#define POLLIN 0x0001
#define POLLOUT 0x0004
#define POLLERR 0x0008
#define POLLHUP 0x0010
#define POLLNVAL 0x0020

struct pollfd
{
    SOCKET fd;
    short events;
    short revents;
};
#endif

/*
 * poll() implementation for Windows using WSAPoll
 * Note: winsock2.h defines POLLIN and struct pollfd, but not a poll() function.
 * We always define poll() to wrap WSAPoll().
 */
#ifndef HAVE_POLL
static inline int poll(struct pollfd *fds, unsigned long nfds, int timeout)
{
    return WSAPoll((WSAPOLLFD *)fds, nfds, timeout);
}
#define HAVE_POLL 1
#endif

/* IPC related constants */
#define IPC_PRIVATE 0
#define IPC_CREAT 01000
#define IPC_EXCL 02000
#define IPC_NOWAIT 04000
#define IPC_RMID 0
#define IPC_SET 1
#define IPC_STAT 2

/* Semaphore constants */
#define SEM_FAILED ((void *)-1)
#define GETVAL 12
#define SETVAL 16
#define GETALL 13
#define SETALL 17

/* For MSG_NOSIGNAL which doesn't exist on Windows */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

/* Scattered I/O (basic definitions, full implementation elsewhere) */
struct iovec
{
    void *iov_base;
    size_t iov_len;
};

/* Initialize Winsock (call once at startup) */
static inline int win32_socket_init(void)
{
    WSADATA wsaData;
    return WSAStartup(MAKEWORD(2, 2), &wsaData);
}

static inline void win32_socket_cleanup(void)
{
    WSACleanup();
}

#endif /* _WIN32 */

#endif /* NCCL_WIN32_DEFS_H_ */
