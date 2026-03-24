#ifndef NCCL_OS_LINUX_H_
#define NCCL_OS_LINUX_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <getopt.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <linux/types.h>
#include <endian.h>
#include <sys/resource.h>
#include <link.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <dirent.h>
#include <sched.h>

#define NCCL_INVALID_SOCKET -1
typedef int ncclSocketDescriptor;

typedef cpu_set_t ncclAffinity;

typedef pid_t ncclPid_t;

#define NCCL_POLLIN POLLIN
#define NCCL_POLLERR POLLHUP

#endif

