/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "debug.h"
#include "nccl_net.h"
#include <unistd.h>
#include <string.h>
#include <stdarg.h>

#include "nvmlwrap.h"
#include "core.h"

// Convert a logical cudaDev index to the NVML device minor number
ncclResult_t getNvmlDevice(int cudaDev, int *nvmlDev) {
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  nvmlDevice_t nvmlDevice;
  unsigned int dev;
  *nvmlDev = -1;
  CUDACHECK(cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev));
  NCCLCHECK(wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlDevice));
  NCCLCHECK(wrapNvmlDeviceGetMinorNumber(nvmlDevice, &dev));

  *nvmlDev = dev;

  return ncclSuccess;
}

ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (ncclDebugLevel <= NCCL_LOG_NONE) return;

  char hostname[1024];
  getHostName(hostname, 1024, '.');
  int cudaDev;
  cudaGetDevice(&cudaDev);

  char buffer[1024];
  size_t len = 0;
  pthread_mutex_lock(&ncclDebugOutputLock);
  if (level == NCCL_LOG_WARN && ncclDebugLevel >= NCCL_LOG_WARN)
    len = snprintf(buffer, sizeof(buffer),
                   "\n%s:%d:%d [%d] %s:%d NCCL WARN ", hostname, getpid(), gettid(), cudaDev, filefunc, line);
  else if (level == NCCL_LOG_INFO && ncclDebugLevel >= NCCL_LOG_INFO && (flags & ncclDebugMask))
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] NCCL INFO ", hostname, getpid(), gettid(), cudaDev);
#ifdef ENABLE_TRACE
  else if (level == NCCL_LOG_TRACE && ncclDebugLevel >= NCCL_LOG_TRACE && (flags & ncclDebugMask)) {
    auto delta = std::chrono::high_resolution_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] %f %s:%d NCCL TRACE ", hostname, getpid(), gettid(), cudaDev, timestamp, filefunc, line);
  }
#endif
  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
    va_end(vargs);
    fprintf(ncclDebugFile,"%s\n", buffer);
    fflush(ncclDebugFile);
  }
  pthread_mutex_unlock(&ncclDebugOutputLock);

  // If ncclDebugLevel == NCCL_LOG_ABORT then WARN() will also call abort()
  if (level == NCCL_LOG_WARN && ncclDebugLevel == NCCL_LOG_ABORT) {
    fprintf(stderr,"\n%s:%d:%d [%d] %s:%d NCCL ABORT\n",
            hostname, getpid(), gettid(), cudaDev, filefunc, line);
    abort();
  }
}

uint64_t getHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname) $(readlink /proc/self/ns/uts) $(readlink /proc/self/ns/mnt)
 */
uint64_t getHostHash(void) {
  char uname[1024];
  // Start off with the full hostname
  (void) getHostName(uname, sizeof(uname), '\0');
  int offset = strlen(uname);
  int len;
  // $(readlink /proc/self/ns/uts)
  len = readlink("/proc/self/ns/uts", uname+offset, sizeof(uname)-1-offset);
  if (len < 0) len = 0;
  offset += len;
  // $(readlink /proc/self/ns/mnt)
  len = readlink("/proc/self/ns/mnt", uname+offset, sizeof(uname)-1-offset);
  if (len < 0) len = 0;
  offset += len;
  // Trailing '\0'
  uname[offset]='\0';
  TRACE(NCCL_INIT,"unique hostname '%s'", uname);

  return getHash(uname);
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long) getpid());
  int plen = strlen(pname);
  int len = readlink("/proc/self/ns/pid", pname+plen, sizeof(pname)-1-plen);
  if (len < 0) len = 0;

  pname[plen+len]='\0';
  TRACE(NCCL_INIT,"unique PID '%s'", pname);

  return getHash(pname);
}

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;
  // Ignore "^" prefix, will be detected outside of this function
  if (ptr[0] == '^') ptr++;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchPrefix(const char* string, const char* prefix) {
  return (strncmp(string, prefix, strlen(prefix)) == 0);
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}


bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchPrefix(string, ifList[i].prefix)
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}
