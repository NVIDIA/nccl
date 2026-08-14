/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "os.h"

#include <fcntl.h>
#include <limits.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>

int ncclNetIfs = -1;
struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

static std::mutex ncclNetSocketDevicesMutex;

static ncclResult_t ncclNetSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = ncclOsRealpath(devicePath, NULL);
  return ncclSuccess;
}

ncclResult_t ncclNetSocketInitDevices(const char* logPrefix) {
  std::lock_guard<std::mutex> lock(ncclNetSocketDevicesMutex);
  ncclResult_t ret = ncclSuccess;
  if (ncclNetIfs != -1) return ncclSuccess;
  char names[MAX_IF_NAME_SIZE * MAX_IFS];
  union ncclSocketAddress addrs[MAX_IFS];
  NCCLCHECKGOTO(ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS, &ncclNetIfs), ret, fail);
  if (ncclNetIfs <= 0) {
    WARN("%s : no interface found", logPrefix);
    ret = ncclInternalError;
    goto fail;
  } else {
    constexpr size_t maxLineLen = 2047;
    char line[maxLineLen + 1];
    char addrline[SOCKET_NAME_MAXLEN + 1];
    line[0] = '\0';
    addrline[SOCKET_NAME_MAXLEN] = '\0';
    for (int i = 0; i < ncclNetIfs; i++) {
      strcpy(ncclNetSocketDevs[i].devName, names + i * MAX_IF_NAME_SIZE);
      memcpy(&ncclNetSocketDevs[i].addr, addrs + i, sizeof(union ncclSocketAddress));
      NCCLCHECKGOTO(ncclNetSocketGetPciPath(ncclNetSocketDevs[i].devName, &ncclNetSocketDevs[i].pciPath), ret, fail);
      snprintf(line + strlen(line), maxLineLen - strlen(line), " [%d]%s:%s", i, names + i * MAX_IF_NAME_SIZE,
               ncclSocketToString(&addrs[i], addrline));
    }
    line[maxLineLen] = '\0';
    INFO(NCCL_INIT | NCCL_NET, "%s : Using%s", logPrefix, line);
  }
  return ncclSuccess;
fail:
  ncclNetIfs = -1;
  return ret;
}

ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed) {
  ncclResult_t ret = ncclSuccess;
  *speed = 0;

#if defined(NCCL_OS_WINDOWS)
  // On Windows, use GetAdaptersAddresses to get network interface speed
  ULONG bufferSize = 15000;
  IP_ADAPTER_ADDRESSES* adapterAddresses = NULL;
  ULONG result;
  int attempts = 0;

  do {
    adapterAddresses = (IP_ADAPTER_ADDRESSES*)malloc(bufferSize);
    if (adapterAddresses == NULL) {
      WARN("Failed to allocate memory for adapter addresses");
      *speed = 10000;
      return ncclSuccess;
    }

    result = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_INCLUDE_PREFIX, NULL, adapterAddresses, &bufferSize);
    if (result == ERROR_BUFFER_OVERFLOW) {
      free(adapterAddresses);
      adapterAddresses = NULL;
    }
    attempts++;
  } while (result == ERROR_BUFFER_OVERFLOW && attempts < 3);

  if (result == NO_ERROR) {
    // Iterate through adapters to find the matching one
    for (IP_ADAPTER_ADDRESSES* adapter = adapterAddresses; adapter != NULL; adapter = adapter->Next) {
      // Convert adapter friendly name to UTF-8 for comparison
      char adapterName[MAX_IF_NAME_SIZE];
      WideCharToMultiByte(CP_UTF8, 0, adapter->FriendlyName, -1, adapterName, sizeof(adapterName), NULL, NULL);

      // Check if this is the adapter we're looking for
      if (strstr(adapterName, devName) != NULL || strstr(devName, adapterName) != NULL) {
        // TransmitLinkSpeed is in bits per second, convert to Mbps
        if (adapter->TransmitLinkSpeed > 0) {
          *speed = (int)(adapter->TransmitLinkSpeed / 1000000);
          INFO(NCCL_NET, "Found network interface %s with speed %d Mbps", devName, *speed);
          break;
        }
      }
    }
  }

  if (adapterAddresses) {
    free(adapterAddresses);
  }

  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed for interface %s. Defaulting to 10 Gbps.", devName);
    *speed = 10000;
  }
#elif defined(NCCL_OS_LINUX)
  char speedPath[PATH_MAX];
  snprintf(speedPath, sizeof(speedPath), "/sys/class/net/%s/speed", devName);
  int fd = -1;
  SYSCHECKSYNC(open(speedPath, O_RDONLY), "open", fd);
  if (fd != -1) {
    char speedStr[] = "        ";
    int n;
    // Allow this to silently fail
    n = read(fd, speedStr, sizeof(speedStr) - 1);
    if (n > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    *speed = 10000;
  }
  if (fd != -1) SYSCHECK(close(fd), "close");
#endif
  return ret;
}

ncclResult_t ncclNetSocketCreateListener(const char* logPrefix, int dev, struct ncclSocket* sock,
                                         union ncclSocketAddress* connectAddr, uint64_t* magic) {
  if (dev < 0 || dev >= ncclNetIfs) {
    WARN("%s : invalid listen device dev=%d ncclNetIfs=%d", logPrefix, dev, ncclNetIfs);
    return ncclInternalError;
  }

  *magic = ncclSocketDefaultMagic();
  NCCLCHECK(ncclSocketInit(sock, &ncclNetSocketDevs[dev].addr, *magic, ncclSocketTypeNetSocket, NULL, 1));
  NCCLCHECK(ncclSocketListen(sock));
  NCCLCHECK(ncclSocketGetAddr(sock, connectAddr));
  return ncclSuccess;
}
