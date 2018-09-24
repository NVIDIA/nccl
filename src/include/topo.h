/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "nccl.h"
#include <ctype.h>

#define MAXPATHSIZE 1024

static ncclResult_t getCudaPath(int cudaDev, char** path) {
  char busId[16];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, 16, cudaDev));
  for (int i=0; i<16; i++) busId[i] = tolower(busId[i]);
  char busPath[] = "/sys/class/pci_bus/0000:00/device";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, sizeof("0000:00")-1);
  char* cudaRpath = realpath(busPath, NULL);
  char pathname[MAXPATHSIZE];
  strncpy(pathname, cudaRpath, MAXPATHSIZE);
  strncpy(pathname+strlen(pathname), "/", MAXPATHSIZE-strlen(pathname));
  strncpy(pathname+strlen(pathname), busId, MAXPATHSIZE-strlen(pathname));
  free(cudaRpath);
  *path = realpath(pathname, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", pathname);
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t getMlxPath(char* ibName, char** path) {
  char devicepath[MAXPATHSIZE];
  snprintf(devicepath, MAXPATHSIZE, "/sys/class/infiniband/%s/device", ibName);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t getSockPath(char* ifName, char** path) {
  char devicepath[MAXPATHSIZE];
  snprintf(devicepath, MAXPATHSIZE, "/sys/class/net/%s/device", ifName);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    INFO(NET|INIT, "Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

enum ncclIbPathDist {
  PATH_PIX = 0,
  PATH_PXB = 1,
  PATH_PHB = 2,
  PATH_SOC = 3
};

static const char* pathDists[] = { "PIX", "PXB", "PHB", "SOC" };

static int pciDistance(char* path1, char* path2) {
  int score = 0;
  int depth = 0;
  int same = 1;
  for (int i=0; i<strlen(path1); i++) {
    if (path1[i] != path2[i]) same = 0;
    if (path1[i] == '/') {
      depth++;
      if (same == 1) score++;
    }
  }
  if (score == 3) return PATH_SOC;
  if (score == 4) return PATH_PHB;
  if (score == depth-1) return PATH_PIX;
  return PATH_PXB;
}

#endif
