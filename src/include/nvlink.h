/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVLINK_H_
#define NCCL_NVLINK_H_

#include <sys/stat.h>
#include <fcntl.h>
#include "nvmlwrap.h"
#include "topo.h"

#define CONNECT_NVLINK 0x10
#define CONNECT_NVSWITCH 0x100

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

static ncclResult_t ncclDeviceType(const char* busId, enum ncclNvLinkDeviceType* type) {
  char classPath[] =  "/sys/bus/pci/devices/0000:00:00.0/class";
  memcpy(classPath+sizeof("/sys/bus/pci/devices/")-1, busId, sizeof("0000:00:00.0")-1);
  char* rPath = realpath(classPath, NULL);
  int fd;
  if ((fd = open(rPath, O_RDONLY)) == -1) {
    // Could not find device. It might be because we're in a VM and
    // we don't see the whole machine. This is handled silently so
    // we don't want to print an INFO error.
    TRACE(NCCL_INIT, "Open of %s failed : %s\n", rPath, strerror(errno));
    return ncclSystemError;
  }
  free(rPath);
  char pciClass[9];
  strncpy(pciClass, "0x000000", 9);
  int len;
  SYSCHECKVAL(read(fd, pciClass, 8), "read", len);
  SYSCHECK(close(fd), "close");
  if (strcmp(pciClass, "0x068000") == 0) {
    // PCI device is of type "Bridge / Other Bridge Device" (NVswitch)
    *type = ncclNvLinkDeviceSwitch;
  } else if (strcmp(pciClass, "0x068001") == 0) {
    // PCI device is of type "Bridge: IBM Device 04ea"
    *type = ncclNvLinkDeviceBridge;
  } else if (strcmp(pciClass, "0x030200") == 0 // "3D Controller" (Tesla)
      || strcmp(pciClass, "0x030000") == 0) {  // "VGA Controller" (GeForce)
    *type = ncclNvLinkDeviceGpu;
  } else {
    // Ignore if we don't know what's on the other side.
    return ncclSystemError;
  }
  return ncclSuccess;
}

/* Get the maximum number of NVLinks based on the GPU generation */
static ncclResult_t getMaxNvlinks(int* maxLinks) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int ccMajor;
  CUDACHECK(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev));
  // 6 for Volta, 4 for Pascal
  *maxLinks = (ccMajor > 6) ? 6 : 4;
  // INFO("Device %d detected %d NVLinks", cudaDev, *maxLinks);
  return ncclSuccess;
}

static int getNvlinkGpu(const char* busId1, const char* busId2) {
  // Determine if that connection is through NVLink
  int links = 0;
  int nvswitch_links = 0;
  int maxNvLinks = ncclCudaCompCap() > 6 ? 6 : 4;
  nvmlDevice_t nvmlDev;
  ncclResult_t res = wrapNvmlDeviceGetHandleByPciBusId(busId1, &nvmlDev);
  if (res != ncclSuccess) return 0;

  for(int l=0; l<maxNvLinks; ++l) {
    // Check whether we can use this NVLink for P2P
    unsigned canP2P;
    if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

    // Make sure the Nvlink is up. The previous call should have trained the link.
    nvmlEnableState_t isActive;
    if ((wrapNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive) != ncclSuccess) || (isActive != NVML_FEATURE_ENABLED)) continue;

    // Try to figure out what's on the other side of the NVLink
    nvmlPciInfo_t remoteProc;
    if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

    // Old versions of NVML return a lowercase PCI ID
    char* p = remoteProc.busId;
    for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
      if (p[c] == 0) break;
      p[c] = toupper(p[c]);
    }

    if (busId2 != NULL && strncmp(busId2, remoteProc.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
      links++;
    } else {
      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        if (p[c] == 0) break;
        lowerId[c] = tolower(p[c]);
      }

      // Determine if the remote side is NVswitch or a GPU
      enum ncclNvLinkDeviceType type;
      ncclResult_t ret = ncclDeviceType(lowerId, &type);
      if (ret == ncclSuccess) {
        if (type == ncclNvLinkDeviceSwitch) {
          //TODO: we are making an assumption that all GPUs are connected to this switch
          //This assumption may change for future architectures
          nvswitch_links++;
        } else if (type == ncclNvLinkDeviceGpu && busId2 == NULL) {
          links++;
        }
      } else {
        // The NVLink is up but we couldn't find the PCI device on the other
        // side. Assume it's an NVswitch outside a VM.
        if (l==0) INFO(NCCL_INIT, "Assuming NVLink is connected to NVswitch");
        nvswitch_links++;
      }
    }
  }
  return nvswitch_links ? CONNECT_NVSWITCH*nvswitch_links : CONNECT_NVLINK*links;
}

#endif
