/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"

NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbDataDirect,"IB_DATA_DIRECT",1);

static std::mutex ncclIbMutex;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);

// Returns 0 if this is the path of two VFs of the same physical device
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // Keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (ncclIbMatchVfPath(p, ncclIbDevs[d].pciPath)) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000, /* NDR */
  200000  /* XDR */
};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

static bool ncclMlx5dvDmaBufCapable(ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (ncclParamIbMergeNics() == 0 && props->ndevs > 1) {
    INFO(NCCL_NET, "NET/IB : Skipping makeVDevice, NCCL_IB_MERGE_NICS=0");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
      WARN("NET/IB : Can't make virtual NIC with 0 devices");
      return ncclInvalidUsage;
  }

  if (ncclNMergedIbDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // Always count up number of merged devices
  ncclIbMergedDev* mDev = ncclIbMergedDevs + ncclNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (mDev->vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidUsage;
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
    }
  }

  // Check link layers
  ncclIbDev* dev0 = ncclIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= ncclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link), props->devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link));
      return ncclInvalidUsage;
    }
  }

  *d = ncclNMergedIbDevs++;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return ncclSuccess;
}

ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclIbMutex);
  ncclResult_t res = ncclIbMakeVDeviceInternal(d, props);
  return res;

}

ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

const char* ibProviderName[] = {
  "None",
  "Mlx5",
};

ncclResult_t ncclIbFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  if (netRefCount++) return ret;
  ncclProfilerFunction = profFunction;
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if(wrap_mlx5dv_symbols() != ncclSuccess) { INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); }

  if (ncclNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(ncclIbMutex);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = /*undefined*/-1, devOffset = 0;
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;

        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            struct ibv_port_attr portAttr;
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }

            // check for mlx5 data direct support only once for a each device
            if (devCount == -1) {
              devCount = 1;
              devOffset = 0;
              if (ncclParamIbDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                if (res == ncclSuccess) {
                  // data direct devices are exposed twice: with the C2C + PCIe link and with the data direct link
                  devCount = 2;
                  // by default only expose the data direct NIC (devOffset = 1), unless set to 2 by the user
                  devOffset = (ncclParamIbDataDirect() == 2) ? 0 : 1;
                  INFO(NCCL_INIT | NCCL_NET, "NET/IB: Data Direct DMA Interface is detected for device %s", devices[d]->name);
                } else if (res == ncclInvalidArgument) {
                  TRACE(NCCL_NET, "NET/IB: Device %s does not support Data Direct DMA.", devices[d]->name);
                } else {
                  WARN("NET/IB: Error in mlx5dv_get_data_direct_sysfs_path with device %s", devices[d]->name);
                  return res;
                }
              }
            }
            for (int dev = devOffset; dev < devCount; ++dev) {
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].ibProvider = ibProvider;
              ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
              ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
              ncclIbDevs[ncclNIbDevs].portNum = port_num;
              ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
              if (portAttr.active_speed_ex) {
                // A non-zero active_speed_ex indicates XDR rate (0x100) or higher
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed_ex) * ncclIbWidth(portAttr.active_width);
              } else {
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
              }
              ncclIbDevs[ncclNIbDevs].context = context;
              ncclIbDevs[ncclNIbDevs].pdRefs = 0;
              ncclIbDevs[ncclNIbDevs].pd = NULL;
              if (dev == 0) {
                strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
                NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort), ret, fail);
              } else {
                snprintf(ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&ncclIbDevs[ncclNIbDevs].pciPath, PATH_MAX));
                strncpy(ncclIbDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                ncclIbDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
              }
              ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
              ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
              NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

              // Enable ADAPTIVE_ROUTING by default on IB networks
              // But allow it to be overloaded by an env parameter
              ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name,
                   ncclIbDevs[ncclNIbDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), ibProviderName[ncclIbDevs[ncclNIbDevs].ibProvider], ncclIbDevs[ncclNIbDevs].speed, context,
                   ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

              ncclIbAsyncThread = std::thread(ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs);
              ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
              ncclIbAsyncThread.detach();

              // Add this plain physical device to the list of virtual devices
              int vDev;
              ncclNetVDeviceProps_t vProps = {0};
              vProps.ndevs = 1;
              vProps.devs[0] = ncclNIbDevs;
              NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));

              ncclNIbDevs++;
              nPorts++;
            }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; }
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }

    // Print out all net devices to the user (in the same format as before)
    char line[2048];
    line[0] = '\0';
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
    for (int d = 0; d < ncclNIbDevs; d++) {
        snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
          ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
          ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));

  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclIbInitDevices(logFunction, profFunction));
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = config->trafficClass;
  *ctx = (void *)netCommConfig;
  return ret;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

ncclResult_t ncclIbFinalize(void* ctx) {
  free(ctx);
  return ncclIbFinalizeDevices();
}
