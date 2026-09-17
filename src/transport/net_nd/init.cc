/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#include <vector>
#include <atomic>
#include <algorithm>
#include <limits.h>
#include <iphlpapi.h>
#include <setupapi.h>
#include <cfgmgr32.h>

#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "cfgmgr32.lib")

static std::atomic<int> netRefCount{0};
static std::mutex ncclNdMutex;
static HANDLE ncclNdInterfaceNotification = NULL;

struct ncclNdAdapterEntry {
  UINT64 adapterId;
  union ncclSocketAddress addr;
};

// Prefer routable IPv6 addresses over IPv4, but never publish an IPv6
// address whose interface-local scope cannot be used by a remote peer.
static int ncclNdAddressRank(const union ncclSocketAddress* addr) {
  if (addr->sa.sa_family == AF_INET) return 1;
  if (addr->sa.sa_family != AF_INET6) return 0;

  const IN6_ADDR* ipv6 = &addr->sin6.sin6_addr;
  if (IN6_IS_ADDR_UNSPECIFIED(ipv6) || IN6_IS_ADDR_LOOPBACK(ipv6) || IN6_IS_ADDR_MULTICAST(ipv6) ||
      IN6_IS_ADDR_LINKLOCAL(ipv6)) {
    return 0;
  }
  return 2;
}

// Compare an ND address with a Windows interface address, including IPv6 scope.
static bool ncclNdSameAddress(const SOCKADDR_INET* lhs, const union ncclSocketAddress* rhs) {
  if (lhs->si_family != rhs->sa.sa_family) return false;
  if (lhs->si_family == AF_INET) {
    return lhs->Ipv4.sin_addr.s_addr == rhs->sin.sin_addr.s_addr;
  }
  if (lhs->si_family == AF_INET6) {
    return memcmp(&lhs->Ipv6.sin6_addr, &rhs->sin6.sin6_addr, sizeof(IN6_ADDR)) == 0 &&
           (rhs->sin6.sin6_scope_id == 0 || lhs->Ipv6.sin6_scope_id == rhs->sin6.sin6_scope_id);
  }
  return false;
}

// Resolve an ND address to its Windows interface row.
static bool ncclNdGetInterfaceRow(const union ncclSocketAddress* addr, MIB_IF_ROW2* ifRow) {
  MIB_UNICASTIPADDRESS_TABLE* table = NULL;
  if (GetUnicastIpAddressTable(addr->sa.sa_family, &table) != NO_ERROR || table == NULL) return false;

  bool found = false;
  for (ULONG i = 0; i < table->NumEntries; i++) {
    MIB_UNICASTIPADDRESS_ROW* ipRow = &table->Table[i];
    if (!ncclNdSameAddress(&ipRow->Address, addr)) continue;
    memset(ifRow, 0, sizeof(*ifRow));
    ifRow->InterfaceLuid = ipRow->InterfaceLuid;
    found = GetIfEntry2(ifRow) == NO_ERROR;
    break;
  }
  FreeMibTable(table);
  return found;
}

// Read a network interface GUID from the Windows device registry.
static bool ncclNdReadNetCfgInstanceId(HDEVINFO deviceSet, SP_DEVINFO_DATA* device, char* value, DWORD valueBytes) {
  HKEY key = SetupDiOpenDevRegKey(deviceSet, device, DICS_FLAG_GLOBAL, 0, DIREG_DRV, KEY_READ);
  if (key == INVALID_HANDLE_VALUE) return false;
  DWORD type = 0;
  DWORD bytes = valueBytes;
  LONG status = RegQueryValueExA(key, "NetCfgInstanceId", NULL, &type, (BYTE*)value, &bytes);
  RegCloseKey(key);
  if (status != ERROR_SUCCESS || type != REG_SZ || bytes == 0) return false;
  value[valueBytes - 1] = '\0';
  return true;
}

// Return a slash-prefixed PCI bus ID path so NCCL's topology loader can use
// the same bus-id leaf convention on Windows that it uses for Linux sysfs.
static bool ncclNdGetPciPath(const GUID* interfaceGuid, char* pciPath, size_t pciPathBytes) {
  WCHAR wideGuid[64];
  char guid[64];
  if (StringFromGUID2(*interfaceGuid, wideGuid, (int)(sizeof(wideGuid) / sizeof(wideGuid[0]))) == 0 ||
      WideCharToMultiByte(CP_UTF8, 0, wideGuid, -1, guid, sizeof(guid), NULL, NULL) == 0) {
    return false;
  }

  HDEVINFO deviceSet = SetupDiGetClassDevsA(NULL, NULL, NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
  if (deviceSet == INVALID_HANDLE_VALUE) return false;
  bool found = false;
  SP_DEVINFO_DATA device = {};
  device.cbSize = sizeof(device);
  for (DWORD index = 0; SetupDiEnumDeviceInfo(deviceSet, index, &device); index++) {
    char instanceGuid[64];
    if (!ncclNdReadNetCfgInstanceId(deviceSet, &device, instanceGuid, sizeof(instanceGuid)) ||
        _stricmp(guid, instanceGuid) != 0) {
      continue;
    }

    ULONG bus = 0;
    ULONG address = 0;
    ULONG bytes = sizeof(bus);
    if (CM_Get_DevNode_Registry_PropertyA(device.DevInst, CM_DRP_BUSNUMBER, NULL, &bus, &bytes, 0) != CR_SUCCESS) {
      break;
    }
    bytes = sizeof(address);
    if (CM_Get_DevNode_Registry_PropertyA(device.DevInst, CM_DRP_ADDRESS, NULL, &address, &bytes, 0) != CR_SUCCESS) {
      break;
    }
    unsigned int pciDevice = (address >> 16) & 0xffff;
    unsigned int pciFunction = address & 0xffff;
    if (bus > 0xff || pciDevice > 0x1f || pciFunction > 7) break;
    snprintf(pciPath, pciPathBytes, "/0000:%02x:%02x.%x", (unsigned int)bus, pciDevice, pciFunction);
    found = true;
    break;
  }
  SetupDiDestroyDeviceInfoList(deviceSet);
  return found;
}

// Collect link speed, interface identity, and PCI topology for an ND address.
static bool ncclNdGetWindowsProperties(const union ncclSocketAddress* addr, int* speed, char* pciPath,
                                       size_t pciPathBytes, char* alias, size_t aliasBytes, UINT64* interfaceLuid) {
  MIB_IF_ROW2 ifRow = {};
  if (!ncclNdGetInterfaceRow(addr, &ifRow)) return false;
  if (interfaceLuid != NULL) *interfaceLuid = ifRow.InterfaceLuid.Value;

  ULONG64 linkBits = ifRow.TransmitLinkSpeed;
  if (linkBits == 0 || (ifRow.ReceiveLinkSpeed != 0 && ifRow.ReceiveLinkSpeed < linkBits)) {
    linkBits = ifRow.ReceiveLinkSpeed;
  }
  if (linkBits == 0) return false;
  ULONG64 speedMbps = linkBits / 1000000;
  if (speedMbps == 0) speedMbps = 1;
  *speed = speedMbps > INT_MAX ? INT_MAX : (int)speedMbps;

  if (alias != NULL && aliasBytes != 0) {
    if (WideCharToMultiByte(CP_UTF8, 0, ifRow.Alias, -1, alias, (int)aliasBytes, NULL, NULL) == 0) {
      alias[0] = '\0';
    }
  }
  pciPath[0] = '\0';
  (void)ncclNdGetPciPath(&ifRow.InterfaceGuid, pciPath, pciPathBytes);
  return true;
}

// Cache a device link transition and record newly observed failures.
void ncclNdRecordDeviceInterfaceChange(int devIndex, bool healthy) {
  if (devIndex < 0 || devIndex >= ncclNNdDevs) return;
  struct ncclNdDev* dev = &ncclNdDevs[devIndex];
  LONG newState = healthy ? 1 : -1;
  LONG previous = InterlockedExchange(&dev->healthState, newState);
  InterlockedExchange64(&dev->nextHealthCheckMs, (LONG64)(GetTickCount64() + 1000));
  if (previous == newState) return;
  dev->stats.linkEvents.fetch_add(1, std::memory_order_relaxed);
  if (!healthy) {
    InterlockedIncrement64(&dev->linkFailureGeneration);
    WARN("NET/ND : Device %d interface link became unavailable", dev->device);
  } else {
    INFO(NCCL_NET, "NET/ND : Device %d interface link is operational", dev->device);
  }
}

// Apply a Windows interface transition to each matching physical ND device.
void ncclNdRecordInterfaceChange(UINT64 interfaceLuid, bool healthy) {
  for (int i = 0; i < ncclNNdDevs; i++) {
    if (ncclNdDevs[i].interfaceLuid != 0 && ncclNdDevs[i].interfaceLuid == interfaceLuid)
      ncclNdRecordDeviceInterfaceChange(i, healthy);
  }
}

// Translate a Windows interface notification into an ND health update.
static VOID CALLBACK ncclNdInterfaceChangeCallback(PVOID, PMIB_IPINTERFACE_ROW row, MIB_NOTIFICATION_TYPE type) {
  if (row == NULL) return;
  bool healthy = false;
  if (type != MibDeleteInstance) {
    MIB_IF_ROW2 ifRow = {};
    ifRow.InterfaceLuid = row->InterfaceLuid;
    healthy = GetIfEntry2(&ifRow) == NO_ERROR && ifRow.OperStatus == IfOperStatusUp &&
              ifRow.MediaConnectState == MediaConnectStateConnected;
  }
  ncclNdRecordInterfaceChange(row->InterfaceLuid.Value, healthy);
}

// Subscribe once to asynchronous Windows interface changes.
static ncclResult_t ncclNdStartInterfaceMonitor() {
  if (ncclNdInterfaceNotification != NULL) return ncclSuccess;
  DWORD status =
    NotifyIpInterfaceChange(AF_UNSPEC, ncclNdInterfaceChangeCallback, NULL, FALSE, &ncclNdInterfaceNotification);
  if (status != NO_ERROR) {
    ncclNdInterfaceNotification = NULL;
    WARN("NET/ND : Could not subscribe to Windows interface changes: %lu; using polling", status);
    return ncclSystemError;
  }
  INFO(NCCL_NET, "NET/ND : Windows interface-change monitoring enabled");
  return ncclSuccess;
}

// Cancel the process-wide Windows interface subscription.
static void ncclNdStopInterfaceMonitor() {
  if (ncclNdInterfaceNotification == NULL) return;
  DWORD status = CancelMibChangeNotify2(ncclNdInterfaceNotification);
  if (status != NO_ERROR) WARN("NET/ND : Could not cancel Windows interface monitoring cleanly: %lu", status);
  ncclNdInterfaceNotification = NULL;
}

// Report whether asynchronous interface monitoring is active.
bool ncclNdInterfaceMonitorActive() {
  return ncclNdInterfaceNotification != NULL;
}

// Revalidate provider and Windows link state, using a short-lived cache.
ncclResult_t ncclNdCheckAdapterHealth(int dev, bool* healthy) {
  if (healthy == NULL || dev < 0 || dev >= ncclNNdDevs) return ncclInvalidArgument;
  *healthy = false;
  struct ncclNdDev* ndDev = &ncclNdDevs[dev];
  if (ndDev->adapter == NULL) return ncclSystemError;

  // Return a recent notification or polling result without taking the device lock.
  static constexpr ULONGLONG kHealthCacheMs = 1000;
  ULONGLONG now = GetTickCount64();
  LONG state = InterlockedCompareExchange(&ndDev->healthState, 0, 0);
  ULONGLONG next = (ULONGLONG)InterlockedCompareExchange64(&ndDev->nextHealthCheckMs, 0, 0);
  if (state != 0 && now < next) {
    *healthy = state > 0;
    return ncclSuccess;
  }

  // Recheck under the lock in case another thread refreshed the cache.
  std::lock_guard<std::mutex> lock(ndDev->mutex);
  now = GetTickCount64();
  state = InterlockedCompareExchange(&ndDev->healthState, 0, 0);
  next = (ULONGLONG)InterlockedCompareExchange64(&ndDev->nextHealthCheckMs, 0, 0);
  if (state != 0 && now < next) {
    *healthy = state > 0;
    return ncclSuccess;
  }

  // A provider query catches adapter resets/removal even when the TCP control
  // connection remains healthy. The interface row catches physical link and
  // administrative state changes that do not necessarily complete an ND WQE.
  ND2_ADAPTER_INFO info = {};
  ncclResult_t result = wrap_nd_query_adapter(ndDev->adapter, &info);
  MIB_IF_ROW2 ifRow = {};
  ifRow.InterfaceLuid.Value = ndDev->interfaceLuid;
  bool found = result == ncclSuccess && ndDev->interfaceLuid != 0 && GetIfEntry2(&ifRow) == NO_ERROR;
  if (!found && result == ncclSuccess) found = ncclNdGetInterfaceRow(&ndDev->addr, &ifRow);
  *healthy = found && ifRow.OperStatus == IfOperStatusUp && ifRow.MediaConnectState == MediaConnectStateConnected;
  InterlockedExchange(&ndDev->healthState, *healthy ? 1 : -1);
  InterlockedExchange64(&ndDev->nextHealthCheckMs, (LONG64)(now + kHealthCacheMs));
  return result == ncclSuccess && found ? ncclSuccess : ncclSystemError;
}

// Unwind partially initialized global state in reverse ownership order.
static ncclResult_t ncclNdCleanupInitFailure(ncclResult_t failure) {
  ncclNdStopInterfaceMonitor();
  for (int i = 0; i < ncclNNdDevs; i++) {
    struct ncclNdDev* dev = &ncclNdDevs[i];
    NCCLCHECKIGNORE(ncclNdFinalizeMrCache(dev), failure);
    if (dev->adapter != NULL) {
      (void)wrap_nd_release(dev->adapter);
      dev->adapter = NULL;
    }
    dev->devName[0] = '\0';
    dev->pciPath[0] = '\0';
  }
  ncclNNdDevs = -1;
  ncclNMergedNdDevs = -1;
  wrap_nd_unload();
  return failure;
}

// Discover and initialize physical ND adapters on the first transport reference.
static ncclResult_t ncclNdInitDevices() {
  ncclResult_t ret = ncclSuccess;

  std::lock_guard<std::mutex> lock(ncclNdMutex);

  // Reuse process-wide device state after the first successful discovery.
  if (netRefCount.load(std::memory_order_acquire) > 0) {
    netRefCount.fetch_add(1, std::memory_order_acq_rel);
    return ret;
  }

  // Reject a disabled transport before loading provider state.
  if (ncclParamNdDisable()) {
    INFO(NCCL_NET, "NET/ND : NetworkDirect transport disabled via NCCL_IB_DISABLE");
    return ncclInternalError;
  }

  // Load the NDv2 provider before starting a fresh discovery pass.
  if (wrap_nd_symbols() != ncclSuccess) {
    INFO(NCCL_NET, "NET/ND : Failed to load NetworkDirect provider");
    return ncclInternalError;
  }

  ncclNNdDevs = 0;
  ncclNMergedNdDevs = 0;

  // Select the TCP bootstrap interface before enumerating ND adapters.
  char socketIfName[MAX_IF_NAME_SIZE];
  int nSocketIfs = 0;
  ret = ncclFindInterfaces(socketIfName, &ncclNdSocketAddr, MAX_IF_NAME_SIZE, 1, &nSocketIfs);
  if (ret != ncclSuccess) return ncclNdCleanupInitFailure(ret);
  if (nSocketIfs <= 0) {
    WARN("NET/ND : no socket interface found for out-of-band connections");
    return ncclNdCleanupInitFailure(ncclInternalError);
  }

  struct IND2Provider* provider = wrap_nd_get_provider();
  if (provider == NULL) {
    INFO(NCCL_NET, "NET/ND : No NetworkDirect provider available");
    return ncclNdCleanupInitFailure(ncclInternalError);
  }

  // Query the provider address list in two passes: size, then contents.
  SOCKET_ADDRESS_LIST* pAddrList = NULL;
  ULONG cbAddrList = 0;
  ret = wrap_nd_query_address_list(provider, pAddrList, &cbAddrList);
  if (ret != ncclSuccess) return ncclNdCleanupInitFailure(ret);
  if (cbAddrList == 0) {
    INFO(NCCL_NET, "NET/ND : Empty address list from provider");
    return ncclNdCleanupInitFailure(ncclInternalError);
  }
  pAddrList = (SOCKET_ADDRESS_LIST*)malloc(cbAddrList);
  if (pAddrList == NULL) return ncclNdCleanupInitFailure(ncclSystemError);
  ncclResult_t listRet = wrap_nd_query_address_list(provider, pAddrList, &cbAddrList);
  if (listRet != ncclSuccess) {
    free(pAddrList);
    return ncclNdCleanupInitFailure(listRet);
  }

  // Deduplicate adapters by AdapterId while keeping an address we can bind to.
  std::vector<ncclNdAdapterEntry> adapterEntries;
  adapterEntries.reserve(pAddrList->iAddressCount);
  for (int i = 0; i < pAddrList->iAddressCount; i++) {
    SOCKET_ADDRESS* sa = &pAddrList->Address[i];
    if (sa->lpSockaddr == NULL) continue;
    int family = sa->lpSockaddr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    union ncclSocketAddress addr;
    memset(&addr, 0, sizeof(addr));
    if (family == AF_INET) {
      if ((size_t)sa->iSockaddrLength < sizeof(struct sockaddr_in)) continue;
      memcpy(&addr.sin, sa->lpSockaddr, sizeof(struct sockaddr_in));
      addr.sin.sin_port = 0;
    } else {
      if ((size_t)sa->iSockaddrLength < sizeof(struct sockaddr_in6)) continue;
      memcpy(&addr.sin6, sa->lpSockaddr, sizeof(struct sockaddr_in6));
      addr.sin6.sin6_port = 0;
    }
    int addrRank = ncclNdAddressRank(&addr);
    if (addrRank == 0) continue;

    UINT64 adapterId = 0;
    if (wrap_nd_resolve_address(provider, sa->lpSockaddr, (ULONG)sa->iSockaddrLength, &adapterId) != ncclSuccess) {
      continue;
    }
    int seenIndex = -1;
    for (size_t entryIndex = 0; entryIndex < adapterEntries.size(); entryIndex++) {
      if (adapterEntries[entryIndex].adapterId == adapterId) {
        seenIndex = (int)entryIndex;
        break;
      }
    }
    if (seenIndex >= 0) {
      if (addrRank > ncclNdAddressRank(&adapterEntries[seenIndex].addr)) {
        adapterEntries[seenIndex].addr = addr;
      }
    } else {
      ncclNdAdapterEntry entry;
      entry.adapterId = adapterId;
      entry.addr = addr;
      adapterEntries.push_back(entry);
    }
  }

  free(pAddrList);

  // Match net-ib's device selection semantics, but use the Windows interface
  // alias because NetworkDirect exposes adapter IDs rather than HCA names.
  // For example, NCCL_IB_HCA="=Ethernet 4,Ethernet 11" selects exactly those
  // two interfaces, while a leading '^' excludes the listed interfaces.
  const char* userNdEnv = ncclGetEnv("NCCL_IB_HCA");
  struct netIf userIfs[MAX_ND_DEVS];
  bool searchNot = userNdEnv != NULL && userNdEnv[0] == '^';
  if (searchNot) userNdEnv++;
  bool searchExact = userNdEnv != NULL && userNdEnv[0] == '=';
  if (searchExact) userNdEnv++;
  int nUserIfs = parseStringList(userNdEnv, userIfs, MAX_ND_DEVS);
  if (userNdEnv != NULL)
    INFO(NCCL_NET | NCCL_ENV, "NET/ND : NCCL_IB_HCA set to %s%s%s", searchNot ? "^" : "", searchExact ? "=" : "",
         userNdEnv);

  // Open each candidate, reject insufficient capabilities, and publish usable devices.
  for (size_t i = 0; i < adapterEntries.size() && ncclNNdDevs < MAX_ND_DEVS; i++) {
    UINT64 adapterId = adapterEntries[i].adapterId;
    struct IND2Adapter* adapter = NULL;
    if (wrap_nd_open_adapter(provider, adapterId, &adapter) != ncclSuccess || adapter == NULL) {
      WARN("NET/ND : Failed opening adapter 0x%llx", adapterId);
      continue;
    }

    ND2_ADAPTER_INFO info;
    if (wrap_nd_query_adapter(adapter, &info) != ncclSuccess) {
      wrap_nd_release(adapter);
      continue;
    }
    const ULONG requiredCqDepth = NCCL_ND_SEND_WR_DEPTH + NCCL_ND_RECV_WR_DEPTH;
    if (info.MaxInitiatorSge < 1 || info.MaxReceiveSge < 1 || info.MaxTransferLength == 0 ||
        info.MaxInitiatorQueueDepth < NCCL_ND_SEND_WR_DEPTH || info.MaxReceiveQueueDepth < NCCL_ND_RECV_WR_DEPTH ||
        info.MaxCompletionQueueDepth < requiredCqDepth) {
      WARN("NET/ND : Adapter 0x%llx cannot satisfy required queue limits "
           "(tx=%lu/%d rx=%lu/%d cq=%lu/%lu)",
           adapterId, info.MaxInitiatorQueueDepth, NCCL_ND_SEND_WR_DEPTH, info.MaxReceiveQueueDepth,
           NCCL_ND_RECV_WR_DEPTH, info.MaxCompletionQueueDepth, requiredCqDepth);
      wrap_nd_release(adapter);
      continue;
    }

    // Populate the next temporary device slot; publish it only after filtering.
    struct ncclNdDev* dev = &ncclNdDevs[ncclNNdDevs];
    dev->device = (int)ncclNNdDevs;
    dev->adapterId = adapterId;
    dev->interfaceLuid = 0;
    InterlockedExchange64(&dev->nextHealthCheckMs, 0);
    InterlockedExchange64(&dev->linkFailureGeneration, 0);
    InterlockedExchange(&dev->healthState, 0);
    dev->adapter = adapter;
    dev->addr = adapterEntries[i].addr;
    dev->adapterInfo = info;
    snprintf(dev->devName, sizeof(dev->devName), "ND-%016llx", (unsigned long long)adapterId);
    char interfaceAlias[MAXNAMESIZE] = {};
    if (!ncclNdGetWindowsProperties(&dev->addr, &dev->speed, dev->pciPath, sizeof(dev->pciPath), interfaceAlias,
                                    sizeof(interfaceAlias), &dev->interfaceLuid)) {
      // Keep a conservative selection hint if Windows cannot map a provider
      // address to an interface, but never invent a topology path. NDv2's
      // MaxTransferLength is a request-size limit, not a bandwidth signal.
      static constexpr int kFallbackSpeedMbps = 10000;
      dev->speed = kFallbackSpeedMbps;
      dev->pciPath[0] = '\0';
      if (nUserIfs != 0 && !searchNot) {
        INFO(NCCL_NET,
             "NET/ND : Could not map adapter 0x%llx to a Windows interface; "
             "an inclusive NCCL_IB_HCA filter will skip it",
             adapterId);
      } else {
        WARN("NET/ND : Could not query Windows interface properties for adapter 0x%llx; "
             "using conservative speed %d Mbps",
             adapterId, dev->speed);
      }
    }
    if (!(matchIfList(interfaceAlias, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      INFO(NCCL_NET, "NET/ND : Skipping adapter %s interface=%s due to NCCL_IB_HCA", dev->devName,
           interfaceAlias[0] ? interfaceAlias : "unknown");
      (void)wrap_nd_release(adapter);
      dev->adapter = NULL;
      continue;
    }
    // ND2 does not report adapter-wide object counts. Use explicit software
    // admission limits and enforce them when communicators and cached MRs are created.
    int64_t configuredMaxComms = NCCL_ND_MAX_COMMS;
    int64_t configuredMaxMrs = NCCL_ND_MAX_MRS;
    // A loopback pair consumes one send and one receive communicator.
    dev->maxComms = (int)std::max<int64_t>(2, std::min<int64_t>(configuredMaxComms, INT_MAX));
    dev->maxMrs = (int)std::max<int64_t>(1, std::min<int64_t>(configuredMaxMrs, INT_MAX));
    dev->port = 1;
    dev->latency = 0.0f; // ND2 exposes no latency metric; zero means unspecified.
    dev->mrCache.slots = NULL;
    dev->mrCache.capacity = 0;
    dev->mrCache.population = 0;
    ret = ncclNdStatsInit(&dev->stats);
    if (ret != ncclSuccess) {
      (void)wrap_nd_release(adapter);
      dev->adapter = NULL;
      return ncclNdCleanupInitFailure(ret);
    }

    // GPUDirect capability is specific to a CUDA device/ND adapter pair and
    // will be populated lazily by an actual registration probe.
    for (int gpu = 0; gpu < NCCL_ND_MAX_CUDA_DEVS; gpu++) {
      dev->gpuCaps.gpuDirectSupported[gpu] = -1;
      dev->gpuCaps.forceFlush[gpu] = 0;
    }

    char addrLine[SOCKET_NAME_MAXLEN + 1];
    INFO(NCCL_NET,
         "NET/ND : Device [%d] %s addr=%s interface=%s pciPath=%s "
         "flags=0x%x speed=%d maxComms=%d maxMrs=%d maxInline=%lu",
         dev->device, dev->devName, ncclSocketToString(&dev->addr, addrLine),
         interfaceAlias[0] ? interfaceAlias : "unknown", dev->pciPath[0] ? dev->pciPath : "unknown",
         dev->adapterInfo.AdapterFlags, dev->speed, dev->maxComms, dev->maxMrs, dev->adapterInfo.MaxInlineDataSize);

    ncclNNdDevs++;
  }

  INFO(NCCL_NET, "NET/ND : Enumerated %d NetworkDirect adapters", ncclNNdDevs);
  INFO(NCCL_NET,
       "NET/ND : Configured tuning qpsPerAdapter=%lld cqPollBatch=%d inline=%lld "
       "providerCheck=%dms linkCheck=%dms failover=%lld",
       (long long)ncclParamNdQpsPerConnection(), NCCL_ND_CQ_POLL_BATCH, (long long)ncclParamNdUseInline(),
       NCCL_ND_PROVIDER_CHECK_INTERVAL_MS, NCCL_ND_LINK_CHECK_INTERVAL_MS,
       (long long)ncclParamNdResiliencyPortFailover());

  // Require one usable adapter before enabling asynchronous link monitoring.
  if (ncclNNdDevs == 0) {
    INFO(NCCL_NET, "NET/ND : No NetworkDirect adapters found");
    return ncclNdCleanupInitFailure(ncclInternalError);
  }

  (void)ncclNdStartInterfaceMonitor();

  netRefCount.store(1, std::memory_order_release);
  return ncclSuccess;
}

// Retain shared device state and allocate one plugin context.
ncclResult_t ncclNdInit(void** ctx, uint64_t, ncclNetCommConfig_t*, ncclDebugLogger_t, ncclProfilerCallback_t) {
  NCCLCHECK(ncclNdInitDevices());

  ncclNetCommConfig_t* netCommConfig = nullptr;
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclCalloc(&netCommConfig, 1), ret, fail);
  *ctx = netCommConfig;

  return ncclSuccess;

fail:
  NCCLCHECKIGNORE(ncclNdFinalizeDevices(), ret);
  return ret;
}

// Return the combined physical and virtual device count.
ncclResult_t ncclNdDevices(int* ndev) {
  *ndev = ncclNNdDevs + (ncclNMergedNdDevs > 0 ? ncclNMergedNdDevs : 0);
  return ncclSuccess;
}

// Build NCCL properties for one physical or multi-adapter virtual device.
ncclResult_t ncclNdGetProperties(int dev, ncclNetProperties_t* props) {
  struct ncclNdDev* ndDev;
  struct ncclNdMergedDev* mDev;

  memset(props, 0, sizeof(*props));

  // Resolve virtual devices through their first physical rail for shared fields.
  if (dev < 0) return ncclInternalError;
  if (dev >= ncclNNdDevs) {
    int vDev = dev - ncclNNdDevs;
    if (vDev < 0 || vDev >= ncclNMergedNdDevs) return ncclInternalError;
    mDev = &ncclNdMergedDevs[vDev];
    ndDev = &ncclNdDevs[mDev->vProps.devs[0]];

    // Use the aggregate identity and speed for a virtual device.
    props->name = mDev->devName;
    props->pciPath = ndDev->pciPath[0] ? ndDev->pciPath : NULL;
    props->speed = mDev->speed;
    props->vProps = mDev->vProps;
  } else {
    if (dev >= ncclNNdDevs) return ncclInternalError;
    ndDev = &ncclNdDevs[dev];

    // A physical device exposes a one-rail virtual-device description.
    props->name = ndDev->devName;
    props->pciPath = ndDev->pciPath[0] ? ndDev->pciPath : NULL;
    props->speed = ndDev->speed;
    props->vProps.ndevs = 1;
    props->vProps.devs[0] = dev;
  }

  // Populate properties shared by physical and virtual devices.
  props->port = ndDev->port;
  props->maxRecvs = NCCL_NET_ND_MAX_RECVS;
  props->latency = ndDev->latency;

  // Probe every physical rail with a real provider registration because CUDA
  // attributes alone cannot prove ND support. Intersect rail support and limits,
  // and hash adapter IDs into a stable virtual-device GUID.
  int gpuDirectSupported = 1;
  int forceFlush = 0;
  props->maxComms = INT_MAX;
  props->maxP2pBytes = (size_t)INT_MAX;
  uint64_t guid = UINT64_C(1469598103934665603);
  for (int i = 0; i < props->vProps.ndevs; i++) {
    int physDev = props->vProps.devs[i];
    if (physDev < 0 || physDev >= ncclNNdDevs) return ncclInternalError;
    struct ncclNdDev* physical = &ncclNdDevs[physDev];
    int physicalGdr = 0;
    int physicalFlush = 0;
    NCCLCHECK(ncclNdGetGpuDirectSupport(physical, &physicalGdr, &physicalFlush));
    gpuDirectSupported &= physicalGdr;
    forceFlush |= physicalFlush;
    props->maxComms = std::min(props->maxComms, physical->maxComms);
    size_t maxTransfer = physical->adapterInfo.MaxTransferLength < (ULONG)INT_MAX ?
                           physical->adapterInfo.MaxTransferLength :
                           (size_t)INT_MAX;
    props->maxP2pBytes = std::min(props->maxP2pBytes, maxTransfer);
    guid = (guid ^ physical->adapterId) * UINT64_C(1099511628211);
  }
  props->guid = props->vProps.ndevs == 1 ? ndDev->adapterId : guid;
  props->ptrSupport = NCCL_PTR_HOST | (gpuDirectSupported ? NCCL_PTR_CUDA : 0);

  props->regIsGlobal = 1; // ND memory regions work across QPs
  props->forceFlush = forceFlush;

  props->maxCollBytes = props->maxP2pBytes;
  props->maxMultiRequestSize = 1;
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;

  return ncclSuccess;
}

// Validate and create one virtual device from unique physical rails.
static ncclResult_t ncclNdMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (d == NULL || props == NULL || props->ndevs <= 0 || props->ndevs > NCCL_ND_MAX_DEVS_PER_NIC) {
    WARN("NET/ND : Invalid virtual device size %d (max %d)", props ? props->ndevs : -1, NCCL_ND_MAX_DEVS_PER_NIC);
    return ncclInvalidUsage;
  }

  if (ncclNMergedNdDevs == -1) ncclNMergedNdDevs = 0;

  if (ncclNMergedNdDevs >= MAX_ND_VDEVS) {
    WARN("NET/ND : Maximum virtual devices (%d) reached", MAX_ND_VDEVS);
    return ncclInternalError;
  }

  // Build in the next slot; publish it only after every physical rail validates.
  int vDev = ncclNMergedNdDevs;
  struct ncclNdMergedDev* mDev = &ncclNdMergedDevs[vDev];
  mDev->vProps = *props;
  mDev->speed = 0;
  mDev->devName[0] = '\0';

  // Validate unique rails while aggregating speed and a diagnostic name.
  for (int i = 0; i < props->ndevs; i++) {
    int dev = props->devs[i];
    if (dev < 0 || dev >= ncclNNdDevs) {
      WARN("NET/ND : Invalid device %d in virtual device props", dev);
      return ncclInvalidUsage;
    }
    for (int j = 0; j < i; j++) {
      if (props->devs[j] == dev) {
        WARN("NET/ND : Duplicate physical device %d in virtual device", dev);
        return ncclInvalidUsage;
      }
    }
    mDev->speed += ncclNdDevs[dev].speed;

    size_t remaining = sizeof(mDev->devName) - strlen(mDev->devName) - 1;
    if (i > 0 && remaining > 1) {
      strncat(mDev->devName, "+", remaining);
      remaining--;
    }
    if (remaining > 0) {
      strncat(mDev->devName, ncclNdDevs[dev].devName, remaining);
    }
  }

  ncclNMergedNdDevs++;
  *d = ncclNNdDevs + vDev;
  INFO(NCCL_NET, "NET/ND : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed,
       mDev->vProps.ndevs);

  return ncclSuccess;
}

// Serialize virtual-device creation against discovery and teardown.
ncclResult_t ncclNdMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclNdMutex);
  return ncclNdMakeVDeviceInternal(d, props);
}

// Accept optional NCCL network attributes; ND currently has none to apply.
ncclResult_t ncclNdSetNetAttr(void*, ncclNetAttr_t*) {
  return ncclSuccess;
}

// Release process-wide ND state after the final plugin reference.
ncclResult_t ncclNdFinalizeDevices(void) {
  std::lock_guard<std::mutex> lock(ncclNdMutex);

  // Keep global devices alive while another plugin context still references them.
  int refs = netRefCount.load(std::memory_order_acquire);
  if (refs <= 0) return ncclSuccess;
  refs--;
  netRefCount.store(refs, std::memory_order_release);
  if (refs != 0) return ncclSuccess;

  ncclResult_t result = ncclSuccess;

  // Stop callbacks before draining per-adapter MRs and COM objects.
  ncclNdStopInterfaceMonitor();
  if (ncclNNdDevs > 0) {
    for (int i = 0; i < ncclNNdDevs; i++) {
      struct ncclNdDev* dev = &ncclNdDevs[i];
      ncclNdLogStats(dev);
      NCCLCHECKIGNORE(ncclNdFinalizeMrCache(dev), result);
      // Release adapter COM object only after all MRs have been drained.
      if (dev->adapter) {
        (void)wrap_nd_release(dev->adapter);
        dev->adapter = NULL;
      }
      dev->devName[0] = '\0';
      dev->pciPath[0] = '\0';
    }
  }
  // Reset counts before unloading so a later init starts from a clean state.
  ncclNNdDevs = -1;
  ncclNMergedNdDevs = -1;
  wrap_nd_unload();
  return result;
}

// Release one plugin context and drop its shared device-state reference.
ncclResult_t ncclNdFinalize(void* ctx) {
  if (ctx) free(ctx);
  return ncclNdFinalizeDevices();
}
