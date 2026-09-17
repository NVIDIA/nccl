/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN's software-atomic proxy. This source deliberately contains the entire
// layering escape hatch: raw public ibverbs for its RC/SRQ plane, dynamic
// GDRCopy for target CPU access, and public NCCL only for bootstrap. It neither
// includes nor inspects GPUNetIO/GDAKI/GIN/NCCL transport-private state.

#include "niin/gpunetio/proxy_atomics/host.h"

#include "niin/context_abi.h"
#include "niin/gpunetio/proxy_atomics/device.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gdrapi.h>
#include <infiniband/verbs.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kBootstrapMagic = 0x4e495053u;  // "NIPS"
constexpr uint16_t kBootstrapVersion = 1;
constexpr uint32_t kMinQueueDepth = 16;
constexpr uint32_t kMaxQueueDepth = 32768;
constexpr uint8_t kQpPsn = 0;
constexpr uint8_t kRetryCount = 7;
constexpr uint8_t kRnrRetryCount = 7;
constexpr uint8_t kAckTimeout = 20;
constexpr uint8_t kMinRnrTimer = 12;
constexpr size_t kCompletionInlineBytes = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t);
constexpr size_t kTicketInlineBytes = sizeof(uint64_t);
constexpr int kCqPollBatch = 16;

struct alignas(64) ProxyReceiveBuffer {
  struct niinGpunetioProxyAtomicRequest request;
};

struct alignas(8) ProxyWire {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved0;
  uint32_t sourcePe;
  uint32_t destinationPe;
  uint32_t qpn;
  uint16_t lid;
  uint8_t gid[16];
  uint8_t linkLayer;
  uint8_t activeMtu;
  uint16_t reserved1;
  uint64_t completionBase;
  uint64_t completionBytes;
  uint32_t completionRkey;
  uint32_t completionStride;
  uint32_t completionOffset;
  uint32_t completionSlots;
  uint64_t heapBytes;
  uint32_t ptrdiffBytes;
  uint32_t reserved2;
};

static_assert(sizeof(ProxyWire) % alignof(uint64_t) == 0,
              "NIIN proxy bootstrap wire must remain 64-bit aligned");

bool isPowerOfTwo(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

int readEnvInt(const char* name, int fallback) {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') return fallback;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max())
    return fallback;
  return static_cast<int>(value);
}

ncclResult_t cudaToNccl(cudaError_t status) {
  return status == cudaSuccess ? ncclSuccess : ncclSystemError;
}

using GdrOpen = gdr_t (*)();
using GdrClose = int (*)(gdr_t);
using GdrPinBuffer = int (*)(gdr_t, unsigned long, size_t, uint64_t, uint32_t, gdr_mh_t*);
using GdrUnpinBuffer = int (*)(gdr_t, gdr_mh_t);
using GdrGetInfo = int (*)(gdr_t, gdr_mh_t, gdr_info_t*);
using GdrMap = int (*)(gdr_t, gdr_mh_t, void**, size_t);
using GdrUnmap = int (*)(gdr_t, gdr_mh_t, void*, size_t);
using GdrRuntimeGetVersion = void (*)(int*, int*);

struct GdrCopyMapping {
  void* library = nullptr;
  gdr_t descriptor = nullptr;
  GdrClose close = nullptr;
  GdrPinBuffer pinBuffer = nullptr;
  GdrUnpinBuffer unpinBuffer = nullptr;
  GdrGetInfo getInfo = nullptr;
  GdrMap map = nullptr;
  GdrUnmap unmap = nullptr;
  gdr_mh_t memoryHandle = {};
  bool pinned = false;
  bool mapped = false;
  void* mappedBase = nullptr;
  size_t mappedBytes = 0;
  void* heapCpuBase = nullptr;
  size_t heapBytes = 0;

  ~GdrCopyMapping() { reset(); }

  void reset() {
    if (mapped && unmap != nullptr) (void)unmap(descriptor, memoryHandle, mappedBase, mappedBytes);
    if (pinned && unpinBuffer != nullptr) (void)unpinBuffer(descriptor, memoryHandle);
    if (descriptor != nullptr && close != nullptr) (void)close(descriptor);
    if (library != nullptr) dlclose(library);
    library = nullptr;
    descriptor = nullptr;
    close = nullptr;
    pinBuffer = nullptr;
    unpinBuffer = nullptr;
    getInfo = nullptr;
    map = nullptr;
    unmap = nullptr;
    memoryHandle = {};
    pinned = false;
    mapped = false;
    mappedBase = nullptr;
    mappedBytes = 0;
    heapCpuBase = nullptr;
    heapBytes = 0;
  }

  template <typename Fn>
  bool loadSymbol(const char* name, Fn* target) {
    *target = reinterpret_cast<Fn>(dlsym(library, name));
    return *target != nullptr;
  }

  bool initialize(const char* requestedLibrary, void* heapBase, size_t bytes) {
    if (heapBase == nullptr || bytes == 0) return false;
    const char* candidates[] = {requestedLibrary, "libgdrapi.so.2", "libgdrapi.so"};
    for (const char* candidate : candidates) {
      if (candidate == nullptr || *candidate == '\0') continue;
      library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
      if (library != nullptr) break;
    }
    if (library == nullptr) return false;

    GdrOpen open = nullptr;
    GdrRuntimeGetVersion runtimeGetVersion = nullptr;
    if (!loadSymbol("gdr_open", &open) || !loadSymbol("gdr_close", &close) ||
        !loadSymbol("gdr_pin_buffer", &pinBuffer) || !loadSymbol("gdr_unpin_buffer", &unpinBuffer) ||
        !loadSymbol("gdr_get_info", &getInfo) || !loadSymbol("gdr_map", &map) ||
        !loadSymbol("gdr_unmap", &unmap) || !loadSymbol("gdr_runtime_get_version", &runtimeGetVersion)) {
      reset();
      return false;
    }
    int runtimeMajor = 0;
    int runtimeMinor = 0;
    runtimeGetVersion(&runtimeMajor, &runtimeMinor);
    if (runtimeMajor != GDR_API_MAJOR_VERSION) {
      std::fprintf(stderr, "NIIN proxy atomics: unsupported GDRCopy runtime %d.%d (need major %d)\n",
                   runtimeMajor, runtimeMinor, GDR_API_MAJOR_VERSION);
      reset();
      return false;
    }
    descriptor = open();
    if (descriptor == nullptr) {
      reset();
      return false;
    }
    if (pinBuffer(descriptor, reinterpret_cast<unsigned long>(heapBase), bytes, 0, 0, &memoryHandle) != 0) {
      reset();
      return false;
    }
    pinned = true;
    if (map(descriptor, memoryHandle, &mappedBase, bytes) != 0 || mappedBase == nullptr) {
      reset();
      return false;
    }
    mapped = true;
    mappedBytes = bytes;
    gdr_info_t info = {};
    if (getInfo(descriptor, memoryHandle, &info) != 0 || reinterpret_cast<uintptr_t>(heapBase) < info.va) {
      reset();
      return false;
    }
    const uint64_t offset = reinterpret_cast<uintptr_t>(heapBase) - info.va;
    if (offset > info.mapped_size || bytes > info.mapped_size - offset ||
        reinterpret_cast<uintptr_t>(mappedBase) > UINTPTR_MAX - offset) {
      reset();
      return false;
    }
    heapCpuBase = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(mappedBase) + offset);
    heapBytes = bytes;
    return true;
  }
};

template <typename T>
T bitsToValue(uint64_t bits) {
  T value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
uint64_t valueToBits(T value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  return bits;
}

float halfBitsToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & UINT16_C(0x8000)) << 16;
  uint32_t exponent = (bits >> 10) & UINT16_C(0x1f);
  uint32_t mantissa = bits & UINT16_C(0x03ff);
  uint32_t result = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      int exp = -14;
      while ((mantissa & UINT32_C(0x0400)) == 0) {
        mantissa <<= 1;
        --exp;
      }
      mantissa &= UINT32_C(0x03ff);
      result = sign | static_cast<uint32_t>(exp + 127) << 23 | mantissa << 13;
    }
  } else if (exponent == 31) {
    result = sign | UINT32_C(0x7f800000) | mantissa << 13;
  } else {
    result = sign | (exponent + 112) << 23 | mantissa << 13;
  }
  return bitsToValue<float>(result);
}

uint16_t floatToHalfBits(float value) {
  const uint32_t bits = static_cast<uint32_t>(valueToBits(value));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & UINT32_C(0x8000));
  const uint32_t exponent = (bits >> 23) & UINT32_C(0xff);
  uint32_t mantissa = bits & UINT32_C(0x007fffff);
  if (exponent == 255) {
    if (mantissa == 0) return static_cast<uint16_t>(sign | UINT16_C(0x7c00));
    // Preserve a nonzero payload so a NaN remains a NaN after narrowing.
    return static_cast<uint16_t>(sign | UINT16_C(0x7c00) |
                                 std::max<uint32_t>(UINT32_C(1), mantissa >> 13));
  }
  int halfExponent = static_cast<int>(exponent) - 127 + 15;
  if (halfExponent >= 31) return static_cast<uint16_t>(sign | UINT16_C(0x7c00));
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    mantissa |= UINT32_C(0x00800000);
    const uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
    const uint32_t rounded = (mantissa + (UINT32_C(1) << (shift - 1)) +
                              ((mantissa >> shift) & UINT32_C(1))) >> shift;
    return static_cast<uint16_t>(sign | rounded);
  }
  const uint32_t rounded = (mantissa + UINT32_C(0x00001000) +
                            ((mantissa >> 13) & UINT32_C(1))) >> 13;
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExponent) << 10) | rounded);
}

}  // namespace

struct niinGpunetioProxyAtomicHostContext {
  ncclComm_t comm = nullptr;
  int rank = -1;
  int nPes = 0;
  int port = 0;
  int gidIndex = 0;
  uint16_t pkeyIndex = 0;
  uint8_t serviceLevel = 0;
  struct ibv_context* verbsContext = nullptr;
  struct ibv_pd* pd = nullptr;
  struct ibv_port_attr portAttr = {};
  union ibv_gid localGid = {};
  struct ibv_cq* sendCq = nullptr;
  struct ibv_cq* receiveCq = nullptr;
  struct ibv_srq* srq = nullptr;
  struct ibv_mr* receiveMr = nullptr;
  struct ibv_mr* slotsMr = nullptr;
  std::vector<struct ibv_qp*> qps;
  std::vector<struct ProxyWire> peers;
  std::vector<struct ProxyReceiveBuffer> receiveBuffers;
  struct niinGpunetioProxyAtomicSlot* slotsHost = nullptr;
  struct niinGpunetioProxyAtomicSlot* slotsDevice = nullptr;
  uint32_t* progressFailureHost = nullptr;
  uint32_t* progressFailureDevice = nullptr;
  size_t slotCount = 0;
  uint32_t slotsPerPe = 1;
  std::vector<uint64_t> lastTickets;
  int* endpointLocksDevice = nullptr;
  struct niinGpunetioProxyAtomicDeviceContext proxyContextHost = {};
  struct niinGpunetioProxyAtomicDeviceContext* proxyContextDevice = nullptr;
  struct niinGpunetioAtomicContext atomicContextHost = {};
  struct niinGpunetioAtomicContext* atomicContextDevice = nullptr;
  struct niinContext* boundDeviceContext = nullptr;
  struct GdrCopyMapping gdr;
  std::atomic<bool> stopProgress{false};
  std::atomic<bool> progressFailed{false};
  std::thread progressThread;
  std::mutex progressMutex;

  ~niinGpunetioProxyAtomicHostContext() { cleanup(); }

  void markProgressFailed() {
    progressFailed.store(true, std::memory_order_release);
    if (progressFailureHost != nullptr)
      __atomic_store_n(progressFailureHost, UINT32_C(1), __ATOMIC_RELEASE);
  }

  void cleanup() {
    // In normal use Finalize is called after kernel completion. Publishing an
    // error first also makes an accidental late device waiter fail closed
    // before the mapped proxy state is released.
    markProgressFailed();
    stopProgress.store(true, std::memory_order_release);
    if (progressThread.joinable()) progressThread.join();
    if (boundDeviceContext != nullptr) {
      const struct niinGpunetioAtomicContext* currentContext = nullptr;
      const struct niinGpunetioAtomicContext* nullContext = nullptr;
      // Do not clear a provider that was explicitly installed after this
      // proxy. The normal bind API rejects that misuse, but match the direct
      // provider's teardown rule so cleanup cannot erase unrelated state.
      const cudaError_t readStatus = cudaMemcpy(
          &currentContext,
          reinterpret_cast<const char*>(boundDeviceContext) +
              offsetof(struct niinContext, gpunetioAtomicContext),
          sizeof(currentContext), cudaMemcpyDeviceToHost);
      if (readStatus == cudaSuccess && currentContext == atomicContextDevice) {
        (void)cudaMemcpy(reinterpret_cast<char*>(boundDeviceContext) +
                             offsetof(struct niinContext, gpunetioAtomicContext),
                         &nullContext, sizeof(nullContext), cudaMemcpyHostToDevice);
      }
      boundDeviceContext = nullptr;
    }
    if (atomicContextDevice != nullptr) cudaFree(atomicContextDevice);
    if (proxyContextDevice != nullptr) cudaFree(proxyContextDevice);
    if (endpointLocksDevice != nullptr) cudaFree(endpointLocksDevice);
    atomicContextDevice = nullptr;
    proxyContextDevice = nullptr;
    endpointLocksDevice = nullptr;
    for (struct ibv_qp* qp : qps) {
      if (qp != nullptr) ibv_destroy_qp(qp);
    }
    qps.clear();
    if (srq != nullptr) ibv_destroy_srq(srq);
    if (slotsMr != nullptr) ibv_dereg_mr(slotsMr);
    if (receiveMr != nullptr) ibv_dereg_mr(receiveMr);
    if (sendCq != nullptr) ibv_destroy_cq(sendCq);
    if (receiveCq != nullptr) ibv_destroy_cq(receiveCq);
    srq = nullptr;
    slotsMr = nullptr;
    receiveMr = nullptr;
    sendCq = nullptr;
    receiveCq = nullptr;
    if (slotsHost != nullptr) cudaFreeHost(slotsHost);
    if (progressFailureHost != nullptr) cudaFreeHost(progressFailureHost);
    slotsHost = nullptr;
    slotsDevice = nullptr;
    progressFailureHost = nullptr;
    progressFailureDevice = nullptr;
    receiveBuffers.clear();
    lastTickets.clear();
    peers.clear();
    gdr.reset();
    if (pd != nullptr) ibv_dealloc_pd(pd);
    if (verbsContext != nullptr) ibv_close_device(verbsContext);
    pd = nullptr;
    verbsContext = nullptr;
  }
};

namespace {

ncclResult_t openVerbsDevice(niinGpunetioProxyAtomicHostContext* provider,
                             const struct niinGpunetioProxyAtomicOptions& options) {
  const char* requestedDevice = options.ibDevice;
  if (requestedDevice == nullptr || *requestedDevice == '\0')
    requestedDevice = std::getenv("NIIN_GPUNETIO_PROXY_IB_DEV");
  int deviceCount = 0;
  struct ibv_device** devices = ibv_get_device_list(&deviceCount);
  if (devices == nullptr || deviceCount <= 0) return ncclSystemError;

  for (int index = 0; index < deviceCount && provider->verbsContext == nullptr; ++index) {
    if (requestedDevice != nullptr && *requestedDevice != '\0' &&
        std::strcmp(ibv_get_device_name(devices[index]), requestedDevice) != 0)
      continue;
    struct ibv_context* context = ibv_open_device(devices[index]);
    if (context == nullptr) continue;
    const int firstPort = options.ibPort > 0 ? options.ibPort : 1;
    const int lastPort = options.ibPort > 0 ? options.ibPort : 255;
    struct ibv_port_attr portAttr = {};
    int selectedPort = 0;
    for (int port = firstPort; port <= lastPort; ++port) {
      if (ibv_query_port(context, static_cast<uint8_t>(port), &portAttr) != 0) {
        if (options.ibPort > 0) break;
        continue;
      }
      if (portAttr.state == IBV_PORT_ACTIVE) {
        selectedPort = port;
        break;
      }
      if (options.ibPort > 0) break;
    }
    if (selectedPort == 0) {
      ibv_close_device(context);
      continue;
    }
    provider->verbsContext = context;
    provider->port = selectedPort;
    provider->portAttr = portAttr;
  }
  ibv_free_device_list(devices);
  if (provider->verbsContext == nullptr) return ncclInvalidUsage;

  provider->gidIndex = options.gidIndex >= 0
                           ? options.gidIndex
                           : readEnvInt("NIIN_GPUNETIO_PROXY_GID_INDEX", 0);
  if (provider->gidIndex < 0 ||
      ibv_query_gid(provider->verbsContext, static_cast<uint8_t>(provider->port), provider->gidIndex,
                    &provider->localGid) != 0)
    return ncclInvalidUsage;
  provider->pd = ibv_alloc_pd(provider->verbsContext);
  return provider->pd == nullptr ? ncclSystemError : ncclSuccess;
}

ncclResult_t createMappedSlots(niinGpunetioProxyAtomicHostContext* provider) {
  if (provider->nPes <= 0 || provider->slotsPerPe == 0 ||
      static_cast<size_t>(provider->nPes) >
          std::numeric_limits<size_t>::max() / provider->slotsPerPe)
    return ncclInvalidArgument;
  provider->slotCount = static_cast<size_t>(provider->nPes) * provider->slotsPerPe;
  if (provider->slotCount > std::numeric_limits<size_t>::max() / sizeof(*provider->slotsHost))
    return ncclInvalidArgument;
  const size_t slotBytes = provider->slotCount * sizeof(*provider->slotsHost);
  cudaError_t cudaStatus = cudaHostAlloc(reinterpret_cast<void**>(&provider->slotsHost), slotBytes,
                                         cudaHostAllocMapped | cudaHostAllocPortable);
  if (cudaStatus != cudaSuccess) return ncclSystemError;
  std::memset(provider->slotsHost, 0, slotBytes);
  cudaStatus = cudaHostGetDevicePointer(reinterpret_cast<void**>(&provider->slotsDevice), provider->slotsHost, 0);
  if (cudaStatus != cudaSuccess) return ncclSystemError;
  cudaStatus = cudaHostAlloc(reinterpret_cast<void**>(&provider->progressFailureHost),
                             sizeof(*provider->progressFailureHost),
                             cudaHostAllocMapped | cudaHostAllocPortable);
  if (cudaStatus != cudaSuccess) return ncclSystemError;
  *provider->progressFailureHost = 0;
  cudaStatus = cudaHostGetDevicePointer(reinterpret_cast<void**>(&provider->progressFailureDevice),
                                        provider->progressFailureHost, 0);
  if (cudaStatus != cudaSuccess) return ncclSystemError;
  provider->slotsMr = ibv_reg_mr(provider->pd, provider->slotsHost, slotBytes,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (provider->slotsMr == nullptr) return ncclSystemError;
  cudaStatus = cudaMalloc(&provider->endpointLocksDevice, provider->nPes * sizeof(int));
  if (cudaStatus != cudaSuccess) return ncclSystemError;
  cudaStatus = cudaMemset(provider->endpointLocksDevice, 0, provider->nPes * sizeof(int));
  if (cudaStatus != cudaSuccess) return ncclSystemError;
  provider->lastTickets.assign(provider->slotCount, 0);
  return ncclSuccess;
}

int postReceive(niinGpunetioProxyAtomicHostContext* provider, struct ProxyReceiveBuffer* buffer) {
  struct ibv_sge sge = {};
  sge.addr = reinterpret_cast<uintptr_t>(&buffer->request);
  sge.length = sizeof(buffer->request);
  sge.lkey = provider->receiveMr->lkey;
  struct ibv_recv_wr wr = {};
  wr.wr_id = reinterpret_cast<uintptr_t>(buffer);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  struct ibv_recv_wr* bad = nullptr;
  return ibv_post_srq_recv(provider->srq, &wr, &bad);
}

ncclResult_t createSrqResources(niinGpunetioProxyAtomicHostContext* provider,
                                 const struct niinGpunetioProxyAtomicOptions& options) {
  const uint64_t peerCount = provider->nPes > 0 ? static_cast<uint64_t>(provider->nPes - 1) : 0;
  // The send CQ is shared by every RC QP. With one leased source slot per
  // destination, an all-peer burst can have one signaled SEND and one
  // signaled completion-ticket write outstanding per peer at once.
  const uint64_t sendCqEntries = std::max<uint64_t>(UINT64_C(32), 2 * peerCount + 8);
  if (options.srqDepth > static_cast<uint32_t>(INT_MAX - 8) ||
      sendCqEntries > static_cast<uint64_t>(INT_MAX))
    return ncclInvalidArgument;
  provider->receiveCq = ibv_create_cq(provider->verbsContext, static_cast<int>(options.srqDepth + 8),
                                      nullptr, nullptr, 0);
  provider->sendCq = ibv_create_cq(provider->verbsContext, static_cast<int>(sendCqEntries), nullptr,
                                   nullptr, 0);
  if (provider->receiveCq == nullptr || provider->sendCq == nullptr) return ncclSystemError;
  struct ibv_srq_init_attr srqAttr = {};
  srqAttr.attr.max_wr = options.srqDepth;
  srqAttr.attr.max_sge = 1;
  provider->srq = ibv_create_srq(provider->pd, &srqAttr);
  if (provider->srq == nullptr) return ncclSystemError;
  provider->receiveBuffers.resize(options.srqDepth);
  provider->receiveMr = ibv_reg_mr(provider->pd, provider->receiveBuffers.data(),
                                    provider->receiveBuffers.size() * sizeof(provider->receiveBuffers[0]),
                                    IBV_ACCESS_LOCAL_WRITE);
  if (provider->receiveMr == nullptr) return ncclSystemError;
  for (struct ProxyReceiveBuffer& buffer : provider->receiveBuffers) {
    if (postReceive(provider, &buffer) != 0) return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t transitionQpInit(niinGpunetioProxyAtomicHostContext* provider, struct ibv_qp* qp) {
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = provider->pkeyIndex;
  attr.port_num = static_cast<uint8_t>(provider->port);
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  const int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  return ibv_modify_qp(qp, &attr, mask) == 0 ? ncclSuccess : ncclSystemError;
}

ncclResult_t createPeerQps(niinGpunetioProxyAtomicHostContext* provider,
                           const struct niinGpunetioProxyAtomicOptions& options,
                           std::vector<struct ProxyWire>* local) {
  provider->qps.assign(provider->nPes, nullptr);
  local->assign(provider->nPes, {});
  for (int peer = 0; peer < provider->nPes; ++peer) {
    struct ProxyWire& wire = (*local)[peer];
    wire.magic = kBootstrapMagic;
    wire.version = kBootstrapVersion;
    wire.sourcePe = provider->rank;
    wire.destinationPe = peer;
    wire.lid = provider->portAttr.lid;
    std::memcpy(wire.gid, provider->localGid.raw, sizeof(wire.gid));
    wire.linkLayer = static_cast<uint8_t>(provider->portAttr.link_layer);
    wire.activeMtu = static_cast<uint8_t>(provider->portAttr.active_mtu);
    wire.completionBase = reinterpret_cast<uintptr_t>(provider->slotsHost);
    wire.completionBytes = provider->slotCount * sizeof(*provider->slotsHost);
    wire.completionRkey = provider->slotsMr->rkey;
    wire.completionStride = sizeof(*provider->slotsHost);
    wire.completionOffset = offsetof(struct niinGpunetioProxyAtomicSlot, completion);
    wire.completionSlots = provider->slotCount;
    wire.heapBytes = provider->gdr.heapBytes;
    wire.ptrdiffBytes = sizeof(ptrdiff_t);
    if (peer == provider->rank) continue;

    struct ibv_qp_init_attr init = {};
    init.send_cq = provider->sendCq;
    init.recv_cq = provider->receiveCq;
    init.srq = provider->srq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = options.sendQueueDepth;
    init.cap.max_recv_wr = 0;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = sizeof(struct niinGpunetioProxyAtomicCompletion);
    struct ibv_qp* qp = ibv_create_qp(provider->pd, &init);
    if (qp == nullptr) return ncclSystemError;
    if (init.cap.max_inline_data < kCompletionInlineBytes ||
        init.cap.max_inline_data < kTicketInlineBytes) {
      ibv_destroy_qp(qp);
      return ncclSystemError;
    }
    provider->qps[peer] = qp;
    ncclResult_t status = transitionQpInit(provider, qp);
    if (status != ncclSuccess) return status;
    wire.qpn = qp->qp_num;
  }
  return ncclSuccess;
}

ncclResult_t allGatherSetupStatus(ncclComm_t comm, int nPes, ncclResult_t localStatus) {
  if (nPes <= 0) return ncclInvalidArgument;
  uint32_t local = static_cast<uint32_t>(localStatus);
  std::vector<uint32_t> remote(nPes, static_cast<uint32_t>(ncclSystemError));
  void* send = nullptr;
  void* recv = nullptr;
  cudaStream_t stream = nullptr;
  ncclResult_t result = cudaToNccl(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&send, sizeof(local)));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&recv, remote.size() * sizeof(remote[0])));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMemcpyAsync(send, &local, sizeof(local), cudaMemcpyHostToDevice, stream));
  if (result != ncclSuccess) goto cleanup;
  result = ncclAllGather(send, recv, sizeof(local), ncclUint8, comm, stream);
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMemcpyAsync(remote.data(), recv, remote.size() * sizeof(remote[0]),
                                      cudaMemcpyDeviceToHost, stream));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaStreamSynchronize(stream));
  if (result != ncclSuccess) goto cleanup;
  for (uint32_t status : remote) {
    if (status != static_cast<uint32_t>(ncclSuccess)) {
      result = localStatus == ncclSuccess ? ncclSystemError : localStatus;
      goto cleanup;
    }
  }
  result = ncclSuccess;
cleanup:
  if (recv != nullptr) cudaFree(recv);
  if (send != nullptr) cudaFree(send);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return result;
}

ncclResult_t allGatherWire(ncclComm_t comm, const std::vector<struct ProxyWire>& local, int nPes,
                           std::vector<struct ProxyWire>* remote) {
  if (local.size() != static_cast<size_t>(nPes)) return ncclInvalidArgument;
  const size_t localBytes = local.size() * sizeof(local[0]);
  if (nPes <= 0 || localBytes > std::numeric_limits<size_t>::max() / nPes) return ncclInvalidArgument;
  void* send = nullptr;
  void* recv = nullptr;
  cudaStream_t stream = nullptr;
  ncclResult_t result = cudaToNccl(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&send, localBytes));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&recv, localBytes * static_cast<size_t>(nPes)));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMemcpyAsync(send, local.data(), localBytes, cudaMemcpyHostToDevice, stream));
  if (result != ncclSuccess) goto cleanup;
  result = ncclAllGather(send, recv, localBytes, ncclUint8, comm, stream);
  if (result != ncclSuccess) goto cleanup;
  remote->resize(static_cast<size_t>(nPes) * nPes);
  result = cudaToNccl(cudaMemcpyAsync(remote->data(), recv, remote->size() * sizeof((*remote)[0]),
                                      cudaMemcpyDeviceToHost, stream));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaStreamSynchronize(stream));
cleanup:
  if (recv != nullptr) cudaFree(recv);
  if (send != nullptr) cudaFree(send);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return result;
}

bool remoteCompletionLayoutIsValid(const niinGpunetioProxyAtomicHostContext* provider,
                                   const struct ProxyWire& remote) {
  constexpr uint64_t slotBytes = sizeof(struct niinGpunetioProxyAtomicSlot);
  constexpr uint64_t completionOffset = offsetof(struct niinGpunetioProxyAtomicSlot, completion);
  if (provider->slotCount > UINT64_MAX / slotBytes) return false;
  const uint64_t expectedSlots = static_cast<uint64_t>(provider->slotCount);
  const uint64_t expectedBytes = expectedSlots * slotBytes;
  return remote.completionBase != 0 && remote.completionRkey != 0 &&
         (remote.completionBase % alignof(struct niinGpunetioProxyAtomicSlot)) == 0 &&
         remote.completionStride == slotBytes && remote.completionOffset == completionOffset &&
         remote.completionSlots == expectedSlots && remote.completionBytes == expectedBytes &&
         remote.completionBase <= UINT64_MAX - remote.completionBytes;
}

ncclResult_t connectPeerQp(niinGpunetioProxyAtomicHostContext* provider, int peer,
                           const struct ProxyWire& remote) {
  if (peer == provider->rank || peer < 0 || peer >= provider->nPes || provider->qps[peer] == nullptr ||
      remote.magic != kBootstrapMagic || remote.version != kBootstrapVersion ||
      remote.sourcePe != static_cast<uint32_t>(peer) || remote.destinationPe != static_cast<uint32_t>(provider->rank) ||
      remote.qpn == 0 || remote.linkLayer != static_cast<uint8_t>(provider->portAttr.link_layer) ||
      !remoteCompletionLayoutIsValid(provider, remote) || remote.heapBytes != provider->gdr.heapBytes ||
      remote.ptrdiffBytes != sizeof(ptrdiff_t))
    return ncclInvalidUsage;

  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = static_cast<enum ibv_mtu>(std::min<int>(provider->portAttr.active_mtu, remote.activeMtu));
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = kQpPsn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = kMinRnrTimer;
  attr.ah_attr.dlid = remote.lid;
  attr.ah_attr.sl = provider->serviceLevel;
  attr.ah_attr.port_num = static_cast<uint8_t>(provider->port);
  const bool isIb = provider->portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND;
  const bool sameIbSubnet =
      isIb && std::memcmp(&provider->localGid.global.subnet_prefix, remote.gid,
                          sizeof(provider->localGid.global.subnet_prefix)) == 0;
  attr.ah_attr.is_global = !isIb || !sameIbSubnet || (provider->portAttr.flags & IBV_QPF_GRH_REQUIRED) != 0;
  if (attr.ah_attr.is_global) {
    std::memcpy(&attr.ah_attr.grh.dgid, remote.gid, sizeof(remote.gid));
    attr.ah_attr.grh.sgid_index = provider->gidIndex;
    attr.ah_attr.grh.hop_limit = 255;
  }
  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
             IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(provider->qps[peer], &attr, mask) != 0) return ncclSystemError;
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = kQpPsn;
  attr.timeout = kAckTimeout;
  attr.retry_cnt = kRetryCount;
  attr.rnr_retry = kRnrRetryCount;
  attr.max_rd_atomic = 1;
  mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
         IBV_QP_MAX_QP_RD_ATOMIC;
  return ibv_modify_qp(provider->qps[peer], &attr, mask) == 0 ? ncclSuccess : ncclSystemError;
}

ncclResult_t exportDeviceContext(niinGpunetioProxyAtomicHostContext* provider) {
  provider->proxyContextHost.version = NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_VERSION;
  provider->proxyContextHost.flags = NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_READY |
                                    NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_PROXY_ALL;
  provider->proxyContextHost.nPes = provider->nPes;
  provider->proxyContextHost.localPe = provider->rank;
  provider->proxyContextHost.heapBytes = provider->gdr.heapBytes;
  provider->proxyContextHost.slotsPerPe = provider->slotsPerPe;
  provider->proxyContextHost.slots = provider->slotsDevice;
  provider->proxyContextHost.endpointLocks = provider->endpointLocksDevice;
  provider->proxyContextHost.progressFailure = provider->progressFailureDevice;
  ncclResult_t result = cudaToNccl(cudaMalloc(&provider->proxyContextDevice, sizeof(provider->proxyContextHost)));
  if (result != ncclSuccess) return result;
  result = cudaToNccl(cudaMemcpy(provider->proxyContextDevice, &provider->proxyContextHost,
                                 sizeof(provider->proxyContextHost), cudaMemcpyHostToDevice));
  if (result != ncclSuccess) return result;

  provider->atomicContextHost.version = NIIN_GPUNETIO_ATOMIC_CONTEXT_VERSION;
  provider->atomicContextHost.flags = NIIN_GPUNETIO_ATOMIC_CONTEXT_READY | NIIN_GPUNETIO_ATOMIC_CONTEXT_PROXY;
  provider->atomicContextHost.nPes = provider->nPes;
  provider->atomicContextHost.localPe = provider->rank;
  provider->atomicContextHost.proxyContext = provider->proxyContextDevice;
  result = cudaToNccl(cudaMalloc(&provider->atomicContextDevice, sizeof(provider->atomicContextHost)));
  if (result != ncclSuccess) return result;
  return cudaToNccl(cudaMemcpy(provider->atomicContextDevice, &provider->atomicContextHost,
                               sizeof(provider->atomicContextHost), cudaMemcpyHostToDevice));
}

void publishLocalCompletion(struct niinGpunetioProxyAtomicSlot* slot, uint64_t result, uint32_t status,
                            uint64_t ticket) {
  slot->completion.resultBits = result;
  slot->completion.status = status;
  slot->completion.reserved = 0;
  __atomic_thread_fence(__ATOMIC_RELEASE);
  __atomic_store_n(&slot->completion.ticket, ticket, __ATOMIC_RELEASE);
}

bool executeIntegral(void* address, uint8_t type, const struct niinGpunetioProxyAtomicRequest& request,
                     uint64_t* result, uint32_t* status) {
  if (type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL32) {
    volatile uint32_t* target = static_cast<volatile uint32_t*>(address);
    const uint32_t old = *target;
    const uint32_t operand = static_cast<uint32_t>(request.operandBits);
    const uint32_t compare = static_cast<uint32_t>(request.compareBits);
    uint32_t next = old;
    bool write = true;
    switch (request.op) {
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD: next = old + operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_COMPARE_SWAP: next = old == compare ? operand : old; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET: next = operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH: write = false; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_INC:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_INC: next = old + 1; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_AND:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_AND: next = old & operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_OR:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_OR: next = old | operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_XOR:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_XOR: next = old ^ operand; break;
      default: *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED; return false;
    }
    if (write) *target = next;
    *result = old;
  } else if (type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL64) {
    volatile uint64_t* target = static_cast<volatile uint64_t*>(address);
    const uint64_t old = *target;
    const uint64_t operand = request.operandBits;
    const uint64_t compare = request.compareBits;
    uint64_t next = old;
    bool write = true;
    switch (request.op) {
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD: next = old + operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_COMPARE_SWAP: next = old == compare ? operand : old; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET: next = operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH: write = false; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_INC:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_INC: next = old + 1; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_AND:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_AND: next = old & operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_OR:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_OR: next = old | operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_XOR:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_XOR: next = old ^ operand; break;
      default: *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED; return false;
    }
    if (write) *target = next;
    *result = old;
  } else {
    *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED;
    return false;
  }
  __sync_synchronize();
  *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_SUCCESS;
  return true;
}

bool executeFloating(void* address, uint8_t type, const struct niinGpunetioProxyAtomicRequest& request,
                     uint64_t* result, uint32_t* status) {
  if (type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT32) {
    volatile float* target = static_cast<volatile float*>(address);
    const float old = *target;
    const float operand = bitsToValue<float>(request.operandBits);
    float next = old;
    bool write = true;
    switch (request.op) {
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD: next = old + operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET: next = operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH: write = false; break;
      default: *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED; return false;
    }
    if (write) *target = next;
    *result = valueToBits(old);
  } else if (type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT64) {
    volatile double* target = static_cast<volatile double*>(address);
    const double old = *target;
    const double operand = bitsToValue<double>(request.operandBits);
    double next = old;
    bool write = true;
    switch (request.op) {
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD: next = old + operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP:
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET: next = operand; break;
      case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH: write = false; break;
      default: *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED; return false;
    }
    if (write) *target = next;
    *result = valueToBits(old);
  } else if (type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT16) {
    volatile uint16_t* target = static_cast<volatile uint16_t*>(address);
    const uint16_t old = *target;
    if (request.op != NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD &&
        request.op != NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD) {
      *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED;
      return false;
    }
    const float sum = halfBitsToFloat(old) + halfBitsToFloat(static_cast<uint16_t>(request.operandBits));
    *target = floatToHalfBits(sum);
    *result = old;
  } else {
    *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED;
    return false;
  }
  __sync_synchronize();
  *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_SUCCESS;
  return true;
}

bool executeRequest(niinGpunetioProxyAtomicHostContext* provider,
                    const struct niinGpunetioProxyAtomicRequest& request, uint64_t* result,
                    uint32_t* status) {
  if (!niinGpunetioProxyAtomicRequestIsWellFormed(request) ||
      request.targetPe != static_cast<uint32_t>(provider->rank)) {
    *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_BAD_REQUEST;
    return false;
  }
  const size_t bytes = niinGpunetioProxyAtomicElementBytes(request.type);
  if (request.remoteOffset > provider->gdr.heapBytes || bytes > provider->gdr.heapBytes - request.remoteOffset) {
    *status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_BAD_ADDRESS;
    return false;
  }
  void* address = static_cast<char*>(provider->gdr.heapCpuBase) + request.remoteOffset;
  if (niinGpunetioProxyAtomicTypeIsIntegral(request.type))
    return executeIntegral(address, request.type, request, result, status);
  return executeFloating(address, request.type, request, result, status);
}

bool postRequest(niinGpunetioProxyAtomicHostContext* provider,
                 struct niinGpunetioProxyAtomicSlot* slot, int targetPe) {
  if (targetPe < 0 || targetPe >= provider->nPes || targetPe == provider->rank ||
      provider->qps[targetPe] == nullptr)
    return false;
  struct ibv_sge sge = {};
  sge.addr = reinterpret_cast<uintptr_t>(&slot->request);
  sge.length = sizeof(slot->request);
  sge.lkey = provider->slotsMr->lkey;
  struct ibv_send_wr wr = {};
  wr.wr_id = reinterpret_cast<uintptr_t>(slot);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  struct ibv_send_wr* bad = nullptr;
  return ibv_post_send(provider->qps[targetPe], &wr, &bad) == 0;
}

bool postRemoteCompletion(niinGpunetioProxyAtomicHostContext* provider, int sourcePe,
                          uint32_t sourceSlot, uint64_t result, uint32_t status, uint64_t ticket) {
  if (sourcePe < 0 || sourcePe >= provider->nPes || sourcePe == provider->rank ||
      provider->qps[sourcePe] == nullptr || sourcePe >= static_cast<int>(provider->peers.size()))
    return false;
  const struct ProxyWire& peer = provider->peers[sourcePe];
  if (!remoteCompletionLayoutIsValid(provider, peer) || sourceSlot >= peer.completionSlots)
    return false;
  const uint64_t slotOffset = static_cast<uint64_t>(sourceSlot) * peer.completionStride;
  if (slotOffset > peer.completionBytes || peer.completionOffset > peer.completionBytes - slotOffset ||
      sizeof(struct niinGpunetioProxyAtomicCompletion) >
          peer.completionBytes - slotOffset - peer.completionOffset ||
      peer.completionBase > UINT64_MAX - slotOffset - peer.completionOffset)
    return false;
  const uint64_t remoteBase = peer.completionBase + slotOffset + peer.completionOffset;
  if (remoteBase > UINT64_MAX - sizeof(uint64_t)) return false;
  struct {
    uint64_t resultBits;
    uint32_t completionStatus;
    uint32_t reserved;
  } body = {result, status, 0};
  struct ibv_sge bodySge = {};
  bodySge.addr = reinterpret_cast<uintptr_t>(&body);
  bodySge.length = sizeof(body);
  struct ibv_sge ticketSge = {};
  ticketSge.addr = reinterpret_cast<uintptr_t>(&ticket);
  ticketSge.length = sizeof(ticket);
  struct ibv_send_wr bodyWr = {};
  bodyWr.wr_id = static_cast<uintptr_t>(NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH);
  bodyWr.sg_list = &bodySge;
  bodyWr.num_sge = 1;
  bodyWr.opcode = IBV_WR_RDMA_WRITE;
  bodyWr.send_flags = IBV_SEND_INLINE;
  bodyWr.wr.rdma.remote_addr = remoteBase;
  bodyWr.wr.rdma.rkey = peer.completionRkey;
  struct ibv_send_wr ticketWr = {};
  ticketWr.wr_id = static_cast<uintptr_t>(NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH);
  ticketWr.sg_list = &ticketSge;
  ticketWr.num_sge = 1;
  ticketWr.opcode = IBV_WR_RDMA_WRITE;
  ticketWr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  ticketWr.wr.rdma.remote_addr = remoteBase + niinGpunetioProxyAtomicCompletionBodyBytes();
  ticketWr.wr.rdma.rkey = peer.completionRkey;
  bodyWr.next = &ticketWr;
  struct ibv_send_wr* bad = nullptr;
  return ibv_post_send(provider->qps[sourcePe], &bodyWr, &bad) == 0;
}

bool processLocalSlots(niinGpunetioProxyAtomicHostContext* provider, bool* didWork) {
  for (size_t index = 0; index < provider->slotCount; ++index) {
    struct niinGpunetioProxyAtomicSlot* slot = provider->slotsHost + index;
    const uint64_t ticket = __atomic_load_n(&slot->request.ticket, __ATOMIC_ACQUIRE);
    if (ticket == NIIN_GPUNETIO_PROXY_ATOMIC_UNPUBLISHED_TICKET || ticket == provider->lastTickets[index]) continue;
    const struct niinGpunetioProxyAtomicRequest request = slot->request;
    provider->lastTickets[index] = ticket;
    *didWork = true;
    uint64_t result = 0;
    uint32_t status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_BAD_REQUEST;
    if (request.ticket != ticket || request.sourcePe != static_cast<uint32_t>(provider->rank) ||
        request.sourceSlot != index || request.targetPe >= static_cast<uint32_t>(provider->nPes) ||
        !niinGpunetioProxyAtomicRequestIsWellFormed(request)) {
      publishLocalCompletion(slot, result, status, ticket);
      continue;
    }
    if (request.targetPe == static_cast<uint32_t>(provider->rank)) {
      (void)executeRequest(provider, request, &result, &status);
      publishLocalCompletion(slot, result, status, ticket);
    } else if (!postRequest(provider, slot, static_cast<int>(request.targetPe))) {
      publishLocalCompletion(slot, result, NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_INTERNAL_ERROR, ticket);
      provider->markProgressFailed();
    }
  }
  return true;
}

int peerFromReceiveQpn(const niinGpunetioProxyAtomicHostContext* provider, uint32_t qpNumber) {
  for (int peer = 0; peer < provider->nPes; ++peer) {
    if (peer != provider->rank && provider->qps[peer] != nullptr &&
        provider->qps[peer]->qp_num == qpNumber)
      return peer;
  }
  return -1;
}

bool processReceiveCq(niinGpunetioProxyAtomicHostContext* provider, bool* didWork) {
  struct ibv_wc completions[kCqPollBatch];
  // Do not drain this CQ indefinitely: each received request posts a
  // signaled completion-ticket write onto the shared send CQ. Returning after
  // one bounded batch lets progressOnce() drain that CQ before a sustained
  // stream can overrun it.
  const int count = ibv_poll_cq(provider->receiveCq, kCqPollBatch, completions);
  if (count < 0) return false;
  for (int index = 0; index < count; ++index) {
    struct ibv_wc& completion = completions[index];
    auto* buffer = reinterpret_cast<struct ProxyReceiveBuffer*>(completion.wr_id);
    if (buffer == nullptr) return false;
    *didWork = true;
    const int sourcePe = peerFromReceiveQpn(provider, completion.qp_num);
    if (completion.status != IBV_WC_SUCCESS || completion.opcode != IBV_WC_RECV ||
        completion.byte_len != sizeof(struct niinGpunetioProxyAtomicRequest) || sourcePe < 0) {
      return false;
    }
    const struct niinGpunetioProxyAtomicRequest request = buffer->request;
    const uint64_t expectedSlot64 =
        static_cast<uint64_t>(provider->rank) * static_cast<uint64_t>(provider->slotsPerPe);
    if (expectedSlot64 > UINT32_MAX) return false;
    const uint32_t expectedSlot = static_cast<uint32_t>(expectedSlot64);
    uint64_t result = 0;
    uint32_t status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_BAD_REQUEST;
    if (request.sourcePe == static_cast<uint32_t>(sourcePe) &&
        request.targetPe == static_cast<uint32_t>(provider->rank) && request.sourceSlot == expectedSlot &&
        request.ticket != NIIN_GPUNETIO_PROXY_ATOMIC_UNPUBLISHED_TICKET) {
      (void)executeRequest(provider, request, &result, &status);
    }
    // Complete to the peer identified by the trusted local QP number, not a
    // request-supplied source identity. A malformed but ticketed request
    // receives BAD_REQUEST on the slot where its sender is waiting.
    if (request.ticket == NIIN_GPUNETIO_PROXY_ATOMIC_UNPUBLISHED_TICKET ||
        !postRemoteCompletion(provider, sourcePe, expectedSlot, result, status, request.ticket))
      return false;
    if (postReceive(provider, buffer) != 0) return false;
  }
  return true;
}

bool processSendCq(niinGpunetioProxyAtomicHostContext* provider, bool* didWork) {
  struct ibv_wc completions[kCqPollBatch];
  for (;;) {
    const int count = ibv_poll_cq(provider->sendCq, kCqPollBatch, completions);
    if (count < 0) return false;
    if (count == 0) return true;
    *didWork = true;
    for (int index = 0; index < count; ++index) {
      if (completions[index].status != IBV_WC_SUCCESS) return false;
    }
  }
}

ncclResult_t progressOnce(niinGpunetioProxyAtomicHostContext* provider, bool* didWork = nullptr) {
  std::lock_guard<std::mutex> guard(provider->progressMutex);
  bool progressed = false;
  if (!processSendCq(provider, &progressed) || !processReceiveCq(provider, &progressed) ||
      !processLocalSlots(provider, &progressed)) {
    provider->markProgressFailed();
    return ncclSystemError;
  }
  if (didWork != nullptr) *didWork = progressed;
  return ncclSuccess;
}

void startProgressThread(niinGpunetioProxyAtomicHostContext* provider) {
  provider->stopProgress.store(false, std::memory_order_release);
  provider->progressThread = std::thread([provider] {
    while (!provider->stopProgress.load(std::memory_order_acquire)) {
      bool progressed = false;
      if (progressOnce(provider, &progressed) != ncclSuccess) break;
      if (!progressed) std::this_thread::yield();
    }
  });
}

}  // namespace

ncclResult_t niinGpunetioProxyAtomicInit(
    ncclComm_t comm, void* heapBase, size_t heapBytes,
    const struct niinGpunetioProxyAtomicOptions* suppliedOptions,
    struct niinGpunetioProxyAtomicHostContext** out) {
  if (comm == nullptr || heapBase == nullptr || heapBytes == 0 || out == nullptr) return ncclInvalidArgument;
  *out = nullptr;
  struct niinGpunetioProxyAtomicOptions options = NIIN_GPUNETIO_PROXY_ATOMIC_OPTIONS_INITIALIZER;
  if (suppliedOptions != nullptr) options = *suppliedOptions;
  if (options.mode != NIIN_GPUNETIO_PROXY_ATOMIC_EXECUTION_PROXY_ALL || options.serviceLevel > 15 ||
      options.requestSlotsPerPe != 1 || !isPowerOfTwo(options.srqDepth) ||
      !isPowerOfTwo(options.sendQueueDepth) || options.srqDepth < kMinQueueDepth ||
      options.sendQueueDepth < kMinQueueDepth || options.srqDepth > kMaxQueueDepth ||
      options.sendQueueDepth > kMaxQueueDepth)
    return ncclInvalidArgument;

  auto* provider = new (std::nothrow) niinGpunetioProxyAtomicHostContext;
  if (provider == nullptr) return ncclSystemError;
  provider->comm = comm;
  provider->slotsPerPe = options.requestSlotsPerPe;
  ncclResult_t result = ncclCommUserRank(comm, &provider->rank);
  if (result != ncclSuccess) goto fail;
  result = ncclCommCount(comm, &provider->nPes);
  if (result != ncclSuccess || provider->nPes <= 0 || provider->nPes > INT32_MAX) {
    if (result == ncclSuccess) result = ncclInvalidArgument;
    goto fail;
  }
  provider->pkeyIndex = options.pkeyIndex;
  provider->serviceLevel = options.serviceLevel;

  {
    ncclResult_t local = ncclSuccess;
    const char* setupStage = "heap alignment/SRQ sizing";
    if ((reinterpret_cast<uintptr_t>(heapBase) & (alignof(uint64_t) - 1)) != 0 ||
        (provider->nPes > 1 && static_cast<uint64_t>(provider->nPes - 1) > options.srqDepth)) {
      local = ncclInvalidArgument;
    }
    if (local == ncclSuccess) {
      setupStage = "verbs device setup";
      local = openVerbsDevice(provider, options);
    }
    if (local == ncclSuccess) {
      setupStage = "GDRCopy CPU mapping setup";
      if (!provider->gdr.initialize(options.gdrCopyLibrary, heapBase, heapBytes)) {
        std::fprintf(stderr,
                     "NIIN proxy atomics: GDRCopy CPU mapping is unavailable; proxy-all AMOs require it\n");
        local = ncclInvalidUsage;
      }
    }
    if (local == ncclSuccess) {
      setupStage = "mapped request-slot setup";
      local = createMappedSlots(provider);
    }
    if (local == ncclSuccess) {
      setupStage = "SRQ setup";
      local = createSrqResources(provider, options);
    }
    if (local != ncclSuccess)
      std::fprintf(stderr, "NIIN proxy atomics [rank %d]: %s failed: %s\n", provider->rank,
                   setupStage, ncclGetErrorString(local));
    result = allGatherSetupStatus(comm, provider->nPes, local);
    if (result != ncclSuccess) goto fail;
  }
  {
    std::vector<struct ProxyWire> localWire;
    std::vector<struct ProxyWire> remoteWire;
    ncclResult_t local = createPeerQps(provider, options, &localWire);
    result = allGatherSetupStatus(comm, provider->nPes, local);
    if (result != ncclSuccess) goto fail;
    result = allGatherWire(comm, localWire, provider->nPes, &remoteWire);
    if (result != ncclSuccess) goto fail;
    provider->peers.assign(provider->nPes, {});
    local = ncclSuccess;
    for (int peer = 0; peer < provider->nPes; ++peer) {
      const struct ProxyWire& remote = remoteWire[static_cast<size_t>(peer) * provider->nPes + provider->rank];
      provider->peers[peer] = remote;
      if (peer != provider->rank && (local = connectPeerQp(provider, peer, remote)) != ncclSuccess) break;
    }
    result = allGatherSetupStatus(comm, provider->nPes, local);
    if (result != ncclSuccess) goto fail;
    local = exportDeviceContext(provider);
    result = allGatherSetupStatus(comm, provider->nPes, local);
    if (result != ncclSuccess) goto fail;
  }
  startProgressThread(provider);
  *out = provider;
  return ncclSuccess;

fail:
  delete provider;
  return result;
}

ncclResult_t niinGpunetioProxyAtomicBind(struct niinGpunetioProxyAtomicHostContext* provider,
                                         struct niinContext* deviceContext) {
  if (provider == nullptr || provider->atomicContextDevice == nullptr || deviceContext == nullptr)
    return ncclInvalidArgument;
  if (provider->boundDeviceContext != nullptr && provider->boundDeviceContext != deviceContext)
    return ncclInvalidArgument;
  const struct niinGpunetioAtomicContext* existing = nullptr;
  if (cudaMemcpy(&existing, reinterpret_cast<const char*>(deviceContext) +
                              offsetof(struct niinContext, gpunetioAtomicContext),
                 sizeof(existing), cudaMemcpyDeviceToHost) != cudaSuccess)
    return ncclSystemError;
  if (existing != nullptr && existing != provider->atomicContextDevice) return ncclInvalidUsage;
  const struct niinGpunetioAtomicContext* context = provider->atomicContextDevice;
  if (cudaMemcpy(reinterpret_cast<char*>(deviceContext) + offsetof(struct niinContext, gpunetioAtomicContext),
                 &context, sizeof(context), cudaMemcpyHostToDevice) != cudaSuccess)
    return ncclSystemError;
  provider->boundDeviceContext = deviceContext;
  return ncclSuccess;
}

ncclResult_t niinGpunetioProxyAtomicProgress(struct niinGpunetioProxyAtomicHostContext* provider) {
  if (provider == nullptr) return ncclInvalidArgument;
  if (provider->progressFailed.load(std::memory_order_acquire)) return ncclSystemError;
  return progressOnce(provider);
}

ncclResult_t niinGpunetioProxyAtomicFinalize(struct niinGpunetioProxyAtomicHostContext* provider) {
  if (provider == nullptr) return ncclInvalidArgument;
  // Every rank must arrive here only after all of its AMO-producing kernels
  // complete. The private RC/SRQ fabric is peer-owned, so this collective
  // quiesce prevents one PE from destroying a target QP/SRQ while another PE
  // is still in the normal teardown path.
  const ncclResult_t quiesce = allGatherSetupStatus(provider->comm, provider->nPes, ncclSuccess);
  if (quiesce != ncclSuccess) return quiesce;
  delete provider;
  return ncclSuccess;
}
