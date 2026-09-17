/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN's direct GPUNetIO atomic provider.
//
// This file intentionally uses only public NCCL, CUDA, ibverbs, and
// GPUNetIO APIs.  In particular, it must not include NCCL transport headers
// or inspect a GIN context: GIN remains the implementation for NIIN RMA and
// signaling, while this provider owns only its atomic QPs and registrations.

#include "niin/gpunetio/host.h"

#include "niin/context_abi.h"
#include "niin/gpunetio/context.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#if __has_include(<gpunetio/doca_gpunetio_host.h>)
#include <gpunetio/doca_gpunetio_host.h>
#elif __has_include(<doca_gpunetio_host.h>)
#include <doca_gpunetio_host.h>
#else
#error "NIIN GPUNetIO atomics requires GPUNetIO's public host header"
#endif
#include <infiniband/verbs.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <endian.h>
#include <cstdio>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kWireMagic = 0x4e49414du;  // "NIAM"
constexpr uint16_t kWireVersion = 1;
constexpr uint32_t kMinSqDepth = 64;
constexpr uint32_t kMaxSqDepth = 32768;
// Eight bytes hold the response value and byte eight is the independent
// scratch byte used by a fenced DUMP. Keep the stride naturally aligned so
// both 32- and 64-bit response loads remain aligned.
constexpr uint32_t kResponseStride = 16;
constexpr uint32_t kCstScratchOffset = sizeof(uint64_t);
constexpr uint8_t kQpPsn = 0;
constexpr uint16_t kAckTimeout = 20;
constexpr uint16_t kRetryCount = 7;
constexpr uint16_t kRnrRetryCount = 7;
constexpr uint16_t kMinRnrTimer = 12;
constexpr uint8_t kMaxOutstandingAtomics = 1;

// The public GPUNetIO headers expose the MLX5 capability layout but purposefully
// do not expose a capability-query convenience API.  Query the standard
// mlx5dv DevX entry point dynamically so this NIIN client remains compatible
// with GPUNetIO's own dynamic libmlx5 loading model and does not need to use
// GPUNetIO private source headers.
using Mlx5dvDevxGeneralCmd = int (*)(struct ibv_context*, const void*, size_t, void*, size_t);

constexpr uint32_t kMlx5AtomicOpCas = 0x1;
constexpr uint32_t kMlx5AtomicOpFetchAdd = 0x2;
constexpr uint32_t kMlx5AtomicOpMaskedCas = 0x4;
constexpr uint32_t kMlx5AtomicOpMaskedFetchAdd = 0x8;
constexpr uint32_t kMlx5AtomicOpsRequired =
    kMlx5AtomicOpCas | kMlx5AtomicOpFetchAdd | kMlx5AtomicOpMaskedCas |
    kMlx5AtomicOpMaskedFetchAdd;
constexpr uint32_t kMlx5AtomicSize4B = 0x4;
constexpr uint32_t kMlx5AtomicSize8B = 0x8;
constexpr uint32_t kMlx5AtomicSizesRequired = kMlx5AtomicSize4B | kMlx5AtomicSize8B;

struct Mlx5AtomicCapabilities {
  // 4-byte atomic responses are always network byte order.  Some HCAs can
  // report 8-byte responses in host/device byte order; retain that negotiated
  // mode in NIIN's device ABI rather than assuming one family of NICs.
  bool atomic8BResultHostEndian = false;
};

// `mlx5_ifc.h` models bit fields as byte arrays whose offsetof/sizeof values
// are bit offsets/sizes.  Keep the tiny encoder/decoder private to this file
// instead of reaching into GPUNetIO's private DevX wrapper.
bool mlx5SetField(uint8_t* command, size_t bitOffset, size_t bitSize, uint32_t value) {
  if (bitSize == 0 || bitSize > 32 || (bitOffset & 31u) + bitSize > 32) return false;
  const uint32_t fieldMask = bitSize == 32 ? UINT32_MAX : (uint32_t{1} << bitSize) - 1;
  uint32_t wordBe = 0;
  std::memcpy(&wordBe, command + (bitOffset / 32) * sizeof(wordBe), sizeof(wordBe));
  const uint32_t shift = 32 - static_cast<uint32_t>(bitSize) - static_cast<uint32_t>(bitOffset & 31u);
  const uint32_t word = be32toh(wordBe);
  wordBe = htobe32((word & ~(fieldMask << shift)) | ((value & fieldMask) << shift));
  std::memcpy(command + (bitOffset / 32) * sizeof(wordBe), &wordBe, sizeof(wordBe));
  return true;
}

bool mlx5GetField(const uint8_t* command, size_t bitOffset, size_t bitSize, uint32_t* value) {
  if (value == nullptr || bitSize == 0 || bitSize > 32 || (bitOffset & 31u) + bitSize > 32) return false;
  const uint32_t fieldMask = bitSize == 32 ? UINT32_MAX : (uint32_t{1} << bitSize) - 1;
  uint32_t wordBe = 0;
  std::memcpy(&wordBe, command + (bitOffset / 32) * sizeof(wordBe), sizeof(wordBe));
  const uint32_t shift = 32 - static_cast<uint32_t>(bitSize) - static_cast<uint32_t>(bitOffset & 31u);
  *value = (be32toh(wordBe) >> shift) & fieldMask;
  return true;
}

bool queryRequiredMlx5AtomicCaps(struct ibv_context* verbsContext, Mlx5AtomicCapabilities* capabilities) {
  if (verbsContext == nullptr || capabilities == nullptr) return false;

  alignas(uint32_t) uint8_t commandIn[MLX5_ST_SZ_BYTES(query_hca_cap_in)] = {};
  alignas(uint32_t) uint8_t commandOut[MLX5_ST_SZ_BYTES(query_hca_cap_out)] = {};
  if (!mlx5SetField(commandIn, offsetof(struct mlx5_ifc_query_hca_cap_in_bits, opcode),
                    MLX5_FLD_SZ_BITS(query_hca_cap_in, opcode), MLX5_CMD_OP_QUERY_HCA_CAP) ||
      !mlx5SetField(commandIn, offsetof(struct mlx5_ifc_query_hca_cap_in_bits, op_mod),
                    MLX5_FLD_SZ_BITS(query_hca_cap_in, op_mod),
                    MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_ATOMIC << 1) |
                        HCA_CAP_OPMOD_GET_CUR))
    return false;

  void* mlx5 = dlopen("libmlx5.so.1", RTLD_NOW | RTLD_LOCAL);
  if (mlx5 == nullptr) mlx5 = dlopen("libmlx5.so", RTLD_NOW | RTLD_LOCAL);
  if (mlx5 == nullptr) return false;
  const auto generalCmd = reinterpret_cast<Mlx5dvDevxGeneralCmd>(dlsym(mlx5, "mlx5dv_devx_general_cmd"));
  const int status = generalCmd == nullptr
                         ? -1
                         : generalCmd(verbsContext, commandIn, sizeof(commandIn), commandOut,
                                      sizeof(commandOut));
  dlclose(mlx5);
  if (status != 0) return false;

  const uint8_t* atomicCaps = commandOut + MLX5_BYTE_OFF(query_hca_cap_out, capability);
  uint32_t operations = 0;
  uint32_t qpSizes = 0;
  uint32_t atomic8BEndiannessMode = 0;
  uint32_t atomic8BHostEndianSupported = 0;
  if (!mlx5GetField(atomicCaps, offsetof(struct mlx5_ifc_atomic_caps_bits, atomic_operations),
                    MLX5_FLD_SZ_BITS(atomic_caps, atomic_operations), &operations) ||
      !mlx5GetField(atomicCaps, offsetof(struct mlx5_ifc_atomic_caps_bits, atomic_size_qp),
                    MLX5_FLD_SZ_BITS(atomic_caps, atomic_size_qp), &qpSizes) ||
      !mlx5GetField(atomicCaps, offsetof(struct mlx5_ifc_atomic_caps_bits,
                                          atomic_req_8B_endianness_mode),
                    MLX5_FLD_SZ_BITS(atomic_caps, atomic_req_8B_endianness_mode),
                    &atomic8BEndiannessMode) ||
      !mlx5GetField(atomicCaps, offsetof(struct mlx5_ifc_atomic_caps_bits,
                                          supported_atomic_req_8B_endianness_mode_1),
                    MLX5_FLD_SZ_BITS(atomic_caps, supported_atomic_req_8B_endianness_mode_1),
                    &atomic8BHostEndianSupported))
    return false;

  // NIIN creates RC QPs, so only QP (not DC) atomic-size capability is a
  // requirement.  All four operation bits are required because the public
  // NIIN surface includes masked set/bitwise AMOs as well as standard CAS/FA.
  if ((operations & kMlx5AtomicOpsRequired) != kMlx5AtomicOpsRequired ||
      (qpSizes & kMlx5AtomicSizesRequired) != kMlx5AtomicSizesRequired)
    return false;

  // This is the same capability interpretation used by NVSHMEM's transport:
  // only a supported, nonzero mode permits host-endian 8-byte responses.
  capabilities->atomic8BResultHostEndian =
      atomic8BHostEndianSupported != 0 && atomic8BEndiannessMode != 0;
  return true;
}

struct niinGpunetioAtomicWire {
  uint32_t magic;
  uint16_t version;
  uint16_t sourcePe;
  uint16_t destinationPe;
  uint16_t lid;
  uint32_t qpn;
  uint8_t gid[DOCA_VERBS_GID_BYTE_LENGTH];
  uint8_t linkLayer;
  uint8_t activeMtu;
  uint16_t reserved0;
  uint64_t heapBase;
  uint64_t heapBytes;
  // This is in the mkey byte order expected by GPUNetIO's device WQE helpers.
  uint32_t heapRkey;
  uint32_t reserved1;
};

static_assert(sizeof(niinGpunetioAtomicWire) % alignof(uint64_t) == 0,
              "NIIN GPUNetIO wire metadata must retain 64-bit alignment");

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

uint32_t toGpunetioMkey(uint32_t key) {
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
  return __builtin_bswap32(key);
#else
  return key;
#endif
}

enum doca_verbs_mtu_size toDocaMtu(int ibvMtu) {
  switch (ibvMtu) {
    case IBV_MTU_256: return DOCA_VERBS_MTU_SIZE_256_BYTES;
    case IBV_MTU_512: return DOCA_VERBS_MTU_SIZE_512_BYTES;
    case IBV_MTU_1024: return DOCA_VERBS_MTU_SIZE_1K_BYTES;
    case IBV_MTU_2048: return DOCA_VERBS_MTU_SIZE_2K_BYTES;
    case IBV_MTU_4096: return DOCA_VERBS_MTU_SIZE_4K_BYTES;
    default: return DOCA_VERBS_MTU_SIZE_256_BYTES;
  }
}

ncclResult_t cudaToNccl(cudaError_t status) {
  return status == cudaSuccess ? ncclSuccess : ncclSystemError;
}

}  // namespace

struct niinGpunetioAtomicHostContext {
  ncclComm_t comm = nullptr;
  int rank = -1;
  int nPes = 0;

  struct ibv_context* verbsContext = nullptr;
  struct ibv_pd* pd = nullptr;
  struct ibv_port_attr portAttr = {};
  union ibv_gid localGid = {};
  int port = 0;
  int gidIndex = 0;
  uint16_t pkeyIndex = 0;
  uint8_t serviceLevel = 0;
  bool atomic8BResultHostEndian = false;

  doca_gpu_t* gpu = nullptr;
  doca_dev_t* netDev = nullptr;

  struct ibv_mr* heapMr = nullptr;
  struct ibv_mr* responseMr = nullptr;
  void* responseBase = nullptr;
  size_t responseBytes = 0;

  // qps[pe] is the private outbound atomic QP to pe. The local-PE entry is
  // intentionally null: NIIN continues to use CUDA atomics for self/LSA.
  std::vector<struct doca_gpu_verbs_qp_hl*> qps;
  std::vector<struct doca_gpu_verbs_qp*> exportedQps;
  struct doca_gpu_dev_verbs_qp* deviceQps = nullptr;

  std::vector<struct niinGpunetioAtomicEndpoint> endpointHost;
  struct niinGpunetioAtomicEndpoint* endpointDevice = nullptr;
  int* endpointLocksDevice = nullptr;
  struct niinGpunetioAtomicContext atomicContextHost = {};
  struct niinGpunetioAtomicContext* atomicContextDevice = nullptr;
  struct niinContext* boundDeviceContext = nullptr;

  std::atomic<bool> stopProgress{false};
  std::atomic<bool> progressFailed{false};
  std::thread progressThread;

  ~niinGpunetioAtomicHostContext() { cleanup(); }

  bool hasCpuProxyQp() const {
    for (const auto* qp : qps) {
      if (qp != nullptr && qp->qp_gverbs != nullptr && qp->qp_gverbs->qp_cpu != nullptr &&
          (qp->qp_gverbs->qp_cpu->nic_handler &
           DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) != 0)
        return true;
    }
    return false;
  }

  ncclResult_t progressOnce(bool* didProgress = nullptr) {
    bool progressed = false;
    for (const auto* qp : qps) {
      if (qp == nullptr || qp->qp_gverbs == nullptr || qp->qp_gverbs->qp_cpu == nullptr ||
          (qp->qp_gverbs->qp_cpu->nic_handler &
           DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY) == 0)
        continue;
      bool thisQpProgressed = false;
      if (doca_gpu_verbs_cpu_proxy_progress(qp->qp_gverbs, &thisQpProgressed) != DOCA_SUCCESS) {
        progressFailed.store(true, std::memory_order_release);
        return ncclSystemError;
      }
      progressed = progressed || thisQpProgressed;
    }
    if (didProgress != nullptr) *didProgress = progressed;
    return ncclSuccess;
  }

  void startProgressThread() {
    if (!hasCpuProxyQp()) return;
    stopProgress.store(false, std::memory_order_release);
    progressThread = std::thread([this] {
      while (!stopProgress.load(std::memory_order_acquire)) {
        bool progressed = false;
        if (progressOnce(&progressed) != ncclSuccess) return;
        if (!progressed) std::this_thread::yield();
      }
    });
  }

  void clearBinding() {
    if (boundDeviceContext == nullptr) return;
    const struct niinGpunetioAtomicContext* currentContext = nullptr;
    const cudaError_t readStatus = cudaMemcpy(
        &currentContext,
        reinterpret_cast<const char*>(boundDeviceContext) +
            offsetof(struct niinContext, gpunetioAtomicContext),
        sizeof(currentContext), cudaMemcpyDeviceToHost);
    const struct niinGpunetioAtomicContext* nullContext = nullptr;
    // Do not clear a newer provider which was bound after this one. Bind()
    // rejects that normal misuse, but this comparison also makes teardown
    // robust when a caller has updated the context explicitly.
    if (readStatus == cudaSuccess && currentContext == atomicContextDevice) {
      cudaMemcpy(reinterpret_cast<char*>(boundDeviceContext) +
                     offsetof(struct niinContext, gpunetioAtomicContext),
                 &nullContext, sizeof(nullContext), cudaMemcpyHostToDevice);
    }
    boundDeviceContext = nullptr;
  }

  void cleanup() {
    stopProgress.store(true, std::memory_order_release);
    if (progressThread.joinable()) progressThread.join();

    clearBinding();

    if (atomicContextDevice != nullptr) cudaFree(atomicContextDevice);
    atomicContextDevice = nullptr;
    if (endpointLocksDevice != nullptr) cudaFree(endpointLocksDevice);
    endpointLocksDevice = nullptr;
    if (endpointDevice != nullptr) cudaFree(endpointDevice);
    endpointDevice = nullptr;

    if (deviceQps != nullptr && gpu != nullptr && !exportedQps.empty()) {
      doca_gpu_verbs_unexport_multi_qps_dev(gpu, exportedQps.data(), exportedQps.size(), deviceQps);
    }
    deviceQps = nullptr;
    exportedQps.clear();

    for (auto*& qp : qps) {
      if (qp != nullptr) doca_gpu_verbs_destroy_qp_hl(qp);
      qp = nullptr;
    }
    qps.clear();

    if (responseMr != nullptr) ibv_dereg_mr(responseMr);
    responseMr = nullptr;
    if (heapMr != nullptr) ibv_dereg_mr(heapMr);
    heapMr = nullptr;
    if (responseBase != nullptr) cudaFree(responseBase);
    responseBase = nullptr;

    if (netDev != nullptr) doca_verbs_dev_close(netDev);
    netDev = nullptr;
    if (pd != nullptr) ibv_dealloc_pd(pd);
    pd = nullptr;
    if (verbsContext != nullptr) ibv_close_device(verbsContext);
    verbsContext = nullptr;
    if (gpu != nullptr) doca_gpu_destroy(gpu);
    gpu = nullptr;
  }
};

namespace {

ncclResult_t openVerbsDevice(niinGpunetioAtomicHostContext* provider,
                             const struct niinGpunetioAtomicOptions& options) {
  const char* requestedDevice = options.ibDevice;
  if (requestedDevice == nullptr || *requestedDevice == '\0') {
    requestedDevice = std::getenv("NIIN_GPUNETIO_IB_DEV");
  }
  const int requestedPort = options.ibPort != 0 ? options.ibPort : readEnvInt("NIIN_GPUNETIO_IB_PORT", 0);
  const int requestedGidIndex =
      options.gidIndex >= 0 ? options.gidIndex : readEnvInt("NIIN_GPUNETIO_GID_INDEX", 0);
  if (requestedPort < 0 || requestedGidIndex < 0 || requestedGidIndex > UINT8_MAX) return ncclInvalidArgument;

  int deviceCount = 0;
  struct ibv_device** devices = ibv_get_device_list(&deviceCount);
  if (devices == nullptr) return ncclSystemError;

  ncclResult_t result = ncclSystemError;
  for (int deviceIndex = 0; deviceIndex < deviceCount && provider->verbsContext == nullptr; ++deviceIndex) {
    if (requestedDevice != nullptr && std::strcmp(ibv_get_device_name(devices[deviceIndex]), requestedDevice) != 0)
      continue;

    struct ibv_context* context = ibv_open_device(devices[deviceIndex]);
    if (context == nullptr) continue;

    struct ibv_device_attr deviceAttr = {};
    if (ibv_query_device(context, &deviceAttr) != 0) {
      ibv_close_device(context);
      continue;
    }

    for (int port = 1; port <= deviceAttr.phys_port_cnt; ++port) {
      if (requestedPort != 0 && port != requestedPort) continue;
      struct ibv_port_attr portAttr = {};
      if (ibv_query_port(context, port, &portAttr) != 0 || portAttr.state != IBV_PORT_ACTIVE) continue;

      union ibv_gid gid = {};
      if (ibv_query_gid(context, port, requestedGidIndex, &gid) != 0) continue;

      struct ibv_pd* pd = ibv_alloc_pd(context);
      if (pd == nullptr) continue;

      // GPUNetIO Open 4.1's public high-level QP path resolves the RoCE
      // address using port 1.  An IB port can use any active port, but reject
      // an Ethernet port other than 1 rather than construct a QP that may
      // resolve the wrong MAC address.
      if (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET && port != 1) {
        ibv_dealloc_pd(pd);
        continue;
      }

      provider->verbsContext = context;
      provider->pd = pd;
      provider->portAttr = portAttr;
      provider->localGid = gid;
      provider->port = port;
      provider->gidIndex = requestedGidIndex;
      result = ncclSuccess;
      break;
    }

    if (provider->verbsContext == nullptr) ibv_close_device(context);
  }

  ibv_free_device_list(devices);
  return result;
}

ncclResult_t openGpunetio(niinGpunetioAtomicHostContext* provider) {
  int cudaDevice = -1;
  if (cudaGetDevice(&cudaDevice) != cudaSuccess) return ncclSystemError;

  char pciBusId[32] = {};
  if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), cudaDevice) != cudaSuccess) return ncclSystemError;
  if (doca_gpu_create(pciBusId, &provider->gpu) != DOCA_SUCCESS) return ncclSystemError;
  if (doca_verbs_dev_open(provider->pd, &provider->netDev) != DOCA_SUCCESS) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t registerAtomicMemory(niinGpunetioAtomicHostContext* provider, void* heapBase,
                                  size_t heapBytes) {
  // This provider exposes atomic WQEs only.  Do not grant ordinary remote
  // read/write access to either its symmetric heap registration or its local
  // response ring.
  const int heapAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  provider->heapMr = ibv_reg_mr(provider->pd, heapBase, heapBytes, heapAccess);
  if (provider->heapMr == nullptr) return ncclSystemError;

  provider->responseBytes = static_cast<size_t>(provider->nPes) * kResponseStride;
  if (cudaMalloc(&provider->responseBase, provider->responseBytes) != cudaSuccess) return ncclSystemError;
  if (cudaMemset(provider->responseBase, 0, provider->responseBytes) != cudaSuccess) return ncclSystemError;

  // The response ring receives Atomic response DMA and is also the source of
  // the terminal DUMP used to establish response visibility.  Giving it the
  // same access flags keeps registration semantics simple; no remote address
  // from this MR is ever exchanged or used for NIIN RMA.
  provider->responseMr = ibv_reg_mr(provider->pd, provider->responseBase, provider->responseBytes,
                                    heapAccess);
  return provider->responseMr != nullptr ? ncclSuccess : ncclSystemError;
}

enum doca_gpu_dev_verbs_nic_handler selectNicHandler(enum niinGpunetioAtomicNicHandler requested) {
  switch (requested) {
    case NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_GPU_SM_DB:
      return DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
    case NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_CPU_PROXY:
      return DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    case NIIN_GPUNETIO_ATOMIC_NIC_HANDLER_AUTO:
    default:
      return DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  }
}

ncclResult_t createPrivateQps(niinGpunetioAtomicHostContext* provider,
                              const struct niinGpunetioAtomicOptions& options,
                              std::vector<niinGpunetioAtomicWire>* localWire) {
  struct doca_gpu_verbs_qp_init_attr_hl initAttr = {};
  initAttr.gpu_dev = provider->gpu;
  initAttr.net_dev = provider->netDev;
  initAttr.ibpd = provider->pd;
  initAttr.sq_nwqe = static_cast<uint16_t>(options.sqDepth);
  initAttr.nic_handler = selectNicHandler(options.nicHandler);
  initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  // Keep the exported device-QP CQ type in sync with the hardware CQ setup.
  // A zero-initialized cq_type means normal 64B CQEs even when cq_collapsed is
  // true, which makes the generic device poller decode collapsed CQEs wrong.
  initAttr.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED;
  initAttr.cq_collapsed = true;

  provider->qps.resize(provider->nPes, nullptr);
  localWire->assign(provider->nPes, {});
  for (int pe = 0; pe < provider->nPes; ++pe) {
    struct niinGpunetioAtomicWire& wire = (*localWire)[pe];
    wire.magic = kWireMagic;
    wire.version = kWireVersion;
    wire.sourcePe = static_cast<uint16_t>(provider->rank);
    wire.destinationPe = static_cast<uint16_t>(pe);
    wire.lid = provider->portAttr.lid;
    std::memcpy(wire.gid, &provider->localGid, sizeof(wire.gid));
    wire.linkLayer = static_cast<uint8_t>(provider->portAttr.link_layer);
    wire.activeMtu = static_cast<uint8_t>(provider->portAttr.active_mtu);
    wire.heapBase = reinterpret_cast<uint64_t>(provider->heapMr->addr);
    wire.heapBytes = provider->heapMr->length;
    wire.heapRkey = toGpunetioMkey(provider->heapMr->rkey);

    if (pe == provider->rank) continue;
    if (doca_gpu_verbs_create_qp_hl(&initAttr, &provider->qps[pe]) != DOCA_SUCCESS)
      return ncclSystemError;
    if (doca_verbs_qp_get_qpn(provider->qps[pe]->qp, &wire.qpn) != DOCA_SUCCESS)
      return ncclSystemError;
  }

  // The direct NIIN provider has been validated with GPU SM doorbells.  The
  // GPUNetIO CPU doorbell proxy is a different host-visible publication path;
  // on the provider version paired with this NCCL build it can leave a device
  // AMO waiting indefinitely.  Reject it collectively during setup instead
  // of publishing a context that can hang a user kernel.  This is unrelated
  // to NIIN's future SRQ software-atomic proxy.
  if (provider->hasCpuProxyQp()) {
    std::fprintf(stderr,
                 "NIIN GPUNetIO atomics: CPU doorbell proxy is not supported by the direct "
                 "atomic provider; require GPU SM doorbells\n");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t allGatherWire(ncclComm_t comm, const std::vector<niinGpunetioAtomicWire>& local,
                           int nPes, std::vector<niinGpunetioAtomicWire>* remote) {
  if (local.size() != static_cast<size_t>(nPes)) return ncclInvalidArgument;
  const size_t localBytes = local.size() * sizeof(niinGpunetioAtomicWire);
  const size_t remoteBytes = localBytes * static_cast<size_t>(nPes);
  void* deviceSend = nullptr;
  void* deviceRecv = nullptr;
  cudaStream_t stream = nullptr;

  ncclResult_t result = cudaToNccl(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&deviceSend, localBytes));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&deviceRecv, remoteBytes));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMemcpyAsync(deviceSend, local.data(), localBytes, cudaMemcpyHostToDevice, stream));
  if (result != ncclSuccess) goto cleanup;
  result = ncclAllGather(deviceSend, deviceRecv, localBytes, ncclUint8, comm, stream);
  if (result != ncclSuccess) goto cleanup;

  remote->resize(static_cast<size_t>(nPes) * static_cast<size_t>(nPes));
  result = cudaToNccl(
      cudaMemcpyAsync(remote->data(), deviceRecv, remoteBytes, cudaMemcpyDeviceToHost, stream));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaStreamSynchronize(stream));

cleanup:
  if (deviceRecv != nullptr) cudaFree(deviceRecv);
  if (deviceSend != nullptr) cudaFree(deviceSend);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return result;
}

// `niinGpunetioAtomicInit` is collective even though most of its setup is
// local.  Do not let one PE return early after a local verbs/GPUNetIO failure
// while another continues into the descriptor all-gather: that would turn an
// ordinary unsupported-HCA error into a peer hang.  A byte all-gather keeps
// this bootstrap independent of NCCL reduction datatype/operator support.
//
// CUDA/NCCL failures while executing this helper are necessarily fatal to the
// collective itself, but all ordinary local setup failures reach this point on
// every PE before the next stage begins.
ncclResult_t allGatherSetupStatus(ncclComm_t comm, int nPes, ncclResult_t localStatus) {
  if (nPes <= 0) return ncclInvalidArgument;

  uint32_t localCode = static_cast<uint32_t>(localStatus);
  std::vector<uint32_t> remoteCodes(static_cast<size_t>(nPes), static_cast<uint32_t>(ncclSystemError));
  void* deviceSend = nullptr;
  void* deviceRecv = nullptr;
  cudaStream_t stream = nullptr;

  ncclResult_t result = cudaToNccl(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&deviceSend, sizeof(localCode)));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMalloc(&deviceRecv, remoteCodes.size() * sizeof(remoteCodes[0])));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMemcpyAsync(deviceSend, &localCode, sizeof(localCode), cudaMemcpyHostToDevice, stream));
  if (result != ncclSuccess) goto cleanup;
  result = ncclAllGather(deviceSend, deviceRecv, sizeof(localCode), ncclUint8, comm, stream);
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaMemcpyAsync(remoteCodes.data(), deviceRecv,
                                      remoteCodes.size() * sizeof(remoteCodes[0]), cudaMemcpyDeviceToHost,
                                      stream));
  if (result != ncclSuccess) goto cleanup;
  result = cudaToNccl(cudaStreamSynchronize(stream));
  if (result != ncclSuccess) goto cleanup;

  for (uint32_t code : remoteCodes) {
    if (code != static_cast<uint32_t>(ncclSuccess)) {
      // Preserve the most useful local error on the failing PE.  A peer only
      // learns that setup failed collectively; it cannot safely infer which
      // external resource (NIC, GPU mapping, or registration) failed there.
      result = localStatus == ncclSuccess ? ncclSystemError : localStatus;
      goto cleanup;
    }
  }
  result = ncclSuccess;

cleanup:
  if (deviceRecv != nullptr) cudaFree(deviceRecv);
  if (deviceSend != nullptr) cudaFree(deviceSend);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return result;
}

ncclResult_t connectPeerQp(niinGpunetioAtomicHostContext* provider, int peer,
                           const struct niinGpunetioAtomicWire& remote) {
  if (peer == provider->rank || provider->qps[peer] == nullptr || remote.magic != kWireMagic ||
      remote.version != kWireVersion || remote.sourcePe != peer || remote.destinationPe != provider->rank ||
      remote.qpn == 0 || remote.heapBase == 0 || remote.heapBytes == 0 ||
      remote.linkLayer != static_cast<uint8_t>(provider->portAttr.link_layer))
    return ncclInvalidArgument;

  struct doca_verbs_ah_attr_t* ah = nullptr;
  doca_verbs_qp_attr_t* qpAttr = nullptr;
  auto fail = [&] {
    if (qpAttr != nullptr) doca_verbs_qp_attr_destroy(qpAttr);
    if (ah != nullptr) doca_verbs_ah_attr_destroy(ah);
    return ncclSystemError;
  };

  if (doca_verbs_ah_attr_create(provider->netDev, &ah) != DOCA_SUCCESS) return fail();
  if (doca_verbs_ah_attr_set_sl(ah, provider->serviceLevel) != DOCA_SUCCESS) return fail();

  struct doca_verbs_gid remoteGid = {};
  std::memcpy(remoteGid.raw, remote.gid, sizeof(remoteGid.raw));
  const bool isIb = provider->portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND;
  // Match normal verbs path selection instead of forcing GRH for every IB
  // connection. A same-subnet IB link ordinarily uses a LID-only AH. Forcing
  // GRH there can leave a QP apparently connected yet unable to complete its
  // first WQE. RoCE always needs the GID-backed path.
  const bool sameIbSubnet =
      isIb && std::memcmp(&provider->localGid.global.subnet_prefix,
                          &remoteGid.raw[0], sizeof(provider->localGid.global.subnet_prefix)) == 0;
  const bool useIbGrh = isIb && (!sameIbSubnet || (provider->portAttr.flags & IBV_QPF_GRH_REQUIRED) != 0);
  if (doca_verbs_ah_attr_set_addr_type(
          ah, isIb ? (useIbGrh ? DOCA_VERBS_ADDR_TYPE_IB_GRH : DOCA_VERBS_ADDR_TYPE_IB_NO_GRH)
                   : DOCA_VERBS_ADDR_TYPE_IPv4) != DOCA_SUCCESS ||
      doca_verbs_ah_attr_set_dlid(ah, remote.lid) != DOCA_SUCCESS)
    return fail();
  if ((!isIb || useIbGrh) &&
      (doca_verbs_ah_attr_set_gid(ah, remoteGid) != DOCA_SUCCESS ||
       doca_verbs_ah_attr_set_sgid_index(ah, static_cast<uint8_t>(provider->gidIndex)) != DOCA_SUCCESS ||
       doca_verbs_ah_attr_set_hop_limit(ah, 255) != DOCA_SUCCESS))
    return fail();

  if (doca_verbs_qp_attr_create(&qpAttr) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_rq_psn(qpAttr, kQpPsn) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_sq_psn(qpAttr, kQpPsn) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_pkey_index(qpAttr, provider->pkeyIndex) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_path_mtu(
          qpAttr, toDocaMtu(std::min<int>(provider->portAttr.active_mtu, remote.activeMtu))) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_port_num(qpAttr, provider->port) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_ack_timeout(qpAttr, kAckTimeout) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_retry_cnt(qpAttr, kRetryCount) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_rnr_retry(qpAttr, kRnrRetryCount) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_min_rnr_timer(qpAttr, kMinRnrTimer) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_INIT) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_allow_remote_write(qpAttr, 0) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_allow_remote_read(qpAttr, 0) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_atomic_mode(qpAttr, DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_8BYTES) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_ah_attr(qpAttr, ah) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_dest_qp_num(qpAttr, remote.qpn) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_max_rd_atomic(qpAttr, kMaxOutstandingAtomics) != DOCA_SUCCESS ||
      doca_verbs_qp_attr_set_max_dest_rd_atomic(qpAttr, kMaxOutstandingAtomics) != DOCA_SUCCESS)
    return fail();

  struct doca_verbs_qp_t* qp = provider->qps[peer]->qp;
  if (doca_verbs_qp_modify(qp, qpAttr,
                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                                DOCA_VERBS_QP_ATTR_PORT_NUM |
                                DOCA_VERBS_QP_ATTR_ATOMIC_MODE) != DOCA_SUCCESS)
    return fail();
  if (doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTR) != DOCA_SUCCESS ||
      doca_verbs_qp_modify(qp, qpAttr,
                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC) != DOCA_SUCCESS)
    return fail();
  if (doca_verbs_qp_attr_set_next_state(qpAttr, DOCA_VERBS_QP_STATE_RTS) != DOCA_SUCCESS ||
      doca_verbs_qp_modify(qp, qpAttr,
                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                DOCA_VERBS_QP_ATTR_RNR_RETRY |
                                DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC) != DOCA_SUCCESS)
    return fail();

  doca_verbs_qp_attr_destroy(qpAttr);
  doca_verbs_ah_attr_destroy(ah);
  return ncclSuccess;
}

ncclResult_t exportDeviceContext(niinGpunetioAtomicHostContext* provider,
                                 const std::vector<niinGpunetioAtomicWire>& remote) {
  provider->exportedQps.clear();
  for (int pe = 0; pe < provider->nPes; ++pe) {
    if (pe != provider->rank) provider->exportedQps.push_back(provider->qps[pe]->qp_gverbs);
  }
  if (!provider->exportedQps.empty() &&
      doca_gpu_verbs_export_multi_qps_dev(provider->gpu, provider->exportedQps.data(),
                                           provider->exportedQps.size(), &provider->deviceQps) != DOCA_SUCCESS)
    return ncclSystemError;

  provider->endpointHost.assign(provider->nPes, {});
  int deviceQpIndex = 0;
  for (int pe = 0; pe < provider->nPes; ++pe) {
    if (pe == provider->rank) continue;
    const struct niinGpunetioAtomicWire& peerWire =
        remote[static_cast<size_t>(pe) * provider->nPes + provider->rank];
    struct niinGpunetioAtomicEndpoint& endpoint = provider->endpointHost[pe];
    endpoint.deviceQp = provider->deviceQps + deviceQpIndex++;
    endpoint.remoteHeapBase = peerWire.heapBase;
    endpoint.remoteHeapBytes = peerWire.heapBytes;
    endpoint.remoteHeapRkey = peerWire.heapRkey;
    endpoint.flags = NIIN_GPUNETIO_ATOMIC_ENDPOINT_CONNECTED |
                     NIIN_GPUNETIO_ATOMIC_ENDPOINT_ATOMIC_4B |
                     NIIN_GPUNETIO_ATOMIC_ENDPOINT_ATOMIC_8B |
                     NIIN_GPUNETIO_ATOMIC_ENDPOINT_EXTENDED;
  }

  ncclResult_t result = cudaToNccl(
      cudaMalloc(&provider->endpointDevice, provider->endpointHost.size() * sizeof(provider->endpointHost[0])));
  if (result != ncclSuccess) return result;
  result = cudaToNccl(cudaMemcpy(provider->endpointDevice, provider->endpointHost.data(),
                                 provider->endpointHost.size() * sizeof(provider->endpointHost[0]),
                                 cudaMemcpyHostToDevice));
  if (result != ncclSuccess) return result;
  result = cudaToNccl(cudaMalloc(&provider->endpointLocksDevice, provider->nPes * sizeof(int)));
  if (result != ncclSuccess) return result;
  result = cudaToNccl(cudaMemset(provider->endpointLocksDevice, 0, provider->nPes * sizeof(int)));
  if (result != ncclSuccess) return result;

  provider->atomicContextHost.version = NIIN_GPUNETIO_ATOMIC_CONTEXT_VERSION;
  provider->atomicContextHost.flags = NIIN_GPUNETIO_ATOMIC_CONTEXT_READY |
                                      NIIN_GPUNETIO_ATOMIC_CONTEXT_DIRECT |
                                      NIIN_GPUNETIO_ATOMIC_CONTEXT_EXTENDED;
  if (provider->atomic8BResultHostEndian)
    provider->atomicContextHost.flags |= NIIN_GPUNETIO_ATOMIC_CONTEXT_8B_RESULT_HOST_ENDIAN;
  if (provider->hasCpuProxyQp())
    provider->atomicContextHost.flags |= NIIN_GPUNETIO_ATOMIC_CONTEXT_CPU_PROXY_DOORBELL;
  provider->atomicContextHost.nPes = provider->nPes;
  provider->atomicContextHost.localPe = provider->rank;
  provider->atomicContextHost.endpoints = provider->endpointDevice;
  provider->atomicContextHost.responseBase = provider->responseBase;
  provider->atomicContextHost.responseBytes = provider->responseBytes;
  provider->atomicContextHost.responseStride = kResponseStride;
  provider->atomicContextHost.cstScratchOffset = kCstScratchOffset;
  provider->atomicContextHost.responseLkey = toGpunetioMkey(provider->responseMr->lkey);
  provider->atomicContextHost.endpointLocks = provider->endpointLocksDevice;
  provider->atomicContextHost.proxyContext = nullptr;

  result = cudaToNccl(cudaMalloc(&provider->atomicContextDevice, sizeof(provider->atomicContextHost)));
  if (result != ncclSuccess) return result;
  return cudaToNccl(cudaMemcpy(provider->atomicContextDevice, &provider->atomicContextHost,
                               sizeof(provider->atomicContextHost), cudaMemcpyHostToDevice));
}

}  // namespace

ncclResult_t niinGpunetioAtomicInit(
    ncclComm_t comm, void* heapBase, size_t heapBytes,
    const struct niinGpunetioAtomicOptions* suppliedOptions,
    struct niinGpunetioAtomicHostContext** out) {
  if (comm == nullptr || heapBase == nullptr || heapBytes == 0 || out == nullptr) return ncclInvalidArgument;
  *out = nullptr;

  struct niinGpunetioAtomicOptions options = NIIN_GPUNETIO_ATOMIC_OPTIONS_INITIALIZER;
  if (suppliedOptions != nullptr) options = *suppliedOptions;
  if (!isPowerOfTwo(options.sqDepth) || options.sqDepth < kMinSqDepth || options.sqDepth > kMaxSqDepth)
    return ncclInvalidArgument;
  if (options.serviceLevel > 15) return ncclInvalidArgument;

  auto* provider = new (std::nothrow) niinGpunetioAtomicHostContext;
  if (provider == nullptr) return ncclSystemError;
  provider->comm = comm;

  ncclResult_t result = ncclCommUserRank(comm, &provider->rank);
  ncclResult_t localStatus = ncclSuccess;
  Mlx5AtomicCapabilities mlx5AtomicCapabilities = {};
  if (result != ncclSuccess) goto fail;
  result = ncclCommCount(comm, &provider->nPes);
  if (result != ncclSuccess || provider->nPes <= 0 || provider->nPes > UINT16_MAX) {
    if (result == ncclSuccess) result = ncclInvalidArgument;
    goto fail;
  }
  provider->pkeyIndex = options.pkeyIndex;
  provider->serviceLevel = options.serviceLevel;

  // Each phase first performs its local work, then collectively reports the
  // outcome before any PE can enter a later NCCL/verbs operation.  This is
  // especially important for the capability gate: HCAs without masked
  // atomics must reject the provider cleanly rather than leave peers waiting
  // in the QP descriptor all-gather.
  localStatus = openVerbsDevice(provider, options);
  if (localStatus == ncclSuccess &&
      !queryRequiredMlx5AtomicCaps(provider->verbsContext, &mlx5AtomicCapabilities)) {
    std::fprintf(stderr,
                 "NIIN GPUNetIO atomics: selected HCA lacks required 4/8-byte "
                 "standard and masked atomic QP capabilities\n");
    localStatus = ncclInvalidUsage;
  }
  if (localStatus == ncclSuccess)
    provider->atomic8BResultHostEndian = mlx5AtomicCapabilities.atomic8BResultHostEndian;
  if (localStatus == ncclSuccess) localStatus = openGpunetio(provider);
  if (localStatus == ncclSuccess) localStatus = registerAtomicMemory(provider, heapBase, heapBytes);
  result = allGatherSetupStatus(comm, provider->nPes, localStatus);
  if (result != ncclSuccess) goto fail;

  {
    std::vector<niinGpunetioAtomicWire> localWire;
    std::vector<niinGpunetioAtomicWire> remoteWire;
    localStatus = createPrivateQps(provider, options, &localWire);
    result = allGatherSetupStatus(comm, provider->nPes, localStatus);
    if (result != ncclSuccess) goto fail;
    result = allGatherWire(comm, localWire, provider->nPes, &remoteWire);
    if (result != ncclSuccess) goto fail;

    localStatus = ncclSuccess;
    for (int peer = 0; peer < provider->nPes; ++peer) {
      if (peer == provider->rank) continue;
      const struct niinGpunetioAtomicWire& wire =
          remoteWire[static_cast<size_t>(peer) * provider->nPes + provider->rank];
      localStatus = connectPeerQp(provider, peer, wire);
      if (localStatus != ncclSuccess) break;
    }
    result = allGatherSetupStatus(comm, provider->nPes, localStatus);
    if (result != ncclSuccess) goto fail;

    localStatus = exportDeviceContext(provider, remoteWire);
    result = allGatherSetupStatus(comm, provider->nPes, localStatus);
    if (result != ncclSuccess) goto fail;
  }

  provider->startProgressThread();
  *out = provider;
  return ncclSuccess;

fail:
  delete provider;
  return result;
}

ncclResult_t niinGpunetioAtomicBind(struct niinGpunetioAtomicHostContext* provider,
                                    struct niinContext* deviceContext) {
  if (provider == nullptr || provider->atomicContextDevice == nullptr || deviceContext == nullptr)
    return ncclInvalidArgument;
  if (provider->boundDeviceContext != nullptr && provider->boundDeviceContext != deviceContext)
    return ncclInvalidArgument;

  const struct niinGpunetioAtomicContext* existingContext = nullptr;
  cudaError_t status = cudaMemcpy(
      &existingContext,
      reinterpret_cast<const char*>(deviceContext) + offsetof(struct niinContext, gpunetioAtomicContext),
      sizeof(existingContext), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return ncclSystemError;
  if (existingContext != nullptr && existingContext != provider->atomicContextDevice)
    return ncclInvalidUsage;

  const struct niinGpunetioAtomicContext* context = provider->atomicContextDevice;
  status = cudaMemcpy(
      reinterpret_cast<char*>(deviceContext) + offsetof(struct niinContext, gpunetioAtomicContext), &context,
      sizeof(context), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) return ncclSystemError;
  provider->boundDeviceContext = deviceContext;
  return ncclSuccess;
}

ncclResult_t niinGpunetioAtomicProgress(struct niinGpunetioAtomicHostContext* provider) {
  if (provider == nullptr) return ncclInvalidArgument;
  if (provider->progressFailed.load(std::memory_order_acquire)) return ncclSystemError;
  return provider->progressOnce();
}

ncclResult_t niinGpunetioAtomicFinalize(struct niinGpunetioAtomicHostContext* provider) {
  if (provider == nullptr) return ncclInvalidArgument;
  delete provider;
  return ncclSuccess;
}
