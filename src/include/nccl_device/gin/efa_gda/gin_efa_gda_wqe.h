/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Register-resident 64-byte EFA RDMA-write WQE for the thread-exclusive Put
 * path (gin_efa_gda.h). The words are packed by hand because filling
 * efa_io_tx_wqe by field compiles to IMAD chains on the issue path and costs
 * 5-25% at small sizes (H200/B200/B300, CUDA 13.1/13.2); every field position
 * and control value is taken from efa_io_defs.h.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_WQE_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_WQE_H_

#include <cstddef>
#include <cstdint>

#include "../../utility.h"
#include "../../transport/net_efa_gda/efa-dp-direct/include/device/efa_cuda_dp_impl.cuh"

namespace nccl {
namespace gin {
namespace efa_gda {

/* A field at byte offset `off` of the WQE sits at bit 8*(off % 8) of 64-bit
 * word off / 8. The _MASK macros of efa_io_defs.h expand to BIT(), which
 * efa_cuda_dp_impl.cuh leaves undefined, so it is re-provided locally. */
namespace efa_gda_wqe {
NCCL_HOST_DEVICE_INLINE constexpr unsigned word(size_t off) {
  return (unsigned)(off / 8u);
}
NCCL_HOST_DEVICE_INLINE constexpr unsigned bit(size_t off) {
  return 8u * (unsigned)(off % 8u);
}
NCCL_HOST_DEVICE_INLINE constexpr bool inWord(size_t off, size_t size) {
  return bit(off) + 8u * size <= 64u;
}

constexpr size_t kMeta = offsetof(struct efa_io_tx_wqe, meta);
constexpr size_t kReqId = kMeta + offsetof(struct efa_io_tx_meta_desc, req_id);
constexpr size_t kCtrl1 = kMeta + offsetof(struct efa_io_tx_meta_desc, ctrl1);
constexpr size_t kCtrl2 = kMeta + offsetof(struct efa_io_tx_meta_desc, ctrl2);
constexpr size_t kDestQp = kMeta + offsetof(struct efa_io_tx_meta_desc, dest_qp_num);
constexpr size_t kLength = kMeta + offsetof(struct efa_io_tx_meta_desc, length);
constexpr size_t kAh = kMeta + offsetof(struct efa_io_tx_meta_desc, ah);
constexpr size_t kCtrl3 = kMeta + offsetof(struct efa_io_tx_meta_desc, ctrl3);
constexpr size_t kQkey = kMeta + offsetof(struct efa_io_tx_meta_desc, qkey);
constexpr size_t kReqIdEx = kMeta + offsetof(struct efa_io_tx_meta_desc, req_id_ex);
constexpr size_t kRemote = offsetof(struct efa_io_tx_wqe, data.rdma_req.remote_mem);
constexpr size_t kRemLen = kRemote + offsetof(struct efa_io_remote_mem_addr, length);
constexpr size_t kRemRkey = kRemote + offsetof(struct efa_io_remote_mem_addr, rkey);
constexpr size_t kRemAddr = kRemote + offsetof(struct efa_io_remote_mem_addr, buf_addr_lo);
constexpr size_t kLocal = offsetof(struct efa_io_tx_wqe, data.rdma_req.local_mem);
constexpr size_t kLocLen = kLocal + offsetof(struct efa_io_tx_buf_desc, length);
constexpr size_t kLocLkey = kLocal + offsetof(struct efa_io_tx_buf_desc, lkey);
constexpr size_t kLocAddr = kLocal + offsetof(struct efa_io_tx_buf_desc, buf_addr_lo);

#pragma push_macro("BIT")
#undef BIT
#define BIT(nr) (1ull << (nr))
constexpr uint64_t kCtrl1Val =
  EFA_IO_TX_META_DESC_META_DESC_MASK | ((uint64_t)EFA_IO_RDMA_WRITE & EFA_IO_TX_META_DESC_OP_TYPE_MASK);
constexpr uint64_t kCtrl2Val =
  EFA_IO_TX_META_DESC_FIRST_MASK | EFA_IO_TX_META_DESC_LAST_MASK | EFA_IO_TX_META_DESC_COMP_REQ_MASK;
constexpr uint32_t kPhaseMask = (uint32_t)EFA_IO_TX_META_DESC_PHASE_MASK;
constexpr uint64_t kCtrl3Val =
  (uint64_t)EFA_IO_PROCESSING_HINT_BURST_PPS_SENSITIVE & EFA_IO_TX_META_DESC_PROCESSING_HINTS_MASK;
#undef BIT
#pragma pop_macro("BIT")

static_assert(sizeof(struct efa_io_tx_wqe) == 64u, "efa_io_tx_wqe must be 64 bytes");
/* EfaGdaWriteWqeRegs builds each 64-bit word as one expression, so the fields
 * it combines must share a word, and the 64-bit addresses span adjacent
 * buf_addr_lo / buf_addr_hi halves. */
static_assert(
  word(kReqId) == 0 && word(kCtrl1) == 0 && word(kCtrl2) == 0 && word(kDestQp) == 0 && word(kLength) == 0 &&
    word(kAh) == 1 && word(kCtrl3) == 1 && word(kQkey) == 2 && word(kReqIdEx) == 3 && word(kRemLen) == 4 &&
    word(kRemRkey) == 4 && word(kRemAddr) == 5 && word(kLocLen) == 6 && word(kLocLkey) == 6 && word(kLocAddr) == 7 &&
    inWord(kReqIdEx, sizeof(efa_io_tx_meta_desc::req_id_ex)) && inWord(kRemAddr, 8) && inWord(kLocAddr, 8) &&
    offsetof(struct efa_io_remote_mem_addr, buf_addr_hi) == offsetof(struct efa_io_remote_mem_addr, buf_addr_lo) + 4 &&
    offsetof(struct efa_io_tx_buf_desc, buf_addr_hi) == offsetof(struct efa_io_tx_buf_desc, buf_addr_lo) + 4,
  "efa_io_tx_wqe fields moved across 64-bit words; regroup EfaGdaWriteWqeRegs");
} // namespace efa_gda_wqe

/* The 64-byte RDMA-write WQE (one SGE) as eight 64-bit register words, written
 * straight to the SQ slot. Requires the 64-byte WQE that the plugin configures
 * on the data and counter endpoints. */
struct EfaGdaWriteWqeRegs {
  uint64_t w[8];

  NCCL_DEVICE_INLINE EfaGdaWriteWqeRegs(uint32_t phase, uint16_t ah, uint16_t qpn, uint32_t qkey, uint64_t dstAddr,
                                        uint32_t dstRkey, uint64_t srcAddr, uint32_t srcLkey, uint32_t bytes) {
    using namespace efa_gda_wqe;
    const uint64_t reqId = threadIdx.x;
    w[0] = ((reqId & 0xFFFFull) << bit(kReqId)) | (kCtrl1Val << bit(kCtrl1)) |
           ((kCtrl2Val | (phase & kPhaseMask)) << bit(kCtrl2)) | ((uint64_t)qpn << bit(kDestQp)) |
           (1ull << bit(kLength)); /* one SGE */
    w[1] = ((uint64_t)ah << bit(kAh)) | (kCtrl3Val << bit(kCtrl3));
    w[2] = (uint64_t)qkey << bit(kQkey);
    w[3] = (reqId >> 16) << bit(kReqIdEx); /* req_id_ex: the request id above its low 16 bits */
    w[4] = ((uint64_t)dstRkey << bit(kRemRkey)) | ((uint64_t)bytes << bit(kRemLen));
    w[5] = dstAddr << bit(kRemAddr);
    w[6] = ((uint64_t)(srcLkey & EFA_IO_TX_BUF_DESC_LKEY_MASK) << bit(kLocLkey)) | ((uint64_t)bytes << bit(kLocLen));
    w[7] = srcAddr << bit(kLocAddr);
  }

  NCCL_DEVICE_INLINE void storeMmio(uint64_t dst) const {
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
      asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" : : "l"(dst + i * 8u), "l"(w[i]) : "memory");
    }
  }
};

} // namespace efa_gda
} // namespace gin
} // namespace nccl

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_WQE_H_ */
