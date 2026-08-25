// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_TYPES_H
#define EFA_CUDA_DP_TYPES_H

#include <stdint.h>

/*
 * Queue-structure layouts, one set per API major version. A version defines a
 * struct only where its layout differs from the previous version and reuses the
 * older type otherwise.
 *
 * These structs are the host/device wire format: the host fills them and the
 * GPU reads them by offset. A published layout is frozen; never modify one.
 * Add the next major version's layouts instead.
 */

struct efa_cuda_cq_v0 {
	uint64_t comp_mask;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t cc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_wq_v0 {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t max_batch;
	uint32_t wqes_pending;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	uint32_t pc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_rq_v0 {
	struct efa_cuda_wq_v0 wq;
};

struct efa_cuda_sq_v0 {
	struct efa_cuda_wq_v0 wq;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
};

struct efa_cuda_qp_v0 {
	uint64_t comp_mask;
	struct efa_cuda_sq_v0 sq;
	struct efa_cuda_rq_v0 rq;
};

struct efa_cuda_wr_ctx_v0 {
	uint8_t remote_mem_offset;
	uint8_t local_mem_offset;
	uint8_t sgl_offset;
	uint8_t send_inline_data_offset;
	uint8_t write_inline_data_offset;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
	uint16_t wqe_size;
};

struct efa_cuda_sq_v1 {
	struct efa_cuda_wq_v0 wq;
	struct efa_cuda_wr_ctx_v0 wr_ctx;
};

struct efa_cuda_qp_v1 {
	uint64_t comp_mask;
	struct efa_cuda_sq_v1 sq;
	struct efa_cuda_rq_v0 rq;
};

#endif
