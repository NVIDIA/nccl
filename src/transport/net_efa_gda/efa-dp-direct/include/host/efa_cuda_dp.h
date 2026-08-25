// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_H
#define EFA_CUDA_DP_H

#include <stdint.h>

#include "../common/efa_cuda_dp_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Host-side API for efa-dp-direct.
 *
 * Translates the EFA queue attributes a caller obtained from the driver into the
 * queue structures that device code reads. Those structures are a wire format
 * between host and device: this library fills them, a GPU kernel consumes them.
 *
 * No CUDA header is included and no CUDA library is linked. Allocating device
 * memory and copying the initialized structure into it are the caller's
 * responsibility.
 */

struct efa_cuda_cq_attrs {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *buffer;
	uint32_t num_entries;
	uint32_t entry_size;
};

enum efa_cuda_wq_caps {
	EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID = 1 << 0,
};

struct efa_cuda_qp_attrs {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *sq_buffer;
	uint8_t *rq_buffer;
	uint32_t *sq_doorbell;
	uint32_t *rq_doorbell;
	uint32_t sq_num_entries;
	uint32_t sq_entry_size;
	uint32_t sq_max_batch;
	uint32_t rq_num_entries;
	uint32_t rq_entry_size;
	uint32_t sq_max_inline_data;
	uint32_t sq_max_rdma_sges;
	uint32_t sq_caps;
	uint32_t rq_caps;
};

struct efa_cuda_host_context;

struct efa_cuda_host_context *efa_cuda_host_context_create(int major, int minor, int subminor);
void efa_cuda_host_context_destroy(struct efa_cuda_host_context *ctx);


int efa_cuda_init_cq(struct efa_cuda_host_context *ctx, void *cq_buf, uint32_t cq_buf_size,
		     const struct efa_cuda_cq_attrs *attrs, uint32_t inlen);
int efa_cuda_init_qp(struct efa_cuda_host_context *ctx, void *qp_buf, uint32_t qp_buf_size,
		     const struct efa_cuda_qp_attrs *attrs, uint32_t inlen);

int efa_cuda_get_cq_size(struct efa_cuda_host_context *ctx);
int efa_cuda_get_qp_size(struct efa_cuda_host_context *ctx);

/* Library version. */
int efa_cuda_get_version(int *major, int *minor, int *subminor);

#ifdef __cplusplus
}
#endif

#endif
