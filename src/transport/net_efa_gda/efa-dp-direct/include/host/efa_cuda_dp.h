// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_H
#define EFA_CUDA_DP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_cuda_cq_attrs {
	uint64_t comp_mask;
	uint64_t flags;
	uint8_t *buffer;
	uint32_t num_entries;
	uint32_t entry_size;
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
	uint32_t reserved;
};

struct efa_cuda_cq;
struct efa_cuda_qp;

struct efa_cuda_cq *efa_cuda_create_cq(struct efa_cuda_cq_attrs *attrs, uint32_t inlen);
void efa_cuda_destroy_cq(struct efa_cuda_cq *d_cq);
struct efa_cuda_qp *efa_cuda_create_qp(struct efa_cuda_qp_attrs *attrs, uint32_t inlen);
void efa_cuda_destroy_qp(struct efa_cuda_qp *d_qp);

int efa_cuda_init_cq(struct efa_cuda_cq *cq, struct efa_cuda_cq_attrs *attrs, uint32_t inlen);
int efa_cuda_init_qp(struct efa_cuda_qp *qp, struct efa_cuda_qp_attrs *attrs, uint32_t inlen);

int efa_cuda_get_version(int *major, int *minor, int *subminor);

#ifdef __cplusplus
}
#endif

#endif
