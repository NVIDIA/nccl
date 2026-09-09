// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#include <stdio.h>
#include <errno.h>
#include <stdint.h>

#include "../include/device/efa_cuda_dp.cuh"
#include "../include/device/efa_cuda_dp_impl.cuh"
#include "../include/common/efa_io_defs.h"

static bool is_buf_cleared(void *buf, size_t len)
{
	int i;

	for (i = 0; i < len; i++) {
		if (((uint8_t *)buf)[i])
			return false;
	}

	return true;
}

#define is_ext_cleared(ptr, inlen) \
	is_buf_cleared((uint8_t *)ptr + sizeof(*ptr), inlen - sizeof(*ptr))


namespace efa_cuda_dp {

struct efa_cuda_cq *create_cq(struct efa_cuda_cq_attrs *attrs, uint32_t inlen)
{
	cudaError_t cuda_err;

	if (inlen > sizeof(*attrs) && !is_ext_cleared(attrs, inlen)) {
		printf("Incompatible attributes struct\n");
		return nullptr;
	}

	if (__builtin_popcount(attrs->num_entries) != 1) {
		printf("CQ size must be positive power of 2\n");
		return nullptr;
	}

	// Allocate device CQ structure
	efa_cuda_cq *d_cq;
	cuda_err = cudaMalloc(&d_cq, sizeof(efa_cuda_cq));
	if (cuda_err != cudaSuccess) {
		printf("Failed to allocate device memory for cq: %s\n",
		       cudaGetErrorString(cuda_err));
		return nullptr;
	}

	// Initialize and copy CQ structure
	efa_cuda_cq h_cq = {};
	h_cq.buf = attrs->buffer;
	h_cq.entry_size = attrs->entry_size;
	h_cq.num_entries = attrs->num_entries;
	h_cq.queue_mask = attrs->num_entries - 1;
	h_cq.queue_size_shift = __builtin_ctz(attrs->num_entries);
	h_cq.cc = 0;
	h_cq.phase = 1;

	cuda_err = cudaMemcpy(d_cq, &h_cq, sizeof(efa_cuda_cq), cudaMemcpyHostToDevice);
	if (cuda_err != cudaSuccess) {
		cudaFree(d_cq);
		printf("Failed to copy cq to device: %s\n",
		       cudaGetErrorString(cuda_err));
		return nullptr;
	}

	return d_cq;
}

void destroy_cq(efa_cuda_cq *d_cq)
{
	cudaFree(d_cq);
}

struct efa_cuda_qp *create_qp(struct efa_cuda_qp_attrs *attrs, uint32_t inlen)
{
	cudaError_t cuda_err;

	if ((inlen > sizeof(*attrs) && !is_ext_cleared(attrs, inlen)) ||
	    attrs->reserved) {
		printf("Incompatible attributes struct\n");
		return nullptr;
	}

	if (__builtin_popcount(attrs->sq_num_entries) != 1 ||
	    __builtin_popcount(attrs->rq_num_entries) != 1) {
		printf("SQ and RQ sizes must be positive powers of 2\n");
		return nullptr;
	}

	// Allocate device QP structure
	efa_cuda_qp *d_qp;
	cuda_err = cudaMalloc(&d_qp, sizeof(efa_cuda_qp));
	if (cuda_err != cudaSuccess) {
		printf("Failed to allocate device memory for qp: %s\n",
		       cudaGetErrorString(cuda_err));
		return nullptr;
	}

	// Initialize QP structure on host
	efa_cuda_qp h_qp = {};

	// Initialize SQ
	h_qp.sq.wq.buf = attrs->sq_buffer;
	h_qp.sq.wq.db = attrs->sq_doorbell;
	h_qp.sq.wq.max_wqes = attrs->sq_num_entries;
	h_qp.sq.wq.max_batch = attrs->sq_max_batch;
	h_qp.sq.wq.queue_mask = attrs->sq_num_entries - 1;
	h_qp.sq.wq.queue_size_shift = __builtin_ctz(attrs->sq_num_entries);
	h_qp.sq.wq.wqes_pending = 0;
	h_qp.sq.wq.wqes_posted = 0;
	h_qp.sq.wq.wqes_completed = 0;
	h_qp.sq.wq.pc = 0;
	h_qp.sq.wq.phase = 0;
	// TODO: get from args or delete:
	h_qp.sq.max_inline_data = 32;
	h_qp.sq.max_rdma_sges = 2;

	// Initialize RQ
	h_qp.rq.wq.buf = attrs->rq_buffer;
	h_qp.rq.wq.db = attrs->rq_doorbell;
	h_qp.rq.wq.max_wqes = attrs->rq_num_entries;
	h_qp.rq.wq.max_batch = attrs->rq_num_entries;
	h_qp.rq.wq.queue_mask = attrs->rq_num_entries - 1;
	h_qp.rq.wq.queue_size_shift = __builtin_ctz(attrs->rq_num_entries);
	h_qp.rq.wq.wqes_pending = 0;
	h_qp.rq.wq.wqes_posted = 0;
	h_qp.rq.wq.wqes_completed = 0;
	h_qp.rq.wq.pc = 0;
	h_qp.rq.wq.phase = 1;

	// Copy QP structure to device
	cuda_err = cudaMemcpy(d_qp, &h_qp, sizeof(efa_cuda_qp), cudaMemcpyHostToDevice);
	if (cuda_err != cudaSuccess) {
		cudaFree(d_qp);
		printf("Failed to copy qp to device: %s\n",
		       cudaGetErrorString(cuda_err));
		return nullptr;
	}

	return d_qp;
}

void destroy_qp(efa_cuda_qp *d_qp)
{
	if (d_qp) {
		cudaFree(d_qp);
	}
}

int get_version(int *major, int *minor, int *subminor) {
	if (!major || !minor || !subminor)
		return -EINVAL;

	*major = EFA_CUDA_DP_VERSION_MAJOR;
	*minor = EFA_CUDA_DP_VERSION_MINOR;
	*subminor = EFA_CUDA_DP_VERSION_SUBMINOR;

	return 0;
}

}
