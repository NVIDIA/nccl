// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#include "../include/host/efa_cuda_dp.h"
#include "../include/device/efa_cuda_dp.cuh"

struct efa_cuda_cq *efa_cuda_create_cq(struct efa_cuda_cq_attrs *attrs, uint32_t inlen)
{
	return efa_cuda_dp::create_cq(attrs, inlen);
}

void efa_cuda_destroy_cq(efa_cuda_cq *d_cq)
{
	return efa_cuda_dp::destroy_cq(d_cq);
}

struct efa_cuda_qp *efa_cuda_create_qp(struct efa_cuda_qp_attrs *attrs, uint32_t inlen)
{
	return efa_cuda_dp::create_qp(attrs, inlen);
}

void efa_cuda_destroy_qp(struct efa_cuda_qp *d_qp)
{
	efa_cuda_dp::destroy_qp(d_qp);
}

int efa_cuda_get_version(int *major, int *minor, int *subminor)
{
	return efa_cuda_dp::get_version(major, minor, subminor);
}
