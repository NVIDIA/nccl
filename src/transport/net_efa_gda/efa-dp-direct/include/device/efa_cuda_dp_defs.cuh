// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_DEFS_CUH
#define EFA_CUDA_DP_DEFS_CUH

enum efa_cuda_wc_opcode {
	EFA_CUDA_WC_SEND,
	EFA_CUDA_WC_RDMA_WRITE,
	EFA_CUDA_WC_RDMA_READ,
/*
 * Set value of EFA_CUDA_WC_RECV so consumers can test if a completion is a
 * receive by testing (opcode & EFA_CUDA_WC_RECV).
 */
	EFA_CUDA_WC_RECV                  = 1 << 7,
	EFA_CUDA_WC_RECV_RDMA_WITH_IMM,
};

enum efa_cuda_processing_hint {
	EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE = 1 << 0,
};

#endif
