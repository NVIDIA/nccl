// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_IMPL_CUH
#define EFA_CUDA_DP_IMPL_CUH

#include <stdio.h>
#include <cstring>
#include <errno.h>
#include <stdint.h>
#include <cstddef>

#include "../common/efa_cuda_dp_types.h"
#include "../common/efa_io_defs.h"
#include "efa_cuda_dp_defs.cuh"

/* Update when the layout of QP/CQ changes */
typedef struct efa_cuda_cq_v0 efa_cuda_cq;
typedef struct efa_cuda_qp_v1 efa_cuda_qp;

typedef decltype(((efa_cuda_qp *)0)->sq) efa_cuda_sq;
typedef decltype(((efa_cuda_qp *)0)->rq) efa_cuda_rq;
typedef decltype(((efa_cuda_sq *)0)->wq) efa_cuda_wq;
typedef decltype(((efa_cuda_sq *)0)->wr_ctx) efa_cuda_wr_ctx;

#pragma push_macro("BIT")
#undef BIT
#define BIT(nr)		(1UL << (nr))

#define __bf_shf(x)	(__builtin_ffsll(x) - 1)

#define EFA_FIELD_GET(_mask, _reg)						\
	({									\
		(typeof(_mask))(((_reg) & (_mask)) >> __bf_shf(_mask));		\
	})

#define EFA_FIELD_PREP(_mask, _val)						\
	({									\
		((typeof(_mask))(_val) << __bf_shf(_mask)) & (_mask);		\
	})

#define BITS_PER_LONG	(8 * sizeof(long))

#define GENMASK(h, l)								\
	(((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BITS_PER_LONG - 1 - (h))))

#define EFA_GET(ptr, mask)							\
	EFA_FIELD_GET(mask##_MASK, *(typeof(*ptr) volatile *)(ptr))

#define EFA_SET(ptr, mask, value)						\
	({									\
		typeof(ptr) _ptr = ptr;				        	\
		*_ptr = (*_ptr & ~(mask##_MASK)) |				\
			EFA_FIELD_PREP(mask##_MASK, value);		        \
	})

#define efa_container_of(ptr, type, field)					\
	((type *) ((char *)ptr - offsetof(type, field)))


__device__ static inline int efa_cuda_cqe_is_pending(const efa_io_cdesc_common *cqe_common, int phase)
{
	return EFA_GET(&cqe_common->flags, EFA_IO_CDESC_COMMON_PHASE) == phase;
}

__device__ static inline efa_io_cdesc_common *efa_cuda_get_cqe(efa_cuda_cq *cq, uint32_t position)
{
	uint32_t index = (cq->cc + position) & cq->queue_mask;
	return (efa_io_cdesc_common *)(cq->buf + (index * cq->entry_size));
}

__device__ static inline int efa_cuda_get_cqe_phase(efa_cuda_cq *cq, uint32_t position)
{
	return cq->phase ^ (((cq->cc & cq->queue_mask) + position) >> cq->queue_size_shift);
}

__device__ static inline void *efa_cuda_cq_poll(efa_cuda_cq *cq, uint32_t position)
{
	efa_io_cdesc_common *cqe = efa_cuda_get_cqe(cq, position);
	int cqe_phase = efa_cuda_get_cqe_phase(cq, position);

	if (efa_cuda_cqe_is_pending(cqe, cqe_phase)) {
		__threadfence_block();

		return cqe;
	}
	return nullptr;
}
__device__ static inline int efa_cuda_cq_pop(efa_cuda_cq *cq, uint32_t amount)
{
	cq->phase = efa_cuda_get_cqe_phase(cq, amount);
	cq->cc += amount;

	return 0;
}

__device__ static inline enum efa_cuda_wc_opcode efa_cuda_wc_read_opcode(void *wc_buf)
{
	enum efa_io_send_op_type op_type;
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	op_type = (enum efa_io_send_op_type)EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_OP_TYPE);

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) == EFA_IO_SEND_QUEUE) {
		if (op_type == EFA_IO_RDMA_WRITE)
			return EFA_CUDA_WC_RDMA_WRITE;

		if (op_type == EFA_IO_RDMA_READ)
			return EFA_CUDA_WC_RDMA_READ;

		return EFA_CUDA_WC_SEND;
	}

	if (op_type == EFA_IO_RDMA_WRITE)
		return EFA_CUDA_WC_RECV_RDMA_WITH_IMM;

	return EFA_CUDA_WC_RECV;
}

__device__ static inline bool efa_cuda_wc_is_unsolicited(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_UNSOLICITED);
}

__device__ static inline uint64_t efa_cuda_wc_read_req_id(void *wc_buf)
{
	struct efa_io_tx_cdesc *tcqe = (struct efa_io_tx_cdesc *)wc_buf;

	return (uint64_t)tcqe->common.req_id |
	       (uint64_t)tcqe->req_id_ex.w[0] << 16 |
	       (uint64_t)tcqe->req_id_ex.w[1] << 32 |
	       (uint64_t)tcqe->req_id_ex.w[2] << 48;
}

__device__ static inline uint32_t efa_cuda_wc_read_vendor_err(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return cqe->status;
}

__device__ static inline bool efa_cuda_wc_has_imm(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_HAS_IMM);
}

__device__ static inline uint32_t efa_cuda_wc_read_imm_data(void *wc_buf)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = efa_container_of(wc_buf, struct efa_io_rx_cdesc, common);

	return rcqe->imm;
}

__device__ static inline uint32_t efa_cuda_wc_read_byte_len(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;
	struct efa_io_rx_cdesc_ex *rcqe;
	uint32_t length;

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) != EFA_IO_RECV_QUEUE)
		return 0;

	rcqe = efa_container_of(cqe, struct efa_io_rx_cdesc_ex, base.common);

	length = rcqe->base.length;
	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_OP_TYPE) == EFA_IO_RDMA_WRITE)
		length |= ((uint32_t)rcqe->u.rdma_write.length_hi << 16);

	return length;
}

__device__ static inline uint32_t efa_cuda_wc_read_qp_num(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return cqe->qp_num;
}

__device__ static inline uint32_t efa_cuda_wc_read_src_qp(void *wc_buf)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = efa_container_of(wc_buf, struct efa_io_rx_cdesc, common);

	return rcqe->src_qp_num;
}

__device__ static inline uint32_t efa_cuda_wc_read_slid(void *wc_buf)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = efa_container_of(wc_buf, struct efa_io_rx_cdesc, common);

	return rcqe->ah;
}

class EfaCudaWrBuilder {
private:
	efa_cuda_wr_ctx *wr_ctx;
	uint8_t *wr_buf;
	struct efa_io_tx_meta_desc *md;

	__device__ inline struct efa_io_remote_mem_addr *remote_mem()
	{
		return (struct efa_io_remote_mem_addr *)(wr_buf +
			__ldg(&wr_ctx->remote_mem_offset));
	}

	__device__ inline void set_remote_mem(uint32_t rkey, uint64_t remote_addr)
	{
		struct efa_io_remote_mem_addr *rmem = remote_mem();

		rmem->rkey = rkey;
		rmem->buf_addr_lo = remote_addr & 0xFFFFFFFF;
		rmem->buf_addr_hi = remote_addr >> 32;
	}

	__device__ inline void set_imm_data(uint32_t imm_data)
	{

		md->immediate_data = imm_data;
		EFA_SET(&md->ctrl1, EFA_IO_TX_META_DESC_HAS_IMM, 1);
	}

	__device__ inline int init_wr(enum efa_io_send_op_type op_type, uint64_t wr_id)
	{
		uint16_t wqe_size = __ldg(&wr_ctx->wqe_size);
		uint64_t *dst = (uint64_t *)wr_buf;

		for (int i = 0; i < wqe_size / sizeof(uint64_t); i++)
			dst[i] = 0;

		EFA_SET(&md->ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
		EFA_SET(&md->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, op_type);
		EFA_SET(&md->ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
		EFA_SET(&md->ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
		EFA_SET(&md->ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);

		md->req_id = (uint16_t)wr_id;
		md->req_id_ex.w[0] = (uint16_t)(wr_id >> 16);
		md->req_id_ex.w[1] = (uint16_t)(wr_id >> 32);
		md->req_id_ex.w[2] = (uint16_t)(wr_id >> 48);

		return 0;
	}

public:
	__device__ EfaCudaWrBuilder(efa_cuda_wr_ctx *wr_ctx, uint8_t *wr_buf)
		: wr_ctx(wr_ctx), wr_buf(wr_buf),
		  md((struct efa_io_tx_meta_desc *)wr_buf) {}

	__device__ inline int init_send(uint64_t wr_id)
	{
		return init_wr(EFA_IO_SEND, wr_id);
	}

	__device__ inline int init_send_imm(uint64_t wr_id, uint32_t imm_data)
	{
		int ret = init_wr(EFA_IO_SEND, wr_id);
		if (ret)
			return ret;

		set_imm_data(imm_data);
		return 0;
	}

	__device__ inline int init_rdma_write(uint64_t wr_id, uint32_t rkey, uint64_t remote_addr)
	{
		int ret = init_wr(EFA_IO_RDMA_WRITE, wr_id);
		if (ret)
			return ret;

		set_remote_mem(rkey, remote_addr);
		return 0;
	}

	__device__ inline int init_rdma_write_imm(uint64_t wr_id, uint32_t rkey,
						  uint64_t remote_addr, uint32_t imm_data)
	{
		int ret = init_rdma_write(wr_id, rkey, remote_addr);
		if (ret)
			return ret;

		set_imm_data(imm_data);
		return 0;
	}

	__device__ inline int init_rdma_read(uint64_t wr_id, uint32_t rkey, uint64_t remote_addr)
	{
		int ret = init_wr(EFA_IO_RDMA_READ, wr_id);
		if (ret)
			return ret;

		set_remote_mem(rkey, remote_addr);
		return 0;
	}

	__device__ inline int set_inline_data(void *addr, size_t length)
	{
		uint32_t max_inline = __ldg(&wr_ctx->max_inline_data);
		uint8_t op_type;
		uint8_t offset;

		if (length > max_inline)
			return -EINVAL;

		op_type = EFA_GET(&md->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
		switch (op_type) {
		case EFA_IO_SEND:
			offset = __ldg(&wr_ctx->send_inline_data_offset);
			break;
		case EFA_IO_RDMA_WRITE:
			offset = __ldg(&wr_ctx->write_inline_data_offset);
			if (!offset)
				return -EINVAL;

			remote_mem()->length = length;
			break;
		default:
			return -EINVAL;
		}

		EFA_SET(&md->ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG, 1);
		md->length = length;
		memcpy(wr_buf + offset, addr, length);
		return 0;
	}

	__device__ inline int set_sge(uint32_t lkey, uint64_t addr, uint32_t length)
	{
		struct efa_io_tx_buf_desc *tx_buf;
		uint8_t op_type;
		uint8_t offset;

		md->length = 1;

		op_type = EFA_GET(&md->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
		switch (op_type) {
		case EFA_IO_SEND:
			offset = __ldg(&wr_ctx->sgl_offset);
			break;
		case EFA_IO_RDMA_READ:
		case EFA_IO_RDMA_WRITE:
			offset = __ldg(&wr_ctx->local_mem_offset);
			remote_mem()->length = length;
			break;
		default:
			return -EINVAL;
		}

		tx_buf = (struct efa_io_tx_buf_desc *)(wr_buf + offset);
		tx_buf->length = length;
		EFA_SET(&tx_buf->lkey, EFA_IO_TX_BUF_DESC_LKEY, lkey);
		tx_buf->buf_addr_lo = addr & 0xFFFFFFFF;
		tx_buf->buf_addr_hi = addr >> 32;
		return 0;
	}

	__device__ inline void set_remote(uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey)
	{

		md->ah = ah;
		md->dest_qp_num = remote_qpn;
		md->qkey = remote_qkey;
	}

	__device__ inline void set_processing_hints(uint32_t hints)
	{
		uint32_t io_hints = 0;

		if (hints & EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE)
			io_hints |= EFA_IO_PROCESSING_HINT_BURST_PPS_SENSITIVE;

		EFA_SET(&md->ctrl3, EFA_IO_TX_META_DESC_PROCESSING_HINTS, io_hints);
	}
};

__device__ static inline int efa_cuda_get_wqe_phase(efa_cuda_wq *wq, uint32_t index_in_batch)
{
	return wq->phase ^ (((wq->pc & wq->queue_mask) + index_in_batch) >> wq->queue_size_shift);
}

__device__ static inline void efa_cuda_flush_sq_wrs(efa_cuda_qp *qp)
{
	if (!qp->sq.wq.wqes_pending)
		return;

	qp->sq.wq.phase = efa_cuda_get_wqe_phase(&qp->sq.wq, qp->sq.wq.wqes_pending);
	qp->sq.wq.pc += qp->sq.wq.wqes_pending;
	qp->sq.wq.wqes_pending = 0;

	__threadfence_system();
	*qp->sq.wq.db = qp->sq.wq.pc;
	__threadfence_system();
}

__device__ static inline int efa_cuda_start_sq_batch(efa_cuda_qp *qp, int batch_size)
{
	// TODO: check free space

	if (qp->sq.wq.wqes_pending + batch_size > qp->sq.wq.max_batch)
		efa_cuda_flush_sq_wrs(qp);

	qp->sq.wq.wqes_pending += batch_size;
	return 0;
}

__device__ static inline int efa_cuda_sq_batch_place_wr(efa_cuda_qp *qp, int index_in_batch, void *wr_buf)
{
	struct efa_io_tx_meta_desc *meta = (struct efa_io_tx_meta_desc *)wr_buf;
	uint32_t sq_desc_offset, queue_mask;
	efa_cuda_sq *sq = &qp->sq;
	uint64_t *src, *dst;
	uint16_t wqe_size;
	uint8_t *sq_buf;
	int wqe_phase;

	wqe_phase = efa_cuda_get_wqe_phase(&sq->wq, index_in_batch);
	sq_buf = (uint8_t *)__ldg((uint64_t *)&sq->wq.buf);
	queue_mask = __ldg(&sq->wq.queue_mask);
	wqe_size = __ldg(&sq->wr_ctx.wqe_size);

	EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase);

	src = (uint64_t *)wr_buf;
	sq_desc_offset = ((sq->wq.pc + index_in_batch) & queue_mask) * wqe_size;
	dst = (uint64_t *)(sq_buf + sq_desc_offset);
	for (int i = 0 ; i < wqe_size / sizeof(uint64_t) ; i++)
		dst[i] = src[i];

	return 0;
}

__device__ static inline int efa_cuda_post_recv_wr(efa_cuda_qp *qp, uint16_t req_id, uint64_t addr, uint32_t length, uint32_t lkey)
{
	struct efa_io_rx_desc wqe = {0};
	uint32_t rq_desc_offset;

	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_FIRST, 1);
	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LAST, 1);

	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LKEY, lkey);
	wqe.buf_addr_lo = addr;
	wqe.buf_addr_hi = addr >> 32;
	wqe.length = length;
	wqe.req_id = req_id;

	/* Copy descriptor to RX ring */
	rq_desc_offset = (qp->rq.wq.pc & qp->rq.wq.queue_mask) * sizeof(wqe);
	memcpy(qp->rq.wq.buf + rq_desc_offset, &wqe, sizeof(wqe));

	qp->rq.wq.pc++;
	if (!(qp->rq.wq.pc & qp->rq.wq.queue_mask))
		qp->rq.wq.phase++;

	qp->rq.wq.wqes_pending++;
	if (qp->rq.wq.wqes_pending == qp->rq.wq.max_batch) {
		__threadfence_system();
		*qp->rq.wq.db = qp->rq.wq.pc;

		qp->rq.wq.wqes_pending = 0;
	}

	return 0;
}

__device__ static inline void efa_cuda_flush_rq_wrs(efa_cuda_qp *qp)
{
	if (!qp->rq.wq.wqes_pending)
		return;

	__threadfence_system();
	*qp->rq.wq.db = qp->rq.wq.pc;
	qp->rq.wq.wqes_pending = 0;
}

__device__ static inline bool efa_cuda_is_cq_compatible(efa_cuda_cq *cq)
{
	return cq->comp_mask == 0;
}

__device__ static inline bool efa_cuda_is_qp_compatible(efa_cuda_qp *qp)
{
	return qp->comp_mask == 0;
}

#undef BIT
#pragma pop_macro("BIT")

#endif
