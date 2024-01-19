#ifndef NCCL_IBV_SYMBOLS_H_
#define NCCL_IBV_SYMBOLS_H_

#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

#include "nccl.h"

/* IB Verbs Function Pointers*/
struct ncclIbvSymbols {
  int (*ibv_internal_fork_init)(void);
  struct ibv_device** (*ibv_internal_get_device_list)(int *num_devices);
  void (*ibv_internal_free_device_list)(struct ibv_device **list);
  const char * (*ibv_internal_get_device_name)(struct ibv_device *device);
  struct ibv_context* (*ibv_internal_open_device)(struct ibv_device* device);
  int (*ibv_internal_close_device)(struct ibv_context *context);
  int (*ibv_internal_get_async_event)(struct ibv_context *context, struct ibv_async_event *event);
  void (*ibv_internal_ack_async_event)(struct ibv_async_event *event);
  int (*ibv_internal_query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
  int (*ibv_internal_query_port)(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
  int (*ibv_internal_query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
  int (*ibv_internal_query_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr);
  struct ibv_pd * (*ibv_internal_alloc_pd)(struct ibv_context *context);
  int (*ibv_internal_dealloc_pd)(struct ibv_pd *pd);
  struct ibv_mr * (*ibv_internal_reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
  struct ibv_mr * (*ibv_internal_reg_mr_iova2)(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access);
  /* DMA-BUF support */
  struct ibv_mr * (*ibv_internal_reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access);
  int (*ibv_internal_dereg_mr)(struct ibv_mr *mr);
  struct ibv_cq * (*ibv_internal_create_cq)(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector);
  int (*ibv_internal_destroy_cq)(struct ibv_cq *cq);
  struct ibv_qp * (*ibv_internal_create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
  int (*ibv_internal_modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
  int (*ibv_internal_destroy_qp)(struct ibv_qp *qp);
  const char * (*ibv_internal_event_type_str)(enum ibv_event_type event);
  int (*ibv_internal_query_ece)(struct ibv_qp *qp, struct ibv_ece *ece);
  int (*ibv_internal_set_ece)(struct ibv_qp *qp, struct ibv_ece *ece);
};

/* Constructs IB verbs symbols per rdma-core linking or dynamic loading mode */
ncclResult_t buildIbvSymbols(struct ncclIbvSymbols* ibvSymbols);

#endif  // NCCL_IBV_SYMBOLS_H_
