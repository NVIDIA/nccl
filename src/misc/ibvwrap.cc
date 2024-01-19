/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ibvwrap.h"
#include <sys/types.h>
#include <unistd.h>

#include "ibvsymbols.h"

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
static ncclResult_t initResult;
struct ncclIbvSymbols ibvSymbols;

ncclResult_t wrap_ibv_symbols(void) {
  pthread_once(&initOnceControl,
               [](){ initResult = buildIbvSymbols(&ibvSymbols); });
  return initResult;
}

/* CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define CHECK_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
     WARN("lib wrapper not initialized."); \
     return ncclInternalError; \
  }

#define IBV_PTR_CHECK_ERRNO(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PTR_CHECK(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO_OPTIONAL(container, internal_name, call, success_retval, name, supported) \
  if (container.internal_name == NULL) { \
    INFO(NCCL_NET, "Call to " name " skipped, internal_name doesn't exist"); \
    *supported = 0; \
    return ncclSuccess; \
  } \
  int ret = container.call; \
  if (ret == ENOTSUP || ret == EOPNOTSUPP) { \
    INFO(NCCL_NET, "Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    *supported = 0; \
    return ncclSuccess; \
  } else if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    *supported = 1; \
    return ncclSystemError; \
  } \
  *supported = 1; \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO(container, internal_name, call, success_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s errno %d", strerror(ret), ret); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK(container, internal_name, call, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PASSTHRU(container, internal_name, call) \
  CHECK_NOT_NULL(container, internal_name); \
  container.call; \
  return ncclSuccess;

ncclResult_t wrap_ibv_fork_init() {
  IBV_INT_CHECK(ibvSymbols, ibv_internal_fork_init, ibv_internal_fork_init(), -1, "ibv_fork_init");
}

ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices) {
  *ret = ibvSymbols.ibv_internal_get_device_list(num_devices);
  if (*ret == NULL) *num_devices = 0;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list) {
  IBV_PASSTHRU(ibvSymbols, ibv_internal_free_device_list, ibv_internal_free_device_list(list));
}

const char *wrap_ibv_get_device_name(struct ibv_device *device) {
  if (ibvSymbols.ibv_internal_get_device_name == NULL) {
    WARN("lib wrapper not initialized.");
    exit(-1);
  }
  return ibvSymbols.ibv_internal_get_device_name(device);
}

ncclResult_t wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device) { /*returns 0 on success, -1 on failure*/
  IBV_PTR_CHECK(ibvSymbols, ibv_internal_open_device, ibv_internal_open_device(device), *ret, NULL, "ibv_open_device");
}

ncclResult_t wrap_ibv_close_device(struct ibv_context *context) { /*returns 0 on success, -1 on failure*/
  IBV_INT_CHECK(ibvSymbols, ibv_internal_close_device, ibv_internal_close_device(context), -1, "ibv_close_device");
}

ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) { /*returns 0 on success, and -1 on error*/
  IBV_INT_CHECK(ibvSymbols, ibv_internal_get_async_event, ibv_internal_get_async_event(context, event), -1, "ibv_get_async_event");
}

ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event) {
  IBV_PASSTHRU(ibvSymbols, ibv_internal_ack_async_event, ibv_internal_ack_async_event(event));
}

ncclResult_t wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_device, ibv_internal_query_device(context, device_attr), 0, "ibv_query_device");
}

ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_port, ibv_internal_query_port(context, port_num, port_attr), 0, "ibv_query_port");
}

ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_gid, ibv_internal_query_gid(context, port_num, index, gid), 0, "ibv_query_gid");
}

ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr) {
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_qp, ibv_internal_query_qp(qp, attr, attr_mask, init_attr), 0, "ibv_query_qp");
}

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context) {
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_alloc_pd, ibv_internal_alloc_pd(context), *ret, NULL, "ibv_alloc_pd");
}

ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_dealloc_pd, ibv_internal_dealloc_pd(pd), 0, "ibv_dealloc_pd");
}

ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access) {
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_reg_mr, ibv_internal_reg_mr(pd, addr, length, access), *ret, NULL, "ibv_reg_mr");
}

struct ibv_mr * wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  if (ibvSymbols.ibv_internal_reg_mr == NULL) {
    WARN("lib wrapper not initialized.");
    return NULL;
  }
  return ibvSymbols.ibv_internal_reg_mr(pd, addr, length, access);
}

ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access) {
  if (ibvSymbols.ibv_internal_reg_mr_iova2 == NULL) {
    return ncclInternalError;
  }
  if (ret == NULL) { return ncclSuccess; } // Assume dummy call
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_reg_mr_iova2, ibv_internal_reg_mr_iova2(pd, addr, length, iova, access), *ret, NULL, "ibv_reg_mr_iova2");
}

/* DMA-BUF support */
ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_reg_dmabuf_mr, ibv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access), *ret, NULL, "ibv_reg_dmabuf_mr");
}

struct ibv_mr * wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  if (ibvSymbols.ibv_internal_reg_dmabuf_mr == NULL) {
    errno = EOPNOTSUPP; // ncclIbDmaBufSupport() requires this errno being set
    return NULL;
  }
  return ibvSymbols.ibv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access);
}

ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_dereg_mr, ibv_internal_dereg_mr(mr), 0, "ibv_dereg_mr");
}

ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_create_cq, ibv_internal_create_cq(context, cqe, cq_context, channel, comp_vector), *ret, NULL, "ibv_create_cq");
}

ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq) {
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_destroy_cq, ibv_internal_destroy_cq(cq), 0, "ibv_destroy_cq");
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp) {
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_destroy_qp, ibv_internal_destroy_qp(qp), 0, "ibv_destroy_qp");
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
  IBV_PTR_CHECK_ERRNO(ibvSymbols, ibv_internal_create_qp, ibv_internal_create_qp(pd, qp_init_attr), *ret, NULL, "ibv_create_qp");
}

ncclResult_t wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_modify_qp, ibv_internal_modify_qp(qp, attr, attr_mask), 0, "ibv_modify_qp");
}

ncclResult_t wrap_ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece, int* supported) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO_OPTIONAL(ibvSymbols, ibv_internal_query_ece, ibv_internal_query_ece(qp, ece), 0, "ibv_query_ece", supported);
}

ncclResult_t wrap_ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece, int* supported) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO_OPTIONAL(ibvSymbols, ibv_internal_set_ece, ibv_internal_set_ece(qp, ece), 0, "ibv_set_ece", supported);
}

ncclResult_t wrap_ibv_event_type_str(char **ret, enum ibv_event_type event) {
  *ret = (char *) ibvSymbols.ibv_internal_event_type_str(event);
  return ncclSuccess;
}
