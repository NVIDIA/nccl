/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ibvwrap.h"
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif
#include "ibvsymbols.h"

static std::once_flag initOnceFlag;
static ncclResult_t initResult;
struct ncclIbvSymbols ibvSymbols;

ncclResult_t wrap_ibv_symbols(void) {
  std::call_once(initOnceFlag,
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
    INFO(NCCL_NET, "Call to " name " not supported"); \
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

NCCL_PARAM(IbMQpRetryAll, "IB_MQP_RETRY_ALL", 0);
NCCL_PARAM(IbMQpRetryCnt, "IB_MQP_RETRY_CNT", 34);
NCCL_PARAM(IbMQpRetryTimeout, "IB_MQP_RETRY_SLEEP_MSEC", 100); // in milliseconds

#define IBV_ERR_EQ(e, code)        (e == code || e == (-code))
#define IBV_MQP_RETRY_ERRNO(e)     (IBV_ERR_EQ(e, ETIMEDOUT))
#define IBV_MQP_RETRY_ERRNO_ALL(e) (ncclParamIbMQpRetryAll() ? (e != 0) : IBV_MQP_RETRY_ERRNO(e))

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

ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) {
#ifndef NCCL_BUILD_RDMA_CORE
  // First try and query the extended port attributes (e.g. active_speed_ex)
  if (ibv_query_port_ex(context, port_num, port_attr) != 0) {
    // Fall back to the original attribute API call, but zero all members first
    memset(port_attr, 0, sizeof(*port_attr));
    IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_port, ibv_internal_query_port(context, port_num, port_attr), 0, "ibv_query_port");
  }
#else
  // When using system rdma-core, use the regular ibv_query_port
  IBV_INT_CHECK_RET_ERRNO(ibvSymbols, ibv_internal_query_port, ibv_internal_query_port(context, port_num, port_attr), 0, "ibv_query_port");
#endif
  return ncclSuccess;
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

static void ibvQpStateName(enum ibv_qp_state state, char* msg, const size_t len) {
  switch (state) {
  case (IBV_QPS_RESET): snprintf(msg, len, "RESET"); break;
  case (IBV_QPS_INIT): snprintf(msg, len, "INIT"); break;
  case (IBV_QPS_RTR): snprintf(msg, len, "RTR"); break;
  case (IBV_QPS_RTS): snprintf(msg, len, "RTS"); break;
  case (IBV_QPS_SQD): snprintf(msg, len, "SQD"); break;
  case (IBV_QPS_SQE): snprintf(msg, len, "SQE"); break;
  case (IBV_QPS_ERR): snprintf(msg, len, "ERR"); break;
  case (IBV_QPS_UNKNOWN): snprintf(msg, len, "UNKNOWN"); break;
  default: snprintf(msg, len, "NOT RECOGNIZED (%d)", state); break;
  }
}

#define QP_ATTR(attr, userAttr, userFlag, mask) ((userFlag & mask) ? (userAttr) : (attr))

static void ibvModifyQpLog(struct ibv_qp* qp, enum ibv_qp_state qpState, struct ibv_qp_attr* userAttr, int userFlag, char* msg, size_t msgLen) {
  ncclResult_t res;
  int portNum = -1, gidIndex = -1;
  char localGidName[INET6_ADDRSTRLEN], remoteGidName[INET6_ADDRSTRLEN];
  const char *localGidRes = NULL, *remoteGidRes = NULL;

  char nextState[32], currState[32];
  ibvQpStateName(qp->state, currState, sizeof(currState));
  ibvQpStateName(qpState, nextState, sizeof(nextState));
  char devName[IBV_SYSFS_NAME_MAX] = "";
  snprintf(devName, sizeof(devName), "%s", (qp->pd->context) ? wrap_ibv_get_device_name(qp->pd->context->device) : "N/A");

  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
  int attr_mask = IBV_QP_PORT | IBV_QP_AV;
  res = wrap_ibv_query_qp(qp, &attr, attr_mask, &init_attr);
  struct ibv_qp_attr *qpAttr = (res == ncclSuccess) ? &attr : NULL;

  // port info, portAttr can be NULL if not given by the user and query_qp failed
  struct ibv_qp_attr *portAttr = QP_ATTR(qpAttr, userAttr, userFlag, IBV_QP_PORT);
  portNum = portAttr ? portAttr->port_num : -1;

  // address info, avAttr can be NULL if not given by the user and query_qp failed
  struct ibv_qp_attr *avAttr = QP_ATTR(qpAttr, userAttr, userFlag, IBV_QP_AV);
  if (avAttr && avAttr->ah_attr.is_global) {
    union ibv_gid *remoteGid = &avAttr->ah_attr.grh.dgid;
    remoteGidRes = ibvGetGidStr(remoteGid, remoteGidName, sizeof(remoteGidName));
    // we need pd->context to retrieve local GID, skip if not there
    if (!qp->pd->context) goto print;
    gidIndex =  avAttr->ah_attr.grh.sgid_index;
    union ibv_gid localGid;
    NCCLCHECKGOTO(wrap_ibv_query_gid(qp->pd->context, portNum, gidIndex, &localGid), res, print);
    localGidRes = ibvGetGidStr(&localGid, localGidName, sizeof(localGidName));
  }

print:
  snprintf(msg, msgLen, "on dev %s:%d, curr state %s, next state %s, local GID index %d, local GID %s, remote GID %s",
           devName, portNum, currState, nextState, gidIndex, localGidRes ? localGidName : "N/A", remoteGidRes ? remoteGidName : "N/A");
  return;
}

ncclResult_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  char qpMsg[1024];
  int ret = 0, attempts = 0;
  int maxCnt = (int)ncclParamIbMQpRetryCnt() + 1; // number of attempts = number of retry + 1
  int timeOut = (int)ncclParamIbMQpRetryTimeout();
  CHECK_NOT_NULL(ibvSymbols, ibv_internal_modify_qp);
  do {
    if (attempts > 0) {
      unsigned int sleepTime = timeOut * attempts;
      ibvModifyQpLog(qp, attr->qp_state, attr, attr_mask, qpMsg, sizeof(qpMsg));
      INFO(NCCL_NET, "Call to ibv_modify_qp failed with %d %s, %s, retrying %d/%d after %u msec of sleep", ret, strerror(ret), qpMsg, attempts, maxCnt, sleepTime);
      // sleep before retrying
      struct timespec tv = {.tv_sec = sleepTime / 1000, .tv_nsec = (sleepTime % 1000) * ((long)1e6)};
      nanosleep(&tv, NULL);
    }
    ret = ibvSymbols.ibv_internal_modify_qp(qp, attr, attr_mask);
    attempts++;
  } while (IBV_MQP_RETRY_ERRNO_ALL(ret) && attempts < maxCnt);
  if (ret != 0) {
    ibvModifyQpLog(qp, attr->qp_state, attr, attr_mask, qpMsg, sizeof(qpMsg));
    WARN("Call to ibv_modify_qp failed with %d %s, %s", ret, strerror(ret), qpMsg);
    return ncclSystemError;
  }
  return ncclSuccess;
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
