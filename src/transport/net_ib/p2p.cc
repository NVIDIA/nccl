/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "p2p.h"
#include "common.h"
#include "compiler.h"

NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);

const char* ncclIbReqTypeStr[] = { "Unused", "Send", "Recv", "Flush", "IPut" };

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex) {
  struct ncclIbNetCommDevBase* base = ncclIbGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r*8);
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif
  }

  // When nreqs==1, the Immediate Data carries the size of the send request.
  // In case of a multi-send (nreqs>1), the Immediate Data is ignored by the
  // receiver, as the size of the send request is written by the sender side
  // directly to the remote completion records array. Therefore, always
  // assigning the Immediate Data with the size, does not harm, and when it's
  // not required - it's ignored by the receiver side.
  uint32_t immData = reqs[0]->send.size;
  if (nreqs > 1) {
    int* sizes = comm->remCmplsRecords.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
  }

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = htobe32(immData);
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
    int devIndex = qp->devIndex;
    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      // Populating the correct gather information based on the device and
      // slot used.
      // Note that the lkey is already correct from the initialization phase.
      lastWr->sg_list = &(comm->devs[devIndex].sge);
      lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      lastWr->sg_list[0].length = nreqs*sizeof(int);
      // Populate the correct RKey based on the device used
      lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP profiling loop
    for (int r=0; r<nreqs; r++) {
      // Store the qpIndex for this request
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = comm->wrs[r].wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = comm->wrs[r].opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = comm->sges[r].length;
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  std::atomic_thread_fence(std::memory_order_seq_cst); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    // Populate events
    int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
    int qpIndex = -1;
    ncclIbQp* qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
      ncclIbAddEvent(req, qp->devIndex);
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r=0; r<nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and sanity checks
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbRequest*));
    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, size_t* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->recv.sizes = comm->cmplsRecords[slot];
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];

  ncclIbQp* ctsQp = NULL;;
  NCCLCHECK(ncclIbRecvCommGetQpForCts(comm, comm->base.fifoHead, &ctsQp));

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];

    // Send all applicable rkeys
    for (int j = 0; j < comm->base.vProps.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->base.fifoHead+1;
  }
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct rkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;

  // Populating the correct gather information based on the device and user
  // provided information
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = n*sizeof(struct ncclIbSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to CTS FIFO slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each CTS posting QP gets drained
  if (slot == ctsQp->devIndex) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    ncclIbAddEvent(req, ctsQp->devIndex);
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  comm->base.fifoHead++;

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
#endif

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);

  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);

  // Post recvs
  struct ibv_recv_wr* bad_wr;
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
    ncclIbAddEvent(req, qp->devIndex);
#ifdef NCCL_ENABLE_NET_PROFILING
    // Start a QP event for every request in the multirecv and every qp
    for (int r = 0; r < n; r++) {
      int nEventHandles = req->pInfo[r].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      req->pInfo[r].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[r].data.type = ncclProfileQp;
      req->pInfo[r].data.qp.device = qp->devIndex;
      req->pInfo[r].data.qp.wr_id = wr.wr_id;
      req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, phandles[r], pluginId, &req->pInfo[r].data));
      req->pInfo[r].nEventHandles++;
    }
#endif
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    ncclIbAddEvent(req, i);
  }

  *request = req;
  return ncclSuccess;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
static int getReqQpIndex(struct ncclIbRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

// Helper function to convert IB work completion status to string
static const char* ibvWcStatusStr(enum ibv_wc_status status) {
  switch (status) {
    case IBV_WC_SUCCESS:            return "IBV_WC_SUCCESS";
    case IBV_WC_LOC_LEN_ERR:        return "IBV_WC_LOC_LEN_ERR";
    case IBV_WC_LOC_QP_OP_ERR:      return "IBV_WC_LOC_QP_OP_ERR";
    case IBV_WC_LOC_EEC_OP_ERR:     return "IBV_WC_LOC_EEC_OP_ERR";
    case IBV_WC_LOC_PROT_ERR:       return "IBV_WC_LOC_PROT_ERR";
    case IBV_WC_WR_FLUSH_ERR:       return "IBV_WC_WR_FLUSH_ERR";
    case IBV_WC_MW_BIND_ERR:        return "IBV_WC_MW_BIND_ERR";
    case IBV_WC_BAD_RESP_ERR:       return "IBV_WC_BAD_RESP_ERR";
    case IBV_WC_LOC_ACCESS_ERR:     return "IBV_WC_LOC_ACCESS_ERR";
    case IBV_WC_REM_INV_REQ_ERR:    return "IBV_WC_REM_INV_REQ_ERR";
    case IBV_WC_REM_ACCESS_ERR:     return "IBV_WC_REM_ACCESS_ERR";
    case IBV_WC_REM_OP_ERR:         return "IBV_WC_REM_OP_ERR";
    case IBV_WC_RETRY_EXC_ERR:      return "IBV_WC_RETRY_EXC_ERR";
    case IBV_WC_RNR_RETRY_EXC_ERR:  return "IBV_WC_RNR_RETRY_EXC_ERR";
    case IBV_WC_LOC_RDD_VIOL_ERR:   return "IBV_WC_LOC_RDD_VIOL_ERR";
    case IBV_WC_REM_INV_RD_REQ_ERR: return "IBV_WC_REM_INV_RD_REQ_ERR";
    case IBV_WC_REM_ABORT_ERR:      return "IBV_WC_REM_ABORT_ERR";
    case IBV_WC_INV_EECN_ERR:       return "IBV_WC_INV_EECN_ERR";
    case IBV_WC_INV_EEC_STATE_ERR:  return "IBV_WC_INV_EEC_STATE_ERR";
    case IBV_WC_FATAL_ERR:          return "IBV_WC_FATAL_ERR";
    case IBV_WC_RESP_TIMEOUT_ERR:   return "IBV_WC_RESP_TIMEOUT_ERR";
    case IBV_WC_GENERAL_ERR:        return "IBV_WC_GENERAL_ERR";
    default:                        return "UNKNOWN_STATUS";
  }
}

// Helper function to convert IB work completion opcode to string
static const char* ibvWcOpcodeStr(enum ibv_wc_opcode opcode) {
  switch (opcode) {
    case IBV_WC_SEND:               return "IBV_WC_SEND";
    case IBV_WC_RDMA_WRITE:         return "IBV_WC_RDMA_WRITE";
    case IBV_WC_RDMA_READ:          return "IBV_WC_RDMA_READ";
    case IBV_WC_COMP_SWAP:          return "IBV_WC_COMP_SWAP";
    case IBV_WC_FETCH_ADD:          return "IBV_WC_FETCH_ADD";
    case IBV_WC_BIND_MW:            return "IBV_WC_BIND_MW";
    case IBV_WC_RECV:               return "IBV_WC_RECV";
    case IBV_WC_RECV_RDMA_WITH_IMM: return "IBV_WC_RECV_RDMA_WITH_IMM";
    default:                        return "UNKNOWN_OPCODE";
  }
}

static inline ncclResult_t ncclIbRequestRetrieveAsIndex(ncclIbRequest* reqs, uint32_t reqIndex, ncclIbRequest** req) {
  if (reqIndex < 0 || reqIndex >= NET_IB_MAX_REQUESTS) {
    WARN("NET/IB: %s: Invalid request index %d. Not in the range [%d, %d). Cannot retrieve request.", __func__, reqIndex, 0, NET_IB_MAX_REQUESTS);
    return ncclInternalError;
  }
  *req = &reqs[reqIndex];
  return ncclSuccess;
}

static inline bool ncclIbRequestIsComplete(struct ncclIbRequest *request) {
  return (request->events[0] == 0 && request->events[1] == 0 && request->events[2] == 0 && request->events[3] == 0);
}

static inline ncclResult_t ncclIbRequestComplete(struct ncclIbRequest* r, int* done, int* sizes) {
  TRACE(NCCL_NET, "r=%p done", r);
  *done = 1;
  if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
    for (int i=0; i<r->nreqs; i++) {
      sizes[i] = r->recv.sizes[i];
#ifdef NCCL_ENABLE_NET_PROFILING
      for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
        NCCLCHECK(ncclProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
      }
#endif
    }
  }
  if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
    sizes[0] = r->send.size;
#ifdef NCCL_ENABLE_NET_PROFILING
    for (int j = 0; j < r->pInfo[0].nEventHandles; j++) {
      NCCLCHECK(ncclProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  // Stop all remaining Qp events for this event
  NCCLCHECK(ncclIbFreeRequest(r));
  return ncclSuccess;
}

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t ncclIbLogCompletionWithError(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(commBase, devIndex);
  char localGidString[INET6_ADDRSTRLEN] = "";
  char remoteGidString[INET6_ADDRSTRLEN] = "";
  const char* localGidStr = NULL, *remoteGidStr = NULL;
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
    remoteGidStr = ibvGetGidStr(&commBase->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
  }

  char sockStr[SOCKET_NAME_MAXLEN+1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr);
  char *hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u %s%s%s%s hca %s",
      sockStr, ibvWcStatusStr(wc->status), wc->status,
      ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err,
      localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
  return ncclSuccess;
}

static inline ncclResult_t ncclIbCompletionEventProcess(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);

  struct ncclIbRequest* req = NULL;
  NCCLCHECK(ncclIbRequestRetrieveAsIndex(commBase->reqs, wc->wr_id & 0xff, &req));

  #ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, devIndex=%d",
    ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], devIndex);
  #endif
  if (req && req->type == NCCL_NET_IB_REQ_SEND) {
    for (int j = 0; j < req->nreqs; j++) {
      struct ncclIbRequest* sendReq = NULL;
      NCCLCHECK(ncclIbRequestRetrieveAsIndex(commBase->reqs, (wc->wr_id >> (j*8)) & 0xff, &sendReq));
      if ((sendReq->events[devIndex] <= 0)) {
        WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], devIndex, j);
        return ncclInternalError;
      }
      sendReq->events[devIndex]--;
#ifdef NCCL_ENABLE_NET_PROFILING
      // Stop Qp event for sendReq
      int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
#endif
    }
  } else {
    if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      if (req->type != NCCL_NET_IB_REQ_RECV) {
        WARN("NET/IB: wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM and req->type=%d", req->type);
        return ncclInternalError;
      }
      if (req->nreqs == 1) {
        req->recv.sizes[0] = be32toh(wc->imm_data);
      }
    }
    req->events[devIndex]--;
#ifdef NCCL_ENABLE_NET_PROFILING
    // Stop Qp event for workFifo
    for (int j = 0; j < req->nreqs; j++) {
      int qpIndex = getReqQpIndex(req, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;
  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[4];
  do {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (ncclIbRequestIsComplete(r)) {
      NCCLCHECK(ncclIbRequestComplete(r, done, sizes));
      return ncclSuccess;
    }

    totalWrDone = 0;
    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      // If we expect any completions from this device's CQ
      if (r->events[i] == 0) {
        continue;
      }
      TIME_START(3);
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
      if (wrDone == 0) continue;
      totalWrDone += wrDone;
      for (int w=0; w<wrDone; w++) {
        struct ibv_wc *wc = wcs+w;
        if (wc->status != IBV_WC_SUCCESS) {
          ncclIbLogCompletionWithError(r->base, wc, i);
          return ncclRemoteError;
        }
        NCCLCHECK(ncclIbCompletionEventProcess(r->base, wc, i));
      }
      // Once the IB fatal event is reported in the async thread, we want to propagate this error
      // to communicator and prevent further polling to reduce error pollution.
      NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
    }
  } while (totalWrDone > 0);

  // If no (more) CQEs found on any device, return and come back later
  return ncclSuccess;
}
