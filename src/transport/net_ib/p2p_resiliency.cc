/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p_resiliency.h"
#include "p2p.h" // For replay (ncclIbMultiSend() and ncclIbPostFifo())
#include "connect.h" // For ncclIbCreateQp()

NCCL_PARAM(IbResiliencyPortFailover, "IB_RESILIENCY_PORT_FAILOVER", 0);
NCCL_PARAM(IbResiliencyPortFailoverMaxAttempts, "IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS", 1);
NCCL_PARAM(IbResiliencyPortFailoverProbeDelay, "IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY", 10); // In milliseconds

#define MSEC_TO_NSEC 1000000ULL

// Checks if the error indicated in the given work completion is fatal or not.
static ncclResult_t ncclIbResiliencyCheckErrorNotFatal(struct ncclIbResiliency* resCtx, struct ibv_wc *wc, int devIndex) {
  int nFailedDevices = 0;
  bool fatalCompletionStatus = true;
  const char* failureReason = NULL;
  for (int i = 0; i < resCtx->ndevs; i++) {
    if (i == devIndex || resCtx->devs[i].state != ncclIbResiliencyDevStateOk) {
      nFailedDevices++;
    }
  }

  switch (wc->status) {
    case IBV_WC_WR_FLUSH_ERR:
    case IBV_WC_RETRY_EXC_ERR:
      fatalCompletionStatus = false;
      break;
    default:
      WARN("NET/IB: %s: Unsupported completion status %s (%d)", __func__, ibvWcStatusStr(wc->status), wc->status);
      break;
  }

  if ((nFailedDevices < resCtx->ndevs) && !fatalCompletionStatus) {
    INFO(NCCL_NET, "NET/IB: %s: The error is not fatal. Trying to continue...", __func__);
    return ncclSuccess;
  }

  if (nFailedDevices == resCtx->ndevs) {
    failureReason = "No functional devices left";
  } else {
    failureReason = "Fatal error status in work completion";
  }

  WARN("NET/IB: %s: The error is fatal (%s). Cannot continue.", __func__, failureReason);
  return ncclRemoteError;
}

// Function to replace all QPs associated with a given failed device.
static ncclResult_t ncclIbResiliencyReplaceQps(struct ncclIbResiliency* resCtx, int failedDevIndex) {
  // Iterate over all active QPs and replace the ones that are associated with
  // the failed device
  struct ncclIbQp** activeQps = resCtx->baseComm->activeQps;
  for (int qpIndex = 0; qpIndex < resCtx->baseComm->nqps; qpIndex++) {
    struct ncclIbQp* failedQp = activeQps[qpIndex];
    if (failedQp->devIndex != failedDevIndex) {
      // This is not a failed QP
      continue;
    }
    // Find the failed QP's index in the baseComm.qps[] array
    int failedQpIndex = failedQp - resCtx->baseComm->qps;
    assert(failedQpIndex >= 0 && failedQpIndex < resCtx->baseComm->nqps);

    // After finding a failed QP, iterate over the QPs to find a replacement QP
    // that is not associated with a failed device.
    bool replaced = false;
    int newQpIndex = -1;
    int newDevIndex = -1;
    int offset = 1;
    enum ncclIbResiliencyDevState newDevState = ncclIbResiliencyDevStateError;
    do {
      newQpIndex = (failedQpIndex + offset) % resCtx->baseComm->nqps;

      // Check if the new QP is on a functional device
      newDevIndex = resCtx->baseComm->qps[newQpIndex].devIndex;
      newDevState = resCtx->devs[newDevIndex].state;
      if (newDevState != ncclIbResiliencyDevStateOk) {
        offset++;
        WARN("NET/IB: %s: Cannot replace QP with qpIndex=%d because the new QP (qpIndex=%d, devIndex=%d) is not on a functional device (state=%d)", __func__, failedQpIndex, newQpIndex, newDevIndex, newDevState);
        continue;
      }

      INFO(NCCL_NET, "NET/IB: %s: Replacing QP: qpIndex=%d (qp_num=%u, devIndex=%d) to qpIndex=%d (qp_num=%u, devIndex=%d) on %s communicator (comm=%p)", __func__, failedQpIndex, resCtx->baseComm->qps[failedQpIndex].qp->qp_num, resCtx->baseComm->qps[failedQpIndex].devIndex, newQpIndex, resCtx->baseComm->qps[newQpIndex].qp->qp_num, resCtx->baseComm->qps[newQpIndex].devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
      activeQps[qpIndex] = &resCtx->baseComm->qps[newQpIndex];
      replaced = true;

    } while (replaced == false && newQpIndex != failedQpIndex);

    if (!replaced) {
      WARN("NET/IB: %s: Could not find a replacement QP for the failed QP with qpIndex=%d (devIndex=%d)", __func__, failedQpIndex, failedQp->devIndex);
      return ncclInternalError;
    }

  }
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencySendRequestInit(struct ncclIbResiliencySend* sendResCtx, ncclIbRequest* request, int devIndex) {
  int slot = request->id % NET_IB_MAX_REQUESTS;
  struct ncclIbResiliencyRequestSend* failedSendRequest = &sendResCtx->failedRequests[slot];

  // Check if the request is/was already tracked
  if (failedSendRequest->id == request->id + 1) {
    if (failedSendRequest->request != NULL) {
      // It might be that a different send request that is part of this
      // multi-send request already got an error and is being replayed.
      // No need to initate a new tracking.
      INFO(NCCL_NET, "NET/IB: %s: No need to add this failed request (req=%p, comm=%p, id=%ld) while another request (req=%p, comm=%p, id=%ld) is already being tracked in the same slot (slot=%d).", __func__, request, request->base, request->id, failedSendRequest->request, failedSendRequest->request->base, failedSendRequest->request->id, slot);
      return ncclSuccess;
    } else {
      // The request was already replayed and released. The CQE should be ignored.
      INFO(NCCL_NET, "NET/IB: %s: Attempting to initiate a replay protocol but the failed request was already handled (req=%p, comm=%p, id=%ld, slot=%d, req.type=%s).", __func__, request, request->base, request->id, slot, ncclIbReqTypeStr[request->type]);
      return ncclSuccess;
    }
  }

  if (request->id + 1 <= failedSendRequest->id) {
    WARN("NET/IB: %s: Attempting to initiate a replay using an old request (req=%p, comm=%p, id=%ld, slot=%d, failedSendRequest.id=%ld).", __func__, request, request->base, request->id, slot, failedSendRequest->id);
    return ncclInternalError;
  }

  if (request->type != NCCL_NET_IB_REQ_SEND) {
    WARN("NET/IB: %s: Attempting to initiate a failed request using a '%s' request while expecting a 'send' request (req=%p, comm=%p, id=%ld, slot=%d, failedSendRequest.id=%ld).", __func__, ncclIbReqTypeStr[request->type], request, request->base, request->id, slot, failedSendRequest->id);
    return ncclInternalError;
  }

  failedSendRequest->state = ncclIbResiliencyRequestStatePending;
  failedSendRequest->request = request;
  failedSendRequest->errorInfo.devIndex = devIndex;
  failedSendRequest->errorInfo.time = clockNano();
  failedSendRequest->failedAttempts = 0;
  failedSendRequest->id = request->id+1;
  sendResCtx->base.outstandingRequests++;
  sendResCtx->base.inProgress = true;
  INFO(NCCL_NET, "NET/IB: %s: Tracking a new failed send request (req=%p, comm=%p, id=%ld, slot=%d, devIndex=%d, time=%ld, total tracked requests: %d).", __func__, request, request->base, request->id, slot, devIndex, failedSendRequest->errorInfo.time, sendResCtx->base.outstandingRequests);
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencySendRequestFree(struct ncclIbResiliencySend* sendResCtx, struct ncclIbResiliencyRequestSend* failedSendRequest) {
  assert(failedSendRequest != NULL);
  if (failedSendRequest->request == NULL) {
    int slot = failedSendRequest - sendResCtx->failedRequests;
    WARN("NET/IB: %s: Attempting to free a non-existent failed request (slot=%d).", __func__, slot);
    return ncclInternalError;
  }
  INFO(NCCL_NET, "NET/IB: %s: Done handling failed send request (req=%p, comm=%p, id=%ld, slot=%ld).", __func__, failedSendRequest->request, failedSendRequest->request->base, failedSendRequest->request->id, failedSendRequest->request->id % NET_IB_MAX_REQUESTS);

  // Note that ID is not reset to allow ignoring old CQEs that might still be
  // in the CQ even after the failed send request was handled completely and
  // freed.
  failedSendRequest->state = ncclIbResiliencyRequestStatePending;
  failedSendRequest->request = NULL;
  failedSendRequest->errorInfo = {0};
  failedSendRequest->failedAttempts = 0;

  sendResCtx->base.outstandingRequests--;
  return ncclSuccess;
}

// Function to repost a given request.
static ncclResult_t ncclIbResiliencyRepostRequest(struct ncclIbRequest* request) {
  if (request->type == NCCL_NET_IB_REQ_UNUSED) {
    WARN("NET/IB: %s: Attempting to repost an unused request (id=%ld).", __func__, request->id);
    return ncclInternalError;
  }
  int slot = request->id % NET_IB_MAX_REQUESTS;
  if (request->type == NCCL_NET_IB_REQ_SEND) {
      struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)request->base->resiliency;
      struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)request->base;
      struct ncclIbRequest** sendReqs = sendComm->sendReqs[slot];
      for (int r = 0; r < request->nreqs; r++) {
        // Clear all event counters and later on increment only the required
        // ones based on the probing results on which QP a retransmission is
        // required.
        memset(sendReqs[r]->events, 0, sizeof(sendReqs[r]->events));

        // Populate events
        int nqps = ncclIbCommBaseGetNqpsPerRequest(sendReqs[r]->base);
        int qpIndex = -1;
        ncclIbQp* qp = NULL;
        for (int i = 0; i < nqps; i++) {
          // TODO: This code does not handle the case where a send request fails twice!
          // If that device that is used for retransmission fails during retransmission,
          // the logic here will retrieve the QP that was used for the first send attempt
          // and not the QP that was used for the second send attempt! Causing
          // probably data corruption or a hang.
          NCCLCHECK(ncclIbCommBaseGetQpForRequest(sendReqs[r]->base, sendReqs[r]->id, i, &qp, &qpIndex));

          // Selective Retransmission:
          // If the probing result shows that the data was delivered successfully on this QP,
          // we don't need to retransmit it.
          if (sendResCtx->probingResults[slot][qpIndex] == true) {
            INFO(NCCL_NET, "NET/IB: %s: Skipping retransmission on QP index %d (req=%p, comm=%p, id=%ld, slot=%d) as it was already delivered.", __func__, qpIndex, sendReqs[r], sendReqs[r]->base, sendReqs[r]->id, slot);
            continue;
          }

          INFO(NCCL_NET, "NET/IB: %s: Retransmitting reqIndex=%d on qp_num=%u (req=%p, comm=%p, id=%ld, slot=%d) as it was not delivered.", __func__, r, qp->qp->qp_num, sendReqs[r], sendReqs[r]->base, sendReqs[r]->id, slot);
          // Reset the sentData for this QP since we are going to retransmit it.
          sendReqs[r]->send.sentData[qpIndex] = false;
          ncclIbAddEvent(sendReqs[r], qp->devIndex);
        }
      }
      INFO(NCCL_NET, "NET/IB: %s: Reposting send request (request=%p, comm=%p, id=%ld, slot=%ld, nreqs=%d)", __func__, request, request->base, request->id, request->id % NET_IB_MAX_REQUESTS, request->nreqs);
      NCCLCHECK(ncclIbMultiSend((struct ncclIbSendComm*)request->base, slot));
  } else if (request->type == NCCL_NET_IB_REQ_RECV) {
    INFO(NCCL_NET, "NET/IB: %s: Reposting CTS (request=%p, comm=%p, id=%ld, slot=%ld)", __func__, request, request->base, request->id, request->id % NET_IB_MAX_REQUESTS);
    NCCLCHECK(ncclIbPostFifo((struct ncclIbRecvComm*)request->base, request, slot));
  } else {
    WARN("NET/IB: %s: Unsupported type of request reposting (type=%d, id=%ld).", __func__, request->type, request->id);
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyHandleCompletionErrorReceiver(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  INFO(NCCL_NET,"NET/IB: %s: Handling an error on the receiver side (comm %p)", __func__, resCtx->baseComm);
  bool inRecvRange = (wc->wr_id >= 0 && wc->wr_id <= NET_IB_MAX_REQUESTS);
  bool inFlushRange = (wc->wr_id >= NCCL_IB_FLUSH_REQ_WR_ID_OFFSET && wc->wr_id < (NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS));
  if (!inRecvRange && !inFlushRange && (wc->wr_id != NCCL_IB_RECV_WR_ID_DUMMY)) {
    WARN("NET/IB: %s: Invalid wr_id (%ld). Unable to retrieve a request on the receiver side (comm=%p)", __func__, wc->wr_id, resCtx->baseComm);
    return ncclInternalError;
  }
  if (wc->wr_id == NCCL_IB_RECV_WR_ID_DUMMY) {
    // The only case where the HW will produce an error on the receiver side on
    // a receive queue is a flush error. The flush error indicates that the QP
    // transitioned into an error state and all outstanding work requests are
    // now flushed.
    assert(wc->status == IBV_WC_WR_FLUSH_ERR);
    // In this case, there is nothing left to do.
    INFO(NCCL_NET, "NET/IB: %s: Ignoring flush error on a QP (comm=%p, wc.wr_id=%ld, wc.status=%s(%d)).", __func__, resCtx->baseComm, wc->wr_id, ibvWcStatusStr(wc->status), wc->status);
    return ncclSuccess;
  }

  ncclIbRequest* request = NULL;
  if (inFlushRange) {
    // Completion for a flush request is offset by NCCL_IB_FLUSH_REQ_WR_ID_OFFSET
    ncclIbRequestRetrieveAsIndex(resCtx->baseComm->reqs, wc->wr_id - NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, &request);
  } else {
    struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)resCtx->baseComm;
    request = recvComm->recvReqs[wc->wr_id];
  }

  INFO(NCCL_NET, "NET/IB: %s: The receiver side request that got an error is %p (req=%p, comm=%p, id=%ld)", __func__, request, request, request->base, request->id);

  switch (request->type) {
    case NCCL_NET_IB_REQ_FLUSH:
      // When a flush request encounters an error, it's ignored and the event
      // counter on that device is set to zero, so the flush request could be
      // completed on other devices if needed.
      request->events[devIndex] = 0;
      INFO(NCCL_NET, "NET/IB: %s: Ignoring error on flush request (req=%p, comm=%p, id=%ld) on device index %d", __func__, request, request->base, request->id, devIndex);
      break;
    case NCCL_NET_IB_REQ_RECV:
      // Assert it's a CTS message that got an error.
      // When error occurs the CQE's opcode is not valid and cannot be read!
      // The only valid fields are: wr_id, status, qp_num, and vendor_err.
      // From: https://www.rdmamojo.com/2013/02/15/ibv_poll_cq/
      // Assert the CQE belongs to a CTS and not a data transfer.
      assert(wc->wr_id != NCCL_IB_RECV_WR_ID_DUMMY);
      // CTS is reposted immediately
      NCCLCHECK(ncclIbResiliencyRepostRequest(request));
      break;
    case (NCCL_NET_IB_REQ_UNUSED):
      // This might happen for a CTS message. Consider a case where a HW ack
      // failed for a CTS message but before the receiver got a CQE with error
      // because of a HW timeout, the sender already completed the data transfer
      // and receiver completed the receive request. Note that receiver does not
      // verify if the CTS was completed for every receive request before
      // completing a receive request.
      // When error occurs the CQE's opcode is not valid and cannot be read!
      WARN("NET/IB: %s: Unrecognized request. It might be a CTS message for which the request was already completed. Continue.", __func__);
      break;
    default:
      WARN("NET/IB: %s: Unrecognized request type. request->type=%d", __func__, request->type);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyHandleCompletionErrorSender(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  ncclResult_t res;
  ncclIbRequest* request = NULL;

  uint64_t slot = (wc->wr_id & 0xff);
  struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)resCtx->baseComm;
  request = sendComm->sendReqs[slot][0];

  if (request == NULL) {
    WARN("NET/IB: %s: Encountered a stale CQE with error for slot=%ld. Slot was already handled (comm=%p, wc.wr_id=%ld, wc.status=%s(%d), wc.opcode=%s(%d)).", __func__, slot, resCtx->baseComm, wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode);
    return ncclSuccess;
  }

  struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
  res = ncclIbResiliencySendRequestInit(sendResCtx, request, devIndex);
  if (res != ncclSuccess) {
    WARN("NET/IB: %s: Failed to initialize a resiliency send request (req=%p, comm=%p, id=%ld, type=%s, wc.wr_id=%ld, wc.status=%s(%d), wc.opcode=%s(%d), slot=%ld).", __func__, request, request->base, request->id, ncclIbReqTypeStr[request->type], wc->wr_id, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, slot);
    return res;
  }

  // The request will be progressed in the main progress function.
  return ncclSuccess;
}

// Mark the device as failed and replace its QPs.
static ncclResult_t ncclIbResiliencyHandleDeviceFailure(struct ncclIbResiliency* resCtx, int devIndex) {
  if (resCtx->devs[devIndex].state == ncclIbResiliencyDevStateOk) {
    WARN("NET/IB: %s: Device %d marked as failed. (%s comm=%p)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
    resCtx->devs[devIndex].state = ncclIbResiliencyDevStateError;
    NCCLCHECK(ncclIbResiliencyReplaceQps(resCtx, devIndex));
  } else {
    INFO(NCCL_NET, "NET/IB: %s: Device %d was already marked as failed.", __func__, devIndex);
  }
  return ncclSuccess;
}

// -----------------------------
// Probing related functions
// -----------------------------

// After a probe is completed, handle the results of the probe. If the probe
// shows that the request was completed successfully on all QPs, the request
// is completed. Otherwise, the request is reposted.
static ncclResult_t ncclIbResiliencyHandleProbeCompleted(struct ncclIbResiliencySend* sendResCtx, struct ncclIbResiliencyRequestSend* failedRequest) {
  int slot = failedRequest->request->id % NET_IB_MAX_REQUESTS;
  bool missingData = false;
  for (int qpIndex = 0; qpIndex < NCCL_IB_MAX_QPS; qpIndex++) {
    if (failedRequest->request->send.sentData[qpIndex] == false) {
      // This QP was not used for this request.
      continue;
    }
    if (sendResCtx->probingResults[slot][qpIndex] == true) {
      // The probing on this QP was successful, so the request was completed
      // successfully on this QP.
      continue;
    }
    missingData = true;
    INFO(NCCL_NET, "NET/IB: %s: Missing data from QP index %d (req=%p, slot=%d devIndex=%d)", __func__, qpIndex, failedRequest->request, slot, failedRequest->request->base->qps[qpIndex].devIndex);
    break;
  }
  if (!missingData) {
    // All data was delivered to the sender and no need for retransmission.
    INFO(NCCL_NET, "NET/IB: %s: Probing conclusion: All data was delivered for request %p (req=%p, comm=%p, id=%ld, slot=%d, nreqs=%d). Completing the request.", __func__, failedRequest->request, failedRequest->request, failedRequest->request->base, failedRequest->request->id, slot, failedRequest->request->nreqs);
    // Clear all events on the request so it could be completed towards the
    // user as well. Note that in case of a multi-send requests, all requests
    // are also cleared.
    struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)failedRequest->request->base;
    struct ncclIbRequest** sendReqs = sendComm->sendReqs[slot];
    for (int r = 0; r < failedRequest->request->nreqs; r++) {
      memset(sendReqs[r]->events, 0, sizeof(sendReqs[r]->events));
      INFO(NCCL_NET, "NET/IB: %s: Clearing events on send request %p (req=%p, comm=%p, id=%ld, slot=%d, reqIdx=%d)", __func__, sendReqs[r], sendReqs[r], sendReqs[r]->base, sendReqs[r]->id, slot, r);
    }
    return ncclSuccess;
  } else {
    // Repost the send request
    NCCLCHECK(ncclIbResiliencyRepostRequest(failedRequest->request));
  }
  return ncclSuccess;
}

// Posts a probe (RDMA Read) operation to check the status of a send request
// that encountered an error.
static ncclResult_t ncclIbResiliencyProbePost(struct ncclIbResiliencySend* sendResCtx, struct ncclIbResiliencyRequestSend* failedSendRequest) {
  assert(failedSendRequest->state == ncclIbResiliencyRequestStatePending);

  if (failedSendRequest->failedAttempts > ncclParamIbResiliencyPortFailoverMaxAttempts()) {
    WARN("NET/IB: %s: Maximum number of probing attempts (%ld) reached for request %p (id=%ld). Cannot post another probe.", __func__, ncclParamIbResiliencyPortFailoverMaxAttempts(), failedSendRequest->request, failedSendRequest->request->id);
    return ncclRemoteError;
  }

  // Get current time and calculate elapsed time since error
  uint64_t currentTime = clockNano();
  uint64_t elapsedTime = currentTime - failedSendRequest->errorInfo.time;
  uint64_t requiredWaitTime = ncclParamIbResiliencyPortFailoverProbeDelay() * MSEC_TO_NSEC;

  if (elapsedTime < requiredWaitTime) {
    return ncclSuccess;
  }

  INFO(NCCL_NET, "NET/IB: %s: Probe delay elapsed (%llu ms) for request %p, posting the probe", __func__, elapsedTime / MSEC_TO_NSEC, failedSendRequest->request);

  int devIndex = 0;
  for (devIndex = 0; devIndex < sendResCtx->base.ndevs; devIndex++) {
    if (sendResCtx->base.devs[devIndex].state == ncclIbResiliencyDevStateOk) {
      // This device is functional. Use it to post the probe.
      break;
    }
  }

  if (devIndex == sendResCtx->base.ndevs) {
    WARN("NET/IB: %s: Could not find a functional device to post the probe for request %p (id=%ld)", __func__, failedSendRequest->request, failedSendRequest->request->id);
    return ncclInternalError;
  }

  struct ncclIbResiliency* resCtx = &sendResCtx->base;
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  struct ncclIbQp* probingQp = &resCtx->probingQps[devIndex];

  struct ibv_send_wr probeWr = {0};
  struct ibv_sge sge = {0};

  int slot = failedSendRequest->request->id % NET_IB_MAX_REQUESTS;

  // The length of the RDMA Read and where to write the response to
  sge.addr = (uint64_t)&(sendResCtx->probingResults[slot]);
  sge.length = NCCL_IB_MAX_QPS * sizeof(bool);
  sge.lkey = resDev->probingResultMr->lkey;

  // Format the RDMA Read WR
  probeWr.sg_list = &sge;
  probeWr.num_sge = 1;
  probeWr.opcode = IBV_WR_RDMA_READ;
  probeWr.wr_id = slot;
  probeWr.send_flags = IBV_SEND_SIGNALED;
  probeWr.next = NULL;

  uint64_t remoteAddr = sendResCtx->remCmplRecordsInfo[probingQp->remDevIdx].addr;
  // Skip "slot" number of completion records
  remoteAddr += sizeof(struct ncclIbRequestCompletionRecord) * slot;
  // Skip the "sizes" array to point to the "completions" array.
  remoteAddr += sizeof(int) * NCCL_NET_IB_MAX_RECVS;
  probeWr.wr.rdma.remote_addr = remoteAddr;
  probeWr.wr.rdma.rkey = sendResCtx->remCmplRecordsInfo[probingQp->remDevIdx].rkey;

  INFO(NCCL_NET, "NET/IB: %s: Posting probe (slot=%d, id=%ld, devIndex=%d, qp_num=%u)", __func__, slot, failedSendRequest->request->id, devIndex, probingQp->qp->qp_num);

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(probingQp->qp, &probeWr, &bad_wr));
  failedSendRequest->state = ncclIbResiliencyRequestStateProbePosted;
  resDev->nOutstandingProbes++;
  assert(resDev->nOutstandingProbes <= NET_IB_MAX_REQUESTS);
  return ncclSuccess;
}

static ncclResult_t ncclIbResiliencyProbeHandleCompletionEvent(struct ncclIbResiliencySend* sendResCtx, struct ibv_wc* probeWc, int devIndex) {

  INFO(NCCL_NET, "NET/IB: %s: Got probing completion (devIndex=%d, wc->status=%d, wc->opcode=%d, wc->wr_id=%ld, wc->qp_num=%u)", __func__, devIndex, probeWc->status, probeWc->opcode, probeWc->wr_id, probeWc->qp_num);

  struct ncclIbResiliencyRequestSend* failedRequest = &sendResCtx->failedRequests[probeWc->wr_id % NET_IB_MAX_REQUESTS];

  if (probeWc->status == IBV_WC_SUCCESS) {
    // The probing was successful. Mark the probe as completed.
    failedRequest->state = ncclIbResiliencyRequestStateProbeCompleted;
    // Further processing will be done in the main progress function.
    return ncclSuccess;
  }

  NCCLCHECK(ncclIbResiliencyCheckErrorNotFatal(&sendResCtx->base, probeWc, devIndex));

  NCCLCHECK(ncclIbResiliencyHandleDeviceFailure(&sendResCtx->base, devIndex));

  // The probe will reposted upon the next call to progress resiliency
  failedRequest->state = ncclIbResiliencyRequestStatePending;
  failedRequest->failedAttempts++;
  return ncclSuccess;
}

#define MAX_PROBE_WC 4

static ncclResult_t ncclIbResiliencyProbeProgress(struct ncclIbResiliencySend* sendResCtx) {
  struct ibv_wc wcs[MAX_PROBE_WC];
  int nCompletions = 0;
  struct ncclIbResiliencyDev *resDev = NULL;
  for (int devIndex = 0; devIndex < sendResCtx->base.ndevs; devIndex++) {
    resDev = &sendResCtx->base.devs[devIndex];
    // Note that even if the device failed, we still want to drain all the CQEs
    // out of it. So the only check before polling is whether there are
    // outstanding probing requests on this CQ or not.
    if (resDev->nOutstandingProbes == 0) {
      continue;
    }
    do {
      NCCLCHECK(wrap_ibv_poll_cq(resDev->probingCq, MAX_PROBE_WC, wcs, &nCompletions));
      if (nCompletions == 0) {
        break;
      }
      assert(nCompletions <= resDev->nOutstandingProbes);
      resDev->nOutstandingProbes -= nCompletions;
      for (int i = 0; i < nCompletions; i++) {
        NCCLCHECK(ncclIbResiliencyProbeHandleCompletionEvent(sendResCtx, &wcs[i], devIndex));
      }
    } while (nCompletions > 0);
  }
  return ncclSuccess;
}

// -----------------------------
// Implementation of entry point functions
// -----------------------------

ncclResult_t ncclIbResiliencyInit(struct ncclIbNetCommBase* baseComm, struct ncclIbResiliency** resCtx) {
  assert(baseComm != NULL);
  assert(resCtx != NULL);
  if (ncclParamIbResiliencyPortFailover() == 0) {
    INFO(NCCL_NET, "NET/IB: %s: Resiliency is disabled on the %s communicator (comm=%p)", __func__, baseComm->isSend ? "send" : "recv", baseComm);
    *resCtx = NULL;
    return ncclSuccess;
  }
  size_t sizeToAlloc = 0;
  if (baseComm->isSend) {
    sizeToAlloc = sizeof(struct ncclIbResiliencySend);
  } else {
    sizeToAlloc = sizeof(struct ncclIbResiliency);
  }
  // TODO: No real need to use IB malloc as this whole memory is not registered,
  // only the probing results should be allocated in memory that should be
  // registered
  NCCLCHECK(ncclIbMalloc((void**)resCtx, sizeToAlloc));
  struct ncclIbResiliency* baseCtx = *resCtx;
  baseCtx->baseComm = baseComm;
  baseCtx->inProgress = false;
  if (baseComm->isSend) {
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)baseCtx;
    memset(sendResCtx->failedRequests, 0, sizeof(sendResCtx->failedRequests));
    memset(sendResCtx->probingResults, 0, sizeof(sendResCtx->probingResults));
    memset(sendResCtx->remCmplRecordsInfo, 0, sizeof(sendResCtx->remCmplRecordsInfo));
  }
  INFO(NCCL_NET, "NET/IB: %s: Resiliency context was initialized on the %s communicator (%p)", __func__, baseComm->isSend ? "send" : "recv", baseComm);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDestroy(struct ncclIbResiliency** resCtx) {
  if (resCtx == NULL || *resCtx == NULL) {
    return ncclSuccess;
  }
  free(*resCtx);
  *resCtx = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDevInit(struct ncclIbResiliency* resCtx, uint devIndex, ncclIbDev* ibDev) {
  assert(resCtx != NULL);
  INFO(NCCL_NET, "NET/IB: %s: Initializing resiliency context on devIndex %d for %s communicator (comm=%p)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  assert(devIndex < resCtx->ndevs);
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  resDev->state = ncclIbResiliencyDevStateOk;
  void* cqContext = (void*)&resCtx->baseComm->stats;
  int cqSize = -1;
  if (resCtx->baseComm->isSend) {
    cqSize = NET_IB_MAX_REQUESTS;
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
    struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(resCtx->baseComm, devIndex);
    NCCLCHECK(wrap_ibv_reg_mr(&resDev->probingResultMr, devBase->pd, &sendResCtx->probingResults, sizeof(sendResCtx->probingResults), IBV_ACCESS_LOCAL_WRITE));
    INFO(NCCL_NET, "NET/IB: %s: Registered probing results memory (%p) on device %d for resiliency context (comm=%p)", __func__, &sendResCtx->probingResults, devIndex, resCtx->baseComm);
  } else {
    // It's not allowed to create a CQ with size 0 so set it to 1 although
    // no CQEs are expected to be generated on this CQ (on the receiver side).
    cqSize = 1;
  }
  INFO(NCCL_NET, "NET/IB: %s: Creating probing CQ on device %d for resiliency context (%s comm=%p, cq_size=%d)", __func__, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm, cqSize);
  NCCLCHECK(wrap_ibv_create_cq(&resDev->probingCq, ibDev->context, cqSize, cqContext, NULL, 0));
  INFO(NCCL_NET, "NET/IB: %s: Created probing CQ (cq=%p) on device %d for resiliency context (%s comm=%p, cq_size=%d)", __func__, resDev->probingCq, devIndex, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm, cqSize);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDevDestroy(struct ncclIbResiliency* resCtx, uint devIndex) {
  assert(resCtx != NULL);
  struct ncclIbResiliencyDev* resDev = &resCtx->devs[devIndex];
  NCCLCHECK(wrap_ibv_destroy_cq(resDev->probingCq));
  INFO(NCCL_NET, "NET/IB: %s: Destroyed probing CQ (cq=%p) on device %d for resiliency context (comm=%p)", __func__, resDev->probingCq, devIndex, resCtx->baseComm);
  if (resCtx->baseComm->isSend) {
    struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
    NCCLCHECK(wrap_ibv_dereg_mr(resDev->probingResultMr));
    INFO(NCCL_NET, "NET/IB: %s: Deregistered probing results memory (%p) on device %d for resiliency context (comm=%p)", __func__, &sendResCtx->probingResults, devIndex, resCtx->baseComm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDataCqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, int* cqSize) {
  assert(cqSize != NULL);
  *cqSize = 0;
  struct ncclIbNetCommBase* baseComm = resCtx->baseComm;
  if (baseComm->isSend) {
    // Every send request generates one completion on every QP it uses.
    *cqSize = NET_IB_MAX_REQUESTS * ncclIbCommBaseGetNqpsPerRequest(baseComm);
  } else {
    // In the worst case, a receive is not a multi-receive request, so every
    // request generates two completions (one for the CTS messages and one for
    // the receive request).
    *cqSize = NET_IB_MAX_REQUESTS * 2 * ncclIbCommBaseGetNqpsPerRequest(baseComm);
  }
  // In the worst case, all devices failed except for one device, so the single
  // device remaining must bear all be able to accommodate for all the
  // completions of all other requests.
  assert(resCtx->ndevs > 0);
  *cqSize = (*cqSize) * resCtx->ndevs;
  INFO(NCCL_NET, "NET/IB: %s: CQ size should be %d on device %d for %s communicator (comm=%p)", __func__, *cqSize, devIndex, baseComm->isSend ? "send" : "recv", baseComm);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDataRqSizeGet(struct ncclIbResiliency* resCtx, uint devIndex, uint32_t* rqSize) {
  assert(rqSize != NULL);
  // This API should only be called on the receiver side.
  assert(resCtx->baseComm->isSend == 0);

  struct ncclIbNetCommBase* baseComm = resCtx->baseComm;
  // The size of a single RQ should accommodate all the receive requests.
  // When resiliency is enabled, the RQ size should accommodate receive requests
  // assuming all other devices have failed so instead of transferring every
  // request on all QPs, a single QP is used and this QP should bear the load.
  *rqSize = NET_IB_MAX_REQUESTS * ncclIbCommBaseGetNqpsPerRequest(baseComm);
  assert(*rqSize > 0);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyDeviceNumSet(struct ncclIbResiliency* resCtx, int nLocalDevs, int nRemDevs) {
  assert(resCtx != NULL);
  assert(nLocalDevs > 0);
  assert(nRemDevs > 0);

  if (nLocalDevs <= 1) {
    INFO(NCCL_NET, "NET/IB: %s: Resiliency is being enabled on a communicator (comm=%p) with a single local device. In case of failure there will be no device to fail-over to.", __func__, resCtx->baseComm);
  }

  INFO(NCCL_NET, "NET/IB: %s: Resiliency context (comm=%p) is configured with %d local devices and %d remote devices", __func__, resCtx->baseComm, nLocalDevs, nRemDevs);

  resCtx->ndevs = nLocalDevs;
  if (resCtx->baseComm->isSend) {
    // On the sender side, the number of probing QPs will be the number of the
    // local devices. The reason for that is that sender should be able to
    // probe on all its devices.
    resCtx->nProbingQps = nLocalDevs;
  } else {
    // On the receiver side, the number of probing QPs should match the number
    // of QPs created on the sender side. So every QP on the sender side is
    // connected to a QP on the receiver side. If the number of devices on the
    // receiver side is less than the number of devices on the sender side,
    // some devices on the receiver side will have more than one probing QP.
    // If the number of devices on the receiver side is more than the number
    // of devices on the sender side, some devices on the receiver side will not
    // have any probing QPs since the sender will not probe on these devices.
    resCtx->nProbingQps = nRemDevs;
  }
  // In any case, the number of probing QPs cannot exceed the maximum number
  // of devices supported.
  assert(resCtx->nProbingQps <= NCCL_IB_MAX_DEVS_PER_NIC);
  if (resCtx->nProbingQps <= 1) {
    WARN("NET/IB: %s: Resiliency is enabled on a %s communicator (comm=%p) with a single device. This does not make sense since there is no other device to fail over to.", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
  }
  INFO(NCCL_NET, "NET/IB: %s: Resiliency context (comm=%p) is configured with %d probing QPs", __func__, resCtx->baseComm, resCtx->nProbingQps);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySenderCreateQps(struct ncclIbResiliency* resCtx, struct ncclIbResiliencyInfo* localResiliencyInfo) {
  ncclIbSendComm* sendComm = (ncclIbSendComm*)resCtx->baseComm;
  void* qpContext = (void*)&sendComm->base.stats;
  struct ncclIbQpCreateAttr qpCreateAttrs = {0};
  qpCreateAttrs.type = IBV_QPT_RC;
  // Probing QPs on the sender side do not require any remote permissions.
  qpCreateAttrs.accessFlags = IBV_ACCESS_LOCAL_WRITE;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  // Every send request can initiate at most one probing request.
  qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS;
  for (int localQpIndex = 0; localQpIndex < resCtx->nProbingQps; localQpIndex++) {
    // Sender creates a single probing QP per local device.
    int localDevIndex = localQpIndex;
    ncclIbSendCommDev* sendCommDev = &sendComm->devs[localDevIndex];
    ncclIbDev* ibDev = &ncclIbDevs[sendCommDev->base.ibDevN];
    ncclIbQp* localQp = &resCtx->probingQps[localQpIndex];
    qpCreateAttrs.ibPort = ibDev->portNum;
    qpCreateAttrs.cq = resCtx->devs[localDevIndex].probingCq;
    qpCreateAttrs.pd = sendCommDev->base.pd;
    NCCLCHECK(ncclIbCreateQp(&qpCreateAttrs, qpContext, localQp));
    // Populate the info that will be delivered to the remote receiver peer
    ncclIbQpInfo* localQpInfo = &localResiliencyInfo->probingQpsInfo[localQpIndex];
    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localDevIndex;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo) {
  ncclIbSendComm* sendComm = (ncclIbSendComm*)resCtx->baseComm;
  ncclIbQp* localQp = NULL;
  ncclIbQpInfo* remQpInfo = NULL;
  for (int localQpIndex = 0; localQpIndex < resCtx->nProbingQps; localQpIndex++) {
    int localDevIndex = localQpIndex;
    ncclIbSendCommDev* sendCommDev = &sendComm->devs[localDevIndex];
    ncclIbDev* ibDev = &ncclIbDevs[sendCommDev->base.ibDevN];
    localQp = &resCtx->probingQps[localQpIndex];
    remQpInfo = &(remInfo->resiliencyInfo.probingQpsInfo[localQpIndex]);
    localQp->remDevIdx = remQpInfo->devIndex;
    // It might be that the remote side has a different number of devices, so
    // finding the correct remote device information is done by checking the
    // remote QP info.
    ncclIbDevInfo* remDevInfo = &remInfo->devs[remQpInfo->devIndex];
    remDevInfo->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu); // TODO: This is bad practice!
    NCCLCHECK(ncclIbRtrQp(localQp->qp, &sendCommDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remInfo->tc, remInfo->sl));
    NCCLCHECK(ncclIbRtsQp(localQp->qp));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo, struct ncclIbResiliencyInfo* localResiliencyInfo) {
  ncclIbRecvComm* recvComm = (ncclIbRecvComm*)resCtx->baseComm;
  void* qpContext = (void*)&recvComm->base.stats;
  struct ncclIbQpCreateAttr qpCreateAttrs = {0};
  qpCreateAttrs.type = IBV_QPT_RC;
  // On the receiver side, probing QPs do not need to send/receive any messages.
  // They are only used as targets of RDMA Read operations.
  qpCreateAttrs.accessFlags = IBV_ACCESS_REMOTE_READ;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  qpCreateAttrs.maxSendWorkRequest = 0;
  for (int localQpIndex = 0; localQpIndex < resCtx->nProbingQps; localQpIndex++) {
    // When number of QPs on the receiver is larger than the number of devices
    // it has, the probing QPs on the receiver side are created in a "striped"
    // manner.
    int localDevIndex = localQpIndex % recvComm->base.vProps.ndevs;
    ncclIbRecvCommDev* recvCommDev = &recvComm->devs[localDevIndex];
    ncclIbDev* ibDev = &ncclIbDevs[recvCommDev->base.ibDevN];
    ncclIbQp* localQp = &resCtx->probingQps[localQpIndex];
    qpCreateAttrs.ibPort = ibDev->portNum;
    qpCreateAttrs.cq = resCtx->devs[localDevIndex].probingCq;
    qpCreateAttrs.pd = recvCommDev->base.pd;
    NCCLCHECK(ncclIbCreateQp(&qpCreateAttrs, qpContext, localQp));
    localResiliencyInfo->probingQpsInfo[localQpIndex].qpn = localQp->qp->qp_num;
    localResiliencyInfo->probingQpsInfo[localQpIndex].devIndex = localDevIndex;

    ncclIbQpInfo* remQpInfo = &remInfo->resiliencyInfo.probingQpsInfo[localQpIndex];
    ncclIbDevInfo* remDevInfo = &remInfo->devs[remQpInfo->devIndex];
    NCCLCHECK(ncclIbRtrQp(localQp->qp, &recvCommDev->base.gidInfo, remQpInfo->qpn, remDevInfo, true, remInfo->tc, remInfo->sl));
    NCCLCHECK(ncclIbRtsQp(localQp->qp));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyClose(struct ncclIbResiliency* resCtx) {
  if (resCtx == NULL) {
    return ncclSuccess;
  }
  INFO(NCCL_NET, "NET/IB: %s: Destroying %d probing QPs for resiliency context (comm=%p)", __func__, resCtx->nProbingQps, resCtx->baseComm);
  for (int qpIndex = 0; qpIndex < resCtx->nProbingQps; qpIndex++) {
    struct ncclIbQp* probingQp = &resCtx->probingQps[qpIndex];
    assert(probingQp != NULL);
    assert(probingQp->qp != NULL);
    INFO(NCCL_NET, "NET/IB: %s: Destroying probing QP (index=%d, qp=%p, qp_num=%u) for resiliency context (comm=%p)", __func__, qpIndex, probingQp->qp, probingQp->qp->qp_num, resCtx->baseComm);
    NCCLCHECK(wrap_ibv_destroy_qp(probingQp->qp));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyRemoteCompletionRecordsSet(struct ncclIbResiliency* resCtx, uint32_t cmplsRecordsRkey, uint64_t cmplsRecordsAddr, uint devIndex) {
  assert(resCtx != NULL);
  assert(resCtx->baseComm->isSend);
  assert(devIndex <= resCtx->nProbingQps);
  struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;
  sendResCtx->remCmplRecordsInfo[devIndex].rkey = cmplsRecordsRkey;
  sendResCtx->remCmplRecordsInfo[devIndex].addr = cmplsRecordsAddr;
  INFO(NCCL_NET, "NET/IB: %s: Set remote completion records info (comm=%p, addr=0x%lx, rkey=0x%x)", __func__, &resCtx->baseComm, cmplsRecordsAddr, cmplsRecordsRkey);
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyRequestIsComplete(struct ncclIbRequest *request, bool *isComplete) {
  assert(isComplete != NULL);

  if (request == NULL) {
    *isComplete = false;
    return ncclSuccess;
  }

  struct ncclIbNetCommBase* baseComm = request->base;
  struct ncclIbResiliency* resCtx = baseComm->resiliency;
  assert(resCtx != NULL);

  int remainingEventsSum = 0;
  int negativeEventsSum = 0;

  // This loop sums the total number of events a request was updated with.
  // Each device on which the request was expected to generate an event was
  // added with the number of events expected on this device.
  // During the request processing, if a device was found to be faulty, the
  // request might have been processed on a different device, generating more
  // events on that device than expected initiatlly, causing the events counter
  // on that device to reach a negative value.
  for (int devIndex = 0; devIndex < resCtx->ndevs; devIndex++) {
    if (request->events[devIndex] < 0) {
      // If the counter < 0 it means that this device was serving requests
      // for a device that failed.
      negativeEventsSum += request->events[devIndex];
    } else {
      remainingEventsSum += request->events[devIndex];
    }
  }
  // A request is considered complete if (1) on every device the number of
  // events is zero, or (2) the absolute value of the sum of all the counters
  // with negative values – is equal to the total remaining completions for a
  // request.
  *isComplete = ((remainingEventsSum == 0) || (abs(negativeEventsSum) == remainingEventsSum));
#ifdef ENABLE_TRACE
  if (*isComplete == 1) {
    TRACE(NCCL_NET, "NET/IB: %s: request=%p, remainingEventsSum=%d, negativeEventsSum=%d, completed? %s", __func__, request, remainingEventsSum, negativeEventsSum, *isComplete ? "yes" : "no");
  }
#endif // ENABLE_TRACE
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyHandleCompletionError(struct ncclIbResiliency* resCtx, struct ibv_wc* wc, int devIndex) {
  INFO(NCCL_NET, "NET/IB: %s: Got completion with error (devIndex=%d, wc->status=(%s)%d, wc->opcode=(%s)%d, wc->wr_id=%ld, wc->qp_num=%u, wc->byte_len=%d)", __func__, devIndex, ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->wr_id, wc->qp_num, wc->byte_len);
  NCCLCHECK(ncclIbResiliencyCheckErrorNotFatal(resCtx, wc, devIndex));

  // Before handling the request that got an error, first the device is
  // transitioned to a "failure" state to make sure that any potential replay
  // will not be done on the failed device.
  NCCLCHECK(ncclIbResiliencyHandleDeviceFailure(resCtx, devIndex));

  if (resCtx->baseComm->isSend) {
    NCCLCHECK(ncclIbResiliencyHandleCompletionErrorSender(resCtx, wc, devIndex));
  } else {
    NCCLCHECK(ncclIbResiliencyHandleCompletionErrorReceiver(resCtx, wc, devIndex));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResiliencyProgress(struct ncclIbResiliency* resCtx) {
  if (resCtx->inProgress == false) {
    // No operations needs to be done
    return ncclSuccess;
  }
  if (!resCtx->baseComm->isSend) {
    // No operations are needed on the receiver side for now.
    return ncclSuccess;
  }

  struct ncclIbResiliencySend* sendResCtx = (struct ncclIbResiliencySend*)resCtx;

  NCCLCHECK(ncclIbResiliencyProbeProgress(sendResCtx));

  // Iterate over all failed requests and progress them.
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbResiliencyRequestSend* failedRequest = &sendResCtx->failedRequests[i];
    if (failedRequest->request == NULL) {
      continue;
    }
    if (failedRequest->state == ncclIbResiliencyRequestStatePending) {
      NCCLCHECK(ncclIbResiliencyProbePost(sendResCtx, failedRequest));
      continue;
    }
    if (failedRequest->state == ncclIbResiliencyRequestStateProbeCompleted) {
      NCCLCHECK(ncclIbResiliencyHandleProbeCompleted(sendResCtx, failedRequest));
      NCCLCHECK(ncclIbResiliencySendRequestFree(sendResCtx, failedRequest));
      continue;
    }
  }
  if (resCtx->outstandingRequests == 0) {
    INFO(NCCL_NET, "NET/IB: %s: All resiliency operations are completed for resiliency context (%s comm=%p). Marking progress as completed.", __func__, resCtx->baseComm->isSend ? "send" : "recv", resCtx->baseComm);
    resCtx->inProgress = false;
  }
  return ncclSuccess;
}
