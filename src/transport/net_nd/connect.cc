/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include <windows.h>
#include <limits.h>
#include "alloc.h"
#include "socket.h"

// Compute an overflow-safe deadline for the complete connection handshake.
static ULONGLONG ncclNdConnectionDeadline(void) {
  int64_t timeoutSeconds = NCCL_ND_CONNECT_TIMEOUT_SECONDS;
  const ULONGLONG now = GetTickCount64();
  const ULONGLONG maxSeconds = (ULLONG_MAX - now) / 1000;
  return now + (ULONGLONG)(timeoutSeconds > (int64_t)maxSeconds ? maxSeconds : timeoutSeconds) * 1000;
}

// Report whether a configured connection deadline has elapsed.
static bool ncclNdConnectionExpired(ULONGLONG deadlineMs) {
  return deadlineMs != 0 && GetTickCount64() >= deadlineMs;
}

// Initialize the fixed protocol preamble advertised to a peer.
void ncclNdInitProtocolHeader(struct ncclNdProtocolHeader* header) {
  memset(header, 0, sizeof(*header));
  header->magic = NCCL_ND_PROTOCOL_MAGIC;
  header->version = NCCL_ND_PROTOCOL_VERSION;
  header->headerSize = sizeof(*header);
  header->metadataSize = sizeof(struct ncclNdConnectionMetadata);
  header->capabilities = NCCL_ND_PROTOCOL_REQUIRED_CAPS;
}

// Reject peers with incompatible protocol layout or capabilities.
ncclResult_t ncclNdValidateProtocolHeader(const struct ncclNdProtocolHeader* header) {
  if (header == NULL || header->magic != NCCL_ND_PROTOCOL_MAGIC || header->version != NCCL_ND_PROTOCOL_VERSION ||
      header->headerSize != sizeof(*header) || header->metadataSize != sizeof(struct ncclNdConnectionMetadata) ||
      header->reserved != 0 ||
      (header->capabilities & NCCL_ND_PROTOCOL_REQUIRED_CAPS) != NCCL_ND_PROTOCOL_REQUIRED_CAPS) {
    WARN("NET/ND : Incompatible peer protocol header "
         "(magic=0x%llx version=%u header=%u metadata=%u caps=0x%llx)",
         header ? (unsigned long long)header->magic : 0, header ? header->version : 0, header ? header->headerSize : 0,
         header ? header->metadataSize : 0, header ? (unsigned long long)header->capabilities : 0);
    return ncclRemoteError;
  }
  return ncclSuccess;
}

// Validate peer metadata shape and remote addresses before installing it.
ncclResult_t ncclNdValidateConnectionMetadata(const struct ncclNdConnectionMetadata* metadata, int expectedDevices) {
  if (metadata == NULL || expectedDevices <= 0 || expectedDevices > NCCL_ND_MAX_DEVS_PER_NIC ||
      metadata->ndevs != (uint32_t)expectedDevices || metadata->options != ncclNdConnectionOptions() ||
      memchr(metadata->devName, '\0', sizeof(metadata->devName)) == NULL) {
    WARN("NET/ND : Invalid peer metadata header "
         "(ndevs=%u expected=%d options=0x%x expected=0x%x)",
         metadata ? metadata->ndevs : 0, expectedDevices, metadata ? metadata->options : 0, ncclNdConnectionOptions());
    return ncclRemoteError;
  }
  for (int i = 0; i < expectedDevices; i++) {
    const struct ncclNdDevInfo* dev = &metadata->devs[i];
    if (dev->ctsFifoAddr == 0 || dev->completionAddr == 0) {
      WARN("NET/ND : Invalid peer metadata for device %d", i);
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

// Validate the peer rail and QP layout received during setup.
ncclResult_t ncclNdValidateConnectionSetup(const struct ncclNdConnectionSetup* setup, int expectedDevices,
                                           int expectedQpsPerDev) {
  if (setup == NULL || expectedDevices <= 0 || expectedDevices > NCCL_ND_MAX_DEVS_PER_NIC ||
      setup->magic != NCCL_ND_PROTOCOL_MAGIC || setup->version != NCCL_ND_PROTOCOL_VERSION || setup->reserved != 0 ||
      setup->ndevs != (uint32_t)expectedDevices ||
      (expectedQpsPerDev > 0 && setup->qpsPerDev != (uint32_t)expectedQpsPerDev) || setup->qpsPerDev == 0 ||
      setup->qpsPerDev > (uint32_t)(NCCL_ND_MAX_QPS / expectedDevices)) {
    WARN("NET/ND : Invalid connection setup (magic=0x%llx version=%u ndevs=%u qps=%u expected=%d/%d)",
         setup ? (unsigned long long)setup->magic : 0, setup ? setup->version : 0, setup ? setup->ndevs : 0,
         setup ? setup->qpsPerDev : 0, expectedDevices, expectedQpsPerDev);
    return ncclRemoteError;
  }
  for (int i = 0; i < expectedDevices; i++) {
    if (setup->addrs[i].sa.sa_family != AF_INET && setup->addrs[i].sa.sa_family != AF_INET6) {
      WARN("NET/ND : Invalid connection setup address family %d for device %d", setup->addrs[i].sa.sa_family, i);
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

// Resolve a selected physical or virtual device into its physical rails.
static ncclResult_t ncclNdSetVProps(int dev, ncclNetVDeviceProps_t* props) {
  memset(props, 0, sizeof(*props));
  if (dev < 0 || dev >= ncclNNdDevs + ncclNMergedNdDevs) return ncclInvalidArgument;
  if (dev >= ncclNNdDevs) {
    *props = ncclNdMergedDevs[dev - ncclNNdDevs].vProps;
  } else {
    props->ndevs = 1;
    props->devs[0] = dev;
  }
  return props->ndevs > 0 && props->ndevs <= NCCL_ND_MAX_DEVS_PER_NIC ? ncclSuccess : ncclInvalidUsage;
}

// Detach and destroy a partially initialized sender connection.
static ncclResult_t ncclNdAbortConnect(struct ncclNdHandle* handle, struct ncclNdSendComm* comm, ncclResult_t result,
                                       bool timedOut = false) {
  handle->sendComm = NULL;
  ncclNdStatsConnectionFailed(&comm->base, timedOut);
  uint64_t commId = comm->base.commId;
  int dev = comm->base.primaryDev;
  ncclResult_t cleanupResult = ncclNdCloseSend(comm);
  if (cleanupResult != ncclSuccess) {
    WARN("NET/ND : Comm %llu dev %d failed to clean up a partial connect", (unsigned long long)commId, dev);
  }
  return result;
}

// Detach and destroy a partially initialized receiver connection.
static ncclResult_t ncclNdAbortAccept(struct ncclNdListenComm* listenComm, struct ncclNdRecvComm* comm,
                                      ncclResult_t result, bool timedOut = false) {
  listenComm->pendingRecvComm = NULL;
  ncclNdStatsConnectionFailed(&comm->base, timedOut);
  uint64_t commId = comm->base.commId;
  int dev = comm->base.primaryDev;
  ncclResult_t cleanupResult = ncclNdCloseRecv(comm);
  if (cleanupResult != ncclSuccess) {
    WARN("NET/ND : Comm %llu dev %d failed to clean up a partial accept", (unsigned long long)commId, dev);
  }
  return result;
}

// Bind and start the asynchronous connect operation for one QP.
static ncclResult_t ncclNdStartConnectQp(struct ncclNdSendComm* comm, int qpIndex) {
  if (qpIndex < 0 || qpIndex >= comm->base.nqps) return ncclInternalError;
  struct ncclNdQp* qp = &comm->base.qps[qpIndex];
  int devIndex = qp->devIndex;
  int remDevIdx = qp->remDevIdx;
  if (devIndex < 0 || devIndex >= comm->base.ndevs || remDevIdx < 0 || remDevIdx >= (int)comm->remSetup.ndevs)
    return ncclInternalError;
  struct ncclNdDev* dev = &ncclNdDevs[comm->base.vProps.devs[devIndex]];
  const union ncclSocketAddress* remote = &comm->remSetup.addrs[remDevIdx];
  ULONG localLen = dev->addr.sa.sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  ULONG remoteLen = remote->sa.sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  NCCLCHECK(wrap_nd_connector_bind(qp->connector, &dev->addr.sa, localLen));
  NCCLCHECK(wrap_nd_prepare_overlapped(&qp->ov));
  NCCLCHECK(wrap_nd_connect(qp->connector, qp->qp, &remote->sa, remoteLen, NCCL_ND_MAX_INBOUND_READS,
                            NCCL_ND_MAX_OUTBOUND_READS, &qp->ov, &qp->opDone));
  TRACE(NCCL_NET, "NET/ND : Comm %llu connecting QP %d localDev=%d remoteDev=%d", (unsigned long long)comm->base.commId,
        qpIndex, devIndex, remDevIdx);
  return ncclSuccess;
}

// Arm asynchronous provider-error notification for a completion queue.
static ncclResult_t ncclNdArmCqErrorNotification(struct ncclNdNetCommDevBase* devBase) {
  NCCLCHECK(wrap_nd_prepare_overlapped(&devBase->cqErrorOv));
  NCCLCHECK(wrap_nd_notify(devBase->cq, ND_CQ_NOTIFY_ERRORS, &devBase->cqErrorOv));
  devBase->cqErrorArmed = true;
  return ncclSuccess;
}

// Cancel notifications and release one completion queue.
static ncclResult_t ncclNdReleaseCq(struct ncclNdNetCommDevBase* devBase) {
  ncclResult_t result = ncclSuccess;
  if (devBase->cq != NULL) {
    if (devBase->cqErrorArmed) NCCLCHECKIGNORE(wrap_nd_cq_cancel_notifications(devBase->cq), result);
    (void)wrap_nd_release(devBase->cq);
    devBase->cq = NULL;
    if (devBase->ndDevN >= 0 && devBase->ndDevN < ncclNNdDevs)
      ncclNdStatsRemoveResource(&ncclNdDevs[devBase->ndDevN].stats.activeCqs);
  }
  devBase->cqErrorArmed = false;
  wrap_nd_close_overlapped(&devBase->cqErrorOv);
  return result;
}

// Create the TCP bootstrap listener and one ND listener per physical rail.
ncclResult_t ncclNdListen(void*, int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t result = ncclSuccess;
  struct ncclNdHandle* handle = (struct ncclNdHandle*)opaqueHandle;
  if (handle == NULL || listenComm == NULL) return ncclInvalidArgument;
  *listenComm = NULL;
  memset(handle, 0, sizeof(*handle));

  struct ncclNdListenComm* lComm;
  NCCLCHECKGOTO(ncclCalloc(&lComm, 1), result, fail_alloc);

  lComm->dev = dev;
  if (dev < 0 || dev >= ncclNNdDevs + ncclNMergedNdDevs) goto fail;
  if (dev >= ncclNNdDevs) {
    lComm->vProps = ncclNdMergedDevs[dev - ncclNNdDevs].vProps;
  } else {
    lComm->vProps.ndevs = 1;
    lComm->vProps.devs[0] = dev;
  }
  lComm->ndevs = lComm->vProps.ndevs;
  if (lComm->ndevs <= 0 || lComm->ndevs > NCCL_ND_MAX_DEVS_PER_NIC) goto fail;

  // Create out-of-band TCP socket on NCCL's selected socket interface.
  NCCLCHECKGOTO(ncclSocketInit(&lComm->sock, &ncclNdSocketAddr, NCCL_SOCKET_MAGIC, ncclSocketTypeNetNd, NULL, 1),
                result, fail);
  NCCLCHECKGOTO(ncclSocketListen(&lComm->sock), result, fail);

  NCCLCHECKGOTO(ncclSocketGetAddr(&lComm->sock, &handle->oobAddr), result, fail);

  // Bind one provider listener per physical rail and publish its assigned address.
  for (int i = 0; i < lComm->ndevs; i++) {
    int physDev = lComm->vProps.devs[i];
    if (physDev < 0 || physDev >= ncclNNdDevs) goto fail;
    struct ncclNdDev* ndDev = &ncclNdDevs[physDev];
    int family = ndDev->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) goto fail;
    NCCLCHECKGOTO(wrap_nd_create_overlapped_file(ndDev->adapter, &lComm->ovFiles[i]), result, fail);
    NCCLCHECKGOTO(wrap_nd_create_listener(ndDev->adapter, lComm->ovFiles[i], &lComm->listeners[i]), result, fail);
    ULONG addrLen = family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    NCCLCHECKGOTO(wrap_nd_bind(lComm->listeners[i], &ndDev->addr.sa, addrLen), result, fail);
    NCCLCHECKGOTO(wrap_nd_listen(lComm->listeners[i], NCCL_ND_LISTENER_BACKLOG), result, fail);
    addrLen = sizeof(lComm->ndAddrs[i]);
    NCCLCHECKGOTO(wrap_nd_get_listener_address(lComm->listeners[i], &lComm->ndAddrs[i].sa, &addrLen), result, fail);
  }

  {
    char oobLine[SOCKET_NAME_MAXLEN + 1];
    INFO(NCCL_NET, "NET/ND : Listen dev %d OOB %s across %d ND adapter(s)", dev,
         ncclSocketToString(&handle->oobAddr, oobLine), lComm->ndevs);
  }

  *listenComm = lComm;

  INFO(NCCL_NET, "NET/ND : Listen on dev %d with %d physical device(s)", dev, lComm->ndevs);
  return ncclSuccess;

fail:
  for (int i = 0; i < NCCL_ND_MAX_DEVS_PER_NIC; i++) {
    if (lComm->listeners[i]) wrap_nd_release(lComm->listeners[i]);
    if (lComm->ovFiles[i]) CloseHandle(lComm->ovFiles[i]);
  }
  ncclSocketClose(&lComm->sock);
  free(lComm);
fail_alloc:
  return result == ncclSuccess ? ncclSystemError : result;
}

// Advance the non-blocking sender handshake and publish a completed communicator.
ncclResult_t ncclNdConnect(void*, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  if (opaqueHandle == NULL || sendComm == NULL || sendDevComm == NULL) return ncclInvalidArgument;
  struct ncclNdHandle* handle = (struct ncclNdHandle*)opaqueHandle;
  struct ncclNdSendComm* sComm = handle->sendComm;
  *sendComm = NULL;
  *sendDevComm = NULL;

  if (sComm == NULL) {
    // Allocate persistent state on the first poll; later calls resume it.
    NCCLCHECK(ncclCalloc(&sComm, 1));
    handle->sendComm = sComm;
    InitializeSRWLock(&sComm->base.reqLock);
    InitializeSRWLock(&sComm->base.progressLock);
    sComm->base.isSend = true;
    ncclNdStatsConnectionStart(&sComm->base, dev);
    InterlockedExchange(&sComm->base.fatalError, ncclSuccess);
    sComm->state = ncclNdConnectInit;
    sComm->deadlineMs = ncclNdConnectionDeadline();
    sComm->setupOffset = 0;
  }
  if (ncclNdConnectionExpired(sComm->deadlineMs)) {
    WARN("NET/ND : Comm %llu dev %d connect handshake timed out in state %d", (unsigned long long)sComm->base.commId,
         sComm->base.primaryDev, sComm->state);
    return ncclNdAbortConnect(handle, sComm, ncclSystemError, true);
  }

  ncclResult_t result = ncclSuccess;

  // Advance one handshake phase per call without blocking NCCL's progress thread.
  switch (sComm->state) {
  case ncclNdConnectInit:
    // Establish the TCP bootstrap channel before creating provider resources.
    NCCLCHECKGOTO(ncclSocketInit(&sComm->sock, &handle->oobAddr, NCCL_SOCKET_MAGIC, ncclSocketTypeNetNd, NULL, 1),
                  result, fail);
    NCCLCHECKGOTO(ncclSocketConnect(&sComm->sock), result, fail);
    sComm->state = ncclNdConnectTcpConnect;
    // Fall through

  case ncclNdConnectTcpConnect:
    {
      int ready = 0;
      NCCLCHECKGOTO(ncclSocketReady(&sComm->sock, &ready), result, fail);
      if (!ready) {
        // Still connecting
        return ncclSuccess;
      }
    }
    if (sComm->sock.state != ncclSocketStateReady) {
      *sendComm = NULL;
      return ncclSuccess;
    }
    sComm->state = ncclNdConnectWaitRequest;
    return ncclSuccess;

  case ncclNdConnectWaitRequest:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &sComm->sock, &sComm->remSetup, sizeof(sComm->remSetup),
                                     &sComm->setupOffset),
                  result, fail);
    if (sComm->setupOffset < (int)sizeof(sComm->remSetup)) {
      *sendComm = NULL;
      return ncclSuccess;
    }
    NCCLCHECKGOTO(ncclNdSetVProps(dev, &sComm->base.vProps), result, fail);
    NCCLCHECKGOTO(ncclNdValidateConnectionSetup(&sComm->remSetup, sComm->base.vProps.ndevs, 0), result, fail);
    sComm->qpsPerDev = (int)sComm->remSetup.qpsPerDev;
    sComm->state = ncclNdConnectTcpConnected;
    // Fall through

  case ncclNdConnectTcpConnected:
    // Resolve local rails before allocating their control resources.
    if (dev < 0 || dev >= ncclNNdDevs + ncclNMergedNdDevs) {
      WARN("NET/ND : Invalid device index %d", dev);
      return ncclNdAbortConnect(handle, sComm, ncclInternalError);
    }
    NCCLCHECKGOTO(ncclNdSetVProps(dev, &sComm->base.vProps), result, fail);

    if (sComm->base.vProps.ndevs <= 0) {
      WARN("NET/ND : No devices in vProps");
      return ncclNdAbortConnect(handle, sComm, ncclInternalError);
    }
    sComm->base.ndevs = sComm->base.vProps.ndevs;
    if (!ncclNdTryAcquireComm(&sComm->base)) {
      *sendComm = NULL;
      return ncclSuccess;
    }
    NCCLCHECKGOTO(ncclCalloc(&sComm->base.localCtsFifo, NET_ND_MAX_REQUESTS * NCCL_NET_ND_MAX_RECVS), result, fail);

    for (int devIndex = 0; devIndex < sComm->base.vProps.ndevs; devIndex++) {
      int ndDevN = sComm->base.vProps.devs[devIndex];
      if (ndDevN < 0 || ndDevN >= ncclNNdDevs) {
        WARN("NET/ND : Invalid ND device %d in vProps", ndDevN);
        return ncclNdAbortConnect(handle, sComm, ncclInternalError);
      }
      struct ncclNdDev* ndDev = &ncclNdDevs[ndDevN];
      struct ncclNdSendCommDev* devComm = &sComm->devs[devIndex];
      devComm->base.ndDevN = ndDevN;

      // Overlapped file handle
      NCCLCHECKGOTO(wrap_nd_create_overlapped_file(ndDev->adapter, &devComm->base.ovFile), result, fail);

      // Completion queue (shared for RX and TX)
      // Size to handle completions from all QPs: send + recv completions
      ULONG cqDepth = NCCL_ND_SEND_WR_DEPTH + NCCL_ND_RECV_WR_DEPTH;
      NCCLCHECKGOTO(wrap_nd_create_completion_queue(ndDev->adapter, devComm->base.ovFile, cqDepth, &devComm->base.cq),
                    result, fail);
      ncclNdStatsAddResource(&ndDev->stats.activeCqs, &ndDev->stats.peakCqs);
      NCCLCHECKGOTO(ncclNdArmCqErrorNotification(&devComm->base), result, fail);

      NCCLCHECKGOTO(wrap_nd_create_memory_region(ndDev->adapter, devComm->base.ovFile, &devComm->ctsFifoMr), result,
                    fail);
      ncclNdStatsAddResource(&ndDev->stats.activeMrs, &ndDev->stats.peakMrs);
      {
        OVERLAPPED ov = {};
        ULONG flags = ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ | ND_MR_FLAG_ALLOW_REMOTE_WRITE;
        NCCLCHECKGOTO(wrap_nd_register_memory(
                        devComm->ctsFifoMr, (const void*)sComm->base.localCtsFifo,
                        sizeof(struct ncclNdSendFifo) * NET_ND_MAX_REQUESTS * NCCL_NET_ND_MAX_RECVS, flags, &ov),
                      result, fail);
        devComm->ctsFifoRegistered = true;
      }
      NCCLCHECKGOTO(wrap_nd_create_memory_region(ndDev->adapter, devComm->base.ovFile, &devComm->cmplsRecordsMr),
                    result, fail);
      ncclNdStatsAddResource(&ndDev->stats.activeMrs, &ndDev->stats.peakMrs);
      {
        OVERLAPPED ov = {};
        ULONG flags = ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ | ND_MR_FLAG_ALLOW_REMOTE_WRITE;
        NCCLCHECKGOTO(wrap_nd_register_memory(devComm->cmplsRecordsMr, (const void*)&sComm->base.remCompletionRecords,
                                              sizeof(sComm->base.remCompletionRecords), flags, &ov),
                      result, fail);
        devComm->cmplsRecordsRegistered = true;
      }
    }

    // Create the configured QPs, striped across local rails.
    {
      const int devIndex = 0;
      int ndDevN = sComm->base.vProps.devs[devIndex];
      struct ncclNdDev* ndDev = &ncclNdDevs[ndDevN];
      int localNqps = sComm->qpsPerDev * sComm->base.ndevs;
      if (localNqps <= 0 || localNqps > NCCL_ND_MAX_QPS) {
        return ncclNdAbortConnect(handle, sComm, ncclInvalidUsage);
      }
      sComm->base.nqps = localNqps;
      sComm->base.qpIndex = 0; // Initialize round-robin index

      const ULONG maxRxSge = 1, maxTxSge = 1;

      for (int qpIndex = 0; qpIndex < sComm->base.nqps; qpIndex++) {
        int qpDevIndex = qpIndex % sComm->base.vProps.ndevs;
        int qpNdDevN = sComm->base.vProps.devs[qpDevIndex];
        struct ncclNdDev* qpNdDev = &ncclNdDevs[qpNdDevN];
        struct ncclNdSendCommDev* qpDevComm = &sComm->devs[qpDevIndex];

        struct IND2QueuePair* qp = NULL;
        struct IND2CompletionQueue* qpCqRx = qpDevComm->base.cq;
        struct IND2CompletionQueue* qpCqTx = qpDevComm->base.cq;
        ULONG inlineBytes = ncclNdConfiguredInlineBytes(qpNdDev);

        NCCLCHECKGOTO(wrap_nd_create_queue_pair(qpNdDev->adapter, qpCqRx, qpCqTx, (void*)sComm, NCCL_ND_RECV_WR_DEPTH,
                                                NCCL_ND_SEND_WR_DEPTH, maxRxSge, maxTxSge, inlineBytes, &qp),
                      result, fail);
        sComm->base.qps[qpIndex].qp = qp;
        sComm->base.qps[qpIndex].devIndex = qpDevIndex;
        sComm->base.qps[qpIndex].remDevIdx = qpDevIndex % sComm->remSetup.ndevs;
        sComm->base.qps[qpIndex].inlineDataSize = inlineBytes;
        ncclNdStatsAddResource(&qpNdDev->stats.activeQps, &qpNdDev->stats.peakQps);
        NCCLCHECKGOTO(wrap_nd_create_connector(qpNdDev->adapter, qpDevComm->base.ovFile,
                                               &sComm->base.qps[qpIndex].connector),
                      result, fail);
        if (ncclParamNdUseInline()) {
          INFO(NCCL_NET, "NET/ND : Sender QP %d inline writes %s (max %lu bytes)", qpIndex,
               inlineBytes ? "enabled" : "unsupported", inlineBytes);
        }
      }

      // Initiate the first ND connection; remaining QPs are connected in order.
      {
        char localLine[SOCKET_NAME_MAXLEN + 1];
        char peerLine[SOCKET_NAME_MAXLEN + 1];
        INFO(NCCL_NET, "NET/ND : Connect dev %d local ND %s peer ND %s (%d QPs)", dev,
             ncclSocketToString(&ndDev->addr, localLine), ncclSocketToString(&sComm->remSetup.addrs[0], peerLine),
             sComm->base.nqps);
      }
      sComm->connectQpIndex = 0;
      NCCLCHECKGOTO(ncclNdStartConnectQp(sComm, 0), result, fail);
    }

    sComm->state = ncclNdConnectNdConnect;
    return ncclSuccess;

  case ncclNdConnectNdConnect:
    {
      struct ncclNdQp* qp = &sComm->base.qps[sComm->connectQpIndex];
      if (!qp->opDone) NCCLCHECKGOTO(wrap_nd_connector_get_status(qp->connector, &qp->ov, &qp->opDone), result, fail);
      if (!qp->opDone) {
        *sendComm = NULL;
        return ncclSuccess;
      }
      NCCLCHECKGOTO(wrap_nd_prepare_overlapped(&qp->ov), result, fail);
      NCCLCHECKGOTO(wrap_nd_complete_connect(qp->connector, &qp->ov, &qp->opDone), result, fail);
      sComm->state = ncclNdConnectNdCompleteConnect;
      return ncclSuccess;
    }

  case ncclNdConnectNdCompleteConnect:
    {
      struct ncclNdQp* qp = &sComm->base.qps[sComm->connectQpIndex];
      if (!qp->opDone) NCCLCHECKGOTO(wrap_nd_connector_get_status(qp->connector, &qp->ov, &qp->opDone), result, fail);
      if (!qp->opDone) {
        *sendComm = NULL;
        return ncclSuccess;
      }
      TRACE(NCCL_NET, "NET/ND : Comm %llu connected QP %d/%d", (unsigned long long)sComm->base.commId,
            sComm->connectQpIndex + 1, sComm->base.nqps);
      sComm->connectQpIndex++;
      if (sComm->connectQpIndex < sComm->base.nqps) {
        NCCLCHECKGOTO(ncclNdStartConnectQp(sComm, sComm->connectQpIndex), result, fail);
        sComm->state = ncclNdConnectNdConnect;
        *sendComm = NULL;
        return ncclSuccess;
      }
    }
    // ND connect completed; prepare metadata for the non-blocking OOB exchange.
    {
      struct ncclNdConnectionMetadata* meta = &sComm->localMeta;
      memset(meta, 0, sizeof(*meta));
      meta->ndevs = sComm->base.ndevs;
      meta->options = ncclNdConnectionOptions();

      // Populate metadata with CTS FIFO info for each device
      for (int i = 0; i < sComm->base.ndevs; i++) {
        struct ncclNdSendCommDev* devComm = &sComm->devs[i];
        struct ncclNdDevInfo* devInfo = &meta->devs[i];

        // Get CTS FIFO address and remote token for receiver to write into
        devInfo->ctsFifoAddr = (UINT64)sComm->base.localCtsFifo;
        devInfo->ctsFifoToken = wrap_nd_get_remote_token(devComm->ctsFifoMr);

        // Get completion records address and token
        devInfo->completionAddr = (UINT64)&sComm->base.remCompletionRecords;
        devInfo->completionToken = wrap_nd_get_remote_token(devComm->cmplsRecordsMr);
      }

      // Get device name for logging
      int physDev = sComm->base.vProps.devs[0];
      if (dev >= ncclNNdDevs && dev < ncclNNdDevs + ncclNMergedNdDevs) {
        strncpy(meta->devName, ncclNdMergedDevs[dev - ncclNNdDevs].devName, MAX_MERGED_DEV_NAME);
      } else if (physDev >= 0 && physDev < ncclNNdDevs) {
        strncpy(meta->devName, ncclNdDevs[physDev].devName, MAX_MERGED_DEV_NAME);
      }
      meta->devName[MAX_MERGED_DEV_NAME - 1] = '\0';
      ncclNdInitProtocolHeader(&sComm->localHeader);
      sComm->headerOffset = 0;
      sComm->metaOffset = 0;
    }
    sComm->state = ncclNdConnectSendHeader;
    // Fall through to progress the header send.

  case ncclNdConnectSendHeader:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &sComm->sock, &sComm->localHeader, sizeof(sComm->localHeader),
                                     &sComm->headerOffset),
                  result, fail);
    if (sComm->headerOffset < (int)sizeof(sComm->localHeader)) {
      *sendComm = NULL;
      return ncclSuccess;
    }
    sComm->state = ncclNdConnectSendMeta;
    // Send the metadata body immediately after its fixed preamble.

  case ncclNdConnectSendMeta:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &sComm->sock, &sComm->localMeta, sizeof(sComm->localMeta),
                                     &sComm->metaOffset),
                  result, fail);
    if (sComm->metaOffset < (int)sizeof(sComm->localMeta)) {
      *sendComm = NULL;
      return ncclSuccess;
    }
    sComm->headerOffset = 0;
    sComm->state = ncclNdConnectRecvHeader;
    // Fall through to receive and validate the peer preamble.

  case ncclNdConnectRecvHeader:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &sComm->sock, &sComm->remHeader, sizeof(sComm->remHeader),
                                     &sComm->headerOffset),
                  result, fail);
    if (sComm->headerOffset < (int)sizeof(sComm->remHeader)) {
      *sendComm = NULL;
      return ncclSuccess;
    }
    NCCLCHECKGOTO(ncclNdValidateProtocolHeader(&sComm->remHeader), result, fail);
    sComm->metaOffset = 0;
    sComm->state = ncclNdConnectRecvMeta;
    // Fall through to receive the versioned metadata body.

  case ncclNdConnectRecvMeta:
    {
      NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &sComm->sock, &sComm->remMeta, sizeof(sComm->remMeta),
                                       &sComm->metaOffset),
                    result, fail);
      if (sComm->metaOffset < (int)sizeof(sComm->remMeta)) {
        *sendComm = NULL;
        return ncclSuccess;
      }

      NCCLCHECKGOTO(ncclNdValidateConnectionMetadata(&sComm->remMeta, sComm->base.ndevs), result, fail);

      // Store remote metadata for use in isend
      sComm->base.nRemDevs = sComm->remMeta.ndevs;
      for (int i = 0; i < sComm->remMeta.ndevs && i < NCCL_ND_MAX_DEVS_PER_NIC; i++) {
        // The sender consumes completion tokens; the receiver writes CTS into local memory.
        if (i == 0) {
          sComm->base.remCompletionRecords.addr = sComm->remMeta.devs[i].completionAddr;
        } else if (sComm->remMeta.devs[i].completionAddr != sComm->base.remCompletionRecords.addr) {
          return ncclNdAbortConnect(handle, sComm, ncclRemoteError);
        }
        sComm->base.remCompletionRecords.tokens[i] = sComm->remMeta.devs[i].completionToken;
      }

      INFO(NCCL_NET, "NET/ND : Sender exchanged metadata with %s (%d devs)", sComm->remMeta.devName,
           sComm->remMeta.ndevs);
    }
    sComm->state = ncclNdConnectReady;
    // Fall through to ready

  case ncclNdConnectReady:
    ncclNdStatsConnectionComplete(&sComm->base);
    *sendComm = sComm;
    handle->sendComm = NULL;
    INFO(NCCL_NET, "NET/ND : Connected to peer on dev %d", dev);
    return ncclSuccess;
  }

  result = ncclInternalError;
fail:
  return ncclNdAbortConnect(handle, sComm, result);
}

// Advance the non-blocking receiver handshake and publish a completed communicator.
ncclResult_t ncclNdAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  if (listenComm == NULL || recvComm == NULL || recvDevComm == NULL) return ncclInvalidArgument;
  struct ncclNdListenComm* lComm = (struct ncclNdListenComm*)listenComm;
  struct ncclNdRecvComm* rComm = lComm->pendingRecvComm;
  *recvComm = NULL;
  *recvDevComm = NULL;

  if (rComm == NULL) {
    // Allocate persistent state on the first poll; later calls resume it.
    NCCLCHECK(ncclCalloc(&rComm, 1));
    lComm->pendingRecvComm = rComm;
    InitializeSRWLock(&rComm->base.reqLock);
    InitializeSRWLock(&rComm->base.progressLock);
    rComm->base.isSend = false;
    ncclNdStatsConnectionStart(&rComm->base, lComm->dev);
    InterlockedExchange(&rComm->base.fatalError, ncclSuccess);
    rComm->state = ncclNdAcceptInit;
    rComm->deadlineMs = ncclNdConnectionDeadline();
  }
  if (ncclNdConnectionExpired(rComm->deadlineMs)) {
    WARN("NET/ND : Comm %llu dev %d accept handshake timed out in state %d", (unsigned long long)rComm->base.commId,
         rComm->base.primaryDev, rComm->state);
    return ncclNdAbortAccept(lComm, rComm, ncclSystemError, true);
  }

  ncclResult_t result = ncclSuccess;

  // Advance one handshake phase per call without blocking NCCL's progress thread.
  switch (rComm->state) {
  case ncclNdAcceptInit:
    // Start the TCP bootstrap accept before creating provider resources.
    NCCLCHECKGOTO(ncclSocketInit(&rComm->sock, NULL, NCCL_SOCKET_MAGIC, ncclSocketTypeNetNd, NULL, 1), result, fail);
    NCCLCHECKGOTO(ncclSocketAccept(&rComm->sock, &lComm->sock), result, fail);
    rComm->state = ncclNdAcceptTcpAccept; // Begin polling for readiness
    // Fall through

  case ncclNdAcceptTcpAccept:
    // Return until the bootstrap socket reports a completed accept.
    {
      int ready = 0;
      NCCLCHECKGOTO(ncclSocketReady(&rComm->sock, &ready), result, fail);
      if (!ready) {
        *recvComm = NULL;
        return ncclSuccess;
      }
    }
    rComm->state = ncclNdAcceptTcpAccepted;
    return ncclSuccess;

  case ncclNdAcceptTcpAccepted:
    // Resolve local rails before allocating their control resources.
    if (lComm->dev < 0 || lComm->dev >= ncclNNdDevs + ncclNMergedNdDevs) {
      WARN("NET/ND : Invalid device index %d for accept", lComm->dev);
      return ncclNdAbortAccept(lComm, rComm, ncclInternalError);
    }
    rComm->base.vProps = lComm->vProps;
    rComm->base.isSend = false;
    if (rComm->base.vProps.ndevs <= 0 || rComm->base.vProps.ndevs > NCCL_ND_MAX_DEVS_PER_NIC) {
      WARN("NET/ND : No devices in vProps for accept");
      return ncclNdAbortAccept(lComm, rComm, ncclInternalError);
    }
    rComm->base.ndevs = rComm->base.vProps.ndevs;
    if (!ncclNdTryAcquireComm(&rComm->base)) {
      *recvComm = NULL;
      return ncclSuccess;
    }
    rComm->qpsPerDev = (int)std::max<int64_t>(1, std::min<int64_t>(ncclParamNdQpsPerConnection(),
                                                                   NCCL_ND_MAX_QPS / rComm->base.vProps.ndevs));
    NCCLCHECKGOTO(ncclCalloc(&rComm->base.localCtsFifo, NET_ND_MAX_REQUESTS * NCCL_NET_ND_MAX_RECVS), result, fail);

    for (int devIndex = 0; devIndex < rComm->base.vProps.ndevs; devIndex++) {
      int ndDevN = rComm->base.vProps.devs[devIndex];
      if (ndDevN < 0 || ndDevN >= ncclNNdDevs) {
        WARN("NET/ND : Invalid ND device %d in vProps (accept)", ndDevN);
        return ncclNdAbortAccept(lComm, rComm, ncclInternalError);
      }
      struct ncclNdDev* ndDev = &ncclNdDevs[ndDevN];
      struct ncclNdRecvCommDev* devComm = &rComm->devs[devIndex];
      devComm->base.ndDevN = ndDevN;

      // Overlapped file handle
      NCCLCHECKGOTO(wrap_nd_create_overlapped_file(ndDev->adapter, &devComm->base.ovFile), result, fail);

      // Completion queue (shared for RX and TX)
      // Size to handle completions from all QPs: send + recv completions
      ULONG cqDepth = NCCL_ND_SEND_WR_DEPTH + NCCL_ND_RECV_WR_DEPTH;
      NCCLCHECKGOTO(wrap_nd_create_completion_queue(ndDev->adapter, devComm->base.ovFile, cqDepth, &devComm->base.cq),
                    result, fail);
      ncclNdStatsAddResource(&ndDev->stats.activeCqs, &ndDev->stats.peakCqs);
      NCCLCHECKGOTO(ncclNdArmCqErrorNotification(&devComm->base), result, fail);

      NCCLCHECKGOTO(wrap_nd_create_memory_region(ndDev->adapter, devComm->base.ovFile, &devComm->ctsFifoMr), result,
                    fail);
      ncclNdStatsAddResource(&ndDev->stats.activeMrs, &ndDev->stats.peakMrs);
      {
        OVERLAPPED ov = {};
        ULONG flags = ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ | ND_MR_FLAG_ALLOW_REMOTE_WRITE;
        NCCLCHECKGOTO(wrap_nd_register_memory(
                        devComm->ctsFifoMr, (const void*)rComm->base.localCtsFifo,
                        sizeof(struct ncclNdSendFifo) * NET_ND_MAX_REQUESTS * NCCL_NET_ND_MAX_RECVS, flags, &ov),
                      result, fail);
        devComm->ctsFifoRegistered = true;
      }
      NCCLCHECKGOTO(wrap_nd_create_memory_region(ndDev->adapter, devComm->base.ovFile, &devComm->cmplsRecordsMr),
                    result, fail);
      ncclNdStatsAddResource(&ndDev->stats.activeMrs, &ndDev->stats.peakMrs);
      {
        OVERLAPPED ov = {};
        ULONG flags = ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ | ND_MR_FLAG_ALLOW_REMOTE_WRITE;
        NCCLCHECKGOTO(wrap_nd_register_memory(devComm->cmplsRecordsMr, (const void*)rComm->base.completionRecords,
                                              offsetof(struct ncclNdNetCommBase, releaseRecords) -
                                                offsetof(struct ncclNdNetCommBase, completionRecords) +
                                                sizeof(rComm->base.releaseRecords),
                                              flags, &ov),
                      result, fail);
        devComm->cmplsRecordsRegistered = true;
      }
    }

    // Create the configured number of QPs on every physical device.
    {
      int localNqps = rComm->qpsPerDev * rComm->base.ndevs;
      if (localNqps <= 0 || localNqps > NCCL_ND_MAX_QPS) {
        return ncclNdAbortAccept(lComm, rComm, ncclInvalidUsage);
      }
      rComm->base.nqps = localNqps;
      rComm->base.qpIndex = 0; // Initialize round-robin index

      const ULONG maxRxSge = 1, maxTxSge = 1;

      for (int qpIndex = 0; qpIndex < rComm->base.nqps; qpIndex++) {
        int qpDevIndex = qpIndex % rComm->base.vProps.ndevs;
        int qpNdDevN = rComm->base.vProps.devs[qpDevIndex];
        struct ncclNdDev* qpNdDev = &ncclNdDevs[qpNdDevN];
        struct ncclNdRecvCommDev* qpDevComm = &rComm->devs[qpDevIndex];

        struct IND2QueuePair* qp = NULL;
        struct IND2CompletionQueue* qpCqRx = qpDevComm->base.cq;
        struct IND2CompletionQueue* qpCqTx = qpDevComm->base.cq;
        ULONG inlineBytes = ncclNdConfiguredInlineBytes(qpNdDev);

        NCCLCHECKGOTO(wrap_nd_create_queue_pair(qpNdDev->adapter, qpCqRx, qpCqTx, (void*)rComm, NCCL_ND_RECV_WR_DEPTH,
                                                NCCL_ND_SEND_WR_DEPTH, maxRxSge, maxTxSge, inlineBytes, &qp),
                      result, fail);
        rComm->base.qps[qpIndex].qp = qp;
        rComm->base.qps[qpIndex].devIndex = qpDevIndex;
        rComm->base.qps[qpIndex].remDevIdx = qpDevIndex;
        rComm->base.qps[qpIndex].inlineDataSize = inlineBytes;
        ncclNdStatsAddResource(&qpNdDev->stats.activeQps, &qpNdDev->stats.peakQps);
        NCCLCHECKGOTO(wrap_nd_create_connector(qpNdDev->adapter, qpDevComm->base.ovFile,
                                               &rComm->base.qps[qpIndex].connector),
                      result, fail);
        if (ncclParamNdUseInline()) {
          INFO(NCCL_NET, "NET/ND : Receiver QP %d inline writes %s (max %lu bytes)", qpIndex,
               inlineBytes ? "enabled" : "unsupported", inlineBytes);
        }
      }

      memset(&rComm->localSetup, 0, sizeof(rComm->localSetup));
      rComm->localSetup.magic = NCCL_ND_PROTOCOL_MAGIC;
      rComm->localSetup.version = NCCL_ND_PROTOCOL_VERSION;
      rComm->localSetup.ndevs = rComm->base.ndevs;
      rComm->localSetup.qpsPerDev = rComm->qpsPerDev;
      for (int i = 0; i < rComm->base.ndevs; i++) rComm->localSetup.addrs[i] = lComm->ndAddrs[i];
      rComm->setupOffset = 0;
      rComm->acceptQpIndex = 0;
    }

    rComm->state = ncclNdAcceptSendSetup;
    return ncclSuccess;

  case ncclNdAcceptSendSetup:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->sock, &rComm->localSetup, sizeof(rComm->localSetup),
                                     &rComm->setupOffset),
                  result, fail);
    if (rComm->setupOffset < (int)sizeof(rComm->localSetup)) {
      *recvComm = NULL;
      return ncclSuccess;
    }
    {
      struct ncclNdQp* qp = &rComm->base.qps[rComm->acceptQpIndex];
      struct IND2Listener* listener = lComm->listeners[qp->devIndex];
      NCCLCHECKGOTO(wrap_nd_prepare_overlapped(&qp->ov), result, fail);
      NCCLCHECKGOTO(wrap_nd_get_connection_request(listener, qp->connector, &qp->ov, &qp->opDone), result, fail);
    }
    rComm->state = ncclNdAcceptNdGetRequest;
    return ncclSuccess;

  case ncclNdAcceptNdGetRequest:
    {
      struct ncclNdQp* qp = &rComm->base.qps[rComm->acceptQpIndex];
      struct IND2Listener* listener = lComm->listeners[qp->devIndex];
      if (!qp->opDone) NCCLCHECKGOTO(wrap_nd_listener_get_request_status(listener, &qp->ov, &qp->opDone), result, fail);
      if (!qp->opDone) {
        *recvComm = NULL;
        return ncclSuccess;
      }
      rComm->state = ncclNdAcceptNdAccept;
      return ncclSuccess;
    }

  case ncclNdAcceptNdAccept:
    {
      struct ncclNdQp* qp = &rComm->base.qps[rComm->acceptQpIndex];
      NCCLCHECKGOTO(wrap_nd_prepare_overlapped(&qp->ov), result, fail);
      NCCLCHECKGOTO(wrap_nd_accept(qp->connector, qp->qp, NCCL_ND_MAX_INBOUND_READS, NCCL_ND_MAX_OUTBOUND_READS,
                                   &qp->ov, &qp->opDone),
                    result, fail);
      rComm->state = ncclNdAcceptNdAcceptWait;
      // Fall through to wait state
    }

  case ncclNdAcceptNdAcceptWait:
    {
      struct ncclNdQp* qp = &rComm->base.qps[rComm->acceptQpIndex];
      if (!qp->opDone) NCCLCHECKGOTO(wrap_nd_connector_get_status(qp->connector, &qp->ov, &qp->opDone), result, fail);
      if (!qp->opDone) {
        *recvComm = NULL;
        return ncclSuccess;
      }
      TRACE(NCCL_NET, "NET/ND : Comm %llu accepted QP %d/%d", (unsigned long long)rComm->base.commId,
            rComm->acceptQpIndex + 1, rComm->base.nqps);
      rComm->acceptQpIndex++;
      if (rComm->acceptQpIndex < rComm->base.nqps) {
        qp = &rComm->base.qps[rComm->acceptQpIndex];
        struct IND2Listener* listener = lComm->listeners[qp->devIndex];
        NCCLCHECKGOTO(wrap_nd_prepare_overlapped(&qp->ov), result, fail);
        NCCLCHECKGOTO(wrap_nd_get_connection_request(listener, qp->connector, &qp->ov, &qp->opDone), result, fail);
        rComm->state = ncclNdAcceptNdGetRequest;
        *recvComm = NULL;
        return ncclSuccess;
      }
      rComm->headerOffset = 0;
      rComm->state = ncclNdAcceptRecvHeader;
      // Fall through to receive and validate the peer preamble.
    }

  case ncclNdAcceptRecvHeader:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->sock, &rComm->remHeader, sizeof(rComm->remHeader),
                                     &rComm->headerOffset),
                  result, fail);
    if (rComm->headerOffset < (int)sizeof(rComm->remHeader)) {
      *recvComm = NULL;
      return ncclSuccess;
    }
    NCCLCHECKGOTO(ncclNdValidateProtocolHeader(&rComm->remHeader), result, fail);
    rComm->metaOffset = 0;
    rComm->state = ncclNdAcceptRecvMeta;
    // Fall through to receive the versioned metadata body.

  case ncclNdAcceptRecvMeta:
    {
      NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->sock, &rComm->remMeta, sizeof(rComm->remMeta),
                                       &rComm->metaOffset),
                    result, fail);
      if (rComm->metaOffset < (int)sizeof(rComm->remMeta)) {
        *recvComm = NULL;
        return ncclSuccess;
      }

      NCCLCHECKGOTO(ncclNdValidateConnectionMetadata(&rComm->remMeta, rComm->base.ndevs), result, fail);

      // Store remote metadata for use in irecv
      rComm->base.nRemDevs = rComm->remMeta.ndevs;
      for (int i = 0; i < rComm->remMeta.ndevs && i < NCCL_ND_MAX_DEVS_PER_NIC; i++) {
        if (i == 0) {
          rComm->base.remCtsFifo = (struct ncclNdSendFifo*)rComm->remMeta.devs[i].ctsFifoAddr;
          rComm->base.remCompletionRecords.addr = rComm->remMeta.devs[i].completionAddr;
        } else if (rComm->remMeta.devs[i].ctsFifoAddr != (UINT64)rComm->base.remCtsFifo ||
                   rComm->remMeta.devs[i].completionAddr != rComm->base.remCompletionRecords.addr) {
          return ncclNdAbortAccept(lComm, rComm, ncclRemoteError);
        }
        rComm->base.remCtsFifoTokens[i] = rComm->remMeta.devs[i].ctsFifoToken;
        rComm->base.remCompletionRecords.tokens[i] = rComm->remMeta.devs[i].completionToken;
      }

      INFO(NCCL_NET, "NET/ND : Receiver got metadata from %s (%d devs)", rComm->remMeta.devName, rComm->remMeta.ndevs);

      struct ncclNdConnectionMetadata* meta = &rComm->localMeta;
      memset(meta, 0, sizeof(*meta));
      meta->ndevs = rComm->base.ndevs;
      meta->options = ncclNdConnectionOptions();

      // Populate metadata with our CTS FIFO info for each device.
      for (int i = 0; i < rComm->base.ndevs; i++) {
        struct ncclNdRecvCommDev* devComm = &rComm->devs[i];
        struct ncclNdDevInfo* devInfo = &meta->devs[i];

        devInfo->ctsFifoAddr = (UINT64)rComm->base.localCtsFifo;
        devInfo->ctsFifoToken = wrap_nd_get_remote_token(devComm->ctsFifoMr);
        devInfo->completionAddr = (UINT64)&rComm->base.completionRecords[0][0];
        devInfo->completionToken = wrap_nd_get_remote_token(devComm->cmplsRecordsMr);
      }

      int physDev = rComm->base.vProps.devs[0];
      if (lComm->dev >= ncclNNdDevs && lComm->dev < ncclNNdDevs + ncclNMergedNdDevs) {
        strncpy(meta->devName, ncclNdMergedDevs[lComm->dev - ncclNNdDevs].devName, MAX_MERGED_DEV_NAME);
      } else if (physDev >= 0 && physDev < ncclNNdDevs) {
        strncpy(meta->devName, ncclNdDevs[physDev].devName, MAX_MERGED_DEV_NAME);
      }
      meta->devName[MAX_MERGED_DEV_NAME - 1] = '\0';
      ncclNdInitProtocolHeader(&rComm->localHeader);
      rComm->headerOffset = 0;
    }
    rComm->state = ncclNdAcceptSendHeader;
    // Fall through to progress the header send.

  case ncclNdAcceptSendHeader:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->sock, &rComm->localHeader, sizeof(rComm->localHeader),
                                     &rComm->headerOffset),
                  result, fail);
    if (rComm->headerOffset < (int)sizeof(rComm->localHeader)) {
      *recvComm = NULL;
      return ncclSuccess;
    }
    rComm->metaOffset = 0;
    rComm->state = ncclNdAcceptSendMeta;
    // Send the metadata body immediately after its fixed preamble.

  case ncclNdAcceptSendMeta:
    NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->sock, &rComm->localMeta, sizeof(rComm->localMeta),
                                     &rComm->metaOffset),
                  result, fail);
    if (rComm->metaOffset < (int)sizeof(rComm->localMeta)) {
      *recvComm = NULL;
      return ncclSuccess;
    }
    INFO(NCCL_NET, "NET/ND : Receiver sent metadata to sender");
    rComm->state = ncclNdAcceptReady;
    // Fall through to ready

  case ncclNdAcceptReady:
    ncclNdStatsConnectionComplete(&rComm->base);
    *recvComm = rComm;
    lComm->pendingRecvComm = NULL;
    INFO(NCCL_NET, "NET/ND : Accepted connection on dev %d", lComm->dev);
    return ncclSuccess;
  }

  result = ncclInternalError;
fail:
  return ncclNdAbortAccept(lComm, rComm, result);
}

// Release request-owned buffers and mark every request-pool slot unused.
static void ncclNdDiscardRequests(struct ncclNdNetCommBase* base) {
  ncclNdScopedSrwLock lock(&base->reqLock);
  for (int i = 0; i < NET_ND_MAX_REQUESTS; i++) {
    struct ncclNdRequest* req = &base->reqs[i];
    if (req->type == NCCL_NET_ND_REQ_RECV && req->recv.sizes != NULL) {
      free(req->recv.sizes);
      req->recv.sizes = NULL;
    }
    if (req->type != NCCL_NET_ND_REQ_UNUSED && base->primaryDev >= 0 && base->primaryDev < ncclNNdDevs) {
      ncclNdStatsRemoveResource(&ncclNdDevs[base->primaryDev].stats.activeRequests);
    }
    req->type = NCCL_NET_ND_REQ_UNUSED;
  }
}

// Stop progress and release a send communicator in dependency order.
ncclResult_t ncclNdCloseSend(void* sendComm) {
  struct ncclNdSendComm* sComm = (struct ncclNdSendComm*)sendComm;
  if (!sComm) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  ncclNdNotifyPeerClosing(&sComm->base);
  ncclNdStatsConnectionFailed(&sComm->base, false);

  // Flush and release QPs before their CQs and overlapped files.
  for (int q = 0; q < sComm->base.nqps; q++) {
    if (sComm->base.qps[q].connector) {
      (void)wrap_nd_release(sComm->base.qps[q].connector);
      sComm->base.qps[q].connector = NULL;
    }
    wrap_nd_close_overlapped(&sComm->base.qps[q].ov);
    if (sComm->base.qps[q].qp) {
      HRESULT flushResult = sComm->base.qps[q].qp->Flush();
      if (FAILED(flushResult) && result == ncclSuccess) result = ncclSystemError;
      (void)wrap_nd_release(sComm->base.qps[q].qp);
      sComm->base.qps[q].qp = NULL;
      int devIndex = sComm->base.qps[q].devIndex;
      if (devIndex >= 0 && devIndex < sComm->base.ndevs) {
        int ndDevN = sComm->base.vProps.devs[devIndex];
        ncclNdStatsRemoveResource(&ncclNdDevs[ndDevN].stats.activeQps);
      }
    }
  }
  ncclNdDiscardRequests(&sComm->base);

  // Deregister control memory before releasing each CQ and file.
  for (int i = 0; i < sComm->base.ndevs; i++) {
    struct ncclNdSendCommDev* devComm = &sComm->devs[i];

    // Deregister memory regions
    OVERLAPPED ov = {};
    if (devComm->ctsFifoMr) {
      if (devComm->ctsFifoRegistered) {
        NCCLCHECKIGNORE(wrap_nd_deregister_memory(devComm->ctsFifoMr, &ov), result);
        devComm->ctsFifoRegistered = false;
      }
      (void)wrap_nd_release(devComm->ctsFifoMr);
      devComm->ctsFifoMr = NULL;
      ncclNdStatsRemoveResource(&ncclNdDevs[devComm->base.ndDevN].stats.activeMrs);
    }
    if (devComm->cmplsRecordsMr) {
      ZeroMemory(&ov, sizeof(ov));
      if (devComm->cmplsRecordsRegistered) {
        NCCLCHECKIGNORE(wrap_nd_deregister_memory(devComm->cmplsRecordsMr, &ov), result);
        devComm->cmplsRecordsRegistered = false;
      }
      (void)wrap_nd_release(devComm->cmplsRecordsMr);
      devComm->cmplsRecordsMr = NULL;
      ncclNdStatsRemoveResource(&ncclNdDevs[devComm->base.ndDevN].stats.activeMrs);
    }

    NCCLCHECKIGNORE(ncclNdReleaseCq(&devComm->base), result);

    // Close overlapped file
    if (devComm->base.ovFile) {
      if (!CloseHandle(devComm->base.ovFile) && result == ncclSuccess) result = ncclSystemError;
      devComm->base.ovFile = NULL;
    }
  }

  // Free local CTS FIFO buffer
  if (sComm->base.localCtsFifo) {
    free(sComm->base.localCtsFifo);
    sComm->base.localCtsFifo = NULL;
  }

  NCCLCHECKIGNORE(ncclSocketClose(&sComm->sock), result);
  ncclNdReleaseComm(&sComm->base);
  free(sComm);
  return result;
}

// Stop progress and release a receive communicator in dependency order.
ncclResult_t ncclNdCloseRecv(void* recvComm) {
  struct ncclNdRecvComm* rComm = (struct ncclNdRecvComm*)recvComm;
  if (!rComm) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  ncclNdNotifyPeerClosing(&rComm->base);
  ncclNdStatsConnectionFailed(&rComm->base, false);

  // Flush and release QPs before their CQs and overlapped files.
  for (int q = 0; q < rComm->base.nqps; q++) {
    if (rComm->base.qps[q].connector) {
      (void)wrap_nd_release(rComm->base.qps[q].connector);
      rComm->base.qps[q].connector = NULL;
    }
    wrap_nd_close_overlapped(&rComm->base.qps[q].ov);
    if (rComm->base.qps[q].qp) {
      HRESULT flushResult = rComm->base.qps[q].qp->Flush();
      if (FAILED(flushResult) && result == ncclSuccess) result = ncclSystemError;
      (void)wrap_nd_release(rComm->base.qps[q].qp);
      rComm->base.qps[q].qp = NULL;
      int devIndex = rComm->base.qps[q].devIndex;
      if (devIndex >= 0 && devIndex < rComm->base.ndevs) {
        int ndDevN = rComm->base.vProps.devs[devIndex];
        ncclNdStatsRemoveResource(&ncclNdDevs[ndDevN].stats.activeQps);
      }
    }
  }
  ncclNdDiscardRequests(&rComm->base);

  // Deregister control memory before releasing each CQ and file.
  for (int i = 0; i < rComm->base.ndevs; i++) {
    struct ncclNdRecvCommDev* devComm = &rComm->devs[i];

    // Deregister memory regions
    OVERLAPPED ov = {};
    if (devComm->ctsFifoMr) {
      if (devComm->ctsFifoRegistered) {
        NCCLCHECKIGNORE(wrap_nd_deregister_memory(devComm->ctsFifoMr, &ov), result);
        devComm->ctsFifoRegistered = false;
      }
      (void)wrap_nd_release(devComm->ctsFifoMr);
      devComm->ctsFifoMr = NULL;
      ncclNdStatsRemoveResource(&ncclNdDevs[devComm->base.ndDevN].stats.activeMrs);
    }
    if (devComm->cmplsRecordsMr) {
      ZeroMemory(&ov, sizeof(ov));
      if (devComm->cmplsRecordsRegistered) {
        NCCLCHECKIGNORE(wrap_nd_deregister_memory(devComm->cmplsRecordsMr, &ov), result);
        devComm->cmplsRecordsRegistered = false;
      }
      (void)wrap_nd_release(devComm->cmplsRecordsMr);
      devComm->cmplsRecordsMr = NULL;
      ncclNdStatsRemoveResource(&ncclNdDevs[devComm->base.ndDevN].stats.activeMrs);
    }

    NCCLCHECKIGNORE(ncclNdReleaseCq(&devComm->base), result);

    // Close overlapped file
    if (devComm->base.ovFile) {
      if (!CloseHandle(devComm->base.ovFile) && result == ncclSuccess) result = ncclSystemError;
      devComm->base.ovFile = NULL;
    }
  }

  // Free local CTS FIFO buffer
  if (rComm->base.localCtsFifo) {
    free(rComm->base.localCtsFifo);
    rComm->base.localCtsFifo = NULL;
  }

  NCCLCHECKIGNORE(ncclSocketClose(&rComm->sock), result);
  ncclNdReleaseComm(&rComm->base);
  free(rComm);
  return result;
}

// Release a listener and any receiver whose accept is still pending.
ncclResult_t ncclNdCloseListen(void* listenComm) {
  struct ncclNdListenComm* lComm = (struct ncclNdListenComm*)listenComm;
  if (!lComm) return ncclSuccess;
  ncclResult_t result = ncclSuccess;

  // Destroy a partial accept before releasing its parent listener state.
  if (lComm->pendingRecvComm) {
    NCCLCHECKIGNORE(ncclNdCloseRecv(lComm->pendingRecvComm), result);
    lComm->pendingRecvComm = NULL;
  }

  // Release provider listeners before closing their overlapped files.
  for (int i = 0; i < lComm->ndevs; i++) {
    if (lComm->listeners[i]) {
      (void)wrap_nd_release(lComm->listeners[i]);
      lComm->listeners[i] = NULL;
    }
    if (lComm->ovFiles[i]) {
      if (!CloseHandle(lComm->ovFiles[i]) && result == ncclSuccess) result = ncclSystemError;
      lComm->ovFiles[i] = NULL;
    }
  }

  NCCLCHECKIGNORE(ncclSocketClose(&lComm->sock), result);
  free(lComm);
  return result;
}
