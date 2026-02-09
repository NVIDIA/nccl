/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"
#define ENABLE_TIMER 0
#include "timer.h"
#include "transport.h"

struct ncclTransport* ncclTransports[NTRANSPORTS+1] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport,
  &profilerTransport // Not really used for transport, only to create proxy ops polling on profiler counters.
};

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSystemError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];
  uint64_t mask = 1UL << channel->id;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

NCCL_PARAM(ConnectRoundMaxPeers, "CONNECT_ROUND_MAX_PEERS", 128);
NCCL_PARAM(ReportConnectProgress, "REPORT_CONNECT_PROGRESS", 0);
#include <sys/time.h>

// Tests communicator for CUDA P2P connectivity (local ranks only).
// *isAllDirectP2p returns 1 if all local ranks have CUDA P2P connectivity with each other
// and are no further than NCCL_P2P_LEVEL apart.
// *directMode returns 1 if *any* two local ranks are managed by the same process.
// *isAllCudaP2p returns 1 if all local ranks have CUDA P2P connectivity with each other, irrespective of the distance.
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* isAllDirectP2p, bool* directMode,
                                       bool* isAllCudaP2p) {
  bool ncclP2pFlag = true;
  bool directFlag = false;
  bool cudaP2pFlag = true;
  for (int i = 0; i < comm->localRanks; ++i) {
    for (int j = i + 1; j < comm->localRanks; ++j) {
      int ipeer = comm->localRankToRank[i];
      int jpeer = comm->localRankToRank[j];
      struct ncclPeerInfo* ipeerInfo = &comm->peerInfo[ipeer];
      struct ncclPeerInfo* jpeerInfo = &comm->peerInfo[jpeer];
      int canConnect = 0;
      int intermediateRank = -1;
      int cudaP2p = 0;
      NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, ipeerInfo->rank, jpeerInfo->rank,
                                 &canConnect, NULL, &intermediateRank, &cudaP2p));
      if (!canConnect || intermediateRank != -1) {
        ncclP2pFlag = false;
      }
      if (!cudaP2p) {
        cudaP2pFlag = false;
      }
      if (ipeerInfo->hostHash == jpeerInfo->hostHash && ipeerInfo->pidHash == jpeerInfo->pidHash) {
        directFlag = true;
      }
      if (!ncclP2pFlag && directFlag && !cudaP2pFlag) {
        break;
      }
    }
  }
  *isAllDirectP2p = ncclP2pFlag;
  *directMode = directFlag;
  *isAllCudaP2p = cudaP2pFlag;
  INFO(NCCL_INIT, "Check P2P Type isAllDirectP2p %d directMode %d isAllCudaP2p %d",
       *isAllDirectP2p, *directMode, *isAllCudaP2p);
  return ncclSuccess;
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  ncclResult_t ret = ncclSuccess;
  struct ncclConnect** data; // Store intermediate send/recvData structs for connect
  struct ncclConnect** recvData = NULL; // Points to entries inside data for given recv connection within a channel
  struct ncclConnect** sendData = NULL; // Points to entries inside data for given send connection within a channel
  int done = 0;
  int maxPeers = ncclParamConnectRoundMaxPeers();

  struct timeval timeStart, timeLast;
  gettimeofday(&timeStart, NULL);
  timeLast = timeStart; // struct copy
  bool timeReported = false;
  cudaStream_t hostStream, deviceStream;

  NCCLCHECK(ncclCalloc(&data, maxPeers));
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);

  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), ret, fail);
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);
  // First time initialization
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint64_t recvMask = comm->connectRecv[recvPeer];
    uint64_t sendMask = comm->connectSend[sendPeer];

    // Data[i] contains all ncclConnect information for all send and receive connections with a given send and recv peer
    // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
    // The first N entries contain recvData, connection information for recv connections
    // The next M entries contain sendData, connection information for send connections
    // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
    int p = i-(done+1);
    if (recvMask || sendMask) {
      if (data[p] == NULL) NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      else memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect));
    }
    recvData[p] = data[p];
    int sendChannels = 0, recvChannels = 0;
    int type;
    TIME_START(0);
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1UL<<c)) {
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
      }
    }
    TIME_STOP(0);
    TIME_START(1);
    sendData[p] = recvData[p]+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1UL<<c)) {
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, connIndex, &type), ret, fail);
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        sendData[p] = data[p];
        recvData[p] = data[p]+sendChannels;
      }
    } else {
      if (recvChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (recvChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
    }
    TIME_STOP(2);

    if (i-done == maxPeers || i == comm->nRanks-1) {
      // Loop until all channels with all ranks have been connected
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;
        for (int j=done+1; j<=i; j++) {
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j-(done+1);
          int sendDataOffset = 0;
          int recvDataOffset = 0;
          for (int c=0; c<MAXCHANNELS; c++) {
            TIME_START(3);
            if (sendMask & (1UL<<c)) {
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is a device memory access. */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;
            }
            TIME_STOP(3);

            // Start with recv channels
            TIME_START(4);
            if (recvMask & (1UL<<c)) {
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;
                  /* comm->channels[c].devPeers[recvPeer]->recv[connIndex] is a device memory access. */
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;
            }
            TIME_STOP(4);
          }
        }
        if (ncclParamReportConnectProgress() && comm->rank == 0 && done > 0) {
          struct timeval now;
          gettimeofday(&now, NULL);
          if (((now.tv_sec - timeLast.tv_sec) * 1.0 + (now.tv_usec - timeLast.tv_usec) * 1e-6) > 1) {
            float elapsed = (now.tv_sec - timeStart.tv_sec) * 1.0 + (now.tv_usec - timeStart.tv_usec) * 1e-6;
            float remaining = elapsed * (comm->nRanks - done) / done;
            printf("%sP2p connect: %g%% Elapsed %d:%02d Remaining %d:%02d                                       ",
              timeReported ? "\r" : "", done * 100.0 / comm->nRanks, ((int)elapsed) / 60, ((int)elapsed) % 60, ((int)remaining) / 60, ((int)remaining) % 60);
            fflush(stdout);
            timeReported = true;
            timeLast = now; // struct copy;
          }
        }
      }
      done = i;
    }
  }

  {
    struct timeval now;
    gettimeofday(&now, NULL);
    float elapsed = (now.tv_sec - timeStart.tv_sec)*1.0 + (now.tv_usec-timeStart.tv_usec)*1e-6;
    if (elapsed > 1.0) INFO(NCCL_PROFILE, "timings: rank %d nranks %d P2p connect done in %.2f", comm->rank, comm->nRanks, elapsed);
    if (timeReported) {
      printf("\rP2p connect done in %d:%02d                                                                       \n",
             ((int)elapsed)/60, ((int)elapsed)%60);
      fflush(stdout);
    }
  }

  /* We need to sync ranks here since some ranks might run too fast after connection setup
   * and start to destroy the connection after returning from this function; however, the
   * others might still be trying to connect and import the buffer. No sync can lead to invalid
   * shmem/cuda buffer. In addition, we also clear all connect masks and free each connectInfo array */
  for (int i = 1; i < comm->nRanks; i++) {
    int bootstrapTag = (i << 8) + (1 << 7) + (graph ? graph->id + 1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    if (recvPeer != sendPeer) {
      if (comm->connectSend[sendPeer] != 0UL) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectSend[sendPeer] != 0UL) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      if (comm->connectRecv[recvPeer] != 0UL) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, NULL, 0), ret, fail);
    } else {
      if (comm->connectSend[sendPeer] != 0UL || comm->connectRecv[recvPeer] != 0UL) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, NULL, 0), ret, fail);
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0UL;
  }

  TIME_PRINT("P2P Setup/Connect");
exit:
  for(int i=0; i<maxPeers; ++i){
    if(data[i]) free(data[i]);
  }
  free(data);
  if (sendData) free(sendData);
  if (recvData) free(recvData);

  NCCLCHECK(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream, /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false));
  return ret;
fail:
  goto exit;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type, ncclConnect* connect) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int isMaster = (rank == masterRank) ? 1 : 0;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  if (isMaster && type == collNetSend) {
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, comm->node, nMasters, masterPeer);
  }

  // select
  struct ncclChannelPeer* root = channel->peers[nranks];
  // connector index: 0 for recv, 1 for send
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect = { 0 };
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }
  // prepare connect handles
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECKGOTO(ncclCalloc(&allConnects, nranks), ret, cleanup);
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), ret, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    if (isMaster) memcpy(masterConnects+comm->node, connect, sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, comm->node, conn), ret, cleanup);
    struct ncclDevChannelPeer* devRoot;
    CUDACHECKGOTO(cudaMemcpy(&devRoot, channel->devPeers + nranks, sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), ret, cleanup);
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv + type : devRoot->send + type;
    CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice), ret, cleanup);
  }
  if (isMaster && type == collNetRecv) {
    memcpy(connect, masterConnects+comm->node, sizeof(struct ncclConnect));
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, comm->node, nMasters, masterPeer);
  }
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return ret != ncclSuccess;
}

ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // AllGather collNet setup results
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
          send->transportResources = NULL; // avoid double free
        }
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
          recv->transportResources = NULL; // avoid double free
        }
      }
    }
  }
  return ncclSuccess;
}
