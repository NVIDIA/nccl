/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int channelId) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint32_t mask = 1 << channel->id;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].recv.connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer].send.connected) continue;
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

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph) {
  struct ncclConnect data[2*MAXCHANNELS];
  for (int i=1; i<comm->nRanks; i++) {
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint32_t recvMask = comm->connectRecv[recvPeer];
    uint32_t sendMask = comm->connectSend[sendPeer];

    struct ncclConnect* recvData = data;
    int sendChannels = 0, recvChannels = 0;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
        NCCLCHECK(selectTransport<0>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, recvData+recvChannels++, conn, c));
      }
    }
    struct ncclConnect* sendData = recvData+recvChannels;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
        NCCLCHECK(selectTransport<1>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, sendData+sendChannels++, conn, c));
      }
    }

    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
         NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, data, sizeof(struct ncclConnect)*(recvChannels+sendChannels)));
         sendData = data;
         recvData = data+sendChannels;
      }
    } else {
      if (recvChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, recvData, sizeof(struct ncclConnect)*recvChannels));
      if (sendChannels) NCCLCHECK(bootstrapSend(comm->bootstrap, sendPeer, sendData, sizeof(struct ncclConnect)*sendChannels));
      if (sendChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, sendPeer, sendData, sizeof(struct ncclConnect)*sendChannels));
      if (recvChannels) NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, recvData, sizeof(struct ncclConnect)*recvChannels));
    }

    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
        NCCLCHECK(conn->transportComm->connect(comm, sendData++, 1, comm->rank, conn));
        conn->connected = 1;
        CUDACHECK(cudaMemcpy(&comm->channels[c].devPeers[sendPeer].send, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
      }
    }
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
        NCCLCHECK(conn->transportComm->connect(comm, recvData++, 1, comm->rank, conn));
        conn->connected = 1;
        CUDACHECK(cudaMemcpy(&comm->channels[c].devPeers[recvPeer].recv, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0;
  }
  return ncclSuccess;
}

