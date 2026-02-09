/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "register.h"
#include "transport.h"
#include "enqueue.h"
#include "register_inline.h"

static ncclResult_t registerCheckP2PConnection(struct ncclComm* comm, struct ncclConnector* conn, struct ncclTopoGraph* graph, int peer, bool* needReg) {
  if (conn->connected) {
    if (conn->conn.flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
      *needReg = true;
    } else {
      // network connection
      *needReg = false;
    }
  } else {
    struct ncclPeerInfo* peerInfo = &comm->peerInfo[peer];
    struct ncclPeerInfo* myInfo = &comm->peerInfo[comm->rank];
    int canConnect = 0;
    NCCLCHECK(ncclTransports[0]->canConnect(&canConnect, comm, graph, myInfo, peerInfo));
    if (canConnect) {
      *needReg = true;
    } else {
      *needReg = false;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclRegisterCollNvlsBuffers(
    struct ncclComm* comm, struct ncclTaskColl* info,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue,
    bool* regNeedConnect
  ) {
  ncclResult_t result = ncclSuccess;

  info->regBufType = NCCL_REGULAR_BUFFER;
  *regNeedConnect = true;
  if (!(ncclParamLocalRegister() || (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
#if CUDART_VERSION >= 11030
  if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    if (!comm->nvlsRegSupport || info->opDev.op == ncclDevPreMulSum) goto exit;
    int nvlsReged = 0;
    int collnetReged = 0;
    const void *sendbuff = info->sendbuff;
    void *recvbuff = info->recvbuff;
    void *recvHandle = NULL, *sendHandle = NULL;
    if (info->func == ncclFuncAllGather) sendbuff = NULL;
    if (info->func == ncclFuncReduceScatter) recvbuff = NULL;
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);

    /* first try graph registration. */
    if (comm->planner.persistent && ncclParamGraphRegister()) {
      ncclNvlsGraphRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv, cleanupQueue, &info->nCleanupQueueElts);
    }

    if (nvlsReged == 0 && ncclParamLocalRegister()) {
      ncclNvlsLocalRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv);
    }

    if (nvlsReged && comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      if (comm->planner.persistent && ncclParamGraphRegister()) {
        if (info->func == ncclFuncAllGather) {
          ncclCollnetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &collnetReged, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
        } else if (info->func == ncclFuncReduceScatter) {
          ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
        } else if (info->func == ncclFuncAllReduce) {
          ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
          if (collnetReged) ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
        }
      }

      if (collnetReged == 0 && ncclParamLocalRegister()) {
        if (info->func == ncclFuncAllGather) {
          ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &collnetReged, &sendHandle);
        } else if (info->func == ncclFuncReduceScatter) {
          ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle);
        } else if (info->func == ncclFuncAllReduce) {
          ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle);
          if (collnetReged) ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle);
        }
      }
    }

    if (nvlsReged) {
      *regNeedConnect = 0;
      /* tweak NVLS channels usage; for registered NVLS buffer to saturate bandwidth. */
      int recChannels;
      NCCLCHECK(ncclNvlsRegResourcesQuery(comm, info, &recChannels));
      info->nMaxChannels = recChannels;
      info->regBufType |= NCCL_NVLS_REG_BUFFER;
    }

    if (collnetReged) {
      info->regBufType |= NCCL_NET_REG_BUFFER;
      info->sendMhandle = sendHandle;
      info->recvMhandle = recvHandle;
    }
  }
exit:
#endif
  return result;
}

ncclResult_t ncclRegisterCollBuffers(
    struct ncclComm* comm, struct ncclTaskColl* info,
    void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
    void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue,
    bool* regNeedConnect
  ) {
  ncclResult_t result = ncclSuccess;

  info->regBufType = NCCL_REGULAR_BUFFER;
  *regNeedConnect = true;
  if (!(ncclParamLocalRegister() || (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
#if CUDART_VERSION >= 11030
  if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    /* this part of nvls reg code is temporarily not used and obsolete. */
    if (!comm->nvlsRegSupport || info->opDev.op == ncclDevPreMulSum) goto exit;
    int nvlsReged = 0;
    int collnetReged = 0;
    const void *sendbuff = info->sendbuff;
    void *recvbuff = info->recvbuff;
    void *recvHandle = NULL, *sendHandle = NULL;
    if (info->func == ncclFuncAllGather) sendbuff = NULL;
    if (info->func == ncclFuncReduceScatter) recvbuff = NULL;
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);

    /* first try local registration. */
    if (ncclParamLocalRegister()) {
      ncclNvlsLocalRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv);
    }

    if (nvlsReged == 0 && comm->planner.persistent && ncclParamGraphRegister()) {
      ncclNvlsGraphRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, &nvlsReged, outRegBufSend, outRegBufRecv, cleanupQueue, &info->nCleanupQueueElts);
    }

    if (comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      if (ncclParamLocalRegister()) {
        ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle);
        if (collnetReged) ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle);
      }

      if (collnetReged == 0 && comm->planner.persistent && ncclParamGraphRegister()) {
        ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetSend, &collnetReged, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
        if (collnetReged) ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &collnetReged, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
      }
    }

    if (nvlsReged) {
      *regNeedConnect = 0;
      /* tweak NVLS channels usage; for registered NVLS buffer, we only need 4/5 channels to
       * saturate bandwidth. */
      if (comm->nNodes == 1) {
        if (info->func == ncclFuncReduceScatter)
          info->nMaxChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 5));
        else
          info->nMaxChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 4));
      } else {
        info->nMaxChannels = std::max(comm->config.minCTAs, std::min(comm->config.maxCTAs, 6));
      }
      info->regBufType |= NCCL_NVLS_REG_BUFFER;
    }

    if (collnetReged) {
      info->regBufType |= NCCL_NET_REG_BUFFER;
      info->sendMhandle = sendHandle;
      info->recvMhandle = recvHandle;
    }
  } else if (info->protocol == NCCL_PROTO_SIMPLE) {
    // IPC buffer registration
    if (info->func == ncclFuncReduceScatter && info->algorithm != NCCL_ALGO_COLLNET_DIRECT) goto exit;
    if (info->algorithm == NCCL_ALGO_RING && ((info->func == ncclFuncAllReduce && info->sendbuff == info->recvbuff) || info->func == ncclFuncReduce)) goto exit;
    if (info->algorithm == NCCL_ALGO_TREE && info->sendbuff == info->recvbuff) goto exit;
    if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN && info->sendbuff == info->recvbuff && comm->maxLocalRanks > 1) goto exit;
    if (info->func == ncclFuncAllGather && info->algorithm == NCCL_ALGO_PAT) goto exit;

    int peerRanks[NCCL_MAX_LOCAL_RANKS];
    int nPeers = 0;
    size_t elementSize = ncclTypeSize(info->datatype);
    size_t sendbuffSize = elementSize*ncclFuncSendCount(info->func, comm->nRanks, info->count);
    size_t recvbuffSize = elementSize*ncclFuncRecvCount(info->func, comm->nRanks, info->count);
    int regBufFlag = 0;
    memset(peerRanks, 0xff, sizeof(int) * NCCL_MAX_LOCAL_RANKS);

    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
      struct ncclChannel* channel = comm->channels;
      int ipcSendRegFlag = 0, ipcRecvRegFlag = 0, netSendRegFlag = 0, netRecvRegFlag = 0;
      void *sendHandle = NULL, *recvHandle = NULL;
      if (info->func != ncclFuncReduceScatter && info->func != ncclFuncAllReduce && comm->isAllDirectP2p) {
        for (int r = 0; r < NCCL_MAX_DIRECT_ARITY; ++r) {
          for (int down = 0; down < 2; ++down) {
            int peer = down ? channel->collnetDirect.down[r] : channel->collnetDirect.up[r];
            if (peer != -1) {
              struct ncclConnector* peerConn = &channel->peers[peer]->recv[0];
              bool needReg = false;

              NCCLCHECK(registerCheckP2PConnection(comm, peerConn, &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], peer, &needReg));
              if (needReg) {
                bool found = false;
                for (int p = 0; p < nPeers; ++p) {
                  if (peerRanks[p] == peer) {
                    found = true;
                    break;
                  }
                }
                if (!found) peerRanks[nPeers++] = peer;
              }
            }
          }
        }

        if (nPeers > 0) {
          if (comm->planner.persistent && ncclParamGraphRegister()) {
            ncclIpcGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcSendRegFlag, &info->sendbuffOffset, &info->sendbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
            ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcRecvRegFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
          }
          if (ncclParamLocalRegister()) {
            if (!ipcSendRegFlag) ncclIpcLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcSendRegFlag, &info->sendbuffOffset, &info->sendbuffRmtAddrs);
            if (!ipcRecvRegFlag) ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &ipcRecvRegFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
          }
        }

        if (info->func == ncclFuncAllGather) {
          if (ipcRecvRegFlag) {
            info->regBufType |= NCCL_IPC_REG_BUFFER;
          }
        }
      }

      // register collnet buffer
      if (info->opDev.op != ncclDevPreMulSum && info->opDev.op != ncclDevSumPostDiv && !(info->func == ncclFuncAllReduce && !comm->isOneRPN)) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          ncclCollnetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
          ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
          info->sendMhandle = sendHandle;
          info->recvMhandle = recvHandle;
        }

        if (ncclParamLocalRegister()) {
          if (!netSendRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle);
            info->sendMhandle = sendHandle;
          }
          if (!netRecvRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle);
            info->recvMhandle = recvHandle;
          }
        }

        if (info->func == ncclFuncAllReduce || info->func == ncclFuncReduceScatter) {
          if (comm->isOneRPN && netSendRegFlag && netRecvRegFlag) {
            info->regBufType |= NCCL_NET_REG_BUFFER;
          }
        } else if (info->func == ncclFuncAllGather) {
          if (comm->isOneRPN) {
            info->regBufType = (netSendRegFlag && netRecvRegFlag) ? (info->regBufType | NCCL_NET_REG_BUFFER) : info->regBufType;
          } else {
            info->regBufType = (netSendRegFlag) ? (info->regBufType | NCCL_NET_REG_BUFFER) : info->regBufType;
          }
        }
      }
      // tweak nChannels for 1RPN net registration
      if (comm->isOneRPN && (info->regBufType & NCCL_NET_REG_BUFFER)) info->nMaxChannels = 1;
    } else if (info->algorithm == NCCL_ALGO_RING) {
      struct ncclReg* recvRegRecord = NULL;
      struct ncclReg* sendRegRecord = NULL;
      int sendNetPeers = comm->nChannels;
      int recvNetPeers = comm->nChannels;
      struct ncclConnector** sendNetConns = NULL;
      struct ncclConnector** recvNetConns = NULL;
      void** sendNetHandles = NULL;
      void** recvNetHandles = NULL;
      void** srecvNetHandles = NULL;
      bool hasRecvNetPeer = false;
      bool hasSendNetPeer = false;

      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &recvRegRecord));
      if (recvRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
      NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &sendRegRecord));
      if (comm->nNodes > 1 && sendRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
      NCCLCHECK(ncclCalloc(&sendNetConns, comm->nChannels));
      NCCLCHECK(ncclCalloc(&sendNetHandles, comm->nChannels));
      NCCLCHECK(ncclCalloc(&recvNetConns, comm->nChannels));
      NCCLCHECK(ncclCalloc(&recvNetHandles, comm->nChannels));
      NCCLCHECK(ncclCalloc(&srecvNetHandles, comm->nChannels));

      for (int c = 0; c < comm->nChannels; ++c) {
        struct ncclChannel* channel = comm->channels + c;
        for (int r = 0; r < 2; ++r) {
          int peer;
          struct ncclConnector* peerConn;
          if (r == 0) {
            peer = channel->ring.prev;
            peerConn = &channel->peers[peer]->recv[0];
            if (peerConn->conn.flags & NCCL_DIRECT_NIC) {
              recvNetConns[c] = peerConn;
              hasRecvNetPeer = true;
            }
          } else {
            peer = channel->ring.next;
            peerConn = &channel->peers[peer]->send[0];
            if (peerConn->conn.flags & NCCL_DIRECT_NIC) {
              sendNetConns[c] = peerConn;
              hasSendNetPeer = true;
            }
          }
          if (peerConn->conn.flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
            bool found = false;
            for (int p = 0; p < nPeers; ++p) {
              if (peerRanks[p] == peer) {
                found = true;
                break;
              }
            }
            if (!found) peerRanks[nPeers++] = peer;
          }
        }
      }
      if (nPeers > 0 && comm->isAllDirectP2p) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
        }
        if (!regBufFlag && ncclParamLocalRegister()) {
          ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
        }
      }
      if (regBufFlag) {
        info->regBufType = NCCL_IPC_REG_BUFFER;
      }

      // start net registration
      regBufFlag = 0;
      if (!comm->useNetPXN && comm->useGdr && comm->netDeviceType != NCCL_NET_DEVICE_UNPACK) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          if (hasSendNetPeer) {
            ncclNetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, sendNetConns, sendNetPeers, &regBufFlag, sendNetHandles, cleanupQueue, &info->nCleanupQueueElts);
            if (regBufFlag)
              ncclNetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, sendNetConns, sendNetPeers, &regBufFlag, srecvNetHandles, cleanupQueue, &info->nCleanupQueueElts);
          }
          if ((regBufFlag || !hasSendNetPeer) && hasRecvNetPeer)
            ncclNetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, recvNetConns, recvNetPeers, &regBufFlag, recvNetHandles, cleanupQueue, &info->nCleanupQueueElts);
        }
        if (!regBufFlag && ncclParamLocalRegister()) {
          if (hasSendNetPeer) {
            ncclNetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, sendNetConns, sendNetPeers, &regBufFlag, sendNetHandles);
            if (regBufFlag)
              ncclNetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, sendNetConns, sendNetPeers, &regBufFlag, srecvNetHandles);
          }
          if ((regBufFlag || !hasSendNetPeer) && hasRecvNetPeer)
            ncclNetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, recvNetConns, recvNetPeers, &regBufFlag, recvNetHandles);
        }
      }

      if (regBufFlag) {
        info->regBufType |= NCCL_NET_REG_BUFFER;
        info->sendNetHandles = sendNetHandles;
        info->recvNetHandles = recvNetHandles;
        info->srecvNetHandles = srecvNetHandles;
        if (comm->isOneRPN && (info->func == ncclFuncAllGather || info->func == ncclFuncBroadcast)) {
          info->nMaxChannels = 1;
        }
      } else {
        free(sendNetHandles);
        free(recvNetHandles);
        free(srecvNetHandles);
      }

      free(sendNetConns);
      free(recvNetConns);
    } else if (info->algorithm == NCCL_ALGO_TREE || info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
      struct ncclReg* recvRegRecord;
      int netSendRegFlag = 0, netRecvRegFlag = 0;
      void *sendHandle, *recvHandle;
      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &recvRegRecord));
      if (recvRegRecord == NULL && !(comm->planner.persistent && ncclParamGraphRegister())) goto exit;
      if (comm->isAllDirectP2p) {
        for (int c = 0; c < comm->nChannels; ++c) {
          struct ncclChannel* channel = comm->channels + c;
          struct ncclTree* tree = NULL;
          int peers[NCCL_MAX_TREE_ARITY + 1];

          if (info->algorithm == NCCL_ALGO_TREE)
            tree = &channel->tree;
          else
            tree = &channel->collnetChain;
          for (int p = 0; p < NCCL_MAX_TREE_ARITY; ++p) peers[p] = tree->down[p];
          peers[NCCL_MAX_TREE_ARITY] = tree->up;
          for (int p = 0; p < NCCL_MAX_TREE_ARITY + 1; ++p) {
            int peer = peers[p];
            bool peerNeedReg = false;
            struct ncclConnector* recvConn = NULL;
            // P2P transport
            if (peer == -1 || peer == comm->nRanks) continue;
            recvConn = &channel->peers[peer]->recv[0];
            NCCLCHECK(registerCheckP2PConnection(comm, recvConn, &comm->graphs[info->algorithm], peer, &peerNeedReg));

            if (peerNeedReg) {
              bool found = false;
              for (int pindex = 0; pindex < nPeers; ++pindex) {
                if (peerRanks[pindex] == peer) {
                  found = true;
                  break;
                }
              }
              if (!found) peerRanks[nPeers++] = peer;
            }
          }
        }
        if (nPeers > 0) {
          if (comm->planner.persistent && ncclParamGraphRegister()) {
            ncclIpcGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs, cleanupQueue, &info->nCleanupQueueElts);
          }
          if (!regBufFlag && ncclParamLocalRegister()) {
            ncclIpcLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regBufFlag, &info->recvbuffOffset, &info->recvbuffRmtAddrs);
          }
        }
        if (regBufFlag) {
          info->regBufType = NCCL_IPC_REG_BUFFER;
        }
      }

      // register collnet chain 1RPN buffer
      if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN && info->opDev.op != ncclDevPreMulSum && info->opDev.op != ncclDevSumPostDiv && comm->isOneRPN) {
        if (comm->planner.persistent && ncclParamGraphRegister()) {
          ncclCollnetGraphRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle, cleanupQueue, &info->nCleanupQueueElts);
          info->sendMhandle = sendHandle;
          if (netSendRegFlag) {
            ncclCollnetGraphRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle, cleanupQueue, &info->nCleanupQueueElts);
            info->recvMhandle = recvHandle;
          }
        }

        if ((netSendRegFlag == 0 || netRecvRegFlag == 0) && ncclParamLocalRegister()) {
          if (!netSendRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->sendbuff, sendbuffSize, collNetSend, &netSendRegFlag, &sendHandle);
            info->sendMhandle = sendHandle;
          }
          if (netSendRegFlag && !netRecvRegFlag) {
            ncclCollnetLocalRegisterBuffer(comm, info->recvbuff, recvbuffSize, collNetRecv, &netRecvRegFlag, &recvHandle);
            info->recvMhandle = recvHandle;
          }
        }
      }

      if (netSendRegFlag && netRecvRegFlag) {
        if (comm->isOneRPN) info->nMaxChannels = 1;
        info->regBufType |= NCCL_NET_REG_BUFFER;
      }
    }

    if (info->regBufType == NCCL_IPC_REG_BUFFER && comm->nNodes == 1 && 16 < info->nMaxChannels && info->nMaxChannels <= 24) {
      info->nMaxChannels = 16;
    }
  }
exit:
#endif
  return result;
}
