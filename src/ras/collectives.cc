/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NDEBUG // Comment out duriyng development only!
#include <cassert>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "utils.h"
#include "ras_internal.h"

// The number of recent collectives to keep track of.  Completely arbitrary.
#define COLL_HISTORY_SIZE 64

// An entry in the rasCollHistory array keeping track of recently completed collectives (to make it possible to
// identify and drop duplicates arriving over different links).
struct rasCollHistoryEntry {
  union ncclSocketAddress rootAddr;
  uint64_t rootId;
};

// Array keeping track of recently completed collectives (to avoid infinite loops).  LRU-based replacement.
static struct rasCollHistoryEntry rasCollHistory[COLL_HISTORY_SIZE];
static int nRasCollHistory, rasCollHistNextIdx;

// Monotonically increased to ensure that each collective originating locally has a unique Id.
static uint64_t rasCollLastId;

// Array keeping track of ongoing collective operations (apart from broadcasts, which have no response so require
// no such tracking).
struct rasCollective* rasCollectives;
static int nRasCollectives;

static ncclResult_t getNewCollEntry(struct rasCollective** pColl);
static ncclResult_t rasLinkSendCollReq(struct rasLink* link, struct rasCollective* coll,
                                       const struct rasCollRequest* req, size_t reqLen, int fromConnIdx);
static ncclResult_t rasConnSendCollReq(struct rasConnection* conn, const struct rasCollRequest* req, size_t reqLen);
static ncclResult_t rasCollReadyResp(struct rasCollective* coll);
static ncclResult_t rasConnSendCollResp(struct rasConnection* conn,
                                        const union ncclSocketAddress* rootAddr, uint64_t rootId,
                                        const union ncclSocketAddress* peers, int nPeers,
                                        const char* data, int nData, int nLegTimeouts);

static ncclResult_t rasCollConnsInit(char** pData, int* pNData);
static ncclResult_t rasCollConnsMerge(struct rasCollective* coll, struct rasMsg* msg);

static ncclResult_t rasCollCommsInit(char** pData, int* pNData);
static ncclResult_t rasCollCommsMerge(struct rasCollective* coll, struct rasMsg* msg);
static int ncclCommsCompare(const void* p1, const void* p2);


///////////////////////////////////////////////////////////////////////////////////////
// Functions related to the initialization of collectives and the message exchanges. //
///////////////////////////////////////////////////////////////////////////////////////

// Returns the index of the first available entry in the rasCollectives array, enlarging the array if necessary.
static ncclResult_t getNewCollEntry(struct rasCollective** pColl) {
  struct rasCollective* coll;
  int i;
  for (i = 0; i < nRasCollectives; i++)
    if (rasCollectives[i].type == RAS_MSG_NONE)
      break;
  if (i == nRasCollectives) {
    NCCLCHECK(ncclRealloc(&rasCollectives, nRasCollectives, nRasCollectives+RAS_INCREMENT));
    nRasCollectives += RAS_INCREMENT;
  }

  coll = rasCollectives+i;
  memset(coll, '\0', sizeof(*coll));
  coll->startTime = clockNano();
  coll->fromConnIdx = -1;
  // We are unlikely to use the whole array, but at least we won't need to realloc.
  NCCLCHECK(ncclCalloc(&coll->fwdConns, nRasConns));

  *pColl = coll;
  return ncclSuccess;
}

// Initializes a collective request by giving it a unique ID.
void rasCollReqInit(struct rasCollRequest* req) {
  memcpy(&req->rootAddr, &rasNetListeningSocket.addr, sizeof(req->rootAddr));
  req->rootId = ++rasCollLastId;
}

// Sends a collective request message through all regular RAS network connections (effectively, broadcasts it).
// Also used for re-broadcasts (on peers receiving the request over the network).
// Checking for duplicates is the responsibility of the caller.
// For collectives other than broadcasts, initializes a rasCollective structure and fills it with local data,
// in preparation for collective response messages.
// pAllDone indicates on return if the collective operation is already finished, which is unusual, but possible
// in scenarios such as a total of two peers.
// pCollIdx provides on return an index of the allocated rasCollective structure to track this collective (unless
// it's a broadcast, which require no such tracking).
ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, size_t reqLen, bool* pAllDone, int* pCollIdx,
                               int fromConnIdx) {
  struct rasCollective* coll = nullptr;
  if (req->type >= RAS_COLL_CONNS) {
    // Keep track of this collective operation so that we can handle the responses appropriately.
    NCCLCHECK(getNewCollEntry(&coll));
    if (pCollIdx)
      *pCollIdx = coll-rasCollectives;
    memcpy(&coll->rootAddr, &req->rootAddr, sizeof(coll->rootAddr));
    coll->rootId = req->rootId;
    coll->type = req->type;
    coll->timeout = req->timeout;
    coll->fromConnIdx = fromConnIdx;
    if (ncclCalloc(&coll->peers, 1) == ncclSuccess) {
      memcpy(coll->peers, &rasNetListeningSocket.addr, sizeof(*coll->peers));
      coll->nPeers = 1;
    }

    // Collective-specific initialization of accumulated data (using local data for now).
    if (req->type == RAS_COLL_CONNS)
      (void)rasCollConnsInit(&coll->data, &coll->nData);
    else if (req->type == RAS_COLL_COMMS)
      (void)rasCollCommsInit(&coll->data, &coll->nData);
  } else { // req->type < RAS_COLL_CONNS
    // Add the info to the collective message history.
    nRasCollHistory = std::min(nRasCollHistory+1, COLL_HISTORY_SIZE);
    memcpy(&rasCollHistory[rasCollHistNextIdx].rootAddr, &req->rootAddr,
           sizeof(rasCollHistory[rasCollHistNextIdx].rootAddr));
    rasCollHistory[rasCollHistNextIdx].rootId = req->rootId;
    rasCollHistNextIdx = (rasCollHistNextIdx + 1) % COLL_HISTORY_SIZE;

    // Collective-specific message handling.
    if (req->type == RAS_BC_DEADPEER) {
      bool done = false;
      rasMsgHandleBCDeadPeer(req, &done);
      if (done)
        goto exit;
    }
  } // req->type < RAS_COLL_CONNS

  for (int connIdx = 0; connIdx < nRasConns; connIdx++)
    rasConns[connIdx].linkFlag = false;

  (void)rasLinkSendCollReq(&rasNextLink, coll, req, reqLen, fromConnIdx);
  (void)rasLinkSendCollReq(&rasPrevLink, coll, req, reqLen, fromConnIdx);

  if (coll && pAllDone)
    *pAllDone = (coll->nFwdSent == coll->nFwdRecv);
exit:
  return ncclSuccess;
}

// Sends the collective message through all connections associated with this link (with the exception of the one
// the message came from, if any).
static ncclResult_t rasLinkSendCollReq(struct rasLink* link, struct rasCollective* coll,
                                       const struct rasCollRequest* req, size_t reqLen, int fromConnIdx) {
  for (int i = 0; i < link->nConns; i++) {
    struct rasLinkConn* linkConn = link->conns+i;
    if (linkConn->connIdx != -1 && linkConn->connIdx != fromConnIdx) {
      struct rasConnection* conn = rasConns+linkConn->connIdx;
      if (!conn->linkFlag) {
        // We send collective messages through fully established and operational connections only.
        if (conn->sockIdx != -1 && rasSockets[conn->sockIdx].status == RAS_SOCK_READY && !conn->experiencingDelays) {
          if (rasConnSendCollReq(conn, req, reqLen) == ncclSuccess && coll != nullptr)
            coll->fwdConns[coll->nFwdSent++] = linkConn->connIdx;
        } // if (conn->sockIdx != -1 && RAS_SOCK_READY)
        conn->linkFlag = true;
      } // if (!conn->linkFlag)
    } // if (linkConn->connIdx != -1 && linkConn->connIdx != fromConnIdx)
  } // for (i)

  return ncclSuccess;
}

// Sends a collective message down a particular connection.
static ncclResult_t rasConnSendCollReq(struct rasConnection* conn, const struct rasCollRequest* req, size_t reqLen) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_COLLREQ) + reqLen;

  NCCLCHECK(rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_COLLREQ;
  memcpy(&msg->collReq, req, reqLen);

  rasConnEnqueueMsg(conn, msg, msgLen);

  return ncclSuccess;
}

// Handles the RAS_MSG_COLLREQ collective message request on the receiver side.  Primarily deals with duplicates and
// re-broadcasts the message to local peers, though in case of a very limited RAS network it might be done right away,
// in which case it can immediately send the response.
ncclResult_t rasMsgHandleCollReq(struct rasMsg* msg, struct rasSocket* sock) {
  bool allDone = false;
  int collIdx = -1;
  assert(sock->connIdx != -1);

  // First check if we've already handled this request (through another connection).
  for (int i = 0; i < nRasCollHistory; i++) {
    // In principle we can use i to index the array but we convert it so that we check the most recent entries first.
    int collHistIdx = (rasCollHistNextIdx + COLL_HISTORY_SIZE - 1 - i) % COLL_HISTORY_SIZE;
    if (memcmp(&msg->collReq.rootAddr, &rasCollHistory[collHistIdx].rootAddr, sizeof(msg->collReq.rootAddr)) == 0 &&
        msg->collReq.rootId == rasCollHistory[collHistIdx].rootId) {
      if (msg->collReq.type >= RAS_COLL_CONNS) {
        // Send an empty response so that the sender can account for it.  The non-empty response has already been
        // sent through the connection that we received the request through first.
        NCCLCHECK(rasConnSendCollResp(rasConns+sock->connIdx, &msg->collReq.rootAddr, msg->collReq.rootId,
                                      /*peers*/nullptr, /*nPeers*/0, /*data*/nullptr, /*nData*/0, /*nLegTimeouts*/0));
      }
      goto exit;
    }
  } // for (i)

  if (msg->collReq.type >= RAS_COLL_CONNS) {
    // Check if we're currently handling this collective request.
    for (int i = 0; i < nRasCollectives; i++) {
      struct rasCollective* coll = rasCollectives+i;
      if (coll->type != RAS_MSG_NONE &&
          memcmp(&msg->collReq.rootAddr, &coll->rootAddr, sizeof(msg->collReq.rootAddr)) == 0 &&
          msg->collReq.rootId == coll->rootId) {
        assert(msg->collReq.type == coll->type);

        // Send an empty response so that the sender can account for it.  The non-empty response will be
        // sent through the connection that we received the request through first.
        NCCLCHECK(rasConnSendCollResp(rasConns+sock->connIdx, &msg->collReq.rootAddr, msg->collReq.rootId,
                                      /*peers*/nullptr, /*nPeers*/0, /*data*/nullptr, /*nData*/0, /*nLegTimeouts*/0));
        goto exit;
      } // if match
    } // for (i)
  } // if (msg->collReq.type >= RAS_COLL_CONNS)

  // Re-broadcast the message to my peers (minus the one it came from) and handle it locally.
  NCCLCHECK(rasNetSendCollReq(&msg->collReq, rasCollDataLength(msg->collReq.type), &allDone, &collIdx, sock->connIdx));

  if (msg->collReq.type >= RAS_COLL_CONNS && allDone) {
    assert(collIdx != -1);
    // We are a leaf process -- send the response right away.  This can probably trigger only for the case of a total
    // of two peers, and hence just one RAS connection, or during communication issues, because normally every peer
    // has more than one connection so there should always be _some_ other peer to forward the request to.
    NCCLCHECK(rasCollReadyResp(rasCollectives+collIdx));
  }
exit:
  return ncclSuccess;
}

// Sends a collective response back to the process we received the collective request from.
// Invoked when we are finished waiting for the collective responses from other peers (i.e., either there weren't
// any peers (unlikely), the peers sent their responses (likely), or we timed out.
static ncclResult_t rasCollReadyResp(struct rasCollective* coll) {
  if (coll->fromConnIdx != -1) {
    // For remotely-initiated collectives, send the response back.
    NCCLCHECK(rasConnSendCollResp(rasConns+coll->fromConnIdx, &coll->rootAddr, coll->rootId,
                                  coll->peers, coll->nPeers, coll->data, coll->nData, coll->nLegTimeouts));

    // Add the identifying info to the collective message history.
    nRasCollHistory = std::min(nRasCollHistory+1, COLL_HISTORY_SIZE);
    memcpy(&rasCollHistory[rasCollHistNextIdx].rootAddr, &coll->rootAddr,
           sizeof(rasCollHistory[rasCollHistNextIdx].rootAddr));
    rasCollHistory[rasCollHistNextIdx].rootId = coll->rootId;
    rasCollHistNextIdx = (rasCollHistNextIdx + 1) % COLL_HISTORY_SIZE;

    rasCollFree(coll);
  } else {
    // For locally-initiated collectives, invoke the client code again (which will release it, once finished).
    NCCLCHECK(rasClientResume(coll));
  }
  return ncclSuccess;
}

// Sends a collective response via the connection we originally received the request from.  The message should be
// a cumulative response from this process and all the processes that we forwarded the request to.
static ncclResult_t rasConnSendCollResp(struct rasConnection* conn,
                                        const union ncclSocketAddress* rootAddr, uint64_t rootId,
                                        const union ncclSocketAddress* peers, int nPeers,
                                        const char* data, int nData, int nLegTimeouts) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_COLLRESP) + nPeers*sizeof(*peers);
  int dataOffset = 0;

  if (nData > 0) {
    ALIGN_SIZE(msgLen, alignof(int64_t));
    dataOffset = msgLen;
    msgLen += nData;
  }

  NCCLCHECK(rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_COLLRESP;
  memcpy(&msg->collResp.rootAddr, rootAddr, sizeof(msg->collResp.rootAddr));
  msg->collResp.rootId = rootId;
  msg->collResp.nLegTimeouts = nLegTimeouts;
  msg->collResp.nPeers = nPeers;
  msg->collResp.nData = nData;
  if (nPeers)
    memcpy(msg->collResp.peers, peers, nPeers*sizeof(*msg->collResp.peers));
  if (nData)
    memcpy(((char*)msg)+dataOffset, data, nData);

  rasConnEnqueueMsg(conn, msg, msgLen);

  return ncclSuccess;
}

// Handles the collective response on the receiver side.  Finds the corresponding rasCollective structure, merges
// the data from the response into the accumulated data.  If all the responses have been accounted for, sends the
// accumulated response back.
ncclResult_t rasMsgHandleCollResp(struct rasMsg* msg, struct rasSocket* sock) {
  int collIdx;
  struct rasCollective* coll = nullptr;
  char line[SOCKET_NAME_MAXLEN+1];

  for (collIdx = 0; collIdx < nRasCollectives; collIdx++) {
    coll = rasCollectives+collIdx;
    if (coll->type != RAS_MSG_NONE &&
        memcmp(&msg->collResp.rootAddr, &coll->rootAddr, sizeof(msg->collResp.rootAddr)) == 0 &&
        msg->collResp.rootId == coll->rootId)
      break;
  }
  if (collIdx == nRasCollectives) {
    INFO(NCCL_RAS, "RAS failed to find a matching ongoing collective for response %s:%ld from %s!",
         ncclSocketToString(&msg->collResp.rootAddr, line), msg->collResp.rootId,
         ncclSocketToString(&sock->sock.addr, rasLine));
    goto exit;
  }

  coll->nLegTimeouts += msg->collResp.nLegTimeouts;
  assert(sock->connIdx != -1);
  // Account for the received response in our collective operation tracking.
  for (int i = 0; i < coll->nFwdSent; i++) {
    if (coll->fwdConns[i] == sock->connIdx) {
      coll->fwdConns[i] = -1;
      break;
    }
  }
  coll->nFwdRecv++;
  if (msg->collResp.nData > 0) {
    // Collective-specific merging of the response into locally accumulated data.
    if (coll->type == RAS_COLL_CONNS)
      NCCLCHECK(rasCollConnsMerge(coll, msg));
    else if (coll->type == RAS_COLL_COMMS)
      NCCLCHECK(rasCollCommsMerge(coll, msg));
  }
  // We merge the peers after merging the data, so that the data merge function can rely on peers being unchanged.
  if (msg->collResp.nPeers > 0) {
    NCCLCHECK(ncclRealloc(&coll->peers, coll->nPeers, coll->nPeers + msg->collResp.nPeers));
    memcpy(coll->peers+coll->nPeers, msg->collResp.peers, msg->collResp.nPeers * sizeof(*coll->peers));
    coll->nPeers += msg->collResp.nPeers;
  }

  // If we received all the data we were waiting for, send our response back.
  if (coll->nFwdSent == coll->nFwdRecv)
    NCCLCHECK(rasCollReadyResp(coll));
exit:
  return ncclSuccess;
}

// Removes a connection from all ongoing collectives.  Called when a connection is experiencing a delay or is being
// terminated.
void rasCollsPurgeConn(int connIdx) {
  for (int i = 0; i < nRasCollectives; i++) {
    struct rasCollective* coll = rasCollectives+i;
    if (coll->type != RAS_MSG_NONE) {
      char line[SOCKET_NAME_MAXLEN+1];
      if (coll->fromConnIdx == connIdx) {
        INFO(NCCL_RAS, "RAS purging collective %s:%ld because it comes from %s",
             ncclSocketToString(&coll->rootAddr, line), coll->rootId,
             ncclSocketToString(&rasConns[connIdx].addr, rasLine));
        rasCollFree(coll);
      } else {
        for (int j = 0; j < coll->nFwdSent; j++) {
          if (coll->fwdConns[j] == connIdx) {
            coll->fwdConns[j] = -1;
            coll->nFwdRecv++;
            coll->nLegTimeouts++;
            INFO(NCCL_RAS, "RAS not waiting for response from %s to collective %s:%ld "
                 "(nFwdSent %d, nFwdRecv %d, nLegTimeouts %d)",
                 ncclSocketToString(&rasConns[connIdx].addr, rasLine), ncclSocketToString(&coll->rootAddr, line),
                 coll->rootId, coll->nFwdSent, coll->nFwdRecv, coll->nLegTimeouts);
            if (coll->nFwdSent == coll->nFwdRecv)
              (void)rasCollReadyResp(coll);
            break;
          }
        } // for (j)
      } // coll->fromConnIdx != connIdx
    } // !RAS_MSG_NONE
  } // for (i)
}

// Frees a rasCollective entry and any memory associated with it.
void rasCollFree(struct rasCollective* coll) {
  free(coll->fwdConns);
  coll->fwdConns = nullptr;
  free(coll->peers);
  coll->peers = nullptr;
  free(coll->data);
  coll->data = nullptr;
  coll->fromConnIdx = -1;
  coll->type = RAS_MSG_NONE;
}

// Invoked from the main RAS thread loop to handle timeouts of the collectives.
// We obviously want to have a reasonable *total* timeout that the RAS client can rely on, but we don't have strict
// global coordination.  So we have, in effect, two timeouts: soft (5s) and hard (10s).  Soft equals the keep-alive
// timeout.
// When sending collective requests, we skip any connections that are experiencing delays.  After the 5s timeout, we
// check again the status of all outstanding connections and if any is now delayed, we give up on it.
// That works fine for directly observable delays, but if the problematic connection is further away from us, all
// we can do is trust that the other peers will "do the right thing soon".  However, if there is a cascade of
// problematic connections, they could still exceed the 5s total.  So after 10s we give up waiting no matter what
// and send back whatever we have.  Unfortunately, the peer that the RAS client is connected to will in all likelihood
// time out first, so at that point any delayed responses that eventually arrive are likely to be too late...
void rasCollsHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  for (int collIdx = 0; collIdx < nRasCollectives; collIdx++) {
    struct rasCollective* coll = rasCollectives+collIdx;
    if (coll->type == RAS_MSG_NONE || coll->timeout == 0)
      continue;

    if (now - coll->startTime > coll->timeout) {
      // We've exceeded the leg timeout.  For all outstanding responses, check their connections.
      if (!coll->timeoutWarned) {
        INFO(NCCL_RAS, "RAS collective %s:%ld timeout warning (%lds) -- %d responses missing",
             ncclSocketToString(&coll->rootAddr, rasLine), coll->rootId,
             (now - coll->startTime) / CLOCK_UNITS_PER_SEC, coll->nFwdSent - coll->nFwdRecv);
        coll->timeoutWarned = true;
      }
      for (int i = 0; i < coll->nFwdSent; i++) {
        if (coll->fwdConns[i] != -1) {
          struct rasConnection* conn = rasConns+coll->fwdConns[i];
          char line[SOCKET_NAME_MAXLEN+1];
          if (!conn->experiencingDelays && conn->sockIdx != -1) {
            struct rasSocket* sock = rasSockets+conn->sockIdx;
            // Ensure that the connection is fully established and operational, and that the socket hasn't been
            // re-created during the handling of the collective (which would suggest that the request may have been
            // lost).
            if (sock->status == RAS_SOCK_READY && sock->createTime < coll->startTime)
              continue;
          }
          // In all other cases we declare a timeout so that we can (hopefully) recover.
          INFO(NCCL_RAS, "RAS not waiting for response from %s to collective %s:%ld "
               "(nFwdSent %d, nFwdRecv %d, nLegTimeouts %d)",
               ncclSocketToString(&conn->addr, rasLine), ncclSocketToString(&coll->rootAddr, line),
               coll->rootId, coll->nFwdSent, coll->nFwdRecv, coll->nLegTimeouts);
          coll->fwdConns[i] = -1;
          coll->nFwdRecv++;
          coll->nLegTimeouts++;
        } // if (coll->fwdConns[i] != -1)
      } // for (i)
      if (coll->nFwdSent == coll->nFwdRecv) {
        (void)rasCollReadyResp(coll);
      } else {
        // At least some of the delays are *not* due to this process' connections experiencing delays, i.e., they
        // must be due to delays at other processes.  Presumably those processes will give up waiting soon and the
        // (incomplete) responses will arrive shortly, so we should wait a little longer.
        if (now - coll->startTime > coll->timeout + RAS_COLLECTIVE_EXTRA_TIMEOUT) {
          // We've exceeded even the longer timeout, which is unexpected.  Try to return whatever we have (though
          // the originator of the collective, if it's not us, may have timed out already anyway).
          INFO(NCCL_RAS, "RAS collective %s:%ld timeout error (%lds) -- giving up on %d missing responses",
               ncclSocketToString(&coll->rootAddr, rasLine), coll->rootId,
               (now - coll->startTime) / CLOCK_UNITS_PER_SEC, coll->nFwdSent - coll->nFwdRecv);
          coll->nLegTimeouts += coll->nFwdSent - coll->nFwdRecv;
          coll->nFwdRecv = coll->nFwdSent;
          (void)rasCollReadyResp(coll);
        } else {
          *nextWakeup = std::min(*nextWakeup, coll->startTime+coll->timeout+RAS_COLLECTIVE_EXTRA_TIMEOUT);
        }
      } // conn->nFwdRecv < conn->nFwdSent
    } else {
      *nextWakeup = std::min(*nextWakeup, coll->startTime+coll->timeout);
    }
  } // for (collIdx)
}


/////////////////////////////////////////////////////////////////////////
// Functions related to the handling of the RAS_COLL_CONNS collective. //
/////////////////////////////////////////////////////////////////////////

// Initializes the accumulated data with just the local data for now.
// For this particular collective, we keep some reduced statistical data (min/max/avg travel time) as well
// as connection-specific info in case we observed a negative min travel time (which, ideally, shouldn't happen,
// but the system clocks may not be perfectly in sync).
static ncclResult_t rasCollConnsInit(char** pData, int* pNData) {
  struct rasCollConns connsData = {.travelTimeMin = INT64_MAX, .travelTimeMax = INT64_MIN};
  struct rasCollConns* pConnsData;

  // Update the statistical data first and in the process also calculate how much connection-specific space we
  // will need.
  for (int i = 0; i < nRasConns; i++) {
    struct rasConnection* conn = rasConns+i;
    if (conn->inUse && conn->travelTimeCount > 0) {
      if (connsData.travelTimeMin > conn->travelTimeMin)
        connsData.travelTimeMin = conn->travelTimeMin;
      if (connsData.travelTimeMax < conn->travelTimeMax)
        connsData.travelTimeMax = conn->travelTimeMax;
      connsData.travelTimeSum += conn->travelTimeSum;
      connsData.travelTimeCount += conn->travelTimeCount;
      connsData.nConns++;
      if (conn->travelTimeMin < 0)
        connsData.nNegativeMins++;
    }
  }

  *pNData = sizeof(connsData) + connsData.nNegativeMins*sizeof(*connsData.negativeMins);
  NCCLCHECK(ncclCalloc(pData, *pNData));
  pConnsData = (struct rasCollConns*)*pData;
  memcpy(pConnsData, &connsData, sizeof(*pConnsData));
  if (connsData.nNegativeMins > 0) {
    for (int i = 0, negMinsIdx = 0; i < nRasConns; i++) {
      struct rasConnection* conn = rasConns+i;
      if (conn->inUse && conn->travelTimeMin < 0) {
        struct rasCollConns::negativeMin* negativeMin = pConnsData->negativeMins+negMinsIdx;
        memcpy(&negativeMin->source, &rasNetListeningSocket.addr, sizeof(negativeMin->source));
        memcpy(&negativeMin->dest, &conn->addr, sizeof(negativeMin->dest));
        negativeMin->travelTimeMin = conn->travelTimeMin;
        negMinsIdx++;
      }
      assert(negMinsIdx <= connsData.nNegativeMins);
    }
  }

  return ncclSuccess;
}

// Merges incoming collective RAS_COLL_CONNS response message into the local accumulated data.
static ncclResult_t rasCollConnsMerge(struct rasCollective* coll, struct rasMsg* msg) {
  struct rasCollConns* collData;
  struct rasCollConns* msgData;
  int dataOffset = rasMsgLength(RAS_MSG_COLLRESP) + msg->collResp.nPeers*sizeof(*msg->collResp.peers);
  ALIGN_SIZE(dataOffset, alignof(int64_t));

  msgData = (struct rasCollConns*)(((char*)msg) + dataOffset);
  collData = (struct rasCollConns*)coll->data;

  // Merge the stats.
  if (collData->travelTimeMin > msgData->travelTimeMin)
    collData->travelTimeMin = msgData->travelTimeMin;
  if (collData->travelTimeMax < msgData->travelTimeMax)
    collData->travelTimeMax = msgData->travelTimeMax;
  collData->travelTimeSum += msgData->travelTimeSum;
  collData->travelTimeCount += msgData->travelTimeCount;
  collData->nConns += msgData->nConns;

  // Append the info about negative minimums.
  if (msgData->nNegativeMins > 0) {
    int nData = sizeof(*collData) +
      (collData->nNegativeMins+msgData->nNegativeMins) * sizeof(*collData->negativeMins);
    NCCLCHECK(ncclRealloc(&coll->data, coll->nData, nData));
    collData = (struct rasCollConns*)coll->data;
    memcpy(coll->data+coll->nData, msgData->negativeMins,
           msgData->nNegativeMins * sizeof(*collData->negativeMins));
    coll->nData = nData;
    collData->nNegativeMins += msgData->nNegativeMins;
  }

  return ncclSuccess;
}


/////////////////////////////////////////////////////////////////////////
// Functions related to the handling of the RAS_COLL_COMMS collective. //
/////////////////////////////////////////////////////////////////////////

// Initializes the accumulated data with just the local data for now.
// For this particular collective, we keep for every communicator information about every rank, to help identify
// the missing ones and the discrepancies between the ones that did respond.
static ncclResult_t rasCollCommsInit(char** pData, int* pNData) {
  struct rasCollComms* commsData;
  int nComms = 0, nRanks = 0;
  std::lock_guard<std::mutex> lock(ncclCommsMutex);

  // Start by counting the communicators so that we know how much space to allocate.
  // We also need to sort the comms array, to make the subsequent merging easier, both between the ranks (in case
  // of multiple GPUs per process) and between the peers.
  if (!ncclCommsSorted) {
    qsort(ncclComms, nNcclComms, sizeof(*ncclComms), &ncclCommsCompare);
    ncclCommsSorted = true;
  }
  for (int i = 0; i < nNcclComms; i++) {
    if (ncclComms[i] == nullptr) // nullptr's are always at the end after sorting.
      break;
    if (i == 0) {
      nComms = 1;
    } else if (ncclComms[i]->commHash != ncclComms[i-1]->commHash) {
      nComms++;
    }
    nRanks++;
  }

  // rasNetCollCommsData has nested variable-length arrays, which makes the size calculation and subsequent
  // pointer manipulations somewhat unwieldy...
  *pNData = sizeof(*commsData) + nComms * sizeof(*commsData->comms) + nRanks * sizeof(*commsData->comms[0].ranks);
  NCCLCHECK(ncclCalloc(pData, *pNData));
  commsData = (struct rasCollComms*)*pData;
  commsData->nComms = nComms;

  // comm points at the space in the accumulated data where the info about the current communicator is to be stored.
  struct rasCollComms::comm* comm = commsData->comms;
  for (int i = 0; i < nNcclComms; i++) {
    struct rasCollComms::comm::rank* rank;
    ncclResult_t asyncError;
    if (ncclComms[i] == nullptr)
      break;
    if (i == 0 || ncclComms[i]->commHash != ncclComms[i-1]->commHash) {
      if (i > 0)
        comm = (struct rasCollComms::comm*)(((char*)(comm+1)) + comm->nRanks * sizeof(*comm->ranks));
      comm->commHash = ncclComms[i]->commHash;
      comm->commNRanks = ncclComms[i]->nRanks;
      comm->nRanks = 0;
    } else if (ncclComms[i]->nRanks != ncclComms[i-1]->nRanks) {
      INFO(NCCL_RAS, "RAS encountered inconsistent communicator data: size %d != %d -- "
           "possible commHash collision (0x%lx)", ncclComms[i-1]->nRanks, ncclComms[i]->nRanks, comm->commHash);
      continue; // Short of failing, the best we can do is skip...
    } else if (ncclComms[i]->rank == ncclComms[i-1]->rank) {
      INFO(NCCL_RAS, "RAS encountered duplicate data for rank %d -- possible commHash collision (0x%lx)",
           ncclComms[i]->rank, comm->commHash);
      continue; // Short of failing, the best we can do is skip...
    }
    if (comm->nRanks == comm->commNRanks) {
      INFO(NCCL_RAS,
           "RAS encountered more ranks than the communicator size (%d) -- possible commHash collision (0x%lx)",
           comm->commNRanks, comm->commHash);
      continue; // Short of failing, the best we can do is skip...
    }
    rank = comm->ranks+comm->nRanks;
    rank->commRank = ncclComms[i]->rank;
    // rasNetSendCollReq initializes coll->peers[0] to our rasNetListeningSocket.addr, so peerIdx is initially
    // always 0.  It will increase after we send this response back to the peer we got the request from.
    rank->peerIdx = 0;
    rank->collOpCount = ncclComms[i]->collOpCount;
    rank->status.initState = ncclComms[i]->initState;
    if (ncclCommGetAsyncError(ncclComms[i], &asyncError) == ncclSuccess)
      rank->status.asyncError = asyncError;
    rank->status.finalizeCalled = (ncclComms[i]->finalizeCalled != 0);
    rank->status.destroyFlag = (ncclComms[i]->destroyFlag != 0);
    rank->status.abortFlag = (__atomic_load_n(ncclComms[i]->abortFlag, __ATOMIC_ACQUIRE) != 0);
    rank->cudaDev = ncclComms[i]->cudaDev;
    rank->nvmlDev = ncclComms[i]->nvmlDev;
    comm->nRanks++;
  }
  assert(nComms == 0 || ((char*)(comm->ranks+comm->nRanks)) - (char*)commsData <= *pNData);

  return ncclSuccess;
}

// Merges incoming collective RAS_COLL_COMMS response message into the local accumulated data.
static ncclResult_t rasCollCommsMerge(struct rasCollective* coll, struct rasMsg* msg) {
  struct rasCollComms* collData;
  struct rasCollComms* msgData;
  int dataOffset = rasMsgLength(RAS_MSG_COLLRESP) + msg->collResp.nPeers*sizeof(*msg->collResp.peers);
  ALIGN_SIZE(dataOffset, alignof(int64_t));

  msgData = (struct rasCollComms*)(((char*)msg) + dataOffset);
  collData = (struct rasCollComms*)coll->data;

  if (msgData->nComms > 0) {
    struct rasCollComms* newData = nullptr;

    // Allocate the new buffer pessimistically (sized as the sum of the two old ones).
    NCCLCHECK(ncclCalloc((char**)&newData, coll->nData + msg->collResp.nData));
    struct rasCollComms::comm* collComm = collData->comms;
    struct rasCollComms::comm* msgComm = msgData->comms;
    struct rasCollComms::comm* newComm = newData->comms;

    for (int collIdx = 0, msgIdx = 0; collIdx < collData->nComms || msgIdx < msgData->nComms; newData->nComms++) {
      int cmp;
      if (collIdx < collData->nComms && msgIdx < msgData->nComms)
        cmp = (collComm->commHash < msgComm->commHash ? -1 : (collComm->commHash > msgComm->commHash ? 1 : 0));
      else
        cmp = (collIdx < collData->nComms ? -1 : 1);

      if (cmp == 0 && collComm->commNRanks != msgComm->commNRanks) {
        INFO(NCCL_RAS, "RAS encountered inconsistent communicator data: size %d != %d -- "
             "possible commHash collision (0x%lx)", collComm->commNRanks, msgComm->commNRanks, collComm->commHash);
        cmp = (collComm->commNRanks < msgComm->commNRanks ? -1 : 1);
        // We try to preserve both separately, although the input data might already be messed up anyway...
      }

      if (cmp == 0) {
        // Merge the comms.
        newComm->commHash = collComm->commHash;
        newComm->commNRanks = collComm->commNRanks;
        if (collComm->nRanks + msgComm->nRanks > collComm->commNRanks) {
          INFO(NCCL_RAS,
               "RAS encountered more ranks (%d) than the communicator size (%d) -- possible commHash collision (0x%lx)",
               collComm->nRanks + msgComm->nRanks, newComm->commNRanks, newComm->commHash);
          // We'll skip the extras in the loop below.
        } else {
          newComm->nRanks = collComm->nRanks + msgComm->nRanks;
        }
        // Merge the ranks.
        for (int newRankIdx = 0, collRankIdx = 0, msgRankIdx = 0;
             collRankIdx < collComm->nRanks || msgRankIdx < msgComm->nRanks;
             newRankIdx++) {
          int cmpRank;
          if (newRankIdx == newComm->commNRanks)
            break; // Short of failing, the best we can do is skip...
          if (collRankIdx < collComm->nRanks && msgRankIdx < msgComm->nRanks)
            cmpRank = (collComm->ranks[collRankIdx].commRank < msgComm->ranks[msgRankIdx].commRank ? -1 :
                       (collComm->ranks[collRankIdx].commRank > msgComm->ranks[msgRankIdx].commRank ? 1 : 0));
          else
            cmpRank = (collRankIdx < collComm->nRanks ? -1 : 1);

          // There shouldn't be any overlaps in ranks between different sources.
          if (cmpRank == 0) {
            INFO(NCCL_RAS, "RAS encountered duplicate data for rank %d -- possible commHash collision (0x%lx)",
                 collComm->ranks[collRankIdx].commRank, newComm->commHash);
            msgRankIdx++; // Short of failing, the best we can do is skip...
          }
          memcpy(newComm->ranks+newRankIdx, (cmpRank <= 0 ? collComm->ranks+collRankIdx++ :
                                             msgComm->ranks+msgRankIdx++), sizeof(*newComm->ranks));
          if (cmpRank > 0) {
            // peerIdx values from msgComm need to shift after merge.
            newComm->ranks[newRankIdx].peerIdx += coll->nPeers;
          }
        } // for (newRankIdx)
        newComm = (struct rasCollComms::comm*)(((char*)(newComm+1)) + newComm->nRanks * sizeof(*newComm->ranks));
        collComm = (struct rasCollComms::comm*)(((char*)(collComm+1)) + collComm->nRanks * sizeof(*collComm->ranks));
        collIdx++;
        msgComm = (struct rasCollComms::comm*)(((char*)(msgComm+1)) + msgComm->nRanks * sizeof(*msgComm->ranks));
        msgIdx++;
      } else if (cmp < 0) {
        // Copy from collComm.
        int commSize = sizeof(*collComm) + collComm->nRanks * sizeof(*collComm->ranks);
        memcpy(newComm, collComm, commSize);
        newComm = (struct rasCollComms::comm*)(((char*)(newComm)) + commSize);
        collComm = (struct rasCollComms::comm*)(((char*)(collComm)) + commSize);
        collIdx++;
      } else { // cmp > 0
        // Copy from msgComm.
        int commSize = sizeof(*msgComm) + msgComm->nRanks * sizeof(*msgComm->ranks);
        memcpy(newComm, msgComm, commSize);
        for (int i = 0; i < newComm->nRanks; i++) {
          // peerIdx values from msgComm need to shift after merge.
          newComm->ranks[i].peerIdx += coll->nPeers;
        }
        newComm = (struct rasCollComms::comm*)(((char*)(newComm)) + commSize);
        msgComm = (struct rasCollComms::comm*)(((char*)(msgComm)) + commSize);
        msgIdx++;
      } // cmp > 0
    } // for (collIdx and msgIdx)

    free(coll->data);
    coll->data = (char*)newData;
    // newComm points at the next element beyond the last one -- exactly what we need.
    coll->nData = ((char*)newComm) - (char*)newData;
  } // if (msgData->nComms > 0)

  return ncclSuccess;
}

// Sorting callback for the ncclComms array.
static int ncclCommsCompare(const void* p1, const void* p2) {
  const ncclComm** pc1 = (const ncclComm**)p1;
  const ncclComm** pc2 = (const ncclComm**)p2;

  // Put nullptr's at the end.
  if (*pc1 == nullptr || *pc2 == nullptr)
    return (*pc1 != nullptr ? -1 : (*pc2 != nullptr ? 1 : 0));

  if ((*pc1)->commHash == (*pc2)->commHash) {
    return ((*pc1)->rank < (*pc2)->rank ? -1 : ((*pc1)->rank > (*pc2)->rank ? 1 : 0));
  } else {
    return ((*pc1)->commHash < (*pc2)->commHash ? -1 : 1);
  }
}
