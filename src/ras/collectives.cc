/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NDEBUG // Comment out during development only!
#include <cassert>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "transport.h"
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

// Keeping track of ongoing collective operations (apart from broadcasts, which have no response so require
// no such tracking).
struct rasCollective* rasCollectivesHead;
struct rasCollective* rasCollectivesTail;

static ncclResult_t getNewCollEntry(struct rasCollective** pColl);
static ncclResult_t rasLinkSendCollReq(struct rasLink* link, struct rasCollective* coll,
                                       const struct rasCollRequest* req, size_t reqLen, struct rasConnection* fromConn);
static ncclResult_t rasConnSendCollReq(struct rasConnection* conn, const struct rasCollRequest* req, size_t reqLen);
static ncclResult_t rasCollReadyResp(struct rasCollective* coll);
static ncclResult_t rasConnSendCollResp(struct rasConnection* conn,
                                        const union ncclSocketAddress* rootAddr, uint64_t rootId,
                                        const union ncclSocketAddress* peers, int nPeers,
                                        const char* data, int nData, int nLegTimeouts);

static ncclResult_t rasCollConnsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData);
static ncclResult_t rasCollConnsMerge(struct rasCollective* coll, struct rasMsg* msg);

static ncclResult_t rasCollCommsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData);
static ncclResult_t rasCollCommsMerge(struct rasCollective* coll, struct rasMsg* msg);
static bool rasCollCommsSkipMissing(const struct rasCollRequest* req, struct ncclComm* comm);
static int ncclCommsCompare(const void* p1, const void* p2);
static int peersHashesCompare(const void* p1, const void* p2);
static int peersHashesSearch(const void* k, const void* e);
static int rasCommIdCompare(const void* p1, const void* p2);
static int rasCollCommsMissingRankSearch(const void* k, const void* e);


///////////////////////////////////////////////////////////////////////////////////////
// Functions related to the initialization of collectives and the message exchanges. //
///////////////////////////////////////////////////////////////////////////////////////

// Returns the index of the first available entry in the rasCollectives array, enlarging the array if necessary.
static ncclResult_t getNewCollEntry(struct rasCollective** pColl) {
  struct rasCollective* coll;
  int nRasConns;

  NCCLCHECK(ncclCalloc(&coll, 1));

  coll->startTime = clockNano();
  coll->fromConn = nullptr;
  // We are unlikely to use the whole array, but at least we won't need to realloc.
  nRasConns = 0;
  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next)
    nRasConns++;
  NCCLCHECK(ncclCalloc(&coll->fwdConns, nRasConns));

  if (rasCollectivesHead) {
    rasCollectivesTail->next = coll;
    coll->prev = rasCollectivesTail;
    rasCollectivesTail = coll;
  } else {
    rasCollectivesHead = rasCollectivesTail = coll;
  }

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
// pColl provides on return a pointer to the allocated rasCollective structure to track this collective (unless
// it's a broadcast, which require no such tracking).
ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, bool* pAllDone,
                               struct rasCollective** pColl, struct rasConnection* fromConn) {
  struct rasCollective* coll = nullptr;
  struct rasCollRequest* reqMod = (struct rasCollRequest*)req;
  size_t reqLen = 0;
  if (req->type >= RAS_COLL_CONNS) {
    // Keep track of this collective operation so that we can handle the responses appropriately.
    NCCLCHECK(getNewCollEntry(&coll));
    if (pColl)
      *pColl = coll;
    memcpy(&coll->rootAddr, &req->rootAddr, sizeof(coll->rootAddr));
    coll->rootId = req->rootId;
    coll->type = req->type;
    coll->timeout = req->timeout;
    coll->fromConn = fromConn;
    if (ncclCalloc(&coll->peers, 1) == ncclSuccess) {
      memcpy(coll->peers, &rasNetListeningSocket.addr, sizeof(*coll->peers));
      coll->nPeers = 1;
    }

    // Collective-specific initialization of accumulated data (using local data for now).
    if (req->type == RAS_COLL_CONNS)
      (void)rasCollConnsInit(&reqMod, &reqLen, &coll->data, &coll->nData);
    else if (req->type == RAS_COLL_COMMS)
      (void)rasCollCommsInit(&reqMod, &reqLen, &coll->data, &coll->nData);
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
      rasMsgHandleBCDeadPeer(&reqMod, &reqLen, &done);
      if (done)
        goto exit;
    }
  } // req->type < RAS_COLL_CONNS

  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next)
    conn->linkFlag = false;

  (void)rasLinkSendCollReq(&rasNextLink, coll, reqMod, reqLen, fromConn);
  (void)rasLinkSendCollReq(&rasPrevLink, coll, reqMod, reqLen, fromConn);

  if (coll && pAllDone)
    *pAllDone = (coll->nFwdSent == coll->nFwdRecv);
exit:
  if (reqMod != req)
    free(reqMod);
  return ncclSuccess;
}

// Sends the collective message through all connections associated with this link (with the exception of the one
// the message came from, if any).
static ncclResult_t rasLinkSendCollReq(struct rasLink* link, struct rasCollective* coll,
                                       const struct rasCollRequest* req, size_t reqLen,
                                       struct rasConnection* fromConn) {
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next) {
    if (linkConn->conn && linkConn->conn != fromConn && !linkConn->conn->linkFlag) {
      // We send collective messages through fully established and operational connections only.
      if (linkConn->conn->sock && linkConn->conn->sock->status == RAS_SOCK_READY &&
          !linkConn->conn->experiencingDelays) {
        if (rasConnSendCollReq(linkConn->conn, req, reqLen) == ncclSuccess && coll != nullptr)
          coll->fwdConns[coll->nFwdSent++] = linkConn->conn;
      } // linkConn->conn is fully established and operational.
      linkConn->conn->linkFlag = true;
    } // if (linkConn->conn && linkConn->conn != fromConn && !linkConn->con->linkFlag)
  } // for (linkConn)

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
  struct rasCollective* coll = nullptr;
  assert(sock->conn);

  // First check if we've already handled this request (through another connection).
  for (int i = 0; i < nRasCollHistory; i++) {
    // In principle we can use i to index the array but we convert it so that we check the most recent entries first.
    int collHistIdx = (rasCollHistNextIdx + COLL_HISTORY_SIZE - 1 - i) % COLL_HISTORY_SIZE;
    if (memcmp(&msg->collReq.rootAddr, &rasCollHistory[collHistIdx].rootAddr, sizeof(msg->collReq.rootAddr)) == 0 &&
        msg->collReq.rootId == rasCollHistory[collHistIdx].rootId) {
      if (msg->collReq.type >= RAS_COLL_CONNS) {
        // Send an empty response so that the sender can account for it.  The non-empty response has already been
        // sent through the connection that we received the request through first.
        NCCLCHECK(rasConnSendCollResp(sock->conn, &msg->collReq.rootAddr, msg->collReq.rootId,
                                      /*peers*/nullptr, /*nPeers*/0, /*data*/nullptr, /*nData*/0, /*nLegTimeouts*/0));
      }
      goto exit;
    }
  } // for (i)

  if (msg->collReq.type >= RAS_COLL_CONNS) {
    // Check if we're currently handling this collective request.
    for (coll = rasCollectivesHead; coll; coll = coll->next) {
      if (memcmp(&msg->collReq.rootAddr, &coll->rootAddr, sizeof(msg->collReq.rootAddr)) == 0 &&
          msg->collReq.rootId == coll->rootId) {
        assert(msg->collReq.type == coll->type);

        // Send an empty response so that the sender can account for it.  The non-empty response will be
        // sent through the connection that we received the request through first.
        NCCLCHECK(rasConnSendCollResp(sock->conn, &msg->collReq.rootAddr, msg->collReq.rootId,
                                      /*peers*/nullptr, /*nPeers*/0, /*data*/nullptr, /*nData*/0, /*nLegTimeouts*/0));
        goto exit;
      } // if match
    } // for (coll)
  } // if (msg->collReq.type >= RAS_COLL_CONNS)

  // Re-broadcast the message to my peers (minus the one it came from) and handle it locally.
  NCCLCHECK(rasNetSendCollReq(&msg->collReq, &allDone, &coll, sock->conn));

  if (msg->collReq.type >= RAS_COLL_CONNS && allDone) {
    assert(coll);
    // We are a leaf process -- send the response right away.  This can probably trigger only for the case of a total
    // of two peers, and hence just one RAS connection, or during communication issues, because normally every peer
    // has more than one connection so there should always be _some_ other peer to forward the request to.
    NCCLCHECK(rasCollReadyResp(coll));
  }
exit:
  return ncclSuccess;
}

// Sends a collective response back to the process we received the collective request from.
// Invoked when we are finished waiting for the collective responses from other peers (i.e., either there weren't
// any peers (unlikely), the peers sent their responses (likely), or we timed out.
static ncclResult_t rasCollReadyResp(struct rasCollective* coll) {
  if (coll->fromConn) {
    // For remotely-initiated collectives, send the response back.
    NCCLCHECK(rasConnSendCollResp(coll->fromConn, &coll->rootAddr, coll->rootId,
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
  struct rasCollective* coll;
  char line[SOCKET_NAME_MAXLEN+1];

  for (coll = rasCollectivesHead; coll; coll = coll->next) {
    if (memcmp(&msg->collResp.rootAddr, &coll->rootAddr, sizeof(msg->collResp.rootAddr)) == 0 &&
        msg->collResp.rootId == coll->rootId)
      break;
  }
  if (coll == nullptr) {
    INFO(NCCL_RAS, "RAS failed to find a matching ongoing collective for response %s:%ld from %s!",
         ncclSocketToString(&msg->collResp.rootAddr, line), msg->collResp.rootId,
         ncclSocketToString(&sock->sock.addr, rasLine));
    goto exit;
  }

  coll->nLegTimeouts += msg->collResp.nLegTimeouts;
  assert(sock->conn);
  // Account for the received response in our collective operations tracking.
  for (int i = 0; i < coll->nFwdSent; i++) {
    if (coll->fwdConns[i] == sock->conn) {
      coll->fwdConns[i] = nullptr;
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
void rasCollsPurgeConn(struct rasConnection* conn) {
  for (struct rasCollective* coll = rasCollectivesHead; coll;) {
    struct rasCollective* collNext = coll->next;
    char line[SOCKET_NAME_MAXLEN+1];
    if (coll->fromConn == conn) {
      INFO(NCCL_RAS, "RAS purging collective %s:%ld because it comes from %s",
           ncclSocketToString(&coll->rootAddr, line), coll->rootId,
           ncclSocketToString(&conn->addr, rasLine));
      rasCollFree(coll);
    } else {
      for (int i = 0; i < coll->nFwdSent; i++) {
        if (coll->fwdConns[i] == conn) {
          coll->fwdConns[i] = nullptr;
          coll->nFwdRecv++;
          coll->nLegTimeouts++;
          INFO(NCCL_RAS, "RAS not waiting for response from %s to collective %s:%ld "
               "(nFwdSent %d, nFwdRecv %d, nLegTimeouts %d)",
               ncclSocketToString(&conn->addr, rasLine), ncclSocketToString(&coll->rootAddr, line), coll->rootId,
               coll->nFwdSent, coll->nFwdRecv, coll->nLegTimeouts);
          if (coll->nFwdSent == coll->nFwdRecv)
            (void)rasCollReadyResp(coll);
          break;
        }
      } // for (i)
    } // coll->fromConn != conn
    coll = collNext;
  } // for (coll)
}

// Frees a rasCollective entry and any memory associated with it.
void rasCollFree(struct rasCollective* coll) {
  if (coll == nullptr)
    return;

  free(coll->fwdConns);
  free(coll->peers);
  free(coll->data);

  if (coll == rasCollectivesHead)
    rasCollectivesHead = rasCollectivesHead->next;
  if (coll == rasCollectivesTail)
    rasCollectivesTail = rasCollectivesTail->prev;
  if (coll->prev)
    coll->prev->next = coll->next;
  if (coll->next)
    coll->next->prev = coll->prev;
  free(coll);
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
  for (struct rasCollective* coll = rasCollectivesHead; coll;) {
    struct rasCollective* collNext = coll->next;
    if (coll->timeout > 0) {
      if (now - coll->startTime > coll->timeout) {
        // We've exceeded the leg timeout.  For all outstanding responses, check their connections.
        if (!coll->timeoutWarned) {
          INFO(NCCL_RAS, "RAS collective %s:%ld timeout warning (%lds) -- %d responses missing",
               ncclSocketToString(&coll->rootAddr, rasLine), coll->rootId,
               (now - coll->startTime) / CLOCK_UNITS_PER_SEC, coll->nFwdSent - coll->nFwdRecv);
          coll->timeoutWarned = true;
        }
        for (int i = 0; i < coll->nFwdSent; i++) {
          if (coll->fwdConns[i]) {
            struct rasConnection* conn = coll->fwdConns[i];
            char line[SOCKET_NAME_MAXLEN+1];
            if (!conn->experiencingDelays && conn->sock) {
              // Ensure that the connection is fully established and operational, and that the socket hasn't been
              // re-created during the handling of the collective (which would suggest that the request may have been
              // lost).
              if (conn->sock->status == RAS_SOCK_READY && conn->sock->createTime < coll->startTime)
                continue;
            }
            // In all other cases we declare a timeout so that we can (hopefully) recover.
            INFO(NCCL_RAS, "RAS not waiting for response from %s to collective %s:%ld "
                 "(nFwdSent %d, nFwdRecv %d, nLegTimeouts %d)",
                 ncclSocketToString(&conn->addr, rasLine), ncclSocketToString(&coll->rootAddr, line),
                 coll->rootId, coll->nFwdSent, coll->nFwdRecv, coll->nLegTimeouts);
            coll->fwdConns[i] = nullptr;
            coll->nFwdRecv++;
            coll->nLegTimeouts++;
          } // if (coll->fwdConns[i])
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
    } // if (coll->timeout > 0)

    coll = collNext;
  } // for (coll)
}


/////////////////////////////////////////////////////////////////////////
// Functions related to the handling of the RAS_COLL_CONNS collective. //
/////////////////////////////////////////////////////////////////////////

// Initializes the accumulated data with just the local data for now.
// For this particular collective, we keep some reduced statistical data (min/max/avg travel time) as well
// as connection-specific info in case we observed a negative min travel time (which, ideally, shouldn't happen,
// but the system clocks may not be perfectly in sync).
static ncclResult_t rasCollConnsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData) {
  struct rasCollConns connsData = {.travelTimeMin = INT64_MAX, .travelTimeMax = INT64_MIN};
  struct rasCollConns* pConnsData;

  *pReqLen = rasCollDataLength(RAS_COLL_CONNS);

  // Update the statistical data first and in the process also calculate how much connection-specific space we
  // will need.
  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next) {
    if (conn->travelTimeCount > 0) {
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
    int negMinsIdx = 0;
    for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next) {
      if (conn->travelTimeMin < 0) {
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
// For any new (previously unseen) communicator we also save the basic identification data about every rank that is
// "missing" (i.e., not part of this process).  During merging, this should be replaced by the actual data from
// those ranks, if they are responsive.  We want to provide this information to the user (so that we can say more
// than "rank xyz missing").
// Every "new" communicator is also recorded in the (updated) request, so that when that request is forwarded to our
// peers, those peers don't needlessly send us the same data.
static ncclResult_t rasCollCommsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData) {
  ncclResult_t ret = ncclSuccess;
  struct rasCollComms* commsData;
  int nComms = 0, nRanks = 0, nMissingRanks = 0;
  bool skipMissing = false;
  std::lock_guard<std::mutex> lock(ncclCommsMutex);
  struct rasCollComms::comm* comm;
  struct rasCollRequest* req = nullptr;
  struct rasPeerInfo** peersReSorted = nullptr;
  int firstNewSkipMissingIdx = -1;

  *pReqLen = rasCollDataLength(RAS_COLL_COMMS) +
    (*pReq)->comms.nSkipMissingRanksComms * sizeof(*(*pReq)->comms.skipMissingRanksComms);
  *pData = nullptr;

  // Start by counting the communicators so that we know how much space to allocate.
  // We also need to sort the comms array, to make the subsequent merging easier, both between the ranks (in case
  // of multiple GPUs per process) and between the peers.
  if (!ncclCommsSorted) {
    qsort(ncclComms, nNcclComms, sizeof(*ncclComms), &ncclCommsCompare);
    ncclCommsSorted = true;
  }
  for (int commIdx = 0; commIdx < nNcclComms; commIdx++) {
    if (ncclComms[commIdx] == nullptr) // nullptr's are always at the end after sorting.
      break;
    // A process may manage multiple GPUs and thus have multiple communicators with the same commHash.
    // Comparing just the commHash is OK though within communicators that are part of the same process.
    if (commIdx == 0 || ncclComms[commIdx]->commHash != ncclComms[commIdx-1]->commHash) {
      skipMissing = rasCollCommsSkipMissing(*pReq, ncclComms[commIdx]);
      if (!skipMissing) {
        // Add this communicator to the request so that the processes we forward the request to know not to fill in
        // the missing rank info.
        struct rasCommId* skipComm;
        if (req == nullptr) {
          // We pessimistically allocate space for all the remaining communicators so that we don't need to reallocate.
          int newSize = *pReqLen + (nNcclComms-commIdx) * sizeof(*req->comms.skipMissingRanksComms);
          NCCLCHECKGOTO(ncclCalloc((char**)&req, newSize), ret, fail);
          memcpy(req, *pReq, *pReqLen);
          *pReq = req;
          firstNewSkipMissingIdx = req->comms.nSkipMissingRanksComms;
        }
        skipComm = req->comms.skipMissingRanksComms + req->comms.nSkipMissingRanksComms++;
        skipComm->commHash = ncclComms[commIdx]->commHash;
        skipComm->hostHash = ncclComms[commIdx]->peerInfo->hostHash;
        skipComm->pidHash = ncclComms[commIdx]->peerInfo->pidHash;

        nMissingRanks += ncclComms[commIdx]->nRanks;
      } // if (!skipMissing)
      nComms++;
    } // if encountered a new communicator
    nRanks++;
    if (!skipMissing)
      nMissingRanks--;
  } // for (commIdx)

  // rasCollComms has nested variable-length arrays, which makes the size calculation and subsequent
  // pointer manipulations somewhat unwieldy...
  // This is extra complicated because of the "hidden" array of struct rasCollCommsMissingRank following the
  // ranks array for each communicator.
  *pNData = sizeof(*commsData) + nComms * sizeof(*commsData->comms) + nRanks * sizeof(*commsData->comms[0].ranks) +
    nMissingRanks * sizeof(struct rasCollCommsMissingRank);
  NCCLCHECKGOTO(ncclCalloc(pData, *pNData), ret, fail);
  commsData = (struct rasCollComms*)*pData;
  commsData->nComms = nComms;

  // comm points at the space in the accumulated data where the info about the current communicator is to be stored.
  comm = commsData->comms;
  // collCommIdx counts rasCollComms::comm (comm); commIdx indexes ncclComms.
  for (int collCommIdx = 0, commIdx = 0; collCommIdx < nComms; collCommIdx++) {
    struct ncclComm* ncclComm = ncclComms[commIdx];

    comm->commId.commHash = ncclComm->commHash;
    comm->commId.hostHash = ncclComm->peerInfo->hostHash;
    comm->commId.pidHash = ncclComm->peerInfo->pidHash;
    comm->commNRanks = ncclComm->nRanks;
    comm->nRanks = comm->nMissingRanks = 0;

    // Fill in the comm->ranks array.
    for (; commIdx < nNcclComms && ncclComms[commIdx] && ncclComms[commIdx]->commHash == comm->commId.commHash;
         commIdx++) {
      ncclComm = ncclComms[commIdx];
      struct rasCollComms::comm::rank* rank = comm->ranks+comm->nRanks;
      ncclResult_t asyncError;
      rank->commRank = ncclComm->rank;
      // rasNetSendCollReq initializes coll->peers[0] to our rasNetListeningSocket.addr, so peerIdx is initially
      // always 0.  It will increase after we send this response back to the peer we got the request from.
      rank->peerIdx = 0;
      memcpy(rank->collOpCounts, ncclComm->seqNumber, sizeof(rank->collOpCounts));
      rank->status.initState = ncclComm->initState;
      if (ncclCommGetAsyncError(ncclComm, &asyncError) == ncclSuccess)
        rank->status.asyncError = asyncError;
      rank->status.finalizeCalled = (ncclComm->finalizeCalled != 0);
      rank->status.destroyFlag = (ncclComm->destroyFlag != 0);
      rank->status.abortFlag = (__atomic_load_n(ncclComm->abortFlag, __ATOMIC_ACQUIRE) != 0);
      rank->cudaDev = ncclComm->cudaDev;
      rank->nvmlDev = ncclComm->nvmlDev;
      comm->nRanks++;
    } // for (commIdx)

    if (firstNewSkipMissingIdx != -1 &&
        memcmp(req->comms.skipMissingRanksComms+firstNewSkipMissingIdx, &comm->commId, sizeof(comm->commId)) == 0) {
      // Fill in the missingRanks array that follows the comm->ranks.
      struct rasCollCommsMissingRank* missingRanks = (struct rasCollCommsMissingRank*)(comm->ranks+comm->nRanks);

      if (peersReSorted == nullptr) {
        // Create a lookup table to rasPeers that is sorted by hostHash and pidHash, to reduce the complexity of the
        // lookups in the missingRankIdx loop below.
        NCCLCHECKGOTO(ncclCalloc(&peersReSorted, nRasPeers), ret, fail);
        for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++)
          peersReSorted[peerIdx] = rasPeers+peerIdx;
        qsort(peersReSorted, nRasPeers, sizeof(*peersReSorted), peersHashesCompare);
      }

      comm->nMissingRanks = comm->commNRanks - comm->nRanks;
      for (int missingRankIdx = 0, rankIdx = 0; missingRankIdx < comm->nMissingRanks; missingRankIdx++) {
        struct rasCollCommsMissingRank* missingRank;
        struct ncclPeerInfo* info;
        struct rasPeerInfo** peer;
        uint64_t key[2];
        // Look for the next "hole" in the ranks array.
        while (rankIdx < comm->nRanks && comm->ranks[rankIdx].commRank == rankIdx+missingRankIdx)
          rankIdx++;

        missingRank = missingRanks + missingRankIdx;
        missingRank->commRank = rankIdx + missingRankIdx;
        info = ncclComm->peerInfo + missingRank->commRank;
        key[0] = info->hostHash - ncclComm->commHash;
        key[1] = info->pidHash - ncclComm->commHash;
        peer = (struct rasPeerInfo**)bsearch(key, peersReSorted, nRasPeers, sizeof(*peersReSorted), peersHashesSearch);
        if (peer)
          memcpy(&missingRank->addr, &(*peer)->addr, sizeof(missingRank->addr));
        missingRank->cudaDev = info->cudaDev;
        missingRank->nvmlDev = info->nvmlDev;
      } // for (missingRankIdx)

      if (++firstNewSkipMissingIdx == req->comms.nSkipMissingRanksComms)
        firstNewSkipMissingIdx = -1;
    } // if need to fill in the missingRanks

    comm = (struct rasCollComms::comm*)(((char*)(comm+1)) + comm->nRanks * sizeof(*comm->ranks) +
                                        comm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
  } // for (collCommIdx)
  assert(((char*)comm) - (char*)commsData <= *pNData);

  if (req) {
    // Finish updating the request.
    *pReqLen = rasCollDataLength(RAS_COLL_COMMS) +
      req->comms.nSkipMissingRanksComms * sizeof(*req->comms.skipMissingRanksComms);
    qsort(req->comms.skipMissingRanksComms, req->comms.nSkipMissingRanksComms,
          sizeof(*req->comms.skipMissingRanksComms), rasCommIdCompare);
  }
ret:
  free(peersReSorted);
  return ret;
fail:
  if (req) {
    free(req);
    *pReq = nullptr;
  }
  free(*pData);
  *pData = nullptr;
  goto ret;
}

// Merges incoming collective RAS_COLL_COMMS response message into the local accumulated data.
static ncclResult_t rasCollCommsMerge(struct rasCollective* coll, struct rasMsg* msg) {
  struct rasCollComms* collData; // Data previously stored (locally) by our process.
  struct rasCollComms* msgData; // Data just received from another process.
  int dataOffset = rasMsgLength(RAS_MSG_COLLRESP) + msg->collResp.nPeers*sizeof(*msg->collResp.peers);
  ALIGN_SIZE(dataOffset, alignof(int64_t));

  msgData = (struct rasCollComms*)(((char*)msg) + dataOffset);
  collData = (struct rasCollComms*)coll->data;

  if (msgData->nComms > 0) {
    struct rasCollComms* newData = nullptr; // Destination buffer for the merged data.

    // Allocate the new buffer pessimistically (sized as the sum of the two old ones).
    NCCLCHECK(ncclCalloc((char**)&newData, coll->nData + msg->collResp.nData));
    struct rasCollComms::comm* collComm = collData->comms;
    struct rasCollComms::comm* msgComm = msgData->comms;
    struct rasCollComms::comm* newComm = newData->comms;

    for (int collIdx = 0, msgIdx = 0; collIdx < collData->nComms || msgIdx < msgData->nComms; newData->nComms++) {
      int cmp;
      if (collIdx < collData->nComms && msgIdx < msgData->nComms)
        cmp = rasCommIdCompare(&collComm->commId, &msgComm->commId);
      else
        cmp = (collIdx < collData->nComms ? -1 : 1);

      if (cmp == 0 && collComm->commNRanks != msgComm->commNRanks) {
        INFO(NCCL_RAS, "RAS encountered inconsistent communicator data: size %d != %d -- "
             "possible hash collision (0x%lx, 0x%lx, 0x%lx)", collComm->commNRanks, msgComm->commNRanks,
             collComm->commId.commHash, collComm->commId.hostHash, collComm->commId.pidHash);
        cmp = (collComm->commNRanks < msgComm->commNRanks ? -1 : 1);
        // We try to preserve them both separately...
      }

      if (cmp == 0) {
        // Merge the comms.
        memcpy(&newComm->commId, &collComm->commId, sizeof(newComm->commId));
        newComm->commNRanks = collComm->commNRanks;
        if (collComm->nRanks + msgComm->nRanks > collComm->commNRanks) {
          INFO(NCCL_RAS,
               "RAS encountered more ranks (%d) than the communicator size (%d) -- possible hash collision "
               "(0x%lx, 0x%lx, 0x%lx)", collComm->nRanks + msgComm->nRanks, newComm->commNRanks,
               collComm->commId.commHash, collComm->commId.hostHash, collComm->commId.pidHash);
          newComm->nRanks = newComm->commNRanks;
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
          if (collRankIdx < collComm->nRanks && msgRankIdx < msgComm->nRanks) {
            cmpRank = (collComm->ranks[collRankIdx].commRank < msgComm->ranks[msgRankIdx].commRank ? -1 :
                       (collComm->ranks[collRankIdx].commRank > msgComm->ranks[msgRankIdx].commRank ? 1 : 0));
          } else {
            cmpRank = (collRankIdx < collComm->nRanks ? -1 : 1);
          }

          // There shouldn't be any overlaps in ranks between different sources.
          if (cmpRank == 0) {
            INFO(NCCL_RAS, "RAS encountered duplicate data for rank %d -- possible hash collision "
                 "(0x%lx, 0x%lx, 0x%lx)", collComm->ranks[collRankIdx].commRank,
                 newComm->commId.commHash, newComm->commId.hostHash, newComm->commId.pidHash);
            msgRankIdx++; // Short of failing, the best we can do is skip...
          }
          memcpy(newComm->ranks+newRankIdx, (cmpRank <= 0 ? collComm->ranks+collRankIdx++ :
                                             msgComm->ranks+msgRankIdx++), sizeof(*newComm->ranks));
          if (cmpRank > 0) {
            // peerIdx values from msgComm need to shift after merge.
            newComm->ranks[newRankIdx].peerIdx += coll->nPeers;

            if (collComm->nMissingRanks > 0) {
              // Remove the corresponding entry from missingRanks.
              struct rasCollCommsMissingRank* missingRank;
              missingRank = (struct rasCollCommsMissingRank*)bsearch(&newComm->ranks[newRankIdx].commRank,
                                                                     collComm->ranks+collComm->nRanks,
                                                                     collComm->nMissingRanks,
                                                                     sizeof(struct rasCollCommsMissingRank),
                                                                     rasCollCommsMissingRankSearch);
              if (missingRank) {
                // Mark the entry as no longer needed.
                memset(&missingRank->addr, '\0', sizeof(missingRank->addr));
              } else {
                INFO(NCCL_RAS, "RAS failed to find missingRank data -- internal error?");
              }
            } // if (collComm->nMissingRanks > 0)
          } // if (cmpRank > 0)
        } // for (newRankIdx)
        if (collComm->nMissingRanks > 0) {
          // Copy the missingRanks to newComm, skipping over any no longer needed entries.
          union ncclSocketAddress emptyAddr;
          struct rasCollCommsMissingRank* collMissingRanks;
          struct rasCollCommsMissingRank* newMissingRanks;
          int newRankIdx;

          memset(&emptyAddr, '\0', sizeof(emptyAddr));
          collMissingRanks = (struct rasCollCommsMissingRank*)(collComm->ranks+collComm->nRanks);
          newMissingRanks = (struct rasCollCommsMissingRank*)(newComm->ranks+newComm->nRanks);
          newRankIdx = 0;
          for (int collRankIdx = 0; collRankIdx < collComm->nMissingRanks; collRankIdx++) {
            if (memcmp(&collMissingRanks[collRankIdx].addr, &emptyAddr, sizeof(emptyAddr))) {
              memcpy(newMissingRanks + newRankIdx++, collMissingRanks + collRankIdx, sizeof(*newMissingRanks));
            }
          }
          newComm->nMissingRanks = newRankIdx;
          assert(newComm->nRanks + newComm->nMissingRanks == newComm->commNRanks);
        }
        newComm = (struct rasCollComms::comm*)(((char*)(newComm+1)) + newComm->nRanks * sizeof(*newComm->ranks) +
                                               newComm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
        collComm = (struct rasCollComms::comm*)(((char*)(collComm+1)) + collComm->nRanks * sizeof(*collComm->ranks) +
                                                collComm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
        collIdx++;
        msgComm = (struct rasCollComms::comm*)(((char*)(msgComm+1)) + msgComm->nRanks * sizeof(*msgComm->ranks) +
                                               msgComm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
        msgIdx++;
      } else if (cmp < 0) {
        // Copy from collComm.
        int commSize = sizeof(*collComm) + collComm->nRanks * sizeof(*collComm->ranks) +
          collComm->nMissingRanks * sizeof(struct rasCollCommsMissingRank);
        memcpy(newComm, collComm, commSize);
        newComm = (struct rasCollComms::comm*)(((char*)(newComm)) + commSize);
        collComm = (struct rasCollComms::comm*)(((char*)(collComm)) + commSize);
        collIdx++;
      } else { // cmp > 0
        // Copy from msgComm.
        int commSize = sizeof(*msgComm) + msgComm->nRanks * sizeof(*msgComm->ranks) +
          msgComm->nMissingRanks * sizeof(struct rasCollCommsMissingRank);
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

// Checks if a given communicator is in the skipMissingRanksComms array of the request.
static bool rasCollCommsSkipMissing(const struct rasCollRequest* req, struct ncclComm* comm) {
  struct rasCommId id;
  id.commHash = comm->commHash;
  id.hostHash = comm->peerInfo->hostHash;
  id.pidHash = comm->peerInfo->pidHash;
  return (bsearch(&id, req->comms.skipMissingRanksComms, req->comms.nSkipMissingRanksComms,
                  sizeof(*req->comms.skipMissingRanksComms), rasCommIdCompare) != nullptr);
}

// Sorting callback for the ncclComms array.
static int ncclCommsCompare(const void* p1, const void* p2) {
  const ncclComm* comm1 = *(const ncclComm**)p1;
  const ncclComm* comm2 = *(const ncclComm**)p2;

  // Put nullptr's at the end.
  if (comm1 == nullptr || comm2 == nullptr)
    return (comm1 != nullptr ? -1 : (comm2 != nullptr ? 1 : 0));

  if (comm1->commHash == comm2->commHash) {
    return (comm1->rank < comm2->rank ? -1 : (comm1->rank > comm2->rank ? 1 : 0));
  } else {
    return (comm1->commHash < comm2->commHash ? -1 : 1);
  }
}

// Sorting callback for a lookup table to rasPeers.  Sorts by the hostHash (primary) and pidHash (secondary).
static int peersHashesCompare(const void* p1, const void* p2) {
  const struct rasPeerInfo* pi1 = *(const struct rasPeerInfo**)p1;
  const struct rasPeerInfo* pi2 = *(const struct rasPeerInfo**)p2;

  if (pi1->hostHash == pi2->hostHash) {
    return (pi1->pidHash < pi2->pidHash ? -1 : (pi1->pidHash > pi2->pidHash ? 1 : 0));
  } else {
    return (pi1->hostHash < pi2->hostHash ? -1 : 1);
  }
}

// Search callback for a lookup table to rasPeers.  Searches by the hostHash and pidHash.  The key is an array
// containing the hostHash at index 0 and the pidHash at index 1.
static int peersHashesSearch(const void* k, const void* e) {
  const uint64_t* key = (const uint64_t*)k;
  const struct rasPeerInfo* elem = *(const struct rasPeerInfo**)e;

  if (key[0] == elem->hostHash) {
    return (key[1] < elem->pidHash ? -1 : (key[1] > elem->pidHash ? 1 : 0));
  } else {
    return (key[0] < elem->hostHash ? -1 : 1);
  }
}

// Sorting/searching callback for struct rasCommId.  Sorts by commHash, then hostHash, then pidHash.
static int rasCommIdCompare(const void* p1, const void* p2) {
  const struct rasCommId* i1 = (const struct rasCommId*)p1;
  const struct rasCommId* i2 = (const struct rasCommId*)p2;
  if (i1->commHash == i2->commHash) {
    if (i1->hostHash == i2->hostHash) {
      return (i1->pidHash < i2->pidHash ? -1 : (i1->pidHash > i2->pidHash ? 1 : 0));
    } else {
      return (i1->hostHash < i2->hostHash ? -1 : 1);
    }
  } else {
    return (i1->commHash < i2->commHash ? -1 : 1);
  }
}

// Search callback for rasCollComms::comm rasCollCommsMissingRank array.  The key is the commRank.
static int rasCollCommsMissingRankSearch(const void* k, const void* e) {
  int key = *(const int*)k;
  const struct rasCollCommsMissingRank* elem = (const struct rasCollCommsMissingRank*)e;

  return (key < elem->commRank ? -1 : (key > elem->commRank ? 1 : 0));
}

// Invoked during RAS termination to release all the allocated resources.
void rasCollectivesTerminate() {
  for (struct rasCollective* coll = rasCollectivesHead; coll;) {
    struct rasCollective* collNext = coll->next;
    rasCollFree(coll);
    coll = collNext;
  }

  // rasCollectivesHead and rasCollectivesTail are taken care of by rasCollFree().
}
