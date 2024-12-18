/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NDEBUG // Comment out during development only!
#include <cassert>

#include "ras_internal.h"

// Links forming the backbone of the RAS network (currently a ring).
struct rasLink rasNextLink = {1}, rasPrevLink = {-1};

// Connections on the RAS network.
struct rasConnection* rasConns;
int nRasConns;

// Sockets implementing the RAS network.
struct rasSocket *rasSockets;
int nRasSockets;

// Magic file descriptor number when we want poll() to ignore an entry.  Anything negative would do, but
// I didn't want to use -1 because it has a special meaning for us.
#define POLL_FD_IGNORE -2

static void rasConnOpen(struct rasConnection* conn);
static ncclResult_t rasConnPrepare(struct rasConnection* conn);
static void rasConnTerminate(struct rasConnection* conn);

static ncclResult_t getNewSockEntry(struct rasSocket** pSock);

static ncclResult_t rasLinkHandleNetTimeouts(struct rasLink* link, int64_t now, int64_t* nextWakeup);
static void rasConnHandleNetTimeouts(int connIdx, int64_t now, int64_t* nextWakeup);
static void rasConnSendKeepAlive(struct rasConnection* conn, bool nack = false);

static ncclResult_t rasLinkAddFallback(struct rasLink* link, int connIdx);
static void rasConnResume(struct rasConnection* conn);
static void rasLinkSanitizeFallbacks(struct rasLink* link);
static void rasLinkDropConn(struct rasLink* link, int connIdx, int linkIdx = -1);
static int rasLinkFindConn(const struct rasLink* link, int connIdx);


///////////////////////////////////////////////
// Functions related to the RAS connections. //
///////////////////////////////////////////////

// Allocates an entry in the rasConns array, enlarging the array if necessary.
ncclResult_t getNewConnEntry(struct rasConnection** pConn) {
  struct rasConnection* conn;
  int i;
  for (i = 0; i < nRasConns; i++)
    if (!rasConns[i].inUse)
      break;
  if (i == nRasConns) {
    NCCLCHECK(ncclRealloc(&rasConns, nRasConns, nRasConns+RAS_INCREMENT));
    nRasConns += RAS_INCREMENT;
  }

  conn = rasConns+i;
  memset(conn, '\0', sizeof(*conn));
  conn->inUse = true;
  conn->sockIdx = -1;
  ncclIntruQueueConstruct(&conn->sendQ);
  conn->travelTimeMin = INT64_MAX;
  conn->travelTimeMax = INT64_MIN;

  *pConn = conn;
  return ncclSuccess;
}

// Creates a new RAS network connection to a remote peer address.
ncclResult_t rasConnCreate(const union ncclSocketAddress* addr, int* pConnIdx) {
  ncclResult_t ret = ncclSuccess;
  struct rasConnection* conn = nullptr;

  // First check if a connection entry for this peer already exists.
  int connIdx = rasConnFind(addr);
  if (connIdx != -1) {
    conn = rasConns+connIdx;
  }

  if (conn && conn->sockIdx != -1) {
    // An entry exists and has a socket associated with it -- nothing left for us to do.
    if (pConnIdx)
      *pConnIdx = connIdx;
    goto exit;
  }

  if (!conn) {
    NCCLCHECKGOTO(getNewConnEntry(&conn), ret, exit);
    memcpy(&conn->addr, addr, sizeof(conn->addr));
    // We are establishing a new connection -- start the timeout.
    conn->startRetryTime = clockNano();
    connIdx = conn - rasConns;
  }

  if (pConnIdx)
    *pConnIdx = connIdx;

  rasConnOpen(conn);

exit:
  return ret;
}

// Opens a connection to a remote peer.
static void rasConnOpen(struct rasConnection* conn) {
  ncclResult_t ret; // Not used.
  struct rasSocket* sock;
  bool closeSocketOnFail = false;
  int ready;

  NCCLCHECKGOTO(getNewSockEntry(&sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&sock->sock, &conn->addr, NCCL_SOCKET_MAGIC, ncclSocketTypeRasNetwork, nullptr,
                               /*asyncFlag*/1, /*customRetry*/1), ret, fail);
  closeSocketOnFail = true;
  NCCLCHECKGOTO(ncclSocketConnect(&sock->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketReady(&sock->sock, &ready), ret, fail);

  NCCLCHECKGOTO(rasGetNewPollEntry(&sock->pfd), ret, fail);

  // We delay the initialization of sockIdx, connIdx and status until this point so that in case of failures
  // we don't need to clean them up.
  conn->sockIdx = sock-rasSockets;
  sock->connIdx = conn-rasConns;
  rasPfds[sock->pfd].fd = sock->sock.fd;

  // We ignore the possibly ready status of the socket at this point and consider it CONNECTING because
  // there are other things we want to do before sending the CONNINIT, such as adding the connection to
  // the network links, etc.
  sock->status = RAS_SOCK_CONNECTING;
  rasPfds[sock->pfd].events = (POLLIN | POLLOUT);
  if (sock->sock.state == ncclSocketStateConnecting)
    rasPfds[sock->pfd].fd = POLL_FD_IGNORE; // Don't poll on this socket before connect().

exit:
  conn->lastRetryTime = clockNano();
  // We deliberately ignore ret as this function will be retried later if needed.
  return;
fail:
  if (closeSocketOnFail)
    (void)ncclSocketClose(&sock->sock);
  goto exit;
}

// Sends an initial RAS message to the peer after connecting to it.
static ncclResult_t rasConnPrepare(struct rasConnection* conn) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_CONNINIT);

  // The first message the RAS threads exchange provides the listening address of the connecting thread
  // and the NCCL version to ensure that users aren't mixing things up.
  NCCLCHECK(rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_CONNINIT;
  msg->connInit.ncclVersion = NCCL_VERSION_CODE;
  memcpy(&msg->connInit.listeningAddr, &rasNetListeningSocket.addr, sizeof(msg->connInit.listeningAddr));
  msg->connInit.peersHash = rasPeersHash;
  msg->connInit.deadPeersHash = rasDeadPeersHash;
  // We don't update lastSent[Dead]PeersHash because we aren't actually sending the peers themselves here.

  rasConnEnqueueMsg(conn, msg, msgLen, /*front*/true);

  // We'll finish the initialization in rasMsgHandleConnInitAck, after the other side responds.
  return ncclSuccess;
}

// Searches through rasConns for a connection with a provided address.
int rasConnFind(const union ncclSocketAddress* addr) {
  // rasConns is not sorted (given the number of indices, it would be a massive hassle to keep it that way)
  // so binary search won't do...
  for (int i = 0; i < nRasConns; i++) {
    struct rasConnection* conn = rasConns+i;
    if (conn->inUse && memcmp(&conn->addr, addr, sizeof(conn->addr)) == 0)
      return i;
  }

  return -1;
}

// Handles any connection-related timeouts.  Many timeouts affect the underlying sockets and thus have been handled
// in the socket timeout handler earlier by terminating the problematic sockets.  If a socket connection doesn't
// exist or needs to be re-established (due to having just been terminated), we handle that here.
// This is also where we declare peers as dead, etc.
// Invoked from the main RAS event loop.
void rasConnsHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  for (int connIdx = 0; connIdx < nRasConns; connIdx++) {
    struct rasConnection* conn = rasConns+connIdx;

    if (!conn->inUse)
      continue;

    if (conn->sockIdx != -1) {
      struct rasSocket* sock = rasSockets+conn->sockIdx;
      bool sockTerminated = false;

      // Retry the socket connections that have been refused.
      if (sock->status == RAS_SOCK_CONNECTING && sock->sock.state == ncclSocketStateConnecting) {
        if (now - sock->lastSendTime > RAS_CONNECT_RETRY) {
          int ready;
          if (ncclSocketReady(&sock->sock, &ready) != ncclSuccess) {
            INFO(NCCL_RAS, "Unexpected error from ncclSocketReady; terminating the socket connection with %s",
                 ncclSocketToString(&sock->sock.addr, rasLine));
            rasSocketTerminate(sock, /*finalize*/true);
            // We will retry below in the same loop.
            sockTerminated = true;
          } else {
            // We update lastSendTime even if !ready because we need it up-to-date for timeout calculations.
            sock->lastSendTime = clockNano();
            if (!ready && sock->sock.state == ncclSocketStateConnecting)
              *nextWakeup = std::min(*nextWakeup, sock->lastSendTime+RAS_CONNECT_RETRY);
            else
              rasPfds[sock->pfd].fd = sock->sock.fd; // Enable the handling via the main loop.
          } // if (ncclSocketReady)
        } else {
          *nextWakeup = std::min(*nextWakeup, sock->lastSendTime+RAS_CONNECT_RETRY);
        }
      } // if (sock->status == RAS_SOCK_CONNECTING && sock->sock.state == ncclSocketStateConnecting)

      // For connections that have data to send but that we've been unable to send a message on for a while,
      // consider their sockets lost and terminate them.
      if (!sockTerminated && !ncclIntruQueueEmpty(&conn->sendQ) && sock->status == RAS_SOCK_READY) {
        if (now - std::max(sock->lastSendTime, ncclIntruQueueHead(&conn->sendQ)->enqueueTime) > RAS_STUCK_TIMEOUT) {
          INFO(NCCL_RAS, "RAS send stuck timeout error (%lds) on socket connection with %s",
               (now - std::max(sock->lastSendTime, ncclIntruQueueHead(&conn->sendQ)->enqueueTime)) /
               CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine));
          rasSocketTerminate(sock, /*finalize*/false, RAS_STUCK_TIMEOUT);
          // We will retry below in the same loop.
        } else {
          *nextWakeup = std::min(*nextWakeup, std::max(sock->lastSendTime,
                                                       ncclIntruQueueHead(&conn->sendQ)->enqueueTime)+RAS_STUCK_TIMEOUT);
        }
      } // if (!ncclIntruQueueEmpty(&conn->sendQ) && sock->status == RAS_SOCK_READY)
    } // if (conn->sockIdx != -1)

    // For connections that are being (re-)established, irrespective of whether there's a valid socket associated
    // with them (conn->startIdx != -1), we need to check if any connection-level timeout has expired.
    if (conn->startRetryTime) {
      // If we've been trying to open a connection for too long (60s), give up and mark the peer as dead
      // so that we don't try again.
      if (now - conn->startRetryTime > RAS_PEER_DEAD_TIMEOUT) {
        struct rasCollRequest bCast;
        INFO(NCCL_RAS, "RAS connect retry timeout (%lds) on socket connection with %s",
             (now-conn->startRetryTime)/CLOCK_UNITS_PER_SEC, ncclSocketToString(&conn->addr, rasLine));

        // Broadcast the info about a dead peer to everybody.  This will handle it locally as well, including
        // declaring the peer dead and terminating the connection.
        rasCollReqInit(&bCast);
        bCast.type = RAS_BC_DEADPEER;
        memcpy(&bCast.deadPeer.addr, &conn->addr, sizeof(bCast.deadPeer.addr));
        (void)rasNetSendCollReq(&bCast, rasCollDataLength(RAS_BC_DEADPEER));

        continue;
      } else {
        *nextWakeup = std::min(*nextWakeup, conn->startRetryTime+RAS_PEER_DEAD_TIMEOUT);
      }

      // RAS_STUCK_TIMEOUT has already been handled in the socket function (we'll pick it up later via
      // the conn->sockIdx == -1 test).

      // We print warnings after the same time as with keep-alive (5s), and we pessimistically immediately try
      // to establish fallback connections.
      if (now - conn->startRetryTime > RAS_CONNECT_WARN) {
        if (!conn->experiencingDelays) {
          INFO(NCCL_RAS, "RAS connect timeout warning (%lds) on socket connection with %s",
               (now-conn->startRetryTime) / CLOCK_UNITS_PER_SEC, ncclSocketToString(&conn->addr, rasLine));

          // See if the connection was meant to be a part of a RAS link and if so, try to initiate fallback
          // connection(s).  At this point, it's mostly just a precaution; we will continue trying to establish
          // the primary connection until RAS_PEER_DEAD_TIMEOUT expires.
          conn->experiencingDelays = true;
          (void)rasLinkAddFallback(&rasNextLink, connIdx);
          (void)rasLinkAddFallback(&rasPrevLink, connIdx);
          // rasConns may have been reallocated by the above calls.
          conn = rasConns+connIdx;

          // Stop collectives from waiting for a response over it.
          rasCollsPurgeConn(connIdx);
        } // if (!conn->experiencingDelays)
      } else {
        *nextWakeup = std::min(*nextWakeup, conn->startRetryTime+RAS_CONNECT_WARN);
      }

      // If a socket was terminated (or never opened, due to some error), try to open it now.
      // We retry once a second.
      if (conn->sockIdx == -1) {
        if (now - conn->lastRetryTime > RAS_CONNECT_RETRY) {
          INFO(NCCL_RAS, "RAS trying to reconnect with %s (experiencingDelays %d, startRetryTime %.2fs)",
               ncclSocketToString(&conn->addr, rasLine), conn->experiencingDelays,
               (conn->startRetryTime ? (now-conn->startRetryTime)/1e9 : 0.0));
          rasConnOpen(conn);
        }
        if (conn->sockIdx == -1)
          *nextWakeup = std::min(*nextWakeup, conn->lastRetryTime+RAS_CONNECT_RETRY);
      }
    } // if (conn->startRetryTime)
  } // for (connIdx)
}

// Checks if we have a connection to a given peer and if so, terminates it.  The connection is removed from the
// RAS links, though fallbacks are initiated if necessary.  Typically called just before declaring a peer dead.
void rasConnDisconnect(const union ncclSocketAddress* addr) {
  int connIdx = rasConnFind(addr);
  if (connIdx != -1) {
    (void)rasLinkAddFallback(&rasNextLink, connIdx);
    (void)rasLinkAddFallback(&rasPrevLink, connIdx);
    rasLinkDropConn(&rasNextLink, connIdx);
    rasLinkDropConn(&rasPrevLink, connIdx);

    rasConnTerminate(rasConns+connIdx);
  }
}

// Terminates a connection and frees the rasConns entry.
static void rasConnTerminate(struct rasConnection* conn) {
  int connIdx = conn - rasConns;

  // Make sure there are no lingering rasSockets pointing to it.
  for (int i = 0; i < nRasSockets; i++) {
    struct rasSocket* sock = rasSockets+i;
    if (sock->status != RAS_SOCK_CLOSED && sock->connIdx == connIdx)
      rasSocketTerminate(sock, /*finalize*/true);
  }

  // Also check any ongoing collectives.
  rasCollsPurgeConn(connIdx);

  while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&conn->sendQ)) {
    free(meta);
  }

  INFO(NCCL_RAS, "RAS terminating a connection with %s", ncclSocketToString(&conn->addr, rasLine));

  conn->inUse = false;
  conn->sockIdx = -1; // Should be that way already, but just to be extra sure...
}


///////////////////////////////////////////
// Functions related to the RAS sockets. //
///////////////////////////////////////////

// Accepts a new RAS network socket connection.  The socket is not usable until after the handshake, as a
// corresponding rasConnection can't be established without knowing the peer's address.
ncclResult_t rasNetAcceptNewSocket() {
  ncclResult_t ret = ncclSuccess;
  struct rasSocket* sock;
  int ready;
  bool socketInitialized = false;
  NCCLCHECKGOTO(getNewSockEntry(&sock), ret, fail);

  NCCLCHECKGOTO(ncclSocketInit(&sock->sock, nullptr, NCCL_SOCKET_MAGIC, ncclSocketTypeRasNetwork, nullptr,
                               /*asyncFlag*/1), ret, fail);
  socketInitialized = true;
  NCCLCHECKGOTO(ncclSocketAccept(&sock->sock, &rasNetListeningSocket), ret, fail);
  NCCLCHECKGOTO(ncclSocketReady(&sock->sock, &ready), ret, fail);

  if (sock->sock.fd != -1) {
    NCCLCHECKGOTO(rasGetNewPollEntry(&sock->pfd), ret, fail);
    rasPfds[sock->pfd].fd = sock->sock.fd;
    rasPfds[sock->pfd].events = POLLIN; // Initially we'll just wait for a handshake from the other side.  This also
                                        // helps the code tell the sides apart.
    sock->status = RAS_SOCK_CONNECTING;

    INFO(NCCL_RAS, "RAS new incoming socket connection from %s", ncclSocketToString(&sock->sock.addr, rasLine));
  }

exit:
  return ret;
fail:
  if (socketInitialized)
    NCCLCHECK(ncclSocketClose(&sock->sock));
  goto exit;
}

// Returns the index of the first available entry in the rasConns array, enlarging the array if necessary.
static ncclResult_t getNewSockEntry(struct rasSocket** pSock) {
  struct rasSocket* sock;
  int i;
  for (i = 0; i < nRasSockets; i++)
    if (rasSockets[i].status == RAS_SOCK_CLOSED)
      break;
  if (i == nRasSockets) {
    NCCLCHECK(ncclRealloc(&rasSockets, nRasSockets, nRasSockets+RAS_INCREMENT));
    nRasSockets += RAS_INCREMENT;
  }

  sock = rasSockets+i;
  memset(sock, '\0', sizeof(*sock));
  sock->pfd = -1;
  sock->connIdx = -1;
  sock->createTime = sock->lastSendTime = sock->lastRecvTime = clockNano();

  *pSock = sock;
  return ncclSuccess;
}

// Invoked from the main RAS event loop to handle RAS socket timeouts.
void rasSocksHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  for (int sockIdx = 0; sockIdx < nRasSockets; sockIdx++) {
    struct rasSocket* sock = rasSockets+sockIdx;

    if (sock->status == RAS_SOCK_CLOSED)
      continue;

    // For socket connections that are still being established, give up on the ones that take too long to initialize.
    if (sock->status == RAS_SOCK_CONNECTING || sock->status == RAS_SOCK_HANDSHAKE) {
      if (now - sock->createTime > RAS_STUCK_TIMEOUT) {
        if (sock->connIdx == -1) {
          INFO(NCCL_RAS, "RAS init timeout error (%lds) on incoming socket connection from %s",
               (now-sock->createTime)/CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine));
        } else {
          struct rasConnection* conn = rasConns+sock->connIdx;
          INFO(NCCL_RAS, "RAS init timeout error (%lds) on socket connection with %s "
               "(experiencingDelays %d, startRetryTime %.2fs, socket status %d)",
               (now-sock->createTime)/CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine),
               conn->experiencingDelays, (conn->startRetryTime ? (now-conn->startRetryTime)/1e9 : 0.0),
               sock->status);
        }
        rasSocketTerminate(sock, /*finalize*/true);
        // We may retry later.
        continue;
      } else {
        *nextWakeup = std::min(*nextWakeup, sock->createTime+RAS_STUCK_TIMEOUT);
      }
    } // if (sock->status == RAS_SOCK_CONNECTING || sock->status == RAS_SOCK_HANDSHAKE)

    // For sockets that are being terminated, force finalization of the ones that haven't made progress in too long.
    if (sock->status == RAS_SOCK_TERMINATING) {
      if (now - std::max(sock->lastSendTime, sock->lastRecvTime) > RAS_STUCK_TIMEOUT) {
        INFO(NCCL_RAS, "RAS termination stuck timeout error (%lds) on socket connection with %s",
             (now-std::max(sock->lastSendTime, sock->lastRecvTime)) / CLOCK_UNITS_PER_SEC,
             ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock, /*finalize*/true);
        // This socket is presumably already being re-established, if needed.
        continue;
      } else {
        *nextWakeup = std::min(*nextWakeup, std::max(sock->lastSendTime, sock->lastRecvTime)+RAS_STUCK_TIMEOUT);
      }
    } // if (sock->status == RAS_SOCK_TERMINATING)

    // Terminate sockets that haven't been used in a good while.  In principle this shouldn't trigger for anything
    // important due to shorter timeouts on RAS network connections, but in case of weird situations like process
    // suspend, rasSocketTerminate will do additional checking.
    if (sock->status == RAS_SOCK_READY) {
      if (now - std::max(sock->lastSendTime, sock->lastRecvTime) > RAS_IDLE_TIMEOUT) {
        INFO(NCCL_RAS, "RAS idle timeout (%lds) on socket connection with %s",
             (now - std::max(sock->lastSendTime, sock->lastRecvTime)) / CLOCK_UNITS_PER_SEC,
             ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock, /*finalize*/false, /*startRetryOffset*/0, /*retry*/false);
        continue;
        // The RAS network timeout handler will terminate the conn it was associated with, if any.
      } else {
        *nextWakeup = std::min(*nextWakeup, std::max(sock->lastSendTime, sock->lastRecvTime)+RAS_IDLE_TIMEOUT);
      }
    } // if (sock->status == RAS_SOCK_READY)
  } // for (sockIdx)
}

// Handles the termination of a RAS socket.
// We try to do it in stages for established sockets (in READY state).  We shut down just the sending side
// for them and change their state to TERMINATING, so that we can still receive data that may be in the buffers.
// Once we get an EOF when receiving data, we finalize the termination.
// For not fully established sockets, we can terminate immediately as there's no useful data to extract.
void rasSocketTerminate(struct rasSocket* sock, bool finalize, uint64_t startRetryOffset, bool retry) {
  assert(sock->status != RAS_SOCK_CLOSED);
  if (sock->connIdx != -1) {
    struct rasConnection* conn = rasConns+sock->connIdx;
    // If the sockIdx of the connection points back to us, it means that we are the current socket of this
    // connection, so we have additional work to do before we can terminate it.
    if (conn->sockIdx == sock-rasSockets) {
      // Reset it to indicate there's no valid socket associated with that connection anymore.
      conn->sockIdx = -1;

      // Don't attempt to retry on sockets that have been unused for so long that the remote peer probably
      // deliberately closed them.  Make an exception for sockets that are part of the RAS network links.
      if ((retry &&
           clockNano() - std::max(sock->lastSendTime, sock->lastRecvTime) < RAS_IDLE_TIMEOUT - RAS_IDLE_GRACE_PERIOD) ||
          rasLinkFindConn(&rasNextLink, sock->connIdx) != -1 || rasLinkFindConn(&rasPrevLink, sock->connIdx) != -1) {
        // For connections that were fine until now, the connection-level timeout starts at termination, and possibly
        // even earlier, depending on what event trigerred the termination -- if it was another timeout expiring, then
        // we need to include that timeout as well.
        if (conn->startRetryTime == 0) {
          conn->startRetryTime = conn->lastRetryTime = clockNano() - startRetryOffset;
        }

        // We also filter through the sendQ, eliminating any messages that won't need to be sent when the socket
        // connection is re-established (that's essentially the server init and keep-alives).
        // As ncclIntruQueue can't be iterated, we transfer the content in bulk to a temporary and then filter the
        // messages as we move them back one-by-one.
        struct ncclIntruQueue<struct rasMsgMeta, &rasMsgMeta::next> sendQTmp;
        ncclIntruQueueConstruct(&sendQTmp);
        ncclIntruQueueTransfer(&sendQTmp, &conn->sendQ);
        while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&sendQTmp)) {
          if (meta->msg.type != RAS_MSG_CONNINIT && meta->msg.type != RAS_MSG_CONNINITACK &&
              meta->msg.type != RAS_MSG_KEEPALIVE) {
            if (meta->offset != 0) {
              // Reset the progress of any partially-sent messages (they will need to be resent from the beginning;
              // in principle that could apply to the first message only).
              meta->offset = 0;
            }
            ncclIntruQueueEnqueue(&conn->sendQ, meta);
          } else { // RAS_MSG_CONNINIT || RAS_MSG_CONNINITACK || RAS_MSG_KEEPALIVE
            free(meta);
          }
        } // while (meta)
      } // if (retry)

      // Stop collectives from waiting for a response over this connection.
      rasCollsPurgeConn(sock->connIdx);
    } // if (conn->sockIdx == sock-rasSockets)
  } // if (sock->connIdx != -1)

  if (sock->status != RAS_SOCK_CONNECTING && sock->connIdx != -1 && !finalize && (rasPfds[sock->pfd].events & POLLIN)) {
    if (sock->status != RAS_SOCK_TERMINATING) {
      // The receiving side is still open -- close just the sending side.
      (void)ncclSocketShutdown(&sock->sock, SHUT_WR);
      rasPfds[sock->pfd].events &= ~POLLOUT; // Nothing more to send.
      // The timeout for this socket starts ticking now...
      sock->lastSendTime = clockNano();
      sock->status = RAS_SOCK_TERMINATING;
    }
    // Else it must be in RAS_SOCK_TERMINATING state already -- in that case we do nothing here and instead
    // we wait for an EOF on the receiving side or for a timeout.
  } else {
    // Either the caller requested finalization or we cannot receive on it.
    (void)ncclSocketClose(&sock->sock);
    sock->status = RAS_SOCK_CLOSED;
    rasPfds[sock->pfd].fd = -1;
    rasPfds[sock->pfd].events = rasPfds[sock->pfd].revents = 0;
    sock->pfd = sock->connIdx = -1;
    sock->recvOffset = sock->recvLength = 0;
    free(sock->recvMsg);
    sock->recvMsg = nullptr;
  }
}

// Handles a ready socket FD from the main event loop.
void rasSockEventLoop(int sockIdx, int pollIdx) {
  struct rasSocket* sock = rasSockets+sockIdx;

  if (sock->status == RAS_SOCK_CONNECTING) {
    int ready;
    // Socket is not yet fully established. Continue the OS or NCCL-level handshake.
    if (ncclSocketReady(&sock->sock, &ready) != ncclSuccess) {
      INFO(NCCL_RAS, "RAS unexpected error from ncclSocketReady; terminating the socket connection with %s",
           ncclSocketToString(&sock->sock.addr, rasLine));
      rasSocketTerminate(sock);
      // We may retry further down.
    } else {
      if (ready) {
        // We can tell the connect-side based on what events is set to.
        bool connectSide = (rasPfds[pollIdx].events & POLLOUT);
        (connectSide ? sock->lastSendTime : sock->lastRecvTime) = clockNano();
        sock->status = RAS_SOCK_HANDSHAKE;
        if (connectSide) {
          assert(sock->connIdx != -1);
          if (rasConns[sock->connIdx].sockIdx == sockIdx) {
            if (rasConnPrepare(rasConns+sock->connIdx) != ncclSuccess) {
              INFO(NCCL_RAS, "RAS unexpected error from rasConnPrepare; terminating the socket connection with %s",
                   ncclSocketToString(&sock->sock.addr, rasLine));
              rasSocketTerminate(sock);
              // We may retry further down.
            }
          } else {
            // The connection this socket is associated with no longer considers it to be the current one.
            // This could possibly happen due to a race condition.  Simply terminate it.
            INFO(NCCL_RAS, "RAS connected with %s via a socket that's no longer current!",
                 ncclSocketToString(&sock->sock.addr, rasLine));
            rasSocketTerminate(sock);
          }
        } // if (connectSide)
      } else { // !ready
        if (sock->sock.state == ncclSocketStateConnecting)
          rasPfds[sock->pfd].fd = POLL_FD_IGNORE; // Don't poll on this socket before connect().
      }
    } // if (ncclSocketReady)
  } else { // RAS_SOCK_HANDSHAKE || RAS_SOCK_READY || RAS_SOCK_TERMINATING.
    // The extra test for TERMINATING is there to take care of a race when the handling of one socket
    // results in another socket being terminated, but one that already has revents waiting from poll.
    if (sock->status != RAS_SOCK_TERMINATING && (rasPfds[pollIdx].revents & POLLOUT)) {
      int closed = 0;
      bool allSent = false;
      assert(sock->connIdx != -1);
      struct rasConnection* conn = rasConns+sock->connIdx;
      assert(conn->sockIdx == sockIdx);
      if (rasConnSendMsg(conn, &closed, &allSent) != ncclSuccess) {
        INFO(NCCL_RAS, "RAS unexpected error from rasConnSendMsg; terminating the socket connection with %s",
             ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock);
        // We may retry further down.
      } else if (closed) {
        INFO(NCCL_RAS, "RAS socket connection with %s closed by peer on send; terminating it",
             ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock);
        // We may retry further down.
      } else {
        sock->lastSendTime = clockNano();
        if (allSent)
          rasPfds[sock->pfd].events &= ~POLLOUT; // Nothing more to send for now.
      }
    }
    if (rasPfds[pollIdx].revents & POLLIN) {
      struct rasMsg* msg;
      do {
        int closed = 0;
        msg = nullptr;
        if (rasMsgRecv(sock, &msg, &closed) != ncclSuccess) {
          INFO(NCCL_RAS, "RAS unexpected error from rasMsgRecv; terminating the socket connection with %s",
               ncclSocketToString(&sock->sock.addr, rasLine));
          rasSocketTerminate(sock, /*finalize*/true);
          // We may retry further down.
        } else if (closed) {
          const char* socketType;
          if (sock->connIdx == -1)
            socketType = "incoming";
          else if (rasConns[sock->connIdx].sockIdx != sockIdx)
            socketType = "old";
          else if (sock->status == RAS_SOCK_HANDSHAKE)
            socketType = "new";
          else
            socketType = "current";
          INFO(NCCL_RAS, "RAS %s socket connection with %s closed by peer on receive; terminating it",
               socketType, ncclSocketToString(&sock->sock.addr, rasLine));
          rasSocketTerminate(sock, /*finalize*/true);
          // We may retry further down.
        } else {
          sock->lastRecvTime = clockNano();
          if (msg) {
            (void)rasMsgHandle(msg, sock);
            free(msg);
            // Message handlers can terminate a socket in certain cases; we need to check for
            // that here so that we don't try to receive from a closed socket.
            // No handlers are currently believed to create new sockets but better to be safe than sorry
            // and re-init the sock variable.
            sock = rasSockets+sockIdx;
            if (sock->status == RAS_SOCK_CLOSED)
              break;
          }
          if (sock->connIdx != -1) {
            struct rasConnection* conn = rasConns+sock->connIdx;
            if (conn->sockIdx == sockIdx && (conn->startRetryTime || conn->experiencingDelays))
              rasConnResume(conn);
          }
        }
      } while (msg);
    } // if (POLLIN)
  } // RAS_SOCK_HANDSHAKE || RAS_SOCK_READY || RAS_SOCK_TERMINATING
}


////////////////////////////////////////////////////////////////
// Functions related to the handling of RAS network timeouts. //
////////////////////////////////////////////////////////////////

// Invoked from the main RAS event loop to handle RAS network timeouts.
void rasNetHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  // A connection can belong to multiple links but, when it comes to various timeouts, we want to handle each
  // connection just once.  We solve that with a simple flag within a connection.  This also allows us to distinguish
  // connections that are part of a link from those that are not.
  for (int connIdx = 0; connIdx < nRasConns; connIdx++)
    rasConns[connIdx].linkFlag = false;

  (void)rasLinkHandleNetTimeouts(&rasNextLink, now, nextWakeup);
  (void)rasLinkHandleNetTimeouts(&rasPrevLink, now, nextWakeup);

  for (int connIdx = 0; connIdx < nRasConns; connIdx++) {
    struct rasConnection* conn = rasConns+connIdx;
    if (conn->inUse && !conn->linkFlag) {
      // The connection is not part of any link.  Check if it should be terminated.
      if (conn->sockIdx == -1 && ncclIntruQueueEmpty(&conn->sendQ)) {
        rasConnTerminate(conn);
        continue;
      }
    }
  }
}

// Checks for and handles timeouts at the link level; primarily the keep-alives for link connections.
static ncclResult_t rasLinkHandleNetTimeouts(struct rasLink* link, int64_t now, int64_t* nextWakeup) {
  for (int i = 0; i < link->nConns; i++) {
    struct rasLinkConn* linkConn = link->conns+i;
    if (linkConn->connIdx != -1) {
      if (!rasConns[linkConn->connIdx].linkFlag) {
        rasConnHandleNetTimeouts(linkConn->connIdx, now, nextWakeup);
        // rasConns may have been reallocated by the above call, which is why we don't have a conn variable here.
        // For the same reason we re-init linkConn.
        linkConn = link->conns+i;
        rasConns[linkConn->connIdx].linkFlag = true;
      }
    } else if (i == 0 && link->lastUpdatePeersTime != 0) {
      // This triggers when rasLinkReinitConns didn't create the primary connection because we have a higher address
      // than the peer.  If that peer fails to initiate within RAS_CONNECT_WARN, we need to take action.
      if (now - link->lastUpdatePeersTime > RAS_CONNECT_WARN) {
        INFO(NCCL_RAS, "RAS peer connect timeout warning (%lds) on socket connection from %s",
             (now-link->lastUpdatePeersTime) / CLOCK_UNITS_PER_SEC,
             ncclSocketToString(&rasPeers[linkConn->peerIdx].addr, rasLine));
        NCCLCHECK(rasConnCreate(&rasPeers[linkConn->peerIdx].addr, &linkConn->connIdx));
        if (linkConn->connIdx != -1) {
          rasConns[linkConn->connIdx].linkFlag = true;
        }
        // We used to connect to the first fallback but I think trying to connect to the calculated primary first
        // in this case is more intuitive.
        //(void)rasLinkTryFallback(link, -1);
        link->lastUpdatePeersTime = 0;
      } else {
        *nextWakeup = std::min(*nextWakeup, link->lastUpdatePeersTime+RAS_CONNECT_WARN);
      }
    } // if (i == 0 && link->lastUpdatePeerTime != 0)
  } // for (i)

  return ncclSuccess;
}

// Handles the sending of keep-alive messages and related timeouts for connections that are part of the RAS links.
static void rasConnHandleNetTimeouts(int connIdx, int64_t now, int64_t* nextWakeup) {
  struct rasConnection* conn = rasConns+connIdx;
  if (conn->sockIdx != -1) {
    struct rasSocket* sock = rasSockets+conn->sockIdx;

    if (sock->status == RAS_SOCK_READY) {
      // Send a regular keep-alive message if we haven't sent anything in a while and we don't have anything queued.
      if (ncclIntruQueueEmpty(&conn->sendQ)) {
        if (now - sock->lastSendTime > RAS_KEEPALIVE_INTERVAL) {
          rasConnSendKeepAlive(conn);
        } else {
          *nextWakeup = std::min(*nextWakeup, sock->lastSendTime+RAS_KEEPALIVE_INTERVAL);
        }
      }

      // For short timeouts print a warning but also pessimistically immediately try to establish fallback connections.
      if (now - sock->lastRecvTime > RAS_KEEPALIVE_TIMEOUT_WARN) {
        if (!conn->experiencingDelays) {
          INFO(NCCL_RAS, "RAS keep-alive timeout warning (%lds) on socket connection with %s",
               (now-sock->lastRecvTime) / CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine));

          // At this point, it's mostly just a precaution; we will continue with the primary connection until
          // RAS_PEER_DEAD_TIMEOUT expires.
          conn->experiencingDelays = true;
          (void)rasLinkAddFallback(&rasNextLink, connIdx);
          (void)rasLinkAddFallback(&rasPrevLink, connIdx);
          // rasConns and rasSockets may have been reallocated by the above calls.
          conn = rasConns+connIdx;
          sock = rasSockets+conn->sockIdx;

          // Stop collectives from waiting for a response over it.
          rasCollsPurgeConn(connIdx);
        }
      } else {
        *nextWakeup = std::min(*nextWakeup, sock->lastRecvTime+RAS_KEEPALIVE_TIMEOUT_WARN);
      }

      // For long timeouts we need to act.
      if (now - sock->lastRecvTime > RAS_KEEPALIVE_TIMEOUT_ERROR) {
        INFO(NCCL_RAS, "RAS keep-alive timeout error (%lds) on socket connection with %s",
             (now-sock->lastRecvTime) / CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock, /*finalize*/true, RAS_KEEPALIVE_TIMEOUT_ERROR);
        *nextWakeup = now; // Retry will be in the next iteration of the main loop so ensure we don't wait.
      } else {
        *nextWakeup = std::min(*nextWakeup, sock->lastRecvTime+RAS_KEEPALIVE_TIMEOUT_ERROR);
      }
    } // if (sock->status == RAS_SOCK_READY)
  } // if (conn->sockIdx != -1)
}

// Sends a keep-alive message to a peer on the RAS network.
static void rasConnSendKeepAlive(struct rasConnection* conn, bool nack) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  if (rasMsgAlloc(&msg, msgLen) == ncclSuccess) {
    int linkIdx;
    msg->type = RAS_MSG_KEEPALIVE;
    msg->keepAlive.peersHash = rasPeersHash;
    msg->keepAlive.deadPeersHash = rasDeadPeersHash;
    msg->keepAlive.nack = (nack ? 1 : 0);

    linkIdx = rasLinkFindConn(&rasNextLink, conn-rasConns);
    if (linkIdx != -1 && !rasNextLink.conns[linkIdx].external)
      msg->keepAlive.linkMask |= 2; // Our rasNextLink should be the peer's rasPrevLink.
    linkIdx = rasLinkFindConn(&rasPrevLink, conn-rasConns);
    if (linkIdx != -1 && !rasPrevLink.conns[linkIdx].external)
      msg->keepAlive.linkMask |= 1; // Our rasPrevLink should be the peer's rasNextLink.

    (void)clock_gettime(CLOCK_REALTIME, &msg->keepAlive.realTime);

    rasConnEnqueueMsg(conn, msg, msgLen);
  }
}

// Handles incoming keep-alive messages.
ncclResult_t rasMsgHandleKeepAlive(const struct rasMsg* msg, struct rasSocket* sock) {
  struct timespec currentTime;
  int64_t travelTime;
  int peerIdx;

  assert(sock->connIdx != -1);
  struct rasConnection* conn = rasConns+sock->connIdx;
  SYSCHECK(clock_gettime(CLOCK_REALTIME, &currentTime), "clock_gettime");
  travelTime = (currentTime.tv_sec-msg->keepAlive.realTime.tv_sec)*1000*1000*1000 +
    (currentTime.tv_nsec-msg->keepAlive.realTime.tv_nsec);

  if (msg->keepAlive.peersHash != conn->lastRecvPeersHash) {
    conn->lastRecvPeersHash = msg->keepAlive.peersHash;
  }
  if (msg->keepAlive.deadPeersHash != conn->lastRecvDeadPeersHash) {
    conn->lastRecvDeadPeersHash = msg->keepAlive.deadPeersHash;
  }

  // Make sure that the connection is part of the appropriate links forming the RAS network.  In particular, this
  // will add any externally-requested connections to the appropriate links (or remove existing ones, if no longer
  // needed).
  peerIdx = rasPeerFind(&conn->addr);
  // Note: it's possible for peerIdx to be -1 at this point if, due to races, the keepAlive arrives before
  // the peers update.
  (void)rasLinkUpdateConn(&rasNextLink, (msg->keepAlive.linkMask & 1) ? sock->connIdx : -1, peerIdx, /*external*/true);
  (void)rasLinkUpdateConn(&rasPrevLink, (msg->keepAlive.linkMask & 2) ? sock->connIdx : -1, peerIdx, /*external*/true);

  // If the keep-alive message is from a peer that doesn't actually need this connection (i.e., for that peer the
  // connection is just an external fallback), we should check if *we* still need it.  It might be that we don't,
  // and because we stopped sending the keep-alives, our peer doesn't know about it.  rasLinkUpdateConn calls above
  // will have wiped any external fallbacks, so anything that remains must be needed.
  if (!msg->keepAlive.nack && msg->keepAlive.linkMask == 0) {
    if (rasLinkFindConn(&rasNextLink, sock->connIdx) == -1 && rasLinkFindConn(&rasPrevLink, sock->connIdx) == -1) {
      // We don't need this connection either.  Notify the peer about it.  To avoid an infinite loop, we set the
      // special nack flag in the message to distinguish it from regular keep-alives.
      rasConnSendKeepAlive(conn, /*nack*/true);
    }
  }

  if (conn->travelTimeMin > travelTime)
    conn->travelTimeMin = travelTime;
  if (conn->travelTimeMax < travelTime)
    conn->travelTimeMax = travelTime;
  conn->travelTimeSum += travelTime;
  conn->travelTimeCount++;

  if (msg->keepAlive.peersHash != rasPeersHash || msg->keepAlive.deadPeersHash != rasDeadPeersHash) {
    // This could happen due to a short-lived race condition between the peers propagation
    // process and the periodic keep-alive messages (perhaps we'll see it regularly at scale?).
    // Just in case there's some unforeseen problem with the peers propagation though, exchange with the
    // remote to get everybody in sync.
    INFO(NCCL_RAS, "RAS keepAlive hash mismatch from %s (peersHash 0x%lx, deadPeersHash 0x%lx)",
         ncclSocketToString(&sock->sock.addr, rasLine), msg->keepAlive.peersHash, msg->keepAlive.deadPeersHash);
    INFO(NCCL_RAS, "RAS my peersHash 0x%lx, deadPeersHash 0x%lx", rasPeersHash, rasDeadPeersHash);
    NCCLCHECK(rasConnSendPeersUpdate(conn, rasPeers, nRasPeers));
  }
  return ncclSuccess;
}


///////////////////////////////////////////////////////////////////////////////
// Functions related to the RAS links and recovery from connection failures. //
///////////////////////////////////////////////////////////////////////////////

// Checks if the connection (that we just detected some problem with) is part of the RAS link and if so,
// tries to initiate a(nother) fallback connection if needed.
// External connections are generally ignored by this whole process: in particular, we don't add fallbacks for
// timing out external connections.  However, we will use an active external connection if it would be a better
// option than whatever we can come up with.
static ncclResult_t rasLinkAddFallback(struct rasLink* link, int connIdx) {
  int peerIdx = -1;
  int linkIdx = -1;
  int firstExtLinkIdx = -1;
  int newPeerIdx;

  // First check if the connection is part of this link.  In the process also check if any of the link's connections
  // might be active -- if so, there's no need to initiate any more fallbacks and we can bail out.
  for (int i = 0; i < link->nConns; i++) {
    struct rasLinkConn* linkConn = link->conns+i;

    if (linkConn->peerIdx == -1) {
      // Such elements are always at the very end of the array and we can't use them so we can just as well break.
      break;
    }

    // Check for any other connection that might be a viable fallback (basically, anything that is not experiencing
    // delays).
    if (linkConn->connIdx != -1 && linkConn->connIdx != connIdx) {
      struct rasConnection* conn = rasConns+linkConn->connIdx;
      if (!conn->experiencingDelays) {
        if (!linkConn->external)
          goto exit; // We don't need to do anything if there's a non-external connection.
        else if (linkConn->peerIdx != -1) {
          // Record the location of the first potentially viable external connection in the chain; we may prefer it
          // over anything we can come up with.
          if (firstExtLinkIdx == -1)
            firstExtLinkIdx = i;
          if (linkIdx != -1)
            break; // Break out of the loop if we already have all the data we might need.
        } // linkConn->external && linkConn->peerIdx != -1
      } // if (!conn->experiencingDelays)
    } // if (linkConn->connIdx != -1)

    if (linkConn->connIdx == connIdx) {
      if (linkConn->external)
        goto exit; // We don't add fallbacks for external connections...
      peerIdx = linkConn->peerIdx;
      linkIdx = i;
      // We are not breaking out of the loop here because we want to check for active connections on *all* potentially
      // viable elements (in particular, there could be some external ones beyond this one).
    }
  }

  if (linkIdx == -1)
    goto exit;

  // We found an existing element so the connection is part of the link.  No existing non-external connections of this
  // link are active, so a fallback is needed.
  assert(peerIdx != -1);
  newPeerIdx = rasLinkCalculatePeer(link, peerIdx, /*isFallback*/linkIdx > 0);
  // In principle we want to add (at most) one fallback.  However, if the found fallback connection already exists
  // and is also experiencing delays, we need to keep iterating.
  while (newPeerIdx != -1) {
    int newConnIdx = rasConnFind(&rasPeers[newPeerIdx].addr);
    // If we previously found a potential external fallback connection, check if it's better than what we just found.
    if (firstExtLinkIdx != -1) {
      linkIdx = -1;
      // Calculate the index that the newly found fallback would have (pretend mode).
      NCCLCHECK(rasLinkUpdateConn(link, newConnIdx, newPeerIdx, /*external*/false, /*insert*/true, /*pretend*/true,
                                  &linkIdx));
      assert(linkIdx != -1);
      if (firstExtLinkIdx < linkIdx) {
        // The external connection *is* better -- use it as a fallback instead and be done.
        link->conns[firstExtLinkIdx].external = false;
        goto exit;
      }
    }
    NCCLCHECK(rasLinkUpdateConn(link, newConnIdx, newPeerIdx, /*external*/false, /*insert*/true, /*pretend*/false,
                                &linkIdx));
    if (firstExtLinkIdx != -1 && linkIdx <= firstExtLinkIdx)
      firstExtLinkIdx++; // Adjust if we inserted a new conn at a lower index.

    INFO(NCCL_RAS, "RAS link %d: %s fallback connection %d with %s",
         link->direction, (newConnIdx == -1 ? "opening new" : "calculated existing"),
         linkIdx, ncclSocketToString(&rasPeers[newPeerIdx].addr, rasLine));
    // Note that we don't follow here our convention of "lower address is the one establishing connections" --
    // that convention is for optimizing regular operations, but we don't want to take chances during fault
    // recovery. It may temporarily result in duplicate connections, but we have a mechanism to deal with those.
    if (newConnIdx == -1)
      NCCLCHECK(rasConnCreate(&rasPeers[newPeerIdx].addr, &link->conns[linkIdx].connIdx));

    struct rasConnection* conn = rasConns+link->conns[linkIdx].connIdx;
    // If the fallback connection is also experiencing delays, we need to keep trying.
    if (!conn->experiencingDelays)
      break;
    INFO(NCCL_RAS, "RAS connection experiencingDelays %d, startRetryTime %.2fs, socket status %d",
         conn->experiencingDelays, (conn->startRetryTime ? (clockNano()-conn->startRetryTime)/1e9 : 0.0),
         (conn->sockIdx == -1 ? -1 : rasSockets[conn->sockIdx].status));

    newPeerIdx = rasLinkCalculatePeer(link, newPeerIdx, /*isFallback*/true);
  }
  if (newPeerIdx == -1)
      INFO(NCCL_RAS, "RAS link %d: no more fallbacks to add (nConns %d)", link->direction, link->nConns);
exit:
  return ncclSuccess;
}

// Invoked when we receive a message over a connection that was just activated or was experiencing delays.
// Cleans up the fallbacks, timers, etc, as appropriate.
static void rasConnResume(struct rasConnection* conn) {
  if (conn->sockIdx != -1 && rasSockets[conn->sockIdx].status == RAS_SOCK_READY) {
    INFO(NCCL_RAS, "RAS %s connection with %s (sendQ %sempty, experiencingDelays %d, startRetryTime %.2fs)",
         (conn->experiencingDelays && conn->startRetryTime == 0 ? "recovered" : "established"),
         ncclSocketToString(&conn->addr, rasLine), (ncclIntruQueueEmpty(&conn->sendQ) ? "" : "not "),
         conn->experiencingDelays, (conn->startRetryTime ? (clockNano()-conn->startRetryTime)/1e9 : 0.0));

    conn->experiencingDelays = false;

    conn->startRetryTime = conn->lastRetryTime = 0;

    rasLinkSanitizeFallbacks(&rasNextLink);
    rasLinkSanitizeFallbacks(&rasPrevLink);

    if (!ncclIntruQueueEmpty(&conn->sendQ))
      rasPfds[rasSockets[conn->sockIdx].pfd].events |= POLLOUT;
  }
}

// Checks if the primary connection is fully established and if so, purges the fallbacks (as they are no longer needed).
static void rasLinkSanitizeFallbacks(struct rasLink* link) {
  if (link->nConns > 0 && link->conns[0].connIdx != -1) {
    struct rasConnection* conn = rasConns+link->conns[0].connIdx;
    if (conn->sockIdx != -1 && rasSockets[conn->sockIdx].status == RAS_SOCK_READY && !conn->experiencingDelays) {
      // We have a good primary.  Simply drop all the fallbacks (the external ones will get recreated via the
      // keepAlive messages).
      for (int i = 1; i < link->nConns; i++) {
        INFO(NCCL_RAS, "RAS link %d: dropping %sfallback connection %d with %s",
             link->direction, (link->conns[i].external ? "external " : ""), i,
             ncclSocketToString(&rasConns[link->conns[i].connIdx].addr, rasLine));
      }
      link->nConns = 1;
      link->lastUpdatePeersTime = 0;
    }
  }
}

// Attempt to drop a connection from a link.
static void rasLinkDropConn(struct rasLink* link, int connIdx, int linkIdx) {
  if (linkIdx == -1)
    linkIdx = rasLinkFindConn(link, connIdx);
  if (linkIdx != -1) {
    if (linkIdx == 0) {
      INFO(NCCL_RAS, "RAS link %d: dropping primary connection with %s",
           link->direction, ncclSocketToString(&rasConns[connIdx].addr, rasLine));
    } else {
      INFO(NCCL_RAS, "RAS link %d: dropping %sfallback connection %d with %s",
           link->direction, (link->conns[linkIdx].external ? "external " : ""), linkIdx,
           ncclSocketToString(&rasConns[connIdx].addr, rasLine));
    }
    memmove(link->conns+linkIdx, link->conns+linkIdx+1, (link->nConns-(linkIdx+1))*sizeof(*link->conns));
    if (link->nConns > 1)
      link->nConns--;
    else {
      link->conns[0].peerIdx = link->conns[0].connIdx = -1;
    }

    if (linkIdx == 0) {
      // First ensure that the conn becoming the primary is not marked as external (we don't want to lose it if
      // the remote peer loses interest in it).
      link->conns[0].external = false;
      if (link->conns[0].connIdx != -1) {
        INFO(NCCL_RAS, "RAS link %d: former fallback connection 1 with %s is the new primary",
             link->direction, ncclSocketToString(&rasConns[link->conns[0].connIdx].addr, rasLine));
      }
      rasLinkSanitizeFallbacks(link);
    }
  }
}

// Checks if a given connection is a member of this link and if so, returns its entry index.
// Returns -1 if connection not found.
static int rasLinkFindConn(const struct rasLink* link, int connIdx) {
  for (int i = 0; i < link->nConns; i++) {
    if (link->conns[i].connIdx == connIdx)
      return i;
  }
  return -1;
}

// Note: the behavior of this function has become super-complex and so it should be considered for refactoring.
// Searches for and updates an entry in a RAS network link.  The conns array is de-facto sorted by peerIdx: it is
// ordered by preference, though peerIdx values can wrap around (given the ring/torus topology) and they can also
// be -1 (the latter are stored at the end).
// external provides an updated value for the entry's external field.  A false value, if requested, is always set;
// a true value, however, is only set if a new entry is added (external == true implies insert), i.e., if an entry
// already exists and the function is invoked with external == true, the new value will be ignored.
// If insert is set, it will, if necessary, insert a new entry if one is not already there.
// If pretend is set, it will not modify the array and will just set *pLinkIdx as appropriate.
// pLinkIdx is a pointer to an (optional) result where the index of the added/updated entry is stored.
// -1 can be passed as peerIdx if unknown (possible in case of race conditions, and only if external).
// -1 can be passed as connIdx if unknown or, if insert is *not* set, to indicate that the entry is to be removed
// (the entry's external must match the argument external for it to be removed).
ncclResult_t rasLinkUpdateConn(struct rasLink* link, int connIdx, int peerIdx, bool external, bool insert,
                               bool pretend, int* pLinkIdx) {
  int i, oldLinkIdx = -1;

  if (external && connIdx != -1)
    insert = true;

  if (connIdx != -1) {
    // Start by checking if we already have an element with this connIdx.
    oldLinkIdx = rasLinkFindConn(link, connIdx);
    if (oldLinkIdx != -1) {
      struct rasLinkConn* linkConn = link->conns+oldLinkIdx;
      if (linkConn->peerIdx != -1)
        assert(linkConn->peerIdx == peerIdx);

      if (linkConn->peerIdx == peerIdx) {
        if (!external && !pretend)
          linkConn->external = false; // Ensure that external is cleared if so requested.
        if (pLinkIdx)
          *pLinkIdx = oldLinkIdx;
        goto exit; // Nothing more to do if both connIdx and peerIdx are up to date.
      }

      // Otherwise (linkConn->peerIdx == -1 && peerIdx != -1) we have a conn that, due to -1 peerIdx, is in a wrong
      // place in the array -- we need to find the right spot.  linkConn->peerIdx == -1 can only happen for external
      // connections.
      assert(external);
    }
  }

  if (peerIdx != -1) {
    // Search for the right spot in the conns array.
    for (i = 0; i < link->nConns; i++) {
      struct rasLinkConn* linkConn = link->conns+i;
      if (peerIdx != -1 && linkConn->peerIdx == peerIdx) {
        // The exact conn element already exists.
        if (connIdx == -1 && !insert) {
          // Drop the connection from the link.
          if (linkConn->external == external) {
            if (!pretend)
              rasLinkDropConn(link, linkConn->connIdx, i);
            else if (pLinkIdx)
              *pLinkIdx = i;
          }
        } else { // connIdx != -1 || insert
          if (!pretend) {
            if (linkConn->connIdx != -1)
              assert(linkConn->connIdx == connIdx);
            else
              linkConn->connIdx = connIdx;
            if (!external)
              linkConn->external = false; // Ensure that external is cleared if so requested.
            if (i == 0) {
              // We received a connection from the remote peer that matches the primary connection we've been
              // waiting for.
              rasLinkSanitizeFallbacks(link);
            }
          } // if (!pretend)
          if (pLinkIdx)
            *pLinkIdx = i;
        } // connIdx != -1 || insert

        goto exit;
      } // if (peerIdx != -1 && linkConn->peerIdx == peerIdx)
      if (!insert)
        continue;
      // Ensure that the i-1 index is also valid.
      if (i == 0)
        continue;
      // Conns with peerIdx == -1 are stored at the end, so anything else needs to go before them.
      if (peerIdx != -1 && linkConn->peerIdx == -1)
        break;
      // Detect a roll-over and handle it specially.
      if (link->direction * (link->conns[i-1].peerIdx - linkConn->peerIdx) > 0) {
        if (link->direction * (peerIdx - link->conns[i-1].peerIdx) > 0 ||
            link->direction * (peerIdx - linkConn->peerIdx) < 0)
          break;
      } else { // Regular, monotonic case with the peerIdx value between two existing elements.
        if (link->direction * (peerIdx - link->conns[i-1].peerIdx) > 0 &&
            link->direction * (peerIdx - linkConn->peerIdx) < 0)
          break;
      }
    } // for (i)
  } else {
    // If peerIdx == -1, insert the new element at the very end.  This can only happen for external connections.
    assert(external && oldLinkIdx == -1);
    i = link->nConns;
  }
  if (!insert)
    goto exit;

  // i holds the index at which to insert a new element.
  if (pretend) {
    if (pLinkIdx)
      *pLinkIdx = i;
    goto exit;
  }

  if (oldLinkIdx == -1) {
    struct rasLinkConn* linkConn;
    if (link->nConns == link->connsSize) {
      NCCLCHECK(ncclRealloc(&link->conns, link->connsSize, link->connsSize+RAS_INCREMENT));
      link->connsSize += RAS_INCREMENT;
    }
    linkConn = link->conns+i;
    // Shift existing conns with indices >= i to make room for the new one.
    memmove(linkConn+1, linkConn, (link->nConns-i)*sizeof(*link->conns));
    linkConn->peerIdx = peerIdx;
    linkConn->connIdx = connIdx;
    linkConn->external = external;
    if (external) {
      INFO(NCCL_RAS, "RAS link %d: adding external fallback connection %d with %s", link->direction, i,
           ncclSocketToString((connIdx != -1 ? &rasConns[connIdx].addr : &rasPeers[peerIdx].addr), rasLine));
    }
    link->nConns++;
  }
  else { // oldLinkIdx > -1
    // We already have the conn, we just need to move it to a new spot.
    struct rasLinkConn* linkConn = link->conns+i;
    assert(i <= oldLinkIdx); // We can only get here if linkConn->peerIdx == -1 && peerIdx != -1.
    if (i != oldLinkIdx) {
      struct rasLinkConn tmp;
      struct rasLinkConn* linkConnNext = link->conns+i+1; // Just to silence the compiler.
      // Move the existing conn from index oldLinkIdx to a (lower) index i, shifting the existing conns
      // with indices in the range [i, oldLinkIdx).
      memcpy(&tmp, link->conns+oldLinkIdx, sizeof(tmp));
      memmove(linkConnNext, linkConn, (oldLinkIdx-i)*sizeof(*linkConn));
      memcpy(linkConn, &tmp, sizeof(*linkConn));
    }
    if (!external)
      linkConn->external = false; // Ensure that external is cleared if so requested.
  } // oldLinkIdx > -1
  if (pLinkIdx)
    *pLinkIdx = i;
exit:
  return ncclSuccess;
}
