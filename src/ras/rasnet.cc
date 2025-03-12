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
struct rasConnection* rasConnsHead;
struct rasConnection* rasConnsTail;

// Sockets implementing the RAS network.
struct rasSocket *rasSocketsHead;
struct rasSocket *rasSocketsTail;

// Magic file descriptor number when we want poll() to ignore an entry.  Anything negative would do, but
// I didn't want to use -1 because it has a special meaning for us.
#define POLL_FD_IGNORE -2

static void freeConnEntry(struct rasConnection* conn);
static void rasConnOpen(struct rasConnection* conn);
static ncclResult_t rasConnPrepare(struct rasConnection* conn);
static void rasConnTerminate(struct rasConnection* conn);

static ncclResult_t getNewSockEntry(struct rasSocket** pSock);
static void freeSockEntry(struct rasSocket* sock);

static ncclResult_t rasLinkHandleNetTimeouts(struct rasLink* link, int64_t now, int64_t* nextWakeup);
static void rasConnHandleNetTimeouts(struct rasConnection* conn, int64_t now, int64_t* nextWakeup);
static void rasConnSendKeepAlive(struct rasConnection* conn, bool nack = false);

static void rasConnResume(struct rasConnection* conn);
static void rasLinkSanitizeFallbacks(struct rasLink* link);
static ncclResult_t rasLinkConnAdd(struct rasLink* link, struct rasConnection* conn, int peerIdx, bool pretend = false,
                                   int* pLinkIdx = nullptr, struct rasLinkConn** pLinkConn = nullptr,
                                   bool insert = true);
static ncclResult_t rasLinkConnAddExternal(struct rasLink* link, struct rasConnection* conn, int peerIdx);
static void rasLinkConnDrop(struct rasLink* link, const struct rasConnection* conn, bool external = false);
static struct rasLinkConn* rasLinkConnFind(const struct rasLink* link, const struct rasConnection* conn,
                                           int* pLinkIdx = nullptr);


///////////////////////////////////////////////
// Functions related to the RAS connections. //
///////////////////////////////////////////////

// Allocates a new entry in the rasConnections list.
ncclResult_t getNewConnEntry(struct rasConnection** pConn) {
  struct rasConnection* conn;

  NCCLCHECK(ncclCalloc(&conn, 1));

  ncclIntruQueueConstruct(&conn->sendQ);
  conn->travelTimeMin = INT64_MAX;
  conn->travelTimeMax = INT64_MIN;

  if (rasConnsHead) {
    rasConnsTail->next = conn;
    conn->prev = rasConnsTail;
    rasConnsTail = conn;
  } else {
    rasConnsHead = rasConnsTail = conn;
  }

  *pConn = conn;
  return ncclSuccess;
}

// Frees an entry from the rasConns list.
static void freeConnEntry(struct rasConnection* conn) {
  if (conn == nullptr)
    return;

  if (conn == rasConnsHead)
    rasConnsHead = rasConnsHead->next;
  if (conn == rasConnsTail)
    rasConnsTail = rasConnsTail->prev;
  if (conn->prev)
    conn->prev->next = conn->next;
  if (conn->next)
    conn->next->prev = conn->prev;
  free(conn);
}

// Creates a new RAS network connection to a remote peer address.
ncclResult_t rasConnCreate(const union ncclSocketAddress* addr, struct rasConnection** pConn) {
  ncclResult_t ret = ncclSuccess;
  struct rasConnection* conn;

  // First check if a connection entry for this peer already exists.
  conn = rasConnFind(addr);

  if (conn && conn->sock) {
    // An entry exists and has a socket associated with it -- nothing left for us to do.
    if (pConn)
      *pConn = conn;
    goto exit;
  }

  if (conn == nullptr) {
    NCCLCHECKGOTO(getNewConnEntry(&conn), ret, exit);
    memcpy(&conn->addr, addr, sizeof(conn->addr));
    // We are establishing a new connection -- start the timeout.
    conn->startRetryTime = clockNano();
  }

  if (pConn)
    *pConn = conn;

  rasConnOpen(conn);

exit:
  return ret;
}

// Opens a connection to a remote peer.
static void rasConnOpen(struct rasConnection* conn) {
  ncclResult_t ret; // Not used.
  struct rasSocket* sock = nullptr;
  bool closeSocketOnFail = false;
  int ready;

  NCCLCHECKGOTO(getNewSockEntry(&sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&sock->sock, &conn->addr, NCCL_SOCKET_MAGIC, ncclSocketTypeRasNetwork, nullptr,
                               /*asyncFlag*/1, /*customRetry*/1), ret, fail);
  closeSocketOnFail = true;
  NCCLCHECKGOTO(ncclSocketConnect(&sock->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketReady(&sock->sock, &ready), ret, fail);

  NCCLCHECKGOTO(rasGetNewPollEntry(&sock->pfd), ret, fail);

  conn->sock = sock;
  sock->conn = conn;
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
  freeSockEntry(sock);
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
struct rasConnection* rasConnFind(const union ncclSocketAddress* addr) {
  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next) {
    if (memcmp(&conn->addr, addr, sizeof(conn->addr)) == 0)
      return conn;
  }

  return nullptr;
}

// Handles any connection-related timeouts.  Many timeouts affect the underlying sockets and thus have been handled
// in the socket timeout handler earlier by terminating the problematic sockets.  If a socket connection doesn't
// exist or needs to be re-established (due to having just been terminated), we handle that here.
// This is also where we declare peers as dead, etc.
// Invoked from the main RAS event loop.
void rasConnsHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  for (struct rasConnection* conn = rasConnsHead; conn;) {
    struct rasConnection* connNext = conn->next;
    if (conn->sock) {
      bool sockTerminated = false;

      // Retry the socket connections that have been refused.
      if (conn->sock->status == RAS_SOCK_CONNECTING && conn->sock->sock.state == ncclSocketStateConnecting) {
        if (now - conn->sock->lastSendTime > RAS_CONNECT_RETRY) {
          int ready;
          if (ncclSocketReady(&conn->sock->sock, &ready) != ncclSuccess) {
            INFO(NCCL_RAS, "Unexpected error from ncclSocketReady; terminating the socket connection with %s",
                 ncclSocketToString(&conn->addr, rasLine));
            rasSocketTerminate(conn->sock, /*finalize*/true);
            // We will retry below in the same loop.
            sockTerminated = true;
          } else {
            // We update lastSendTime even if !ready because we need it up-to-date for timeout calculations.
            conn->sock->lastSendTime = clockNano();
            if (!ready && conn->sock->sock.state == ncclSocketStateConnecting)
              *nextWakeup = std::min(*nextWakeup, conn->sock->lastSendTime+RAS_CONNECT_RETRY);
            else
              rasPfds[conn->sock->pfd].fd = conn->sock->sock.fd; // Enable the handling via the main loop.
          } // if (ncclSocketReady)
        } else {
          *nextWakeup = std::min(*nextWakeup, conn->sock->lastSendTime+RAS_CONNECT_RETRY);
        }
      } // if (conn->sock->status == RAS_SOCK_CONNECTING && conn->sock->sock.state == ncclSocketStateConnecting)

      // For connections that have data to send but that we've been unable to send a message on for a while,
      // consider their sockets lost and terminate them.
      if (!sockTerminated && !ncclIntruQueueEmpty(&conn->sendQ) && conn->sock->status == RAS_SOCK_READY) {
        if (now - std::max(conn->sock->lastSendTime,
                           ncclIntruQueueHead(&conn->sendQ)->enqueueTime) > RAS_STUCK_TIMEOUT) {
          INFO(NCCL_RAS, "RAS send stuck timeout error (%lds) on socket connection with %s",
               (now - std::max(conn->sock->lastSendTime, ncclIntruQueueHead(&conn->sendQ)->enqueueTime)) /
               CLOCK_UNITS_PER_SEC, ncclSocketToString(&conn->addr, rasLine));
          rasSocketTerminate(conn->sock, /*finalize*/false, RAS_STUCK_TIMEOUT);
          // We will retry below in the same loop.
        } else {
          *nextWakeup = std::min(*nextWakeup,
                                 std::max(conn->sock->lastSendTime, ncclIntruQueueHead(&conn->sendQ)->enqueueTime)+
                                 RAS_STUCK_TIMEOUT);
        }
      } // if (!ncclIntruQueueEmpty(&conn->sendQ) && conn->sock->status == RAS_SOCK_READY)
    } // if (conn->sock)

    // For connections that are being (re-)established, irrespective of whether there's a valid socket associated
    // with them, we need to check if any connection-level timeout has expired.
    if (conn->startRetryTime) {
      bool connTerminated = false;
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
        (void)rasNetSendCollReq(&bCast);

        connTerminated = true;
      } else {
        *nextWakeup = std::min(*nextWakeup, conn->startRetryTime+RAS_PEER_DEAD_TIMEOUT);
      }

      // RAS_STUCK_TIMEOUT has already been handled in the socket function (we'll pick it up later via
      // the conn->sock == nullptr test).

      if (!connTerminated) {
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
            (void)rasLinkAddFallback(&rasNextLink, conn);
            (void)rasLinkAddFallback(&rasPrevLink, conn);

            // Stop collectives from waiting for a response over it.
            rasCollsPurgeConn(conn);
          } // if (!conn->experiencingDelays)
        } else {
          *nextWakeup = std::min(*nextWakeup, conn->startRetryTime+RAS_CONNECT_WARN);
        }

        // If a socket was terminated (or never opened, due to some error), try to open it now.
        // We retry once a second.
        if (conn->sock == nullptr) {
          if (now - conn->lastRetryTime > RAS_CONNECT_RETRY) {
            INFO(NCCL_RAS, "RAS trying to reconnect with %s (experiencingDelays %d, startRetryTime %.2fs)",
                 ncclSocketToString(&conn->addr, rasLine), conn->experiencingDelays,
                 (conn->startRetryTime ? (now-conn->startRetryTime)/1e9 : 0.0));
            rasConnOpen(conn);
          }
          if (conn->sock == nullptr)
            *nextWakeup = std::min(*nextWakeup, conn->lastRetryTime+RAS_CONNECT_RETRY);
        }
      } // if (!connTerminated)
    } // if (conn->startRetryTime)

    conn = connNext;
  } // for (conn)
}

// Checks if we have a connection to a given peer and if so, terminates it.  The connection is removed from the
// RAS links, though fallbacks are initiated if necessary.  Typically called just before declaring a peer dead.
void rasConnDisconnect(const union ncclSocketAddress* addr) {
  struct rasConnection* conn = rasConnFind(addr);
  if (conn) {
    (void)rasLinkAddFallback(&rasNextLink, conn);
    (void)rasLinkAddFallback(&rasPrevLink, conn);
    rasLinkConnDrop(&rasNextLink, conn);
    rasLinkConnDrop(&rasPrevLink, conn);

    rasConnTerminate(conn);
  }
}

// Terminates a connection and frees the rasConns entry.
static void rasConnTerminate(struct rasConnection* conn) {
  // Make sure there are no lingering rasSockets pointing to it.
  for (struct rasSocket* sock = rasSocketsHead; sock;) {
    struct rasSocket* sockNext = sock->next;
    if (sock->conn == conn)
      rasSocketTerminate(sock, /*finalize*/true);
    sock = sockNext;
  }

  // Also check any ongoing collectives.
  rasCollsPurgeConn(conn);

  while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&conn->sendQ)) {
    free(meta);
  }

  INFO(NCCL_RAS, "RAS terminating a connection with %s", ncclSocketToString(&conn->addr, rasLine));

  freeConnEntry(conn);
}


///////////////////////////////////////////
// Functions related to the RAS sockets. //
///////////////////////////////////////////

// Accepts a new RAS network socket connection.  The socket is not usable until after the handshake, as a
// corresponding rasConnection can't be established without knowing the peer's address.
ncclResult_t rasNetAcceptNewSocket() {
  ncclResult_t ret = ncclSuccess;
  struct rasSocket* sock = nullptr;
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
  freeSockEntry(sock);
  goto exit;
}

// Allocates a new entry in the rasSockets list.
static ncclResult_t getNewSockEntry(struct rasSocket** pSock) {
  struct rasSocket* sock;

  NCCLCHECK(ncclCalloc(&sock, 1));

  sock->pfd = -1;
  sock->createTime = sock->lastSendTime = sock->lastRecvTime = clockNano();

  if (rasSocketsHead) {
    rasSocketsTail->next = sock;
    sock->prev = rasSocketsTail;
    rasSocketsTail = sock;
  } else {
    rasSocketsHead = rasSocketsTail = sock;
  }

  *pSock = sock;
  return ncclSuccess;
}

// Frees an entry from the rasSockets list.
static void freeSockEntry(struct rasSocket* sock) {
  if (sock == nullptr)
    return;

  if (sock == rasSocketsHead)
    rasSocketsHead = rasSocketsHead->next;
  if (sock == rasSocketsTail)
    rasSocketsTail = rasSocketsTail->prev;
  if (sock->prev)
    sock->prev->next = sock->next;
  if (sock->next)
    sock->next->prev = sock->prev;
  free(sock);
}

// Invoked from the main RAS event loop to handle RAS socket timeouts.
void rasSocksHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  for (struct rasSocket* sock = rasSocketsHead; sock;) {
    struct rasSocket* sockNext = sock->next;

    if (sock->status == RAS_SOCK_CONNECTING || sock->status == RAS_SOCK_HANDSHAKE) {
      // For socket connections that are still being established, give up on the ones that take too long to initialize.
      if (now - sock->createTime > RAS_STUCK_TIMEOUT) {
        if (sock->conn == nullptr) {
          INFO(NCCL_RAS, "RAS init timeout error (%lds) on incoming socket connection from %s",
               (now-sock->createTime)/CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine));
        } else {
          INFO(NCCL_RAS, "RAS init timeout error (%lds) on socket connection with %s "
               "(experiencingDelays %d, startRetryTime %.2fs, socket status %d)",
               (now-sock->createTime)/CLOCK_UNITS_PER_SEC, ncclSocketToString(&sock->sock.addr, rasLine),
               sock->conn->experiencingDelays,
               (sock->conn->startRetryTime ? (now-sock->conn->startRetryTime)/1e9 : 0.0), sock->status);
        }
        rasSocketTerminate(sock, /*finalize*/true);
        // We may retry later.
      } else {
        *nextWakeup = std::min(*nextWakeup, sock->createTime+RAS_STUCK_TIMEOUT);
      }
    } else if (sock->status == RAS_SOCK_TERMINATING) {
      // For sockets that are being terminated, force finalization of the ones that haven't made progress in too long.
      if (now - std::max(sock->lastSendTime, sock->lastRecvTime) > RAS_STUCK_TIMEOUT) {
        INFO(NCCL_RAS, "RAS termination stuck timeout error (%lds) on socket connection with %s",
             (now-std::max(sock->lastSendTime, sock->lastRecvTime)) / CLOCK_UNITS_PER_SEC,
             ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock, /*finalize*/true);
        // This socket is presumably already being re-established, if needed.
      } else {
        *nextWakeup = std::min(*nextWakeup, std::max(sock->lastSendTime, sock->lastRecvTime)+RAS_STUCK_TIMEOUT);
      }
    } else if (sock->status == RAS_SOCK_READY) {
      // Terminate sockets that haven't been used in a good while.  In principle this shouldn't trigger for anything
      // important due to shorter timeouts on RAS network connections, but in case of weird situations like process
      // suspend, rasSocketTerminate will do additional checking.
      if (now - std::max(sock->lastSendTime, sock->lastRecvTime) > RAS_IDLE_TIMEOUT) {
        INFO(NCCL_RAS, "RAS idle timeout (%lds) on socket connection with %s",
             (now - std::max(sock->lastSendTime, sock->lastRecvTime)) / CLOCK_UNITS_PER_SEC,
             ncclSocketToString(&sock->sock.addr, rasLine));
        rasSocketTerminate(sock, /*finalize*/false, /*startRetryOffset*/0, /*retry*/false);
        // The RAS network timeout handler will terminate the conn it was associated with, if any.
      } else {
        *nextWakeup = std::min(*nextWakeup, std::max(sock->lastSendTime, sock->lastRecvTime)+RAS_IDLE_TIMEOUT);
      }
    } // if (sock->status == RAS_SOCK_READY)

    sock = sockNext;
  } // for (sock)
}

// Handles the termination of a RAS socket.
// We try to do it in stages for established sockets (in READY state).  We shut down just the sending side
// for them and change their state to TERMINATING, so that we can still receive data that may be in the buffers.
// Once we get an EOF when receiving data, we finalize the termination.
// For not fully established sockets, we can terminate immediately as there's no useful data to extract.
void rasSocketTerminate(struct rasSocket* sock, bool finalize, uint64_t startRetryOffset, bool retry) {
  assert(sock->status != RAS_SOCK_CLOSED);
  if (sock->conn) {
    struct rasConnection* conn = sock->conn;
    // If the sock of the connection points back to us, it means that we are the current socket of this
    // connection, so we have additional work to do before we can terminate it.
    if (conn->sock == sock) {
      // Reset it to indicate there's no valid socket associated with that connection anymore.
      conn->sock = nullptr;

      // Don't attempt to retry on sockets that have been unused for so long that the remote peer probably
      // deliberately closed them.  Make an exception for sockets that are part of the RAS network links.
      if ((retry &&
           clockNano() - std::max(sock->lastSendTime, sock->lastRecvTime) < RAS_IDLE_TIMEOUT - RAS_IDLE_GRACE_PERIOD) ||
          rasLinkConnFind(&rasNextLink, sock->conn) || rasLinkConnFind(&rasPrevLink, sock->conn)) {
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
      rasCollsPurgeConn(sock->conn);
    } // if (conn->sock == sock)
  } // if (sock->conn)

  if (sock->status != RAS_SOCK_CONNECTING && sock->conn && !finalize && (rasPfds[sock->pfd].events & POLLIN)) {
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
    rasPfds[sock->pfd].fd = -1;
    rasPfds[sock->pfd].events = rasPfds[sock->pfd].revents = 0;
    free(sock->recvMsg);
    freeSockEntry(sock);
  }
}

// Handles a ready socket FD from the main event loop.
void rasSockEventLoop(struct rasSocket* sock, int pollIdx) {
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
          assert(sock->conn);
          if (sock->conn->sock == sock) {
            if (rasConnPrepare(sock->conn) != ncclSuccess) {
              INFO(NCCL_RAS, "RAS unexpected error from rasConnPrepare; terminating the socket connection with %s",
                   ncclSocketToString(&sock->sock.addr, rasLine));
              rasSocketTerminate(sock);
              // We may retry further down.
            }
          } else { // sock->conn->sock != sock
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
      assert(sock->conn);
      assert(sock->conn->sock == sock);
      if (rasConnSendMsg(sock->conn, &closed, &allSent) != ncclSuccess) {
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
          if (sock->conn == nullptr)
            socketType = "incoming";
          else if (sock->conn->sock != sock)
            socketType = "old";
          else if (sock->status == RAS_SOCK_HANDSHAKE)
            socketType = "new";
          else
            socketType = "current";
          INFO(NCCL_RAS, "RAS %s socket connection with %s closed by peer on receive; terminating it",
               socketType, ncclSocketToString(&sock->sock.addr, rasLine));
          rasSocketTerminate(sock, /*finalize*/true);
          // We may retry further down.
        } else { // !closed
          sock->lastRecvTime = clockNano();
          if (msg) {
            (void)rasMsgHandle(msg, sock);
            free(msg);
            // Message handlers can terminate a socket in various cases.  We re-check rasPfds.events to ensure that
            // this hasn't happened here (rasSocketTerminate will reset it when finalizing a socket).
            if (!(rasPfds[pollIdx].revents & POLLIN))
              break;
          }
          if (sock->conn) {
            if (sock->conn->sock == sock && (sock->conn->startRetryTime || sock->conn->experiencingDelays))
              rasConnResume(sock->conn);
          }
        } // !closed
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
  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next)
    conn->linkFlag = false;

  (void)rasLinkHandleNetTimeouts(&rasNextLink, now, nextWakeup);
  (void)rasLinkHandleNetTimeouts(&rasPrevLink, now, nextWakeup);

  for (struct rasConnection* conn = rasConnsHead; conn;) {
    struct rasConnection* connNext = conn->next;
    if (!conn->linkFlag) {
      // The connection is not part of any link.  Check if it should be terminated.
      if (conn->sock == nullptr && ncclIntruQueueEmpty(&conn->sendQ))
        rasConnTerminate(conn);
    }
    conn = connNext;
  }
}

// Checks for and handles timeouts at the link level; primarily the keep-alives for link connections.
static ncclResult_t rasLinkHandleNetTimeouts(struct rasLink* link, int64_t now, int64_t* nextWakeup) {
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next) {
    if (linkConn->conn) {
      if (!linkConn->conn->linkFlag) {
        rasConnHandleNetTimeouts(linkConn->conn, now, nextWakeup);
        linkConn->conn->linkFlag = true;
      }
    } else if (linkConn == link->conns && link->lastUpdatePeersTime != 0) {
      // This triggers when rasLinkReinitConns didn't create the primary connection because we have a higher address
      // than the peer.  If that peer fails to initiate within RAS_CONNECT_WARN, we need to take action.
      if (now - link->lastUpdatePeersTime > RAS_CONNECT_WARN) {
        INFO(NCCL_RAS, "RAS peer connect timeout warning (%lds) on socket connection from %s",
             (now-link->lastUpdatePeersTime) / CLOCK_UNITS_PER_SEC,
             ncclSocketToString(&rasPeers[linkConn->peerIdx].addr, rasLine));
        NCCLCHECK(rasConnCreate(&rasPeers[linkConn->peerIdx].addr, &linkConn->conn));
        if (linkConn->conn) {
          linkConn->conn->linkFlag = true;
        }
        link->lastUpdatePeersTime = 0;
      } else {
        *nextWakeup = std::min(*nextWakeup, link->lastUpdatePeersTime+RAS_CONNECT_WARN);
      }
    } // if (linkConn == link->conns && link->lastUpdatePeerTime != 0)
  } // for (linkConn)

  return ncclSuccess;
}

// Handles the sending of keep-alive messages and related timeouts for connections that are part of the RAS links.
static void rasConnHandleNetTimeouts(struct rasConnection* conn, int64_t now, int64_t* nextWakeup) {
  if (conn->sock) {
    if (conn->sock->status == RAS_SOCK_READY) {
      // Send a regular keep-alive message if we haven't sent anything in a while and we don't have anything queued.
      if (ncclIntruQueueEmpty(&conn->sendQ)) {
        if (now - conn->sock->lastSendTime > RAS_KEEPALIVE_INTERVAL) {
          rasConnSendKeepAlive(conn);
        } else {
          *nextWakeup = std::min(*nextWakeup, conn->sock->lastSendTime+RAS_KEEPALIVE_INTERVAL);
        }
      }

      // For short timeouts print a warning but also pessimistically immediately try to establish fallback connections.
      if (now - conn->sock->lastRecvTime > RAS_KEEPALIVE_TIMEOUT_WARN) {
        if (!conn->experiencingDelays) {
          INFO(NCCL_RAS, "RAS keep-alive timeout warning (%lds) on socket connection with %s",
               (now-conn->sock->lastRecvTime) / CLOCK_UNITS_PER_SEC, ncclSocketToString(&conn->addr, rasLine));

          // At this point, it's mostly just a precaution; we will continue with the primary connection until
          // RAS_PEER_DEAD_TIMEOUT expires.
          conn->experiencingDelays = true;
          (void)rasLinkAddFallback(&rasNextLink, conn);
          (void)rasLinkAddFallback(&rasPrevLink, conn);

          // Stop ongoing collectives from waiting for a response over this connection.
          rasCollsPurgeConn(conn);
        }
      } else {
        *nextWakeup = std::min(*nextWakeup, conn->sock->lastRecvTime+RAS_KEEPALIVE_TIMEOUT_WARN);
      }

      // For long timeouts we need to act.
      if (now - conn->sock->lastRecvTime > RAS_KEEPALIVE_TIMEOUT_ERROR) {
        INFO(NCCL_RAS, "RAS keep-alive timeout error (%lds) on socket connection with %s",
             (now-conn->sock->lastRecvTime) / CLOCK_UNITS_PER_SEC, ncclSocketToString(&conn->addr, rasLine));
        rasSocketTerminate(conn->sock, /*finalize*/true, RAS_KEEPALIVE_TIMEOUT_ERROR);
        *nextWakeup = now; // Retry will be in the next iteration of the main loop so ensure we don't wait.
      } else {
        *nextWakeup = std::min(*nextWakeup, conn->sock->lastRecvTime+RAS_KEEPALIVE_TIMEOUT_ERROR);
      }
    } // if (conn->sock->status == RAS_SOCK_READY)
  } // if (conn->sock)
}

// Sends a keep-alive message to a peer on the RAS network.
static void rasConnSendKeepAlive(struct rasConnection* conn, bool nack) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  if (rasMsgAlloc(&msg, msgLen) == ncclSuccess) {
    struct rasLinkConn* linkConn;
    msg->type = RAS_MSG_KEEPALIVE;
    msg->keepAlive.peersHash = rasPeersHash;
    msg->keepAlive.deadPeersHash = rasDeadPeersHash;
    msg->keepAlive.nack = (nack ? 1 : 0);

    linkConn = rasLinkConnFind(&rasNextLink, conn);
    if (linkConn && !linkConn->external)
      msg->keepAlive.linkMask |= 2; // Our rasNextLink should be the peer's rasPrevLink.
    linkConn = rasLinkConnFind(&rasPrevLink, conn);
    if (linkConn && !linkConn->external)
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

  assert(sock->conn);
  SYSCHECK(clock_gettime(CLOCK_REALTIME, &currentTime), "clock_gettime");
  travelTime = (currentTime.tv_sec-msg->keepAlive.realTime.tv_sec)*1000*1000*1000 +
    (currentTime.tv_nsec-msg->keepAlive.realTime.tv_nsec);

  if (msg->keepAlive.peersHash != sock->conn->lastRecvPeersHash) {
    sock->conn->lastRecvPeersHash = msg->keepAlive.peersHash;
  }
  if (msg->keepAlive.deadPeersHash != sock->conn->lastRecvDeadPeersHash) {
    sock->conn->lastRecvDeadPeersHash = msg->keepAlive.deadPeersHash;
  }

  // Make sure that the connection is part of the appropriate links forming the RAS network.  In particular, this
  // will add any externally-requested connections to the appropriate links (or remove existing ones, if no longer
  // needed).
  peerIdx = rasPeerFind(&sock->conn->addr);
  // Note: it's possible for peerIdx to be -1 at this point if, due to races, the keepAlive arrives before
  // the peers update.
  if (msg->keepAlive.linkMask & 1)
    (void)rasLinkConnAddExternal(&rasNextLink, sock->conn, peerIdx);
  else
    rasLinkConnDrop(&rasNextLink, sock->conn, /*external*/true);
  if (msg->keepAlive.linkMask & 2)
    (void)rasLinkConnAddExternal(&rasPrevLink, sock->conn, peerIdx);
  else
    rasLinkConnDrop(&rasPrevLink, sock->conn, /*external*/true);

  // If the keep-alive message is from a peer that doesn't actually need this connection (i.e., for that peer the
  // connection is just an external fallback), we should check if *we* still need it.  It might be that we don't,
  // and because we stopped sending the keep-alives, our peer doesn't know about it.  The rasLinkConnDrop calls
  // above will have wiped any external fallbacks, so anything that remains must be needed.
  if (!msg->keepAlive.nack && msg->keepAlive.linkMask == 0) {
    if (rasLinkConnFind(&rasNextLink, sock->conn) == nullptr && rasLinkConnFind(&rasPrevLink, sock->conn) == nullptr) {
      // We don't need this connection either.  Notify the peer about it.  To avoid an infinite loop, we set the
      // special nack flag in the message to distinguish it from regular keep-alives.
      rasConnSendKeepAlive(sock->conn, /*nack*/true);
    }
  }

  if (sock->conn->travelTimeMin > travelTime)
    sock->conn->travelTimeMin = travelTime;
  if (sock->conn->travelTimeMax < travelTime)
    sock->conn->travelTimeMax = travelTime;
  sock->conn->travelTimeSum += travelTime;
  sock->conn->travelTimeCount++;

  if (msg->keepAlive.peersHash != rasPeersHash || msg->keepAlive.deadPeersHash != rasDeadPeersHash) {
    // This could happen due to a short-lived race condition between the peers propagation
    // process and the periodic keep-alive messages (perhaps we'll see it regularly at scale?).
    // Just in case there's some unforeseen problem with the peers propagation though, exchange with the
    // remote to get everybody in sync.
    INFO(NCCL_RAS, "RAS keepAlive hash mismatch from %s (peersHash 0x%lx, deadPeersHash 0x%lx)",
         ncclSocketToString(&sock->sock.addr, rasLine), msg->keepAlive.peersHash, msg->keepAlive.deadPeersHash);
    INFO(NCCL_RAS, "RAS my peersHash 0x%lx, deadPeersHash 0x%lx", rasPeersHash, rasDeadPeersHash);
    NCCLCHECK(rasConnSendPeersUpdate(sock->conn, rasPeers, nRasPeers));
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
ncclResult_t rasLinkAddFallback(struct rasLink* link, const struct rasConnection* conn) {
  struct rasLinkConn* foundLinkConn = nullptr;
  struct rasLinkConn* firstExtLinkConn = nullptr;
  int firstExtLinkIdx = -1;
  int newPeerIdx, i;

  // First check if the connection is part of this link.  In the process also check if any of the link's connections
  // might be active -- if so, there's no need to initiate any more fallbacks and we can bail out.
  i = 0;
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next, i++) {
    if (linkConn->peerIdx == -1) {
      // Such elements are always at the end and we can't use them so we can just as well break.
      break;
    }

    // Check for any other connection that might be a viable fallback (basically, anything that is not experiencing
    // delays).
    if (linkConn->conn && linkConn->conn != conn) {
      if (!linkConn->conn->experiencingDelays) {
        if (!linkConn->external) {
          goto exit; // We don't need to do anything if there's a non-external connection.
        } else if (linkConn->peerIdx != -1) {
          // Record the location of the first potentially viable external connection in the chain; we may prefer it
          // over anything we can come up with.
          if (firstExtLinkConn == nullptr) {
            firstExtLinkConn = linkConn;
            firstExtLinkIdx = i;
          }
          if (foundLinkConn)
            break; // Break out of the loop if we already have all the data we might need.
        } // linkConn->external && linkConn->peerIdx != -1
      } // if (!linkConn->conn->experiencingDelays)
    } // if (linkConn->conn && linkConn->conn != conn)

    if (linkConn->conn == conn) {
      if (linkConn->external)
        goto exit; // We don't add fallbacks for external connections...
      foundLinkConn = linkConn;
      // We are not breaking out of the loop here because we want to check for active connections on *all* potentially
      // viable elements (in particular, there could be some external ones beyond this one).
    }
  }

  if (foundLinkConn == nullptr)
    goto exit;

  // We found an existing element so the connection is part of the link.  No existing non-external connections of this
  // link are active, so a fallback is needed.
  assert(foundLinkConn->peerIdx != -1);
  newPeerIdx = rasLinkCalculatePeer(link, foundLinkConn->peerIdx, /*isFallback*/(foundLinkConn != link->conns));
  // In principle we want to add (at most) one fallback.  However, if the found fallback connection already exists
  // and is also experiencing delays, we need to keep iterating.
  while (newPeerIdx != -1) {
    struct rasConnection* newConn = rasConnFind(&rasPeers[newPeerIdx].addr);
    int linkIdx;
    struct rasLinkConn* newLinkConn;
    // If we previously found a potential external fallback connection, check if it's better than what we just found.
    if (firstExtLinkConn) {
      linkIdx = -1;
      // Calculate the index that the newly found fallback would have (pretend mode).
      NCCLCHECK(rasLinkConnAdd(link, newConn, newPeerIdx, /*pretend*/true, &linkIdx));
      assert(linkIdx != -1);
      if (firstExtLinkIdx < linkIdx) {
        // The external connection *is* better -- use it as a fallback instead and be done.
        firstExtLinkConn->external = false;
        goto exit;
      }
    }
    NCCLCHECK(rasLinkConnAdd(link, newConn, newPeerIdx, /*pretend*/false, &linkIdx, &newLinkConn));
    if (firstExtLinkConn && linkIdx <= firstExtLinkIdx)
      firstExtLinkIdx++; // Adjust if we inserted a new entry ahead of this one.

    INFO(NCCL_RAS, "RAS link %d: %s fallback connection %d with %s",
         link->direction, (newConn == nullptr ? "opening new" : "calculated existing"),
         linkIdx, ncclSocketToString(&rasPeers[newPeerIdx].addr, rasLine));
    // Note that we don't follow here our convention of "lower address is the one establishing connections" --
    // that convention is for optimizing regular operations, but we don't want to take chances during fault
    // recovery. It may temporarily result in duplicate connections, but we have a mechanism to deal with those.
    if (newConn == nullptr) {
      NCCLCHECK(rasConnCreate(&rasPeers[newPeerIdx].addr, &newConn));
      newLinkConn->conn = newConn;
    }

    // If the fallback connection is also experiencing delays, we need to keep trying.
    if (!newConn->experiencingDelays)
      break;
    INFO(NCCL_RAS, "RAS connection experiencingDelays %d, startRetryTime %.2fs, socket status %d",
         newConn->experiencingDelays, (newConn->startRetryTime ? (clockNano()-newConn->startRetryTime)/1e9 : 0.0),
         (newConn->sock ? newConn->sock->status : -1));

    newPeerIdx = rasLinkCalculatePeer(link, newPeerIdx, /*isFallback*/true);
  }
  if (newPeerIdx == -1) {
    int nConns = 0;
    for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next)
      nConns++;
    INFO(NCCL_RAS, "RAS link %d: no more fallbacks to add (total %d)", link->direction, nConns);
  }
exit:
  return ncclSuccess;
}

// Invoked when we receive a message over a connection that was just activated or was experiencing delays.
// Cleans up the fallbacks, timers, etc, as appropriate.
static void rasConnResume(struct rasConnection* conn) {
  if (conn->sock && conn->sock->status == RAS_SOCK_READY) {
    INFO(NCCL_RAS, "RAS %s connection with %s (sendQ %sempty, experiencingDelays %d, startRetryTime %.2fs)",
         (conn->experiencingDelays && conn->startRetryTime == 0 ? "recovered" : "established"),
         ncclSocketToString(&conn->addr, rasLine), (ncclIntruQueueEmpty(&conn->sendQ) ? "" : "not "),
         conn->experiencingDelays, (conn->startRetryTime ? (clockNano()-conn->startRetryTime)/1e9 : 0.0));

    conn->experiencingDelays = false;

    conn->startRetryTime = conn->lastRetryTime = 0;

    rasLinkSanitizeFallbacks(&rasNextLink);
    rasLinkSanitizeFallbacks(&rasPrevLink);

    if (!ncclIntruQueueEmpty(&conn->sendQ))
      rasPfds[conn->sock->pfd].events |= POLLOUT;
  }
}

// Checks if the primary connection is fully established and if so, purges the fallbacks (as they are no longer needed).
static void rasLinkSanitizeFallbacks(struct rasLink* link) {
  if (link->conns && link->conns->conn) {
    struct rasConnection* conn = link->conns->conn;
    if (conn->sock && conn->sock->status == RAS_SOCK_READY && !conn->experiencingDelays) {
      // We have a good primary.  Simply drop all the fallbacks (the external ones will get recreated via the
      // keepAlive messages).
      int i = 1;
      for (struct rasLinkConn* linkConn = link->conns->next; linkConn; i++) {
        struct rasLinkConn* linkConnNext = linkConn->next;
        INFO(NCCL_RAS, "RAS link %d: dropping %sfallback connection %d with %s",
             link->direction, (linkConn->external ? "external " : ""), i,
             ncclSocketToString(&linkConn->conn->addr, rasLine));
        free(linkConn);
        linkConn = linkConnNext;
      }
      link->conns->next = nullptr;
      link->lastUpdatePeersTime = 0;
    }
  }
}

// Adds an entry to a RAS network link (or updates one, if it already exists).
// conn can be nullptr if the connection doesn't exist (yet).
// peerIdx *cannot* be -1 when this function is invoked.
// If pretend is true, the function will not modify the list and will just set *pLinkIdx and *pLinkConn as appropriate.
// pLinkIdx and pLinkConn are (optional) pointers to the results; the index/address of the added/updated entry are
// stored there.
// insert (true by default) determines whether this is an "add" function (as implied by the name) or an "update" --
// if set to false, it will refuse to add a new entry (but will update an existing one as needed).
// Note: there is some code duplication between this function and rasLinkConnAddExternal so changes to one of them
// may need to be sync'ed to the other one as well.  They used to be a single function that could do it all but the
// logic was extremely difficult to follow then.
static ncclResult_t rasLinkConnAdd(struct rasLink* link, struct rasConnection* conn, int peerIdx, bool pretend,
                                   int* pLinkIdx, struct rasLinkConn** pLinkConn, bool insert) {
  struct rasLinkConn* oldLinkConn = nullptr;
  struct rasLinkConn* linkConnPrev = nullptr;
  int i, oldLinkIdx = -1;

  assert(peerIdx != -1);
  if (conn) {
    // Start by checking if we already have an element with this conn.
    oldLinkConn = rasLinkConnFind(link, conn, &oldLinkIdx);
    if (oldLinkConn) {
      if (pLinkConn)
        *pLinkConn = oldLinkConn;
      if (oldLinkConn->peerIdx != -1) {
        assert(oldLinkConn->peerIdx == peerIdx);

        if (!pretend)
          oldLinkConn->external = false; // Ensure that external is cleared.
        if (pLinkIdx)
          *pLinkIdx = oldLinkIdx;
        goto exit; // Nothing more to do if both conn and peerIdx are up to date.
      } // if (oldLinkConn->peerIdx != -1)

      // Otherwise oldLinkConn->peerIdx == -1.  The oldLinkConn is in a wrong place in the list -- we need to find
      // the right spot.  This can happen only for external connections.
    } // if (oldLinkConn)
  } // if (conn)

  // Search for the right spot in the conns list.
  i = 0;
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConnPrev = linkConn, linkConn = linkConn->next, i++) {
    if (linkConn->peerIdx == peerIdx) {
      // The exact linkConn element already exists.
      if (linkConn->conn)
        assert(linkConn->conn == conn);
      if (!pretend) {
        if (linkConn->conn == nullptr)
          linkConn->conn = conn;
        linkConn->external = false; // Ensure that external is cleared.
        if (linkConn == link->conns) {
          // We received a connection from the remote peer that matches the primary connection we've been
          // waiting for.
          rasLinkSanitizeFallbacks(link);
        }
      } // if (!pretend)
      if (pLinkIdx)
        *pLinkIdx = i;
      if (pLinkConn)
        *pLinkConn = linkConn;
      goto exit;
    } // if (linkConn->peerIdx == peerIdx)

    // Ensure that the previous element is valid.
    if (linkConnPrev == nullptr)
      continue;
    // linkConns with peerIdx == -1 are stored at the end, so if we reach one of them, we are done.
    if (linkConn->peerIdx == -1)
      break;
    // Detect a roll-over and handle it specially.
    if (link->direction * (linkConnPrev->peerIdx - linkConn->peerIdx) > 0) {
      if (link->direction * (peerIdx - linkConnPrev->peerIdx) > 0 ||
          link->direction * (peerIdx - linkConn->peerIdx) < 0)
        break;
    } else { // Regular, monotonic case with the peerIdx value between two existing elements.
      if (link->direction * (peerIdx - linkConnPrev->peerIdx) > 0 &&
          link->direction * (peerIdx - linkConn->peerIdx) < 0)
        break;
    }
  } // for (linkConn)

  // The new element should be inserted after linkConnPrev (which is at index i-1).
  if (pLinkIdx)
    *pLinkIdx = i;
  if (pretend)
    goto exit;

  if (oldLinkConn) {
    if (i != oldLinkIdx) {
      // We already have the entry, but we need to move it to a new spot (which must be earlier in the list).
      assert(i < oldLinkIdx);
      // Remove oldLinkConn from its old spot.
      for (struct rasLinkConn* linkConn = linkConnPrev; linkConn->next; linkConn = linkConn->next) {
        if (linkConn->next == oldLinkConn) {
          linkConn->next = oldLinkConn->next;
          break;
        }
      } // for (linkConn)
      // Insert it at its new spot.
      oldLinkConn->next = linkConnPrev->next;
      linkConnPrev->next = oldLinkConn;
    } // if (i != oldLinkIdx)
    oldLinkConn->peerIdx = peerIdx;
    oldLinkConn->external = false;
  } else if (insert) {
    struct rasLinkConn* linkConn;
    NCCLCHECK(ncclCalloc(&linkConn, 1));
    if (linkConnPrev) {
      linkConn->next = linkConnPrev->next;
      linkConnPrev->next = linkConn;
    } else {
      assert(link->conns == nullptr); // We never add an element that would replace an existing primary.
      link->conns = linkConn;
      // linkConn->next is already nullptr.
    }
    linkConn->peerIdx = peerIdx;
    linkConn->conn = conn;
    linkConn->external = false;
    if (pLinkConn)
      *pLinkConn = linkConn;
  } // oldLinkConn == nullptr && insert

exit:
  return ncclSuccess;
}

// Adds an external entry in a RAS network link (or updates one, if already exists).
// conn *cannot* be nullptr when this function is invoked.
// peerIdx can be -1 if unknown (possible in case of a race condition between keepAlive and peers update).
// Note: there is some code duplication between this function and rasLinkConnAdd so changes to one of them
// may need to be sync'ed to the other one as well.  They used to be a single function that could do it all but the
// logic was extremely difficult to follow then.
static ncclResult_t rasLinkConnAddExternal(struct rasLink* link, struct rasConnection* conn, int peerIdx) {
  struct rasLinkConn* oldLinkConn = nullptr;
  struct rasLinkConn* linkConnPrev = nullptr;
  int i, oldLinkIdx = -1;

  assert(conn);
  oldLinkConn = rasLinkConnFind(link, conn, &oldLinkIdx);
  if (oldLinkConn) {
    if (oldLinkConn->peerIdx != -1)
      assert(oldLinkConn->peerIdx == peerIdx);

    if (oldLinkConn->peerIdx == peerIdx)
      goto exit; // Nothing more to do if both conn and peerIdx are up to date.  Note that we neither check nor
                 // update the value of external here.

    // Otherwise (oldLinkConn->peerIdx == -1 && peerIdx != -1) oldLinkConn, due to its -1 peerIdx, is in
    // a wrong place in the array -- we need to find the right spot.  oldLinkConn->peerIdx == -1 can only happen for
    // external connections.
  } // if (oldLinkConn)

  // Search for the right spot in the conns list.
  i = 0;
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConnPrev = linkConn, linkConn = linkConn->next, i++) {
    if (peerIdx == -1) {
      // We simply want to find the end of the list so that we can insert a new entry with -1 peerIdx there.
      continue;
    }
    if (linkConn->peerIdx == peerIdx) {
      // The exact linkConn element already exists.
      if (linkConn->conn)
        assert(linkConn->conn == conn);
      if (linkConn->conn == nullptr)
        linkConn->conn = conn;
      if (linkConn == link->conns) {
        // We received a connection from the remote peer that matches the primary connection we've been
        // waiting for.  This shouldn't trigger for external connections (rasLinkConnUpdate should be invoked first,
        // which will update the entry's conn, so rasLinkConnFind invoked at the top of this function should succeed),
        // but better safe than sorry...
        rasLinkSanitizeFallbacks(link);
      }
      goto exit;
    } // if (linkConn->peerIdx == peerIdx)

    // Ensure that the previous element is valid.
    if (linkConnPrev == nullptr)
      continue;
    // linkConns with peerIdx == -1 are stored at the end, so if we reach one of them, we are done.
    if (linkConn->peerIdx == -1)
      break;
    // Detect a roll-over and handle it specially.
    if (link->direction * (linkConnPrev->peerIdx - linkConn->peerIdx) > 0) {
      if (link->direction * (peerIdx - linkConnPrev->peerIdx) > 0 ||
          link->direction * (peerIdx - linkConn->peerIdx) < 0)
        break;
    } else { // Regular, monotonic case with the peerIdx value between two existing elements.
      if (link->direction * (peerIdx - linkConnPrev->peerIdx) > 0 &&
          link->direction * (peerIdx - linkConn->peerIdx) < 0)
        break;
    }
  } // for (linkConn)

  // The new element should be inserted after linkConnPrev (which is at index i-1).
  if (oldLinkConn) {
    if (i != oldLinkIdx) {
      // We already have the entry, but we need to move it to a new spot (which must be earlier in the list).
      assert(i < oldLinkIdx);
      INFO(NCCL_RAS, "RAS link %d: moving %sfallback connection with %s from %d to %d", link->direction,
           (oldLinkConn->external ? "external " : ""), ncclSocketToString(&conn->addr, rasLine), oldLinkIdx, i);
      // Remove oldLinkConn from its old spot.
      for (struct rasLinkConn* linkConn = linkConnPrev; linkConn->next; linkConn = linkConn->next) {
        if (linkConn->next == oldLinkConn) {
          linkConn->next = oldLinkConn->next;
          break;
        }
      } // for (linkConn)
      // Insert it at its new spot.
      oldLinkConn->next = linkConnPrev->next;
      linkConnPrev->next = oldLinkConn;
    } // if (i != oldLinkIdx)
    oldLinkConn->peerIdx = peerIdx;
    oldLinkConn->external = false;
  } else { // oldLinkConn == nullptr
    struct rasLinkConn* linkConn;
    NCCLCHECK(ncclCalloc(&linkConn, 1));
    if (linkConnPrev) {
      INFO(NCCL_RAS, "RAS link %d: adding external fallback connection %d with %s", link->direction, i,
           ncclSocketToString(&conn->addr, rasLine));
      linkConn->next = linkConnPrev->next;
      linkConnPrev->next = linkConn;
      linkConn->external = true;
    } else {
      INFO(NCCL_RAS, "RAS link %d: adding external fallback with %s as a new primary connection", link->direction,
           ncclSocketToString(&conn->addr, rasLine));
      linkConn->next = link->conns;
      link->conns = linkConn;
      linkConn->external = false; // Primary connections are never external.
    }
    linkConn->peerIdx = peerIdx;
    linkConn->conn = conn;
  } // oldLinkConn == nullptr

exit:
  return ncclSuccess;
}

// Updates an existing entry in a RAS network link, if any.
// Basically an easy-to-use variant of rasLinkConnAdd.
// For this function, conn cannot be a nullptr and peerIdx cannot be -1.
ncclResult_t rasLinkConnUpdate(struct rasLink* link, struct rasConnection* conn, int peerIdx) {
  assert(conn && peerIdx != -1);

  NCCLCHECK(rasLinkConnAdd(link, conn, peerIdx, /*pretend*/false, /*pLinkIdx*/nullptr, /*pLinkConn*/nullptr,
                           /*insert*/false));
  return ncclSuccess;
}

// Attempts to drop a connection from a link.
// If the optional external argument is true, it will drop a connection only if its external flag is set
// (otherwise the flag is ignored and a connection is always dropped if found).
static void rasLinkConnDrop(struct rasLink* link, const struct rasConnection* conn, bool external) {
  struct rasLinkConn* linkConnPrev = nullptr;
  int i = 0;
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConnPrev = linkConn, linkConn = linkConn->next, i++) {
    if (linkConn->conn == conn && (!external || linkConn->external)) {
      if (linkConnPrev) {
        INFO(NCCL_RAS, "RAS link %d: dropping %sfallback connection %d with %s",
             link->direction, (linkConn->external ? "external " : ""), i,
             ncclSocketToString(&conn->addr, rasLine));
        linkConnPrev->next = linkConn->next;
        free(linkConn);
      } else { // linkConnPrev == nullptr
        INFO(NCCL_RAS, "RAS link %d: dropping primary connection with %s",
             link->direction, ncclSocketToString(&conn->addr, rasLine));
        if (linkConn->next) {
          link->conns = linkConn->next;
          // Ensure that the conn becoming the primary is not marked as external (we don't want to lose it if
          // the remote peer loses interest in it).
          link->conns->external = false;
          if (link->conns->conn)
            INFO(NCCL_RAS, "RAS link %d: former fallback connection 1 with %s is the new primary",
                 link->direction, ncclSocketToString(&link->conns->conn->addr, rasLine));
          rasLinkSanitizeFallbacks(link);
          free(linkConn);
        } else { // linkConn->next == nullptr
          // We prefer the primary entry to always be present, even if empty.
          linkConn->peerIdx = -1;
          linkConn->conn = nullptr;
        } // linkConn->next == nullptr
      } // linkConnPrev == nullptr
      break;
    } // if (linkConn->conn == conn)
  } // for (linkConn)
}

// Checks if a given connection is a member of this link and if so, returns its link entry.
// Optionally returns the position of the connection in the conns list.
// Returns nullptr if connection not found.
static struct rasLinkConn* rasLinkConnFind(const struct rasLink* link, const struct rasConnection* conn,
                                           int* pLinkIdx) {
  int i = 0;
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next, i++) {
    if (linkConn->conn == conn) {
      if (pLinkIdx)
        *pLinkIdx = i;
      return linkConn;
    }
  }
  if (pLinkIdx)
    *pLinkIdx = -1;
  return nullptr;
}

// Invoked during RAS termination to release all the allocated resources.
void rasNetTerminate() {
  for (struct rasLinkConn* linkConn = rasNextLink.conns; linkConn;) {
    struct rasLinkConn* linkConnNext = linkConn->next;
    free(linkConn);
    linkConn = linkConnNext;
  }
  for (struct rasLinkConn* linkConn = rasPrevLink.conns; linkConn;) {
    struct rasLinkConn* linkConnNext = linkConn->next;
    free(linkConn);
    linkConn = linkConnNext;
  }
  rasNextLink.conns = rasPrevLink.conns = nullptr;
  rasNextLink.lastUpdatePeersTime = rasPrevLink.lastUpdatePeersTime = 0;

  for (struct rasConnection* conn = rasConnsHead; conn;) {
    struct rasConnection* connNext = conn->next;
    rasConnTerminate(conn);
    conn = connNext;
  }
  // rasConnsHead and rasConnsTail are taken care of by rasConnTerminate().

  for (struct rasSocket* sock = rasSocketsHead; sock;) {
    struct rasSocket* sockNext = sock->next;
    rasSocketTerminate(sock);
    sock = sockNext;
  }
  // rasSocketsHead and rasSocketsTail are taken care of by rasSocketTerminate().
}
