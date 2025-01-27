/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RAS_INTERNAL_H_
#define NCCL_RAS_INTERNAL_H_

#define NCCL_RAS_CLIENT_PORT 28028
#define NCCL_RAS_CLIENT_PROTOCOL 2

#define RAS_COLLECTIVE_LEG_TIMEOUT_SEC 5
#define RAS_COLLECTIVE_EXTRA_TIMEOUT_SEC RAS_COLLECTIVE_LEG_TIMEOUT_SEC

// End of the client section; everything below is meant for the NCCL threads only.
#ifndef NCCL_RAS_CLIENT

#include <mutex>

#include "nccl.h"
#include "ras.h"
#include "socket.h"
#include "utils.h"

// Type of a RAS network or client message.
typedef enum {
  RAS_MSG_CONNINIT = 1,
  RAS_MSG_CONNINITACK = 2,
  RAS_MSG_KEEPALIVE = 3,
  RAS_MSG_PEERSUPDATE = 4,
  RAS_MSG_COLLREQ = 5,
  RAS_MSG_COLLRESP = 6,
} rasMsgType;

// Type of a RAS network collective message.
typedef enum {
  RAS_MSG_NONE = 0,
  RAS_BC_DEADPEER = 1,
  // Broadcast operations above this line; collective operations below (1000 is the demarcation line).
  RAS_COLL_CONNS = 1001, // Collect data about all RAS connections.
  RAS_COLL_COMMS = 1002, // Collect data about all communicators.
} rasCollectiveType;

// Payload of a collective request message (RAS_MSG_COLLREQ).
struct rasCollRequest {
  union ncclSocketAddress rootAddr;
  uint64_t rootId;

  int64_t timeout;
  rasCollectiveType type;
  union {
    struct {
      union ncclSocketAddress addr;
    } deadPeer;
    struct {
    } conns;
    struct {
    } comms;
  };
};

// Payload of a collective response message (RAS_MSG_COLLRESP).
struct rasCollResponse {
  union ncclSocketAddress rootAddr;
  uint64_t rootId;

  int nLegTimeouts; // If >0, indicates incomplete data.
  int nPeers;
  int nData; // Size of data in bytes.
  union ncclSocketAddress peers[0]; // Variable length.
  // The peersAddrs array is followed by:
  //alignas(int64_t) char data[0]; // Variable length, collective-dependent.
};

// Describes a peer NCCL process.  Every RAS thread keeps an (identical) array of them, one entry for each
// NCCL process.
struct rasPeerInfo {
  union ncclSocketAddress addr;
  pid_t pid;
  uint64_t cudaDevs; // Bitmask.  This is for local devices so 64 bits is enough.
  uint64_t nvmlDevs; // Same, but not affected by CUDA_VISIBLE_DEVICES.
};

// Describes a RAS message.  Every message is preceded by a (32-bit) message length.  All data in the host
// byte order.  Depending on the message type, the length of the message will vary.
struct rasMsg {
  rasMsgType type;
  union {
    struct {
      int ncclVersion;
      union ncclSocketAddress listeningAddr;
      uint64_t peersHash;
      uint64_t deadPeersHash;
    } connInit; // Sent by the connecting side as the first message.
    struct {
      int nack; // If non-0, we should stop trying to reconnect.
    } connInitAck; // Response from the accepting side to the above.
    struct {
      uint64_t peersHash;
      uint64_t deadPeersHash;
      int linkMask; // What links at the destination peer should the connection be part of
                    // (bit 0: nextLink; bit 1: prevLink).
      struct timespec realTime; // Wallclock time at the source, for statistical purposes (in principle there's
                                // no guarantee that the nodes have synchronized clocks so we can't really rely
                                // on it for anything important)..
      int nack; // If non-0, it means that this message is a response to an unexpected keepAlive message.
    } keepAlive;
    struct {
      uint64_t peersHash;
      uint64_t deadPeersHash;
      int nPeers;
      int nDeadPeers;
      struct rasPeerInfo peers[0]; // Variable length.
      // The peers array is followed by the following:
      //union ncclSocketAddress deadPeers[0]; // Variable length.
    } peersUpdate;
    struct {
      int protocol; // Protocol version, sent to the client.
    } clientInit;
    struct {
      int nData;
      char data[0]; // Variable length.
    } clientDump;
    struct rasCollRequest collReq; // Variable length.
    struct rasCollResponse collResp; // Variable length.
  };
};

// Returns the size of the collective portion of a collective request message.
static inline size_t rasCollDataLength(rasCollectiveType type) {
  struct rasCollRequest* data;
  switch (type) {
    case RAS_BC_DEADPEER:
      return offsetof(struct rasCollRequest, deadPeer) + sizeof(data->deadPeer);
    case RAS_COLL_CONNS:
      return offsetof(struct rasCollRequest, conns) + sizeof(data->conns);
    case RAS_COLL_COMMS:
      return offsetof(struct rasCollRequest, comms) + sizeof(data->comms);
    case RAS_MSG_NONE:
      return 0;
  };
  return 0;
}

// Returns the size for a message of a particular type.
static inline size_t rasMsgLength(rasMsgType type, rasCollectiveType collType = RAS_MSG_NONE) {
  struct rasMsg* msg;
  switch (type) {
    case RAS_MSG_CONNINIT:
      return offsetof(struct rasMsg, connInit) + sizeof(msg->connInit);
    case RAS_MSG_CONNINITACK:
      return offsetof(struct rasMsg, connInitAck) + sizeof(msg->connInitAck);
    case RAS_MSG_KEEPALIVE:
      return offsetof(struct rasMsg, keepAlive) + sizeof(msg->keepAlive);
    case RAS_MSG_PEERSUPDATE:
      return offsetof(struct rasMsg, peersUpdate) + sizeof(msg->peersUpdate);
    case RAS_MSG_COLLREQ:
      return offsetof(struct rasMsg, collReq) + rasCollDataLength(collType);
    case RAS_MSG_COLLRESP:
      return offsetof(struct rasMsg, collResp) + sizeof(msg->collResp);
  };
  return 0;
}

// How much to enlarge any RAS array by if we run out of space.
#define RAS_INCREMENT 4

// Our clock has nanosecond resolution.
#define CLOCK_UNITS_PER_SEC 1000000000L

// Keep-alive messages are sent no sooner than a second after the last message was sent down a particular connection.
#define RAS_KEEPALIVE_INTERVAL (1*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// If no message arrives in 5 seconds via a particular connection that uses keep-alive messages, generate a warning
// and try alternative connections.
#define RAS_KEEPALIVE_TIMEOUT_WARN (5*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// Abort a socket that uses keep-alive messages if no message arrives in 20 seconds.
// We will try to re-establish communication via that connection (until RAS_PEER_DEAD_TIMEOUT).
#define RAS_KEEPALIVE_TIMEOUT_ERROR RAS_STUCK_TIMEOUT

// Retry connecting on failing sockets (ECONNREFUSED, etc.) once a second.
#define RAS_CONNECT_RETRY (1*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// If we can't connect in 5 seconds, we generate a warning and try alternative connections.
#define RAS_CONNECT_WARN RAS_KEEPALIVE_TIMEOUT_WARN

// Abort a busy socket (one we are trying to send on, or one that was being established) if there's been
// no sign of progress in 20 second.  We will try to re-establish communication (up to RAS_PEER_DEAD_TIMEOUT).
#define RAS_STUCK_TIMEOUT (20*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// Terminate ad-hoc connections that have not been used in 60 seconds.
#define RAS_IDLE_TIMEOUT (60*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// If the socket is closed by peer within 5 seconds from the idle timeout, do not attempt to re-establish.
#define RAS_IDLE_GRACE_PERIOD (5*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// Declare a peer as dead and don't retry communicating with it if we couldn't reach it for 60 seconds.
#define RAS_PEER_DEAD_TIMEOUT (60*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// Abort a leg of a collective operation if the response takes more than 5 seconds to arrive *and* one of the
// connections experiences delays.
#define RAS_COLLECTIVE_LEG_TIMEOUT (RAS_COLLECTIVE_LEG_TIMEOUT_SEC*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// Abort a whole collective operation after at most RAS_COLLECTIVE_LEG_TIMEOUT+RAS_COLLECTIVE_EXTRA_TIMEOUT (10s).
#define RAS_COLLECTIVE_EXTRA_TIMEOUT (RAS_COLLECTIVE_EXTRA_TIMEOUT_SEC*CLOCK_UNITS_PER_SEC*ncclParamRasTimeoutFactor())

// Structure used for tracking the progress of sending a RAS message.
struct rasMsgMeta {
  struct rasMsgMeta* next;
  int64_t enqueueTime;
  int offset; // Progress sending the message (including the message size itself (an int, which is sent first)).
  int length; // Length of the message (*excluding* the message size).
  struct rasMsg msg; // Variable length.
};

// Describes an ongoing collective RAS operation (apart from broadcasts, which don't need a response).
// For every collective operation, each participating RAS thread will create its own.
struct rasCollective {
  union ncclSocketAddress rootAddr;
  uint64_t rootId;

  rasCollectiveType type;

  int64_t timeout;
  bool timeoutWarned;

  int64_t startTime; // For timeout calculations.
  int fromConnIdx; // The connection we received the request from.

  int* fwdConns; // Indices of the connections we forwarded the request to; replaced by -1 as the responses arrive.
  int nFwdSent; // Count of the above (local process only).
  int nFwdRecv; // Count of the responses received or timeouts (local process only).

  int nLegTimeouts; // Collective (from this process and the responses we received).

  union ncclSocketAddress* peers; // Collective (from this process and the responses we received).
  int nPeers;

  char* data; // Collective (from this process and the responses we received).
  int nData;
};

// Collective data in RAS_COLL_CONNS responses.
struct rasCollConns {
  int64_t travelTimeMin;
  int64_t travelTimeMax;
  int64_t travelTimeSum;
  int64_t travelTimeCount;
  int nConns;
  int nNegativeMins;
  struct negativeMin {
    union ncclSocketAddress source;
    union ncclSocketAddress dest;
    int64_t travelTimeMin;
  } negativeMins[0]; // Variable length.
};

// Collective data in RAS_COLL_COMMS responses.
struct rasCollComms {
  int nComms;
  struct comm {
    uint64_t commHash;
    int commNRanks;
    int nRanks; // number of elements in the array below, *not* in the communicator.
    struct rank {
      int commRank;
      int peerIdx; // Index within rasCollective->peers, *not* rasPeers.
      uint64_t collOpCount;
      struct {
        ncclResult_t initState:4;
        ncclResult_t asyncError:4;
        bool finalizeCalled:1;
        bool destroyFlag:1;
        bool abortFlag:1;
      } status;
      char cudaDev;
      char nvmlDev;
    } ranks[0]; // Variable length. Sorted by commRank.  Optimized for 1 GPU/process.
  } comms[0]; // Variable length. Sorted by commHash.
};

// Holds data needed to keep track of a connection belonging to a RAS network link (either the primary one
// or one of the fallbacks).
struct rasLinkConn {
  int peerIdx; // Index in the rasPeers array of the peer this entry describes.  Could be -1 (an entry initiated
               // by an as of yet unknown peer -- should be a temporary situation that resolves via peer updates).
  int connIdx; // Index in the rasConns array of the connection to the above peer.  Could be -1 (a placeholder
               // for a connection to be started by the remote peer).
  bool external; // true if the entry exists only due to an external request (requested by a remote peer, most
                 // likely as part of fault recovery).  Such connections are kept as fallbacks even if there's a
                 // valid primary connection, in order to ensure that keep-alive messages are sent.
};

// Describes a link that forms the backbone of the RAS network.  Links focus on direction (previous/next in
// case of 1-D topology) rather than a particular destination.  The are implemented using rasConnections, but
// they are persistent through the life of the RAS threads, whereas rasConnections can be terminated if the RAS
// network is reconfigured or a peer dies.
struct rasLink {
  int direction; // 1 for nextLink, -1 for prevLink.

  // Index 0 is the primary connection; any additional ones are fallbacks (that get created if we are having
  // problems with the primary connection).  The elements are de-facto ordered (highest-preference ones have
  // the lowest indices).
  struct rasLinkConn* conns;
  int nConns;
  int connsSize; // Array size; could be larger than nConns.

  // Keep track of a timeout in case we did not create a connection during the last peers update (because we expect
  // the peer on the other side to do so) but that peer failed to initiate.
  int64_t lastUpdatePeersTime;
};

// Describes a connection to another peer on the RAS network.  It is meant to be more persistent than a volatile
// socket (described by the rasSocket structure), which can be affected by transient network issues.
struct rasConnection {
  bool inUse;

  union ncclSocketAddress addr;

  // Index of the current rasSocket in the rasSockets array.  Note that multiple rasSocket entries may point back
  // to a single entry here, for sockets that are in the process of being terminated and re-established.
  // We use indices, not pointers, because the arrays holding these structures can be re-alloced at run time.
  // -1 if there is no such socket.
  int sockIdx;

  // We keep the rasPeersHash of remote connections to minimize the number of needless exchanges.
  // There is a subtle difference in the meaning of lastSentPeersHash and lastRecvPeersHash.
  // lastSentPeersHash stores *our* rasPeersHash from the time we last sent a peers *update* through this connection
  // (which is different than sending just the hash, like we do in KEEPALIVE, etc.).
  // lastRecvPeersHash stores the latest known rasPeersHash of the peer (received via KEEPALIVE, etc.).
  uint64_t lastSentPeersHash;
  uint64_t lastRecvPeersHash;

  // Same but for rasDeadPeersHash.
  uint64_t lastSentDeadPeersHash;
  uint64_t lastRecvDeadPeersHash;

  // Queue of messages to send.
  struct ncclIntruQueue<struct rasMsgMeta, &rasMsgMeta::next> sendQ;

  // Used for keeping track of timeouts that may extend beyond the lifetime of a socket.
  // The timeout starts when the connection is being created (and is turned off when the initialization is completed
  // successfully) or when we detect a problem, such as a socket timeout (in the latter case, we may need to
  // retroactively calculate the start time).
  // A value of 0 indicates that they are not currently in use.
  int64_t startRetryTime;
  int64_t lastRetryTime;

  bool experiencingDelays; // A flag indicating that the connection is currently subject to RAS_KEEPALIVE_TIMEOUT_WARN
                           // or RAS_CONNECT_WARN timeout.  If set, the warnings have been issued and the fallbacks
                           // have been initiated if needed.
  bool linkFlag; // Used within rasNet* calls to mark whether this connection was already handled when iterating over
                 // multiple links (since a connection can belong to more than one link).
  // The below four fields are for statistical purposes only.
  int64_t travelTimeMin;
  int64_t travelTimeMax;
  int64_t travelTimeSum;
  int64_t travelTimeCount;
};

// Status of a RAS socket.
typedef enum {
  RAS_SOCK_CLOSED = 0,
  RAS_SOCK_CONNECTING = 1,
  RAS_SOCK_HANDSHAKE = 2,
  RAS_SOCK_READY = 3,
  RAS_SOCK_TERMINATING = 4
} rasSocketStatus;

// Describes a socket implementing communication between two peers.
struct rasSocket {
  struct ncclSocket sock;

  rasSocketStatus status;

  int pfd; // Index in the rasPfds array.

 // Index of the corresponding entry in the rasConns array.
  // We use indices, not pointers, because the arrays holding these structures can be re-alloced at run time.
  // -1 if there is no connection (normal condition on the accept side before the connInit message).
  int connIdx;

  int64_t createTime;
  int64_t lastSendTime;
  int64_t lastRecvTime;

  // Data on the message currently being received.
  int recvOffset;
  int recvLength;
  struct rasMsg* recvMsg;
};

// Status of a RAS client.
typedef enum {
  RAS_CLIENT_CLOSED = 0,
  RAS_CLIENT_CONNECTED = 1,
  RAS_CLIENT_INIT = 2,
  RAS_CLIENT_CONNS = 3,
  RAS_CLIENT_COMMS = 4,
  RAS_CLIENT_FINISHED = 99
} rasClientStatus;

// Describes a RAS client.
struct rasClient {
  int sock;

  rasClientStatus status;

  int pfd; // Index in the rasPfds array.

  char recvBuffer[1024];
  int recvOffset;

  // Queue of messages to send.
  struct ncclIntruQueue<struct rasMsgMeta, &rasMsgMeta::next> sendQ;

  int verbose;
  int64_t timeout;

  // State stored during asynchronous operations such as collectives.
  int collIdx; // Index to the onging rasCollective.
};


// ras.cc
extern struct pollfd* rasPfds;
extern struct ncclSocket rasNetListeningSocket;
extern std::mutex ncclCommsMutex;
extern struct ncclComm** ncclComms;
extern int nNcclComms;
extern  bool ncclCommsSorted;
extern char rasLine[SOCKET_NAME_MAXLEN+1];

int64_t ncclParamRasTimeoutFactor();
ncclResult_t rasMsgAlloc(struct rasMsg** msg, size_t msgLen);
void rasMsgFree(struct rasMsg* msg);
void rasConnEnqueueMsg(struct rasConnection* conn, struct rasMsg* msg, size_t msgLen, bool front = false);
ncclResult_t rasConnSendMsg(struct rasConnection* conn, int* closed, bool* allSent);
ncclResult_t rasMsgRecv(struct rasSocket* sock, struct rasMsg** msg, int* closed);
ncclResult_t rasMsgHandle(struct rasMsg* msg, struct rasSocket* sock);
void rasMsgHandleBCDeadPeer(const struct rasCollRequest* req, bool* pDone);
ncclResult_t rasGetNewPollEntry(int* index);


// rasnet.cc
extern struct rasLink rasNextLink, rasPrevLink;
extern struct rasConnection* rasConns;
extern int nRasConns;
extern struct rasSocket *rasSockets;
extern int nRasSockets;

ncclResult_t getNewConnEntry(struct rasConnection** pConn);
ncclResult_t rasConnCreate(const union ncclSocketAddress* addr, int* pConnIdx);
int rasConnFind(const union ncclSocketAddress* addr);
void rasConnsHandleTimeouts(int64_t now, int64_t* nextWakeup);
void rasConnDisconnect(const union ncclSocketAddress* addr);
ncclResult_t rasNetAcceptNewSocket();
void rasSocksHandleTimeouts(int64_t now, int64_t* nextWakeup);
void rasSocketTerminate(struct rasSocket* sock, bool finalize = false, uint64_t startRetryOffset = 0,
                        bool retry = true);
void rasSockEventLoop(int sockIdx, int pollIdx);
void rasNetHandleTimeouts(int64_t now, int64_t* nextWakeup);
ncclResult_t rasMsgHandleKeepAlive(const struct rasMsg* msg, struct rasSocket* sock);
ncclResult_t rasLinkUpdateConn(struct rasLink* link, int connIdx, int peerIdx, bool external = false,
                               bool insert = false, bool pretend = false, int* pLinkIdx = nullptr);

// peers.cc
extern struct rasPeerInfo* rasPeers;
extern int nRasPeers;
extern uint64_t rasPeersHash;
extern union ncclSocketAddress* rasDeadPeers;
extern int nRasDeadPeers;
extern uint64_t rasDeadPeersHash;

ncclResult_t rasLocalHandleAddRanks(struct rasRankInit* ranks, int nranks);
int rasPeerFind(const union ncclSocketAddress* addr);
ncclResult_t rasConnSendPeersUpdate(struct rasConnection* conn, const struct rasPeerInfo* peers, int nPeers);
ncclResult_t rasMsgHandlePeersUpdate(struct rasMsg* msg, struct rasSocket* sock);
int rasLinkCalculatePeer(const struct rasLink* link, int peerIdx, bool isFallback = false);
ncclResult_t rasPeerDeclareDead(const union ncclSocketAddress* addr);
bool rasPeerIsDead(const union ncclSocketAddress* addr);
int ncclSocketsCompare(const void* p1, const void* p2);
bool ncclSocketsSameNode(const union ncclSocketAddress* a1, const union ncclSocketAddress* a2);


// collectives.cc
extern struct rasCollective* rasCollectives;

void rasCollReqInit(struct rasCollRequest* req);
ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, size_t reqLen, bool* pAllDone = nullptr,
                               int* pCollIdx = nullptr, int fromConnIdx = -1);
ncclResult_t rasMsgHandleCollReq(struct rasMsg* msg, struct rasSocket* sock);
ncclResult_t rasMsgHandleCollResp(struct rasMsg* msg, struct rasSocket* sock);
void rasCollsPurgeConn(int connIdx);
void rasCollFree(struct rasCollective* coll);
void rasCollsHandleTimeouts(int64_t now, int64_t* nextWakeup);

// client_support.cc
extern int rasClientListeningSocket;
extern struct rasClient* rasClients;
extern int nRasClients;
ncclResult_t rasClientInitSocket();
ncclResult_t rasClientAcceptNewSocket();
ncclResult_t rasClientResume(struct rasCollective* coll);
void rasClientEventLoop(int clientIdx, int pollIdx);
const char* rasGpuDevsToString(uint64_t cudaDevs, uint64_t nvmlDevs, char* buf, size_t size);

#endif // !NCCL_RAS_CLIENT

#endif // !NCCL_RAS_INTERNAL_H_
