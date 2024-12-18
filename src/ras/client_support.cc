/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NDEBUG // Comment out duriyng development only!
#include <cassert>
#include <cstdarg>
#include <cstddef>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "utils.h"
#include "ras_internal.h"

// Outlier count above which we don't print individual details about each of them.
#define RAS_CLIENT_DETAIL_THRESHOLD 10
// Fraction of the count of the total above which we don't consider another set to be an outlier.
#define RAS_CLIENT_OUTLIER_FRACTION 0.25
// Fraction of the count of the total below which a set is considered to be an outlier.
#define RAS_CLIENT_VERBOSE_OUTLIER_FRACTION 0.5

#define STR2(v) #v
#define STR(v) STR2(v)

// The RAS client listening socket of this RAS thread (normally port 28028).
int rasClientListeningSocket = -1;

// Auxiliary structure used when processing the results.  Helps with statistics gathering and sorting.
struct rasValCount {
  uint64_t value; // The observed value.
  int count; // The number of occurences of this value in the results.
  int firstIdx; // The index of the first occurence of this value in the results.
};

// Used in rasAuxComm below.  The values are bitmasks so that they can be combined.
typedef enum {
  RAS_ACS_UNKNOWN = 1, // Set if a peer did not provide info about a given communicator.
  RAS_ACS_INIT = 2,
  RAS_ACS_RUNNING = 4,
  RAS_ACS_FINALIZE = 8,
  RAS_ACS_ABORT = 16
} rasACStatus;

// Used in rasAuxComm below.  The values are bitmasks so that they can be combined (with the exception of RAS_ACE_OK).
typedef enum {
  RAS_ACE_OK = 0,
  RAS_ACE_MISMATCH = 1,
  RAS_ACE_ERROR = 2,
  RAS_ACE_INCOMPLETE = 4
} rasACError;

// Auxiliary structure used when processing the results.  Helps with sorting and includes additional statistics
// on the number of peers and nodes for a communicator.
struct rasAuxComm {
  struct rasCollComms::comm* comm;
  int nPeers;
  int nNodes;
  int ranksPerNodeMin;
  int ranksPerNodeMax;
  unsigned int status; // Bitmask of rasACStatus values.
  unsigned int errors; // Bitmask of rasACError values.
  uint64_t firstCollOpCount; // collOpCount of the first rank, to compare against.
};

// Connected RAS clients.
struct rasClient* rasClients;
int nRasClients;

// Minimum byte count to increment the output buffer size by if it's too small.
#define RAS_OUT_INCREMENT 4096

// Internal buffer for storing the formatted results.
static char* rasOutBuffer = nullptr;
static int nRasOutBuffer = 0; // Does _not_ include the terminating '\0' (which _is_ present in the buffer).
static int rasOutBufferSize = 0;

// We use them all over the place; no point in wasting the stack...
static char lineBuf[1024]; // Temporary buffer used for printing at most 10 (RAS_CLIENT_DETAIL_THRESHOLD) rank numbers
                           // or for printing the local GPU devices, which can't be more than 64 (NCCL_MAX_LOCAL_RANKS)
                           // small numbers (times two if the NVML mask is different than the CUDA mask).
                           // Still, 1024 should normally be plenty (verbose output may make things more difficult,
                           // but we do check for overflows, so it will just be trimmed).

static ncclResult_t getNewClientEntry(struct rasClient** pClient);
static void rasClientEnqueueMsg(struct rasClient* client, char* msg, size_t msgLen);
static void rasClientTerminate(struct rasClient* client);

static ncclResult_t rasClientRun(struct rasClient* client);
static ncclResult_t rasClientRunInit(struct rasClient* client);
static ncclResult_t rasClientRunConns(struct rasClient* client);
static ncclResult_t rasClientRunComms(struct rasClient* client);
static void rasClientBreakDownErrors(struct rasClient* client, struct rasCollComms::comm* comm,
                                     const int* peerIdxConv, int ncclErrors[ncclNumResults], bool isAsync = false);

static void rasOutAppend(const char* format, ...) __attribute__ ((format(printf, 1, 2)));
static void rasOutExtract(char* buffer);
static int rasOutLength();
static void rasOutReset();

static int rasPeersNGpuCompare(const void* e1, const void* e2);
static int rasPeersNProcsCompare(const void* e1, const void* e2);
static int rasPeersHostPidCompare(const void* e1, const void* e2);
static int ncclSocketsHostCompare(const void* p1, const void* p2);
static int rasValCountsCompareRev(const void* p1, const void* p2);
static int rasAuxCommsCompareRev(const void* p1, const void* p2);
static int rasCommRanksPeerCompare(const void* p1, const void* p2);
static int rasCommRanksCollOpCompare(const void* p1, const void* p2);

static const char* rasCommRankGpuToString(const struct rasCollComms::comm::rank* rank, char* buf, size_t size);
static const char* ncclErrorToString(ncclResult_t err);
static const char* ncclSocketToHost(const union ncclSocketAddress* addr, char* buf, size_t size);
static bool rasCountIsOutlier(int count, bool verbose, int totalCount = -1);


///////////////////////////////////
// General rasClients functions. //
///////////////////////////////////

// Creates a listening socket for clients to connect to.
ncclResult_t rasClientInitSocket() {
  ncclResult_t ret = ncclSuccess;
  const char* clientAddr = "localhost:" STR(NCCL_RAS_CLIENT_PORT);
  union ncclSocketAddress addr;
  const int opt = 1;
  if (const char* env = ncclGetEnv("NCCL_RAS_ADDR"))
    clientAddr = env;
  NCCLCHECKGOTO(ncclSocketGetAddrFromString(&addr, clientAddr), ret, fail);
  SYSCHECKGOTO(rasClientListeningSocket = socket(addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, fail);
  SYSCHECKGOTO(setsockopt(rasClientListeningSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)),
               "setsockopt", ret, fail);
#if defined(SO_REUSEPORT)
  SYSCHECKGOTO(setsockopt(rasClientListeningSocket, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)),
               "setsockopt", ret, fail);
#endif
  SYSCHECKGOTO(bind(rasClientListeningSocket, &addr.sa, (addr.sa.sa_family == AF_INET ? sizeof(struct sockaddr_in) :
                                                          sizeof(struct sockaddr_in6))), "bind", ret, fail);
  SYSCHECKGOTO(listen(rasClientListeningSocket, 16384), "listen", ret, fail);
  INFO(NCCL_INIT|NCCL_RAS, "RAS client listening socket at %s", ncclSocketToString(&addr, rasLine));
exit:
  return ret;
fail:
  INFO(NCCL_INIT|NCCL_RAS, "RAS failed to establish a client listening socket at %s", clientAddr);
  if (rasClientListeningSocket != -1) {
    (void)close(rasClientListeningSocket);
    rasClientListeningSocket = -1;
  }
  goto exit;
}

// Accepts a new RAS client connection.  The acceptance process may need to continue in the main event loop.
ncclResult_t rasClientAcceptNewSocket() {
  ncclResult_t ret = ncclSuccess;
  struct rasClient* client = nullptr;
  union ncclSocketAddress addr;
  socklen_t addrlen = sizeof(addr);
  int flags;

  NCCLCHECKGOTO(getNewClientEntry(&client), ret, fail);

  SYSCHECKGOTO(client->sock = accept(rasClientListeningSocket, (struct sockaddr*)&addr, &addrlen), "accept", ret, fail);

  SYSCHECKGOTO(flags = fcntl(client->sock, F_GETFL), "fcntl", ret, fail);
  SYSCHECKGOTO(fcntl(client->sock, F_SETFL, flags | O_NONBLOCK), "fcntl", ret, fail);

  NCCLCHECKGOTO(rasGetNewPollEntry(&client->pfd), ret, fail);
  rasPfds[client->pfd].fd = client->sock;
  rasPfds[client->pfd].events = POLLIN;
  client->status = RAS_CLIENT_CONNECTED;
exit:
  return ret;
fail:
  if (client && client->sock != -1)
    (void)close(client->sock);
  goto exit;
}

// Returns the index of the first available entry in the rasClients array, enlarging the array if necessary.
static ncclResult_t getNewClientEntry(struct rasClient** pClient) {
  struct rasClient* client;
  int i;
  for (i = 0; i < nRasClients; i++)
    if (rasClients[i].status == RAS_CLIENT_CLOSED)
      break;
  if (i == nRasClients) {
    NCCLCHECK(ncclRealloc(&rasClients, nRasClients, nRasClients+RAS_INCREMENT));
    nRasClients += RAS_INCREMENT;
  }

  client = rasClients+i;
  memset(client, '\0', sizeof(*client));
  client->sock = client->pfd = -1;
  ncclIntruQueueConstruct(&client->sendQ);
  client->timeout =  RAS_COLLECTIVE_LEG_TIMEOUT;
  client->collIdx = -1;

  *pClient = client;
  return ncclSuccess;
}

// Allocates a message of the desired length for sending.
// Behind the scenes uses rasMsgAlloc.
// Must use rasClientFreeMsg to free.
static ncclResult_t rasClientAllocMsg(char** msg, size_t msgLen) {
  return rasMsgAlloc((struct rasMsg**)msg, msgLen);
}

// To be used only with messages allocated with rasClientAllocMsg, i.e., for messages meant for sending.
static void rasClientFreeMsg(char* msg) {
  rasMsgFree((struct rasMsg*)msg);
}

// Enqueues a message for sending to a RAS client.  The message *must* have been allocated using rasClientAllocMsg.
static void rasClientEnqueueMsg(struct rasClient* client, char* msg, size_t msgLen) {
  // Get to the metadata of this message.
  struct rasMsgMeta* meta = (struct rasMsgMeta*)((char*)msg - offsetof(struct rasMsgMeta, msg));
  meta->offset = 0;
  meta->length = msgLen;
  ncclIntruQueueEnqueue(&client->sendQ, meta);
  assert(client->status != RAS_CLIENT_CLOSED && client->status < RAS_CLIENT_FINISHED);
  rasPfds[client->pfd].events |= POLLOUT;
}

// Terminates a connection with a RAS client.
static void rasClientTerminate(struct rasClient* client) {
  (void)close(client->sock);
  client->sock = -1;
  client->status = RAS_CLIENT_CLOSED;
  rasPfds[client->pfd].fd = -1;
  rasPfds[client->pfd].events = rasPfds[client->pfd].revents = 0;
  client->pfd = -1;
  while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&client->sendQ)) {
    free(meta);
  }
}


//////////////////////////////////////////////////////////////////////
// Functions related to the asynchronous operations of RAS clients. //
//////////////////////////////////////////////////////////////////////

// Invoked when an asynchronous operation that a client was waiting on completes.  Finds the right client and
// reinvokes rasClientRun.
ncclResult_t rasClientResume(struct rasCollective* coll) {
  int collIdx = coll-rasCollectives;
  int i;
  struct rasClient* client = nullptr;
  for (i = 0; i < nRasClients; i++) {
    client = rasClients+i;
    if (client->status != RAS_CLIENT_CLOSED && client->collIdx == collIdx) {
      break;
    }
  }
  if (i == nRasClients) {
    INFO(NCCL_RAS, "RAS failed to find a matching client!");
    rasCollFree(coll);
    goto exit;
  }

  NCCLCHECK(rasClientRun(client));
exit:
  return ncclSuccess;
}

// Handles a ready client FD from the main event loop.
void rasClientEventLoop(int clientIdx, int pollIdx) {
  struct rasClient* client = rasClients+clientIdx;
  bool closed = false;

  if (client->status == RAS_CLIENT_CONNECTED) {
    char* cmd;
    char* cmdEnd;
    if (rasPfds[pollIdx].revents & POLLIN) {
      if (client->recvOffset < sizeof(client->recvBuffer)) {
        ssize_t nRecv;
        nRecv = recv(client->sock, client->recvBuffer+client->recvOffset,
                     sizeof(client->recvBuffer) - client->recvOffset, MSG_DONTWAIT);
        if (nRecv == 0) {
          closed = true;
        } else if (nRecv == -1) {
          if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
            if (errno == ECONNRESET)
              INFO(NCCL_RAS, "RAS socket closed by the client on receive; terminating it");
            else
              INFO(NCCL_RAS, "RAS unexpected error from recv; terminating the client socket");
            closed = true;
          }
        } else { // nRecv > 0
          client->recvOffset += nRecv;
        }
      } else { // client->recvOffset == sizeof(client->recvBuffer)
        rasPfds[client->pfd].events &= ~POLLIN; // No room to receive for now.
      }
    } // if (rasPfds[pollIdx].revents & POLLIN)
    if (closed) {
      rasClientTerminate(client);
      return;
    }
    cmd = client->recvBuffer;
    while ((cmdEnd = (char*)memchr(cmd, '\n', client->recvOffset - (cmd-client->recvBuffer))) != nullptr) {
      char* msg;
      int msgLen;
      *cmdEnd = '\0'; // Replaces '\n'.
      if (cmdEnd > cmd && cmdEnd[-1] == '\r')
        cmdEnd[-1] = '\0'; // Replaces '\r' (e.g., in case of a telnet connection).

      if (strncasecmp(cmd, "client protocol ", strlen("client protocol ")) == 0) {
        // We ignore the protocol version for now; we just send our version back.
        snprintf(rasLine, sizeof(rasLine), "SERVER PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n");
        msgLen = strlen(rasLine);
        if (rasClientAllocMsg(&msg, msgLen) != ncclSuccess) {
          rasClientTerminate(client);
          return;
        }
        // We don't copy the terminating '\0', hence memcpy rather than strcpy.
        memcpy(msg, rasLine, msgLen);
        rasClientEnqueueMsg(client, msg, msgLen);
      } else if (strncasecmp(cmd, "timeout ", strlen("timeout ")) == 0) {
        char* endPtr = nullptr;
        int timeout = strtol(cmd+strlen("timeout "), &endPtr, 10);
        if (timeout < 0 || !endPtr || *endPtr != '\0') {
          snprintf(rasLine, sizeof(rasLine), "ERROR: Invalid timeout value %s\n", cmd+strlen("timeout "));
        } else {
          client->timeout = timeout * CLOCK_UNITS_PER_SEC;
          strcpy(rasLine, "OK\n");
        }
        msgLen = strlen(rasLine);
        if (rasClientAllocMsg(&msg, msgLen) != ncclSuccess) {
          rasClientTerminate(client);
          return;
        }
        // We don't copy the terminating '\0', hence memcpy rather than strcpy.
        memcpy(msg, rasLine, msgLen);
        rasClientEnqueueMsg(client, msg, msgLen);
      } else if (strcasecmp(cmd, "status") == 0) {
        client->status = RAS_CLIENT_INIT;
        (void)rasClientRun(client);
      } else if (strcasecmp(cmd, "verbose status") == 0) {
        client->status = RAS_CLIENT_INIT;
        client->verbose = 1;
        (void)rasClientRun(client);
      } else {
        snprintf(rasLine, sizeof(rasLine), "ERROR: Unknown command %s\n", cmd);
        msgLen = strlen(rasLine);
        if (rasClientAllocMsg(&msg, msgLen) != ncclSuccess)
          return; // It should be non-fatal if we don't return a response...
        // We don't copy the terminating '\0', hence memcpy rather than strcpy.
        memcpy(msg, rasLine, msgLen);
        rasClientEnqueueMsg(client, msg, msgLen);
      }

      cmd = cmdEnd+1;
    } // while newline found

    if (cmd == client->recvBuffer) {
      if (client->recvOffset == sizeof(client->recvBuffer)) {
        // We didn't find any newlines and the buffer is full.
        INFO(NCCL_RAS, "RAS excessively long input line; terminating the client socket");
        rasClientTerminate(client);
        return;
      }
      // Otherwise it's an incomplete command; we need to wait for the rest of it.
    } else { // cmd > client->recvBuffer
      // Shift whatever remains (if anything) to the beginning of the buffer.
      memmove(client->recvBuffer, cmd, client->recvOffset - (cmd-client->recvBuffer));
      client->recvOffset -= cmd-client->recvBuffer;
    }
  } // if (client->status == RAS_CLIENT_CONNECTED)

  if (rasPfds[pollIdx].revents & POLLOUT) {
    struct rasMsgMeta* meta;
    while ((meta = ncclIntruQueueHead(&client->sendQ)) != nullptr) {
      ssize_t nSend;
      nSend = send(client->sock, ((char*)&meta->msg)+meta->offset, meta->length-meta->offset,
                   MSG_DONTWAIT | MSG_NOSIGNAL);
      if (nSend < 1) {
        if (nSend == -1 && errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
          if (errno == EPIPE)
            INFO(NCCL_RAS, "RAS socket closed by the client on send; terminating it");
          else
            INFO(NCCL_RAS, "RAS unexpected error from send; terminating the client socket");
          closed = true;
        }
        break;
      }

      meta->offset += nSend;
      if (meta->offset < meta->length)
        break;

      ncclIntruQueueDequeue(&client->sendQ);
      free(meta);
    } // while (meta)

    if (closed) {
      rasClientTerminate(client);
      return;
    }

    if (!meta) {
      rasPfds[client->pfd].events &= ~POLLOUT; // Nothing more to send for now.
      if (client->status == RAS_CLIENT_FINISHED)
        rasClientTerminate(client);
    }
  } // if (rasPfds[pollIdx].revents & POLLOUT)
}


//////////////////////////////////////////////////////////
// Functions driving data gathering for the RAS client. //
//////////////////////////////////////////////////////////

// Main function that drives the whole data gathering process and sends it back to the client.
// There are multiple asynchronous aspects of it (getting the data on connections and on communicators), so the
// function may exit early and needs to be reinvoked when the asynchronous responses arrive or the timeout expires.
// The state tracking the progress of such operations is kept in the rasClient.
static ncclResult_t rasClientRun(struct rasClient* client) {
  ncclResult_t ret = ncclSuccess;

  switch (client->status) {
    case RAS_CLIENT_INIT:
      NCCLCHECKGOTO(rasClientRunInit(client), ret, exit);
#if 0 // Commented out for now to focus the summary status report on the information most relevant to the users.
      // To be revisited with future extensions to RAS.
      client->status = RAS_CLIENT_CONNS;
      if (ret == ncclInProgress) {
        ret = ncclSuccess;
        break;
      }
    case RAS_CLIENT_CONNS:
      assert(client->collIdx != -1);
      NCCLCHECKGOTO(rasClientRunConns(client), ret, exit);
#endif
      client->status = RAS_CLIENT_COMMS;
      if (ret == ncclInProgress) {
        ret = ncclSuccess;
        break;
      }
    case RAS_CLIENT_COMMS:
      assert(client->collIdx != -1);
      NCCLCHECKGOTO(rasClientRunComms(client), ret, exit);
      client->status = RAS_CLIENT_FINISHED;
      break;
    default:
      WARN("Invalid client status %d", client->status);
      ret = ncclInternalError;
      goto exit;
  }
exit:
  return ret;
}

// Sends to the client the initial data that can be obtained locally -- version info, stats on rasPeers,
// dump of rasDeadPeers.  Initiates the RAS_COLL_CONNS collective operation.
static ncclResult_t rasClientRunInit(struct rasClient* client) {
  ncclResult_t ret = ncclSuccess;
  char* msg = nullptr;
  int msgLen;
  struct rasPeerInfo* peersReSorted = nullptr;
  int totalGpus, totalNodes, firstNGpusNode, firstNGpusGlobal, firstNPeersGlobal;
  bool consistentNGpusNode, consistentNGpusGlobal, consistentNPeersGlobal;
  int firstIdx, nPeers;
  struct rasValCount valCounts[NCCL_MAX_LOCAL_RANKS];
  int nValCounts;
  static int cudaDriver = -1, cudaRuntime = -1;

  rasOutReset();
  rasOutAppend("NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX
               " compiled with CUDA " STR(CUDA_MAJOR) "." STR(CUDA_MINOR) "\n");
  if (cudaRuntime == -1)
    cudaRuntimeGetVersion(&cudaRuntime);
  if (cudaDriver == -1)
    cudaDriverGetVersion(&cudaDriver);
  rasOutAppend("CUDA runtime version %d, driver version %d\n\n", cudaRuntime, cudaDriver);
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;

  rasOutReset();
  totalGpus = totalNodes = 0;
  firstNGpusNode = 0; // #GPUs on the first peer of a node.
  firstNGpusGlobal = 0; // #GPUs on peerIdx 0.
  consistentNGpusNode = true; // Whether #GPUs/peer is consistent between the peers *on any one node*.
  consistentNGpusGlobal = true; // Whether #GPUs/peer is consistent between the peers *on all nodes*.
  consistentNPeersGlobal = true; // Whether #peers/node is consistent between all nodes.
  nPeers = 0; // #peers on a node.
  firstNPeersGlobal = 0;
  for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
    int nGpus = __builtin_popcountll(rasPeers[peerIdx].cudaDevs);
    totalGpus += nGpus;
    if (peerIdx == 0) {
      totalNodes = 1;
      nPeers = 1;
      firstNGpusGlobal = firstNGpusNode = nGpus;
    } else { // peerIdx > 0
      if (nGpus != firstNGpusGlobal)
        consistentNGpusGlobal = false;
      if (!ncclSocketsSameNode(&rasPeers[peerIdx].addr, &rasPeers[peerIdx-1].addr)) {
        totalNodes++;
        if (firstNPeersGlobal == 0)
          firstNPeersGlobal = nPeers;
        else if (nPeers != firstNPeersGlobal)
          consistentNPeersGlobal = false;
        nPeers = 1;
        firstNGpusNode = nGpus;
      } else { // Same node.
        if (nGpus != firstNGpusNode)
          consistentNGpusNode = false;
        nPeers++;
      } // Same node
    } // peerIdx > 0
    if (peerIdx == nRasPeers-1) {
      if (firstNPeersGlobal == 0)
        firstNPeersGlobal = nPeers;
      else if (nPeers != firstNPeersGlobal)
        consistentNPeersGlobal = false;
    }
  } // for (peerIdx)

  rasOutAppend("Job summary\n"
               "===========\n\n");

  if (consistentNGpusNode && consistentNGpusGlobal && consistentNPeersGlobal) {
    rasOutAppend("  Nodes  Processes         GPUs  Processes     GPUs\n"
                 "(total)   per node  per process    (total)  (total)\n"
                 "%7d"  "  %9d"    "  %11d"     "  %9d"    "  %7d\n",
                 totalNodes, firstNPeersGlobal, firstNGpusGlobal, nRasPeers, totalGpus);
  } else {
    // Gather the stats on the number of processes per node.  However, that number is not a property of a peer,
    // but of a group of peers, so calculating it is more involved.  We make a copy of rasPeers and creatively
    // misuse it: cudaDevs of each element will be repurposed to store the number of processes on the node.
    NCCLCHECKGOTO(ncclCalloc(&peersReSorted, nRasPeers), ret, fail);
    memcpy(peersReSorted, rasPeers, nRasPeers * sizeof(*peersReSorted));

    firstIdx = 0;
    nPeers = 0;
    for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
      if (peerIdx == 0) {
        nPeers = 1;
        firstIdx = 0;
      } else { // peerIdx > 0
        if (!ncclSocketsSameNode(&peersReSorted[peerIdx].addr, &peersReSorted[peerIdx-1].addr)) {
          for (int i = firstIdx; i < peerIdx; i++) {
            // Go back and update the number of processes of all the elements of that node.
            peersReSorted[i].cudaDevs = nPeers;
          }
          nPeers = 1;
          firstIdx = peerIdx;
        } else {
          nPeers++;
        }
      } // peerIdx > 0
      if (peerIdx == nRasPeers-1) {
        // Last iteration of the loop.
        for (int i = firstIdx; i < nRasPeers; i++) {
          peersReSorted[i].cudaDevs = nPeers;
        }
      }
    } // for (peerIdx)

    // Re-sort it now using the number of processes on the node (cudaDevs) as the primary key, host IP as the
    // secondary, and process id as the tertiary.
    qsort(peersReSorted, nRasPeers, sizeof(*peersReSorted), rasPeersNProcsCompare);

    // Calculate the distribution of different numbers of peers per node.
    nValCounts = 0;
    for (int peerIdx = 0; peerIdx < nRasPeers;) {
      if (peerIdx == 0 || peersReSorted[peerIdx].cudaDevs != peersReSorted[peerIdx-1].cudaDevs) {
        valCounts[nValCounts].value = peersReSorted[peerIdx].cudaDevs;
        valCounts[nValCounts].count = 1;
        valCounts[nValCounts].firstIdx = peerIdx;
        nValCounts++;
      } else {
        valCounts[nValCounts-1].count++;
      }
      // Advance peerIdx to the next node.
      peerIdx += peersReSorted[peerIdx].cudaDevs;
    }
    // valCounts is currently sorted by value (the number of peers per node).  Sort it by the count (most frequent
    // number of peers first).
    qsort(valCounts, nValCounts, sizeof(*valCounts), rasValCountsCompareRev);

    // Print it out, the most frequent peer counts first.
    if (consistentNGpusNode && consistentNGpusGlobal) {
      rasOutAppend("  Nodes  Processes         GPUs\n"
                   "          per node  per process\n");
      for (int i = 0; i < nValCounts; i++) {
        struct rasValCount* vc = valCounts+i;
        rasOutAppend("%7d  %9ld  %11d\n",
                     vc->count, vc->value, firstNGpusGlobal);
      }
    } else {
      rasOutAppend("  Nodes  Processes\n"
                   "          per node\n");
      for (int i = 0; i < nValCounts; i++) {
        struct rasValCount* vc = valCounts+i;
        rasOutAppend("%7d  %9ld\n",
                     vc->count, vc->value);
      }

      // We calculate and print the GPUs/process separately.  This is required for !consistentNGpusNode and
      // it also makes our life easier above for !consistentNGpusGlobal (which could require a larger valCounts).

      // Sort peers by the GPU count, to simplify data extraction.
      memcpy(peersReSorted, rasPeers, nRasPeers * sizeof(*peersReSorted));
      // GPU count is the primary key, host IP is the secondary, and process id is the tertiary.
      qsort(peersReSorted, nRasPeers, sizeof(*peersReSorted), rasPeersNGpuCompare);

      // Calculate the distribution of different numbers of GPUs per peer.
      nValCounts = 0;
      for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
        if (peerIdx == 0 || __builtin_popcountll(peersReSorted[peerIdx].cudaDevs) !=
                            __builtin_popcountll(peersReSorted[peerIdx-1].cudaDevs)) {
          valCounts[nValCounts].value = __builtin_popcountll(peersReSorted[peerIdx].cudaDevs);
          valCounts[nValCounts].count = 1;
          valCounts[nValCounts].firstIdx = peerIdx;
          nValCounts++;
        } else {
          valCounts[nValCounts-1].count++;
        }
      }
      // valCounts is currently sorted by value (number of GPUs per peer).  Sort it by the count (most frequent
      // GPU counts first).
      qsort(valCounts, nValCounts, sizeof(*valCounts), rasValCountsCompareRev);

      // Print it out, the most frequent GPU counts first.
      rasOutAppend("\n"
                   "         Processes         GPUs\n"
                   "                    per process\n");
      for (int i = 0; i < nValCounts; i++) {
        struct rasValCount* vc = valCounts+i;
        rasOutAppend("         %9d  %11ld\n",
                     vc->count, vc->value);
      }
    }
    rasOutAppend("\n"
                 "  Nodes  Processes         GPUs\n"
                 "(total)    (total)      (total)\n"
                 "%7d"  "  %9d"    "  %11d\n",
                 totalNodes, nRasPeers, totalGpus);

    if (consistentNGpusNode && consistentNGpusGlobal) {
      // In this simpler case, also print the node outliers.
      for (int i = 1; i < nValCounts; i++) {
        struct rasValCount* vc = valCounts+i;
        // We assume that the most frequent group is correct; for the remaining ones, we try to provide more info,
        // provided that they meet our definition of an outlier.
        if (rasCountIsOutlier(vc->count, client->verbose, totalNodes)) {
          rasOutAppend("\nThe outlier node%s:\n", (vc->count > 1 ? "s" : ""));
          // peersReSorted is sorted by the node IP address (not port!) as the secondary key and the pid as
          // the tertiary, which comes in handy when printing...
          for (int peerIdx = vc->firstIdx; peerIdx < vc->count*vc->value + vc->firstIdx; peerIdx += vc->value) {
            lineBuf[0] = '\0';
            for (int j = 0; j < vc->value; j++) {
              snprintf(lineBuf+strlen(lineBuf), sizeof(lineBuf)-strlen(lineBuf), "%s%d",
                       (j > 0 ? "," : ""), peersReSorted[j].pid);
            }
            rasOutAppend("  Node %s running process%s %s\n",
                         ncclSocketToHost(&peersReSorted[peerIdx].addr, rasLine, sizeof(rasLine)),
                         (vc->value > 1 ? "es" : ""), lineBuf);
          } // for (peerIdx)
        } // if (rasCountIsOutlier(vc->count))
      } // for (i)
    } // !consistentNPeersGlobal
  } // !consistentNGpusNode || !consistentNGpusGlobal || !consistentNPeersGlobal

#if 0 // Commented out for now to focus the summary status report on the information most relevant to the users.
      // To be revisited with future extensions to RAS.
  rasOutAppend("\nGathering data about the RAS network (timeout %lds)...", client->timeout / CLOCK_UNITS_PER_SEC);
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;
  {
    struct rasCollRequest collReq;
    bool allDone = false;
    rasCollReqInit(&collReq);
    collReq.timeout = client->timeout;
    collReq.type = RAS_COLL_CONNS;
    NCCLCHECKGOTO(rasNetSendCollReq(&collReq, rasCollDataLength(RAS_COLL_CONNS), &allDone, &client->collIdx),
                  ret, fail);
    if (!allDone)
      ret = ncclInProgress; // We need to wait for async. responses.
  }
#endif
  rasOutAppend("\nCommunicators...");
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;
  {
    struct rasCollRequest collReq;
    bool allDone = false;
    rasCollReqInit(&collReq);
    collReq.timeout = client->timeout;
    collReq.type = RAS_COLL_COMMS;
    NCCLCHECKGOTO(rasNetSendCollReq(&collReq, rasCollDataLength(RAS_COLL_COMMS), &allDone, &client->collIdx),
                  ret, fail);
    if (!allDone)
      ret = ncclInProgress;
  }
exit:
  free(peersReSorted);
  return ret;
fail:
  goto exit;
}

#if 0 // Commented out for now to focus the summary status report on the information most relevant to the users.
      // To be revisited with future extensions to RAS.
// Processes the response from the RAS_COLL_CONNS collective operation and sends the data to the client (for now
// primarily the list of missing processes).  Initiates the RAS_COLL_COMMS collective operation.
static ncclResult_t rasClientRunConns(struct rasClient* client) {
  ncclResult_t ret = ncclSuccess;
  char* msg = nullptr;
  int msgLen;
  struct rasCollective* coll = rasCollectives+client->collIdx;
  struct rasCollConns* connsData = (struct rasCollConns*)coll->data;
  int expected;
  struct rasPeerInfo* peersBuf = nullptr;

  assert(coll->nFwdSent == coll->nFwdRecv);
  client->collIdx = -1;

  rasOutReset();
  rasOutAppend(" obtained a result in %.2fs\n", (clockNano()-coll->startTime)/1e9);
  if (coll->nLegTimeouts > 0) {
    rasOutAppend(" Warning: encountered %d communication timeout%s while gathering data\n", coll->nLegTimeouts,
                 (coll->nLegTimeouts > 1 ? "s" : ""));
  }

  expected = nRasPeers - nRasDeadPeers;
  if (coll->nPeers != expected) {
    int missing = expected - coll->nPeers;
    rasOutAppend(" Warning: missing data from %d process%s (received from %d, expected %d)\n",
                 missing, (missing > 1 ? "es" : ""), coll->nPeers, expected);
    if (missing <= RAS_CLIENT_DETAIL_THRESHOLD) {
      // Extract a list of missing peers.  We don't want to print it right away because it would be sorted
      // by address (including port, which isn't meaningful to end users).
      int nPeersBuf = 0;
      NCCLCHECKGOTO(ncclCalloc(&peersBuf, missing), ret, fail);
      // Ensure both arrays are sorted (rasPeers already is, by addr); makes finding missing records a breeze.
      qsort(coll->peers, coll->nPeers, sizeof(*coll->peers), &ncclSocketsCompare);
      for (int rasPeerIdx = 0, collPeerIdx = 0; rasPeerIdx < nRasPeers || collPeerIdx < coll->nPeers;) {
        int cmp;
        if (rasPeerIdx < nRasPeers && collPeerIdx < coll->nPeers)
          cmp = ncclSocketsCompare(&rasPeers[rasPeerIdx].addr, coll->peers+collPeerIdx);
        else
          cmp = (rasPeerIdx < nRasPeers ? -1 : 1);

        if (cmp == 0) {
          rasPeerIdx++;
          collPeerIdx++;
        } else if (cmp < 0) {
          memcpy(peersBuf+(nPeersBuf++), rasPeers+rasPeerIdx, sizeof(*peersBuf));
          rasPeerIdx++;
        } else { // cmp > 0
          // Process not found in rasPeers -- shouldn't happen.
          collPeerIdx++;
        } // cmp > 0
      } // for (rasPeerIdx, collPeerIdx)

      // Sort the output by host and pid, not host and port.
      qsort(peersBuf, nPeersBuf, sizeof(*peersBuf), rasPeersHostPidCompare);
      rasOutAppend("  The missing process%s:\n", (missing > 1 ? "es" : ""));
      for (int peerIdx = 0; peerIdx < nPeersBuf; peerIdx++) {
        rasOutAppend("  Process %d on node %s managing GPU%s %s\n", peersBuf[peerIdx].pid,
                     ncclSocketToHost(&peersBuf[peerIdx].addr, rasLine, sizeof(rasLine)),
                     (__builtin_popcountll(peersBuf[peerIdx].cudaDevs) > 1 ? "s" : ""),
                     rasGpuDevsToString(peersBuf[peerIdx].cudaDevs, peersBuf[peerIdx].nvmlDevs, lineBuf,
                                        sizeof(lineBuf)));
      }
      if (nPeersBuf != missing)
        rasOutAppend("  [could not find information on %d process%s]\n",
                     missing-nPeersBuf, (missing-nPeersBuf > 1 ? "es" : ""));
    } // if (expected - coll->nPeers <= RAS_CLIENT_DETAIL_THRESHOLD)
  } // if (coll->nPeers != expected)

  if (connsData->nConns > 0) {
    rasOutAppend(" Collected data about %d unidirectional connection%s\n",
                 connsData->nConns, (connsData->nConns > 1 ? "s" : ""));
    rasOutAppend(" Travel times (valid only if system clocks are synchronized between nodes):\n"
                 "  Minimum %fs, maximum %fs, average %fs\n",
                 connsData->travelTimeMin/1e9, connsData->travelTimeMax/1e9,
                 connsData->travelTimeSum/(1e9*connsData->travelTimeCount));
  } else {
    rasOutAppend(" No connection data collected!\n");
  }
  if (connsData->nNegativeMins > 0) {
    rasOutAppend(" Warning: negative travel times were observed across %d connection%s,\n"
                 " indicating that the system clocks are *not* synchronized.\n"
                 " Ordering of events based on local timestamps should be considered unreliable\n",
                 connsData->nNegativeMins, (connsData->nNegativeMins > 1 ? "s" : ""));
    if (connsData->nNegativeMins <= RAS_CLIENT_DETAIL_THRESHOLD) {
      rasOutAppend("  The affected connection%s:\n", (connsData->nNegativeMins > 1 ? "s" : ""));
      for (int i = 0; i < connsData->nNegativeMins; i++) {
        struct rasCollConns::negativeMin* negativeMin = connsData->negativeMins+i;
        int sourcePeerIdx = rasPeerFind(&negativeMin->source);
        int destPeerIdx = rasPeerFind(&negativeMin->dest);
        if (sourcePeerIdx != -1 && destPeerIdx != -1)
          rasOutAppend("  From node %s process %d to node %s process %d: observed travel time of %fs\n",
                       ncclSocketToHost(&negativeMin->source, rasLine, sizeof(rasLine)), rasPeers[sourcePeerIdx].pid,
                       ncclSocketToHost(&negativeMin->dest, lineBuf, sizeof(lineBuf)), rasPeers[destPeerIdx].pid,
                       negativeMin->travelTimeMin/1e9);
      }
    }
  }
  rasCollFree(coll);

  rasOutAppend("\nGathering data about the NCCL communicators (timeout %lds)...",
               client->timeout / CLOCK_UNITS_PER_SEC);
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;
  {
    struct rasCollRequest collReq;
    bool allDone = false;
    rasCollReqInit(&collReq);
    collReq.timeout = client->timeout;
    collReq.type = RAS_COLL_COMMS;
    NCCLCHECKGOTO(rasNetSendCollReq(&collReq, rasCollDataLength(RAS_COLL_COMMS), &allDone, &client->collIdx),
                  ret, fail);
    if (!allDone)
      ret = ncclInProgress;
  }
exit:
  free(peersBuf);
  return ret;
fail:
  goto exit;
}
#endif

// Processes the response from the RAS_COLL_COMMS collective operation and sends the data to the client:
// statistics on the communicators, missing data from ranks, inconsistent collective operation counts,
// initialization and asynchronous errors, and inconsistent initialization/termination status.
static ncclResult_t rasClientRunComms(struct rasClient* client) {
  ncclResult_t ret = ncclSuccess;
  char* msg = nullptr;
  int msgLen;
  struct rasCollective* coll = rasCollectives+client->collIdx;
  struct rasCollComms* commsData = (struct rasCollComms*)coll->data;
  struct rasCollComms::comm* comm;
  struct rasCollComms::comm::rank* ranksReSorted = nullptr;
  struct rasValCount* valCounts = nullptr;
  int nValCounts;
  struct rasValCount* collOpCounts = nullptr;
  struct rasAuxComm* auxComms = nullptr;
  int maxCommSize;
  int* peerIdxConv = nullptr;
  int vcIdx;
  int nPeersMissing;
  uint64_t* peerNvmlDevs = nullptr;
  const char*const statusStr[] = { "UNKNOWN", "INIT", "RUNNING", "FINALIZE", "ABORT" };
  const char*const errorStr[] = {
    // Listing them all like this, while a bit of a hassle, is less effort than formatting in a temporary buffer.
    "OK",
    "MISMATCH",
    "ERROR",
    "ERROR,MISMATCH",
    "INCOMPLETE",
    "INCOMPLETE,MISMATCH",
    "INCOMPLETE,ERROR",
    "INCOMPLETE,ERROR,MISMATCH"
  };

  assert(coll->nFwdSent == coll->nFwdRecv);
  client->collIdx = -1;

  rasOutReset();
  rasOutAppend(" (%.2fs)\n=============\n\n", (clockNano()-coll->startTime)/1e9);

  // Calculate the number of missing peers early as we rely on it for other things.
  nPeersMissing = nRasPeers - nRasDeadPeers - coll->nPeers;

  // Sort the communicators by size.  As the structure is inconvenient to move around due to the elements being
  // of variable length, we create an auxiliary array that includes pointers to individual elements and simply sort
  // that array while keeping the data intact.
  NCCLCHECKGOTO(ncclCalloc(&auxComms, commsData->nComms), ret, fail);
  // While initializing the just allocated array, also find out the size of the largest communicator so that we know
  // how much memory to allocate for another temporary array.
  maxCommSize = 0;
  comm = commsData->comms;
  for (int commIdx = 0; commIdx < commsData->nComms; commIdx++) {
    if (maxCommSize < comm->commNRanks)
      maxCommSize = comm->commNRanks;
    auxComms[commIdx].comm = comm;
    comm = (struct rasCollComms::comm*)(((char*)(comm+1)) + comm->nRanks * sizeof(*comm->ranks));
  }
  NCCLCHECKGOTO(ncclCalloc(&ranksReSorted, maxCommSize), ret, fail);

  // For convenience, create a translation table from rasCollective's peerIdx to rasPeers peerIdx.
  NCCLCHECKGOTO(ncclCalloc(&peerIdxConv, coll->nPeers), ret, fail);
  for (int peerIdx = 0; peerIdx < coll->nPeers; peerIdx++)
    peerIdxConv[peerIdx] = rasPeerFind(coll->peers+peerIdx);
  // Sort coll->peers to match the ordering of rasPeers -- we may need it later...
  qsort(coll->peers, coll->nPeers, sizeof(*coll->peers), &ncclSocketsCompare);

  // Fill in the remaining fields of auxComm's.
  for (int commIdx = 0; commIdx < commsData->nComms; commIdx++) {
    struct rasAuxComm* auxComm = auxComms+commIdx;
    int nRanks = 0;
    comm = auxComm->comm;

    if (comm->commNRanks > comm->nRanks) {
      // There are two possibilities here.  Either we are missing the data on some ranks because the processes are
      // unreachable, or the processes _are_ reachable but didn't report to be part of this communicator (which
      // could definitely happen if some processes have already called ncclCommDestroy or ncclCommAbort).  Because we
      // currently don't collect data about missing ranks, we can't reliably distinguish these two cases.
      // For now we rely on an approximation: if we _know_ that some peers failed to respond, we mark this
      // as an INCOMPLETE error; otherwise as a MISMATCH warning.
      if (nPeersMissing > 0 || nRasDeadPeers > 0)
        auxComm->errors |= RAS_ACE_INCOMPLETE;
      else {
        auxComm->errors |= RAS_ACE_MISMATCH;
        auxComm->status |= RAS_ACS_UNKNOWN;
      }
    }

    memcpy(ranksReSorted, comm->ranks, comm->nRanks * sizeof(*ranksReSorted));
    // Convert ranksReSorted' peerIdx to rasPeers and sort by it -- that way we will have the ranks sorted
    // by process _and_ node, which makes counting easy.
    for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++)
      ranksReSorted[rankIdx].peerIdx = peerIdxConv[ranksReSorted[rankIdx].peerIdx];
    qsort(ranksReSorted, comm->nRanks, sizeof(*ranksReSorted), rasCommRanksPeerCompare);

    // Count the peers and nodes, get the status/error indicators.
    for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
      struct rasCollComms::comm::rank* rank = ranksReSorted+rankIdx;
      if (rankIdx == 0) {
        auxComm->nPeers = auxComm->nNodes = 1;
        auxComm->ranksPerNodeMin = NCCL_MAX_LOCAL_RANKS;
        auxComm->ranksPerNodeMax = 0;
        auxComm->firstCollOpCount = rank->collOpCount;
        nRanks = 1;
      } else { // rankIdx > 0
        if (rank->peerIdx != rank[-1].peerIdx) {
          auxComm->nPeers++;
          if (!ncclSocketsSameNode(&rasPeers[rank->peerIdx].addr, &rasPeers[rank[-1].peerIdx].addr)) {
            auxComm->nNodes++;
            if (auxComm->ranksPerNodeMin > nRanks)
              auxComm->ranksPerNodeMin = nRanks;
            if (auxComm->ranksPerNodeMax < nRanks)
              auxComm->ranksPerNodeMax = nRanks;
            nRanks = 0;
          }
        } // if (rank->peerIdx != rank[-1].peerIdx)
        nRanks++;
      } // rankIdx > 0
      if (rankIdx == comm->nRanks-1) {
        // Last iteration of the loop.
        if (auxComm->ranksPerNodeMin > nRanks)
          auxComm->ranksPerNodeMin = nRanks;
        if (auxComm->ranksPerNodeMax < nRanks)
          auxComm->ranksPerNodeMax = nRanks;
      }

      if (rank->status.abortFlag)
        auxComm->status |= RAS_ACS_ABORT;
      else if (rank->status.finalizeCalled || rank->status.destroyFlag) {
        // destroyFlag is set by ncclCommDestroy and ncclCommAbort.  finalizeCalled appears to be set by
        // ncclCommFinalize only.  According to the docs, ncclCommDestroy *can* be called without calling
        // ncclCommFinalize first.  The code structure here ensures that we attribute destroyFlag properly
        // as a finalize state indicator (and ignore it in case of ncclCommAbort).
        auxComm->status |= RAS_ACS_FINALIZE;
      }
      else if (rank->status.initState == ncclSuccess)
        auxComm->status |= RAS_ACS_RUNNING;
      else // rank->initState != ncclSuccess
        auxComm->status |= RAS_ACS_INIT;

      if (rank->collOpCount != auxComm->firstCollOpCount)
        auxComm->errors |= RAS_ACE_MISMATCH;
      if (rank->status.initState != ncclSuccess && rank->status.initState != ncclInProgress)
        auxComm->errors |= RAS_ACE_ERROR;
      if (rank->status.asyncError != ncclSuccess && rank->status.asyncError != ncclInProgress)
        auxComm->errors |= RAS_ACE_ERROR;
    } // for (rankIdx)

    if (__builtin_popcount(auxComm->status) > 1) {
      // We've got a status mismatch between ranks.
      auxComm->errors |= RAS_ACE_MISMATCH;
    }
  } // for (commIdx)
  // Sort it by size/nNodes/status/errors/missing ranks.
  qsort(auxComms, commsData->nComms, sizeof(*auxComms), &rasAuxCommsCompareRev);

  // Calculate the distribution of different communicator sizes.
  NCCLCHECKGOTO(ncclCalloc(&valCounts, commsData->nComms), ret, fail);
  nValCounts = 0;
  for (int commIdx = 0; commIdx < commsData->nComms; commIdx++) {
    if (commIdx == 0 ||
        auxComms[commIdx].comm->commNRanks != auxComms[commIdx-1].comm->commNRanks ||
        auxComms[commIdx].nNodes != auxComms[commIdx-1].nNodes ||
        // __builtin_clz returns the number of leading 0-bits, which is a proxy for the index of the highest 1-bit.
        __builtin_clz(auxComms[commIdx].status) != __builtin_clz(auxComms[commIdx-1].status) ||
        auxComms[commIdx].errors != auxComms[commIdx-1].errors) {
      valCounts[nValCounts].value = 0; // We have many distinguishing values but only one field to store them.
                                       // It doesn't really matter, given that we can extract them via firstIdx.
      valCounts[nValCounts].count = 1;
      valCounts[nValCounts].firstIdx = commIdx;
      nValCounts++;
    } else {
      valCounts[nValCounts-1].count++;
    }
  }

  rasOutAppend("Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors\n"
               "    #  in group  per comm  per node  per comm  in group\n");
  if (commsData->nComms == 0)
    rasOutAppend("No communicator data collected!\n");

  // Allocate an auxiliary structure used for counting the number of ranks (unique GPUs) in a group.
  NCCLCHECKGOTO(ncclCalloc(&peerNvmlDevs, coll->nPeers), ret, fail);

  // Print it out, the largest communicators first.
  for (int vcIdx = 0; vcIdx < nValCounts; vcIdx++) {
    struct rasValCount* vc = valCounts+vcIdx;
    struct rasAuxComm* auxComm = auxComms+vc->firstIdx;
    int ranksPerNodeMin, ranksPerNodeMax;
    int ranksTotal;

    ranksPerNodeMin = NCCL_MAX_LOCAL_RANKS;
    ranksPerNodeMax = 0;
    memset(peerNvmlDevs, '\0', coll->nPeers * sizeof(*peerNvmlDevs));
    // We don't group comms by ranksPerNodeMin/Max, so the values may differ between comms in one group.
    // Calculate the group's min/max.
    // Also calculate the number of unique ranks in the group.
    for (int commIdx = 0; commIdx < vc->count; commIdx++) {
      if (ranksPerNodeMin > auxComm[commIdx].ranksPerNodeMin)
        ranksPerNodeMin = auxComm[commIdx].ranksPerNodeMin;
      if (ranksPerNodeMax < auxComm[commIdx].ranksPerNodeMax)
        ranksPerNodeMax = auxComm[commIdx].ranksPerNodeMax;
      for (int rankIdx = 0; rankIdx < auxComm[commIdx].comm->nRanks; rankIdx++) {
        struct rasCollComms::comm::rank* rank = auxComm[commIdx].comm->ranks+rankIdx;
        peerNvmlDevs[rank->peerIdx] |= (1UL << rank->nvmlDev);
      }
    }
    ranksTotal = 0;
    for (int peerIdx = 0; peerIdx < coll->nPeers; peerIdx++)
      ranksTotal += __builtin_popcountll(peerNvmlDevs[peerIdx]);
    if (ranksPerNodeMin == ranksPerNodeMax)
      snprintf(rasLine, sizeof(rasLine), "%d", ranksPerNodeMin);
    else
      snprintf(rasLine, sizeof(rasLine), "%d-%d", ranksPerNodeMin, ranksPerNodeMax);
    rasOutAppend("%5d  %8d  %8d  %8s  %8d  %8d  %8s  %6s\n",
                 vcIdx, vc->count, auxComm->nNodes, rasLine, auxComm->comm->commNRanks, ranksTotal,
                 // __builtin_clz returns the number of leading 0-bits.  This makes it possible to translate the
                 // status (which is a bitmask) into an array index.
                 statusStr[(sizeof(unsigned int)*8-1)-__builtin_clz(auxComm->status)], errorStr[auxComm->errors]);
  }

  rasOutAppend("\nErrors\n"
               "======\n\n");

  if (nPeersMissing > 0) {
    rasOutAppend("INCOMPLETE\n"
                 "  Missing communicator data from %d job process%s\n", nPeersMissing, (nPeersMissing > 1 ? "es" : ""));
    if (rasCountIsOutlier(nPeersMissing, client->verbose)) {
      // Extract a list of missing peers.  We don't want to print it right away because it would be sorted
      // by address (including port, which isn't meaningful to end users).
      struct rasPeerInfo* peersBuf = nullptr;
      int nPeersBuf;

      // Both rasPeers and coll->peers are sorted by address (the latter we sorted above) which makes comparing
      // them much easier.
      NCCLCHECKGOTO(ncclCalloc(&peersBuf, nPeersMissing), ret, fail);
      nPeersBuf = 0;
      for (int rasPeerIdx = 0, collPeerIdx = 0; rasPeerIdx < nRasPeers || collPeerIdx < coll->nPeers;) {
        int cmp;
        if (rasPeerIdx < nRasPeers && collPeerIdx < coll->nPeers)
          cmp = ncclSocketsCompare(&rasPeers[rasPeerIdx].addr, coll->peers+collPeerIdx);
        else
          cmp = (rasPeerIdx < nRasPeers ? -1 : 1);

        if (cmp == 0) {
          rasPeerIdx++;
          collPeerIdx++;
        } else if (cmp < 0) {
          // Process missing from coll->peers.  Don't report dead ones though, as they are not included
          // in nPeersMissing and are reported separately below.
          if (!rasPeerIsDead(&rasPeers[rasPeerIdx].addr)) {
            assert(nPeersBuf < nPeersMissing);
            memcpy(peersBuf+(nPeersBuf++), rasPeers+rasPeerIdx, sizeof(*peersBuf));
          }
          rasPeerIdx++;
        } else { // cmp > 0
          // Process not found in rasPeers -- shouldn't happen, unless during a race?
          collPeerIdx++;
        } // cmp > 0
      } // for (rasPeerIdx, collPeerIdx)

      // Sort the output by host and pid.
      qsort(peersBuf, nPeersBuf, sizeof(*peersBuf), rasPeersHostPidCompare);
      for (int peerIdx = 0; peerIdx < nPeersBuf; peerIdx++) {
        rasOutAppend("  Process %d on node %s managing GPU%s %s\n", peersBuf[peerIdx].pid,
                     ncclSocketToHost(&peersBuf[peerIdx].addr, rasLine, sizeof(rasLine)),
                     (__builtin_popcountll(peersBuf[peerIdx].cudaDevs) > 1 ? "s" : ""),
                     rasGpuDevsToString(peersBuf[peerIdx].cudaDevs, peersBuf[peerIdx].nvmlDevs, lineBuf,
                                        sizeof(lineBuf)));
      }
      if (nPeersBuf != nPeersMissing)
        rasOutAppend("  [could not find information on %d process%s]\n",
                     nPeersMissing-nPeersBuf, (nPeersMissing-nPeersBuf > 1 ? "es" : ""));
      free(peersBuf);
    } // if (rasCountIsOutlier(nPeersMissing))
    rasOutAppend("\n");
  }

  if (nRasDeadPeers > 0) {
    rasOutAppend("DEAD\n"
                 "  %d job process%s considered dead (unreachable via the RAS network)\n", nRasDeadPeers,
                 (nRasDeadPeers > 1 ? "es are" : " is"));
    if (rasCountIsOutlier(nRasDeadPeers, client->verbose)) {
      struct rasPeerInfo* peersReSorted = nullptr;
      int nPeersReSorted = 0;
      NCCLCHECKGOTO(ncclCalloc(&peersReSorted, nRasDeadPeers), ret, fail);
      for (int i = 0; i < nRasDeadPeers; i++) {
        int peerIdx = rasPeerFind(rasDeadPeers+i);
        if (peerIdx != -1)
          memcpy(peersReSorted+(nPeersReSorted++), rasPeers+peerIdx, sizeof(*peersReSorted));
      }
      // Sort the output by host and pid, not host and port.
      qsort(peersReSorted, nPeersReSorted, sizeof(*peersReSorted), rasPeersHostPidCompare);
      for (int peerIdx = 0; peerIdx < nPeersReSorted; peerIdx++) {
        rasOutAppend("  Process %d on node %s managing GPU%s %s\n", peersReSorted[peerIdx].pid,
                     ncclSocketToHost(&peersReSorted[peerIdx].addr, rasLine, sizeof(rasLine)),
                     (__builtin_popcountll(peersReSorted[peerIdx].cudaDevs) > 1 ? "s" : ""),
                     rasGpuDevsToString(peersReSorted[peerIdx].cudaDevs, peersReSorted[peerIdx].nvmlDevs, lineBuf,
                                        sizeof(lineBuf)));
      }
      if (nPeersReSorted != nRasDeadPeers)
        rasOutAppend("  [could not find information on %d process%s]\n",
                     nRasDeadPeers-nPeersReSorted, (nRasDeadPeers-nPeersReSorted > 1 ? "es" : ""));
      free(peersReSorted);
    } // if (rasCountIsOutlier(nRasDeadPeers)
    rasOutAppend("\n");
  }

  for (vcIdx = 0; vcIdx < nValCounts; vcIdx++) {
    struct rasValCount* vc;
    vc = valCounts+vcIdx;
    for (int commIdx = vc->firstIdx; commIdx < vc->count + vc->firstIdx; commIdx++) {
      struct rasAuxComm* auxComm = auxComms+commIdx;
      comm = auxComm->comm;

      if (auxComm->errors & RAS_ACE_INCOMPLETE) {
        int nRanksMissing = comm->commNRanks - comm->nRanks;
        rasOutAppend("#%d-%d (%016lx) INCOMPLETE\n"
                     "  Missing communicator data from %d rank%s\n", vcIdx, commIdx - vc->firstIdx,
                     comm->commHash, nRanksMissing, (nRanksMissing > 1 ? "s" : ""));
        if (rasCountIsOutlier(nRanksMissing, client->verbose)) {
          lineBuf[0] = '\0';
          // rankIdx indexes the comm->ranks array; in principle it should be the same as commRank, with the
          // exception of the missing ranks...
          for (int commRank = 0, rankIdx = 0; commRank < comm->commNRanks; commRank++) {
            if (rankIdx < comm->nRanks && comm->ranks[rankIdx].commRank == commRank) {
              rankIdx++;
            } else {
              snprintf(lineBuf+strlen(lineBuf), sizeof(lineBuf)-strlen(lineBuf), "%s%d",
                       (rankIdx == commRank ? "" : ","), commRank);
            }
          } // for (commRank)
          rasOutAppend("  The missing rank%s: %s\n", (nRanksMissing > 1 ? "s" : ""), lineBuf);
        } // if (rasCountIsOutlier(nRanksMissing))
        rasOutAppend("\n");
      } // if (auxComm->errors & RAS_ACE_INCOMPLETE)

      if (auxComm->errors & RAS_ACE_ERROR) {
        int ncclErrors[ncclNumResults];
        int nErrors;
        rasOutAppend("#%d-%d (%016lx) ERROR\n", vcIdx, commIdx - vc->firstIdx, comm->commHash);

        memset(ncclErrors, '\0', sizeof(ncclErrors));
        for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++)
          ncclErrors[comm->ranks[rankIdx].status.initState]++;
        nErrors = comm->nRanks - (ncclErrors[ncclSuccess] + ncclErrors[ncclInProgress]);
        if (nErrors > 0) {
          rasOutAppend("  Initialization error%s on %d rank%s\n",
                       (nErrors > 1 ? "s" : ""), nErrors, (nErrors > 1 ? "s" : ""));
          rasClientBreakDownErrors(client, comm, peerIdxConv, ncclErrors);
        }

        memset(ncclErrors, '\0', sizeof(ncclErrors));
        for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++)
          ncclErrors[comm->ranks[rankIdx].status.asyncError]++;
        nErrors = comm->nRanks - (ncclErrors[ncclSuccess] + ncclErrors[ncclInProgress]);
        if (nErrors > 0) {
          rasOutAppend("  Asynchronous error%s on %d rank%s\n",
                       (nErrors > 1 ? "s" : ""), nErrors, (nErrors > 1 ? "s" : ""));
          rasClientBreakDownErrors(client, comm, peerIdxConv, ncclErrors, /*isAsync*/true);
        }
        rasOutAppend("\n");
      } // if (auxComm->errors & RAS_ACE_ERROR)
    } // for (commIdx)
  } // for (vcIdx)

  rasOutAppend("Warnings\n"
               "========\n\n");

  if (coll->nLegTimeouts > 0) {
    rasOutAppend("TIMEOUT\n"
                 "  Encountered %d communication timeout%s while gathering communicator data\n\n",
                 coll->nLegTimeouts, (coll->nLegTimeouts > 1 ? "s" : ""));
  }

  for (int vcIdx = 0; vcIdx < nValCounts; vcIdx++) {
    struct rasValCount* vc = valCounts+vcIdx;
    for (int commIdx = vc->firstIdx; commIdx < vc->count + vc->firstIdx; commIdx++) {
      bool inconsistent;
      struct rasAuxComm* auxComm = auxComms+commIdx;
      comm = auxComm->comm;

      if (auxComm->errors & RAS_ACE_MISMATCH) {
        rasOutAppend("#%d-%d (%016lx) MISMATCH\n", vcIdx, commIdx - vc->firstIdx, comm->commHash);

        if (collOpCounts == nullptr) {
          // Allocating comm->commNRanks elements ensures that we won't need to reallocate, because the valCounts
          // array is reverse-sorted by commNRanks.  On the other hand, for this purpose allocating commNRanks
          // elements may be massively overpessimistic...
          NCCLCHECKGOTO(ncclCalloc(&collOpCounts, comm->commNRanks), ret, fail);
        }

        if (__builtin_popcount(auxComm->status) > 1) {
          rasOutAppend("  Communicator ranks have different status\n");

          // We need to sort the ranks by status.  However, status is normally calculated from other fields.
          // We will copy the ranks and reuse collOpCount to store it.
          memcpy(ranksReSorted, comm->ranks, comm->nRanks * sizeof(*ranksReSorted));
          for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
            struct rasCollComms::comm::rank* rank = ranksReSorted+rankIdx;

            if (rank->status.abortFlag)
              rank->collOpCount = RAS_ACS_ABORT;
            else if (rank->status.finalizeCalled || rank->status.destroyFlag)
              rank->collOpCount = RAS_ACS_FINALIZE;
            else if (rank->status.initState == ncclSuccess)
              rank->collOpCount = RAS_ACS_RUNNING;
            else
              rank->collOpCount = RAS_ACS_INIT;
          }
          qsort(ranksReSorted, comm->nRanks, sizeof(*ranksReSorted), rasCommRanksCollOpCompare);
          // Calculate the frequency of different status values.
          int nCollOpCounts = 0;
          for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
            if (rankIdx == 0 || ranksReSorted[rankIdx].collOpCount != ranksReSorted[rankIdx-1].collOpCount) {
              // __builtin_clz returns the number of leading 0-bits.  This makes it possible to translate the
              // status (which is a bitmask) into an array index.
              collOpCounts[nCollOpCounts].value = (sizeof(unsigned int)*8-1) - __builtin_clz(ranksReSorted[rankIdx].collOpCount);
              collOpCounts[nCollOpCounts].count = 1;
              collOpCounts[nCollOpCounts].firstIdx = rankIdx;
              nCollOpCounts++;
            } else {
              collOpCounts[nCollOpCounts-1].count++;
            }
          }
          if (comm->nRanks < comm->commNRanks) {
            // Add a "fake" element corresponding to the missing entries.  The statusStr array contains the "UNKNOWN"
            // string at index 0.
            collOpCounts[nCollOpCounts].value = 0;
            collOpCounts[nCollOpCounts].count = comm->commNRanks - comm->nRanks;
            collOpCounts[nCollOpCounts].firstIdx = -1; // "Fake" entry identifier.
            nCollOpCounts++;
          }
          // Sort by that frequency (most frequent first).
          qsort(collOpCounts, nCollOpCounts, sizeof(*collOpCounts), rasValCountsCompareRev);

          for (int coc = 0; coc < nCollOpCounts; coc++) {
            struct rasValCount* vcc = collOpCounts+coc;
            if (vcc->count > 1)
              rasOutAppend("  %d ranks have status %s\n", vcc->count, statusStr[vcc->value]);
            if (rasCountIsOutlier(vcc->count, client->verbose, comm->commNRanks)) {
              if (vcc->firstIdx != -1) {
                // ranksReSorted is sorted by rank as the secondary key, which comes in handy when printing...
                for (int rankIdx = vcc->firstIdx; rankIdx < vcc->count+vcc->firstIdx; rankIdx++) {
                  int peerIdx = peerIdxConv[ranksReSorted[rankIdx].peerIdx];
                  if (peerIdx != -1) {
                    if (vcc->count > 1)
                      rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                                   ranksReSorted[rankIdx].commRank,
                                   rasCommRankGpuToString(ranksReSorted+rankIdx, lineBuf, sizeof(lineBuf)),
                                   rasPeers[peerIdx].pid,
                                   ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                    else
                      rasOutAppend("  Rank %d has status %s -- GPU %s managed by process %d on node %s\n",
                                   ranksReSorted[rankIdx].commRank, statusStr[vcc->value],
                                   rasCommRankGpuToString(ranksReSorted+rankIdx, lineBuf, sizeof(lineBuf)),
                                   rasPeers[peerIdx].pid,
                                   ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                  } else { // peerIdx == -1
                    if (vcc->count > 1)
                      rasOutAppend("  Rank %d -- [process information not found]\n", ranksReSorted[rankIdx].commRank);
                    else
                      rasOutAppend("  Rank %d has status %s -- [process information not found]\n",
                                   ranksReSorted[rankIdx].commRank, statusStr[vcc->value]);
                  } // peerIdx == -1
                } // for (rankIdx)
              } else {
                // UNKNOWN ranks.  Format a string with their rank numbers (we don't know anything more).
                lineBuf[0] = '\0';
                // rankIdx indexes the comm->ranks array; in principle it should be the same as commRank, with the
                // exception of the missing ranks...
                for (int commRank = 0, rankIdx = 0; commRank < comm->commNRanks; commRank++) {
                  if (rankIdx < comm->nRanks && comm->ranks[rankIdx].commRank == commRank) {
                    rankIdx++;
                  } else {
                    snprintf(lineBuf+strlen(lineBuf), sizeof(lineBuf)-strlen(lineBuf), "%s%d",
                             (rankIdx == commRank ? "" : ","), commRank);
                  }
                } // for (commRank)
                if (vcc->count > 1) {
                  rasOutAppend("  The unknown ranks: %s\n", lineBuf);
                } else {
                  rasOutAppend("  Rank %s has status %s\n", lineBuf, statusStr[vcc->value]);
                }
              }
            } // if (rasCountIsOutlier(vcc->count))
          } // for (coc)
        } // if (__builtin_popcount(auxComm->status) > 1)

        inconsistent = false;
        for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
          if (comm->ranks[rankIdx].collOpCount != auxComm->firstCollOpCount) {
            inconsistent = true;
            break;
          }
        }
        if (inconsistent) {
          rasOutAppend("  Communicator ranks have different collective operation counts\n");

          // Sort the ranks by collOpCount and rank for easy counting.
          memcpy(ranksReSorted, comm->ranks, comm->nRanks * sizeof(*ranksReSorted));
          qsort(ranksReSorted, comm->nRanks, sizeof(*ranksReSorted), rasCommRanksCollOpCompare);
          // Calculate the frequency of different collOpCount values.
          int nCollOpCounts = 0;
          for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
            if (rankIdx == 0 || ranksReSorted[rankIdx].collOpCount != ranksReSorted[rankIdx-1].collOpCount) {
              collOpCounts[nCollOpCounts].value = ranksReSorted[rankIdx].collOpCount;
              collOpCounts[nCollOpCounts].count = 1;
              collOpCounts[nCollOpCounts].firstIdx = rankIdx;
              nCollOpCounts++;
            } else {
              collOpCounts[nCollOpCounts-1].count++;
            }
          }
          // Sort by that frequency (most frequent first).
          qsort(collOpCounts, nCollOpCounts, sizeof(*collOpCounts), rasValCountsCompareRev);

          for (int coc = 0; coc < nCollOpCounts; coc++) {
            struct rasValCount* vcc = collOpCounts+coc;
            if (vcc->count > 1)
              rasOutAppend("  %d ranks have launched up to operation %ld\n", vcc->count, vcc->value);
            if (rasCountIsOutlier(vcc->count, client->verbose, comm->commNRanks)) {
              // ranksReSorted is sorted by rank as the secondary key, which comes in handy when printing...
              for (int rankIdx = vcc->firstIdx; rankIdx < vcc->count+vcc->firstIdx; rankIdx++) {
                int peerIdx = peerIdxConv[ranksReSorted[rankIdx].peerIdx];
                if (peerIdx != -1) {
                  if (vcc->count > 1)
                    rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                                 ranksReSorted[rankIdx].commRank,
                                 rasCommRankGpuToString(ranksReSorted+rankIdx, lineBuf, sizeof(lineBuf)),
                                 rasPeers[peerIdx].pid,
                                 ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                  else
                    rasOutAppend("  Rank %d has launched up to operation %ld -- GPU %s managed by process %d on node %s\n",
                                 ranksReSorted[rankIdx].commRank, vcc->value,
                                 rasCommRankGpuToString(ranksReSorted+rankIdx, lineBuf, sizeof(lineBuf)),
                                 rasPeers[peerIdx].pid,
                                 ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                } else { // peerIdx == -1
                  if (vcc->count > 1)
                    rasOutAppend("  Rank %d -- [process information not found]\n", ranksReSorted[rankIdx].commRank);
                  else
                     rasOutAppend("  Rank %d has launched up to operation %ld -- [process information not found]\n",
                                  ranksReSorted[rankIdx].commRank, vcc->value);
                } // peerIdx == -1
              } // for (rankIdx)
            } // if (rasCountIsOutlier(vcc->count))
          } // for (coc)
        } // if (inconsistent)
        rasOutAppend("\n");
      } // if (auxComm->errors & RAS_ACE_MISMATCH)
    } // for (commIdx)
  } // for (vcIdx)
  rasCollFree(coll);

  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;
exit:
  free(peerNvmlDevs);
  free(collOpCounts);
  free(valCounts);
  free(peerIdxConv);
  free(ranksReSorted);
  free(auxComms);
  return ret;
fail:
  goto exit;
}

static void rasClientBreakDownErrors(struct rasClient* client, struct rasCollComms::comm* comm,
                                     const int* peerIdxConv, int ncclErrors[ncclNumResults], bool isAsync) {
  for (;;) {
    int maxCount = 0;
    ncclResult_t maxCountIdx = ncclSuccess;
    for (int i = ncclUnhandledCudaError; i < ncclInProgress; i++) {
      if (maxCount < ncclErrors[i]) {
        maxCount = ncclErrors[i];
        maxCountIdx = (ncclResult_t)i;
      }
    } // for (i)
    if (maxCountIdx == ncclSuccess)
      break;
    if (maxCount > 1)
      rasOutAppend("  %d ranks reported %s\n", maxCount, ncclErrorToString(maxCountIdx));
    if (rasCountIsOutlier(maxCount, client->verbose)) {
      for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
        if ((isAsync ? comm->ranks[rankIdx].status.asyncError : comm->ranks[rankIdx].status.initState) == maxCountIdx) {
          int peerIdx = peerIdxConv[comm->ranks[rankIdx].peerIdx];
          if (peerIdx != -1) {
            if (maxCount > 1)
              rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                           comm->ranks[rankIdx].commRank,
                           rasCommRankGpuToString(comm->ranks+rankIdx, lineBuf, sizeof(lineBuf)),
                           rasPeers[peerIdx].pid,
                           ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
            else
              rasOutAppend("  Rank %d reported %s -- GPU %s managed by process %d on node %s\n",
                           comm->ranks[rankIdx].commRank, ncclErrorToString(maxCountIdx),
                           rasCommRankGpuToString(comm->ranks+rankIdx, lineBuf, sizeof(lineBuf)),
                           rasPeers[peerIdx].pid,
                           ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
          } else { // peerIdx == -1
            if (maxCount > 1)
              rasOutAppend("  Rank %d -- [process information not found]\n", comm->ranks[rankIdx].commRank);
            else
              rasOutAppend("  Rank %d reported %s -- [process information not found]\n",
                           comm->ranks[rankIdx].commRank, ncclErrorToString(maxCountIdx));
          } // peerIdx == -1
        } // if rank's error matches
      } // for (rankIdx)
    } // if (rasCountIsOutlier(maxCount))
    ncclErrors[maxCountIdx] = 0;
  } // for (;;)
}


//////////////////////////////////////////////////////////////////////
// Functions related to the handling of the internal output buffer. //
//////////////////////////////////////////////////////////////////////

// Appends a printf-formatted string to the output buffer.
// Unlike with INFO or WARN messages, the caller should terminate lines with '\n' as appropriate.
static void rasOutAppend(const char* format, ...) {
  ncclResult_t ret; // Ignored.
  va_list vargs;
  int needed;
  va_start(vargs, format);
  needed = vsnprintf(rasOutBuffer+nRasOutBuffer, rasOutBufferSize-nRasOutBuffer, format, vargs);
  va_end(vargs);

  if (needed < 0) // Output error (whatever that might be...)
    return;

  // The +1 below accounts for the terminating '\0'.
  if (needed + 1 > rasOutBufferSize-nRasOutBuffer) {
    int newBufferSize = ROUNDUP(nRasOutBuffer+needed+1, RAS_OUT_INCREMENT);
    NCCLCHECKGOTO(ncclRealloc(&rasOutBuffer, rasOutBufferSize, newBufferSize), ret, exit);
    rasOutBufferSize = newBufferSize;

    va_start(vargs, format);
    needed = vsnprintf(rasOutBuffer+nRasOutBuffer, rasOutBufferSize-nRasOutBuffer, format, vargs);
    va_end(vargs);

    if (needed < 0) // Output error (whatever that might be...)
      return;
  }

  nRasOutBuffer += needed;
  assert(nRasOutBuffer <= rasOutBufferSize);
exit:
  ;
}

// Copies the output data from an internal buffer to a user-supplied one, including the terminating '\0'.
// The user buffer must already be allocated and be at least rasOutLength() bytes long (which includes
// the terminating '\0').
static void rasOutExtract(char* buffer) {
  if (rasOutBuffer)
    memcpy(buffer, rasOutBuffer, rasOutLength());
}

// Returns the current length of the used portion of the output buffer, *not* including the terminating '\0'.
static int rasOutLength() {
  return nRasOutBuffer;
}

// Resets the output buffer position to the beginning (effectively clearing the buffer).
static void rasOutReset() {
  ncclResult_t ret; // Ignored.
  nRasOutBuffer = 0;
  if (rasOutBuffer == nullptr) {
    NCCLCHECKGOTO(ncclCalloc(&rasOutBuffer, RAS_OUT_INCREMENT), ret, exit);
    rasOutBufferSize = RAS_OUT_INCREMENT;
  }
exit:
  ;
}


///////////////////////////////////////////////////////////////////
// Various sorting callbacks used when grouping/formatting data. //
///////////////////////////////////////////////////////////////////

// Sorting callback for rasPeerInfo elements.  Sorts by the number of bits set in cudaDevs.  Uses the host IP as the
// secondary key and the process id as the tertiary key.
static int rasPeersNGpuCompare(const void* e1, const void* e2) {
  const struct rasPeerInfo* p1 = (const struct rasPeerInfo*)e1;
  const struct rasPeerInfo* p2 = (const struct rasPeerInfo*)e2;
  int c1 = __builtin_popcountll(p1->cudaDevs);
  int c2 = __builtin_popcountll(p2->cudaDevs);

  if (c1 == c2) {
    // Host IP address is the secondary key.
    int cmp = ncclSocketsHostCompare(&p1->addr, &p2->addr);
    if (cmp == 0) {
      // Process ID is the tertiary key.
      cmp = (p1->pid < p2->pid ? -1 : (p1->pid > p2->pid ? 1 : 0));
    }
    return cmp;
  } else {
    return (c1 < c2 ? -1 : 1);
  }
}

// Sorting callback for rasPeerInfo elements.  Sorts by the number of peers per node, which we store in cudaDevs.
// Uses the host IP as the secondary key and the process id as the tertiary key.
static int rasPeersNProcsCompare(const void* e1, const void* e2) {
  const struct rasPeerInfo* p1 = (const struct rasPeerInfo*)e1;
  const struct rasPeerInfo* p2 = (const struct rasPeerInfo*)e2;

  if (p1->cudaDevs == p2->cudaDevs) {
    // Host IP address is the secondary key.
    int cmp = ncclSocketsHostCompare(&p1->addr, &p2->addr);
    if (cmp == 0) {
      // Process ID is the tertiary key.
      cmp = (p1->pid < p2->pid ? -1 : (p1->pid > p2->pid ? 1 : 0));
    }
    return cmp;
  } else {
    return (p1->cudaDevs < p2->cudaDevs ? -1 : 1);
  }
}

// Sorting callback for rasPeerInfo elements.  Sorts by the host IP and the process id as the secondary key (rather
// than the port).
static int rasPeersHostPidCompare(const void* e1, const void* e2) {
  const struct rasPeerInfo* p1 = (const struct rasPeerInfo*)e1;
  const struct rasPeerInfo* p2 = (const struct rasPeerInfo*)e2;

  int cmp = ncclSocketsHostCompare(&p1->addr, &p2->addr);
  if (cmp == 0) {
    // Process ID is the secondary key.
    cmp = (p1->pid < p2->pid ? -1 : (p1->pid > p2->pid ? 1 : 0));
  }
  return cmp;
}

// Sorting callback for ncclSocketAddress.  Unlike the ncclSocketsCompare, it ignores the port.
static int ncclSocketsHostCompare(const void* p1, const void* p2) {
  const union ncclSocketAddress* a1 = (const union ncclSocketAddress*)p1;
  const union ncclSocketAddress* a2 = (const union ncclSocketAddress*)p2;
  // AF_INET (2) is less than AF_INET6 (10).
  int family = a1->sa.sa_family;
  if (family != a2->sa.sa_family) {
    if (family > 0 && a2->sa.sa_family > 0)
      return (family < a2->sa.sa_family ? -1 : 1);
    else // Put empty addresses at the end (not that it matters...).
      return (family > 0 ? -1 : 1);
  }

  int cmp;
  if (family == AF_INET) {
    cmp = memcmp(&a1->sin.sin_addr, &a2->sin.sin_addr, sizeof(a1->sin.sin_addr));
  }
  else if (family == AF_INET6) {
    cmp = memcmp(&a1->sin6.sin6_addr, &a2->sin6.sin6_addr, sizeof(a1->sin6.sin6_addr));
  } else {
    // The only remaining valid case are empty addresses.
    assert(family == 0);
    cmp = 0; // Two empty addresses are equal...
  }

  return cmp;
}

// Sorting callback for rasValCount elements.  Sorts by the count, largest first.  Value is the secondary key.
static int rasValCountsCompareRev(const void* p1, const void* p2) {
  const struct rasValCount* r1 = (const struct rasValCount*)p1;
  const struct rasValCount* r2 = (const struct rasValCount*)p2;

  if (r1->count == r2->count) {
    return (r1->value > r2->value ? -1 : (r1->value < r2->value ? 1: 0));
  } else {
    return (r1->count > r2->count ? -1 : 1);
  }
}

// Sorting callback for rasAuxComm elements.
// Sorts the comms by the rank count (commNRanks), nNodes as secondary key, status as the tertiary, and errors as
// the quaternary.  Sorts in reverse (largest first).
// The final key is the comm's nRanks, sorted in reverse to the other keys, so comms with the largest number
// of ranks *missing* will be first.
static int rasAuxCommsCompareRev(const void* p1, const void* p2) {
  const struct rasAuxComm* c1 = (const struct rasAuxComm*)p1;
  const struct rasAuxComm* c2 = (const struct rasAuxComm*)p2;

  if (c1->comm->commNRanks == c2->comm->commNRanks) {
    if (c1->nNodes == c2->nNodes) {
      // We don't want to compare the status values directly because they could be bitmasks and we are only
      // interested in the highest bit set.
      // __builtin_clz returns the number of leading 0-bits, so in our case the value will be the *smallest*
      // if RAS_ACS_ABORT (8) is set and the *largest* if only RAS_ACS_INIT (1) is set, so we reverse the
      // comparison to get the desired sorting order.
      int s1 = __builtin_clz(c1->status);
      int s2 = __builtin_clz(c2->status);
      if (s1 == s2) {
        if (c1->errors == c2->errors) {
          if (c1->comm->nRanks == c2->comm->nRanks) {
            return 0;
          } else {
            return (c1->comm->nRanks < c2->comm->nRanks ? -1 : 1);
          }
        } else {
          return (c1->errors > c2->errors ? -1 : 1);
        }
      } else {
        return (s1 < s2 ? -1 : 1);
      }
    } else {
      return (c1->nNodes > c2->nNodes ? -1 : 1);
    }
  } else {
    return (c1->comm->commNRanks > c2->comm->commNRanks ? -1 : 1);
  }
}

// Sorting callback for rasCollComms::comm::rank elements.  Sorts by the peerIdx.
static int rasCommRanksPeerCompare(const void* p1, const void* p2) {
  const struct rasCollComms::comm::rank* r1 = (const struct rasCollComms::comm::rank*)p1;
  const struct rasCollComms::comm::rank* r2 = (const struct rasCollComms::comm::rank*)p2;

  return (r1->peerIdx < r2->peerIdx ? -1 : (r1->peerIdx > r2->peerIdx ? 1 : 0));
}

// Sorting callback for rasCollComms::comm::rank elements.  Sorts by the collOpCount, with rank as the secondary key.
static int rasCommRanksCollOpCompare(const void* p1, const void* p2) {
  const struct rasCollComms::comm::rank* r1 = (const struct rasCollComms::comm::rank*)p1;
  const struct rasCollComms::comm::rank* r2 = (const struct rasCollComms::comm::rank*)p2;

  if (r1->collOpCount == r2->collOpCount) {
    // Use the rank as the secondary key.
    return (r1->commRank < r2->commRank ? -1 : (r1->commRank > r2->commRank ? 1 : 0));
  } else {
    return (r1->collOpCount < r2->collOpCount ? -1 : 1);
  }
}


////////////////////////////////////////////////////////////
// String formatting functions for various types of data. //
////////////////////////////////////////////////////////////

// Coverts a GPU mask(s) to a string.  If the CUDA mask is different from the NVML mask, both are printed.
const char* rasGpuDevsToString(uint64_t cudaDevs, uint64_t nvmlDevs, char* buf, size_t size) {
  bool first = true;
  buf[0] = '\0';
  for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; i++)
    if (cudaDevs & (1UL << i)) {
      snprintf(buf+strlen(buf), size-strlen(buf), "%s%d", (first ? "" : ","), i);
      first = false;
    }
  if (cudaDevs != nvmlDevs) {
    snprintf(buf+strlen(buf), size-strlen(buf), " (NVML ");
    first = true;
    for (int i = 0; i < NCCL_MAX_LOCAL_RANKS; i++)
      if (nvmlDevs & (1UL << i)) {
        snprintf(buf+strlen(buf), size-strlen(buf), "%s%d", (first ? "" : ","), i);
        first = false;
      }
    snprintf(buf+strlen(buf), size-strlen(buf), ")");
  }
  return buf;
}

// Formats a GPU string based on the rasCollComms's rank.  If the CUDA id is different from the NVML id, both are
// printed.
static const char* rasCommRankGpuToString(const struct rasCollComms::comm::rank* rank, char* buf, size_t size) {
  snprintf(buf, size, "%d", rank->cudaDev);
  if (rank->cudaDev != rank->nvmlDev) {
    snprintf(buf+strlen(buf), size-strlen(buf), " (NVML %d)", rank->nvmlDev);
  }
  return buf;
}

// Converts a NCCL error result to a string.
static const char* ncclErrorToString(ncclResult_t err) {
  switch (err) {
    case ncclUnhandledCudaError     : return "Unhandled CUDA error";
    case ncclSystemError            : return "System error";
    case ncclInternalError          : return "Internal error";
    case ncclInvalidArgument        : return "Invalid argument";
    case ncclInvalidUsage           : return "Invalid usage";
    case ncclRemoteError            : return "Remote process error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "Unexpected error";
  }
}

// Converts the IP number of a NCCL address to a string (the port part is ignored and no DNS resolution is attempted).
static const char* ncclSocketToHost(const union ncclSocketAddress* addr, char* buf, size_t size) {
  if (addr->sa.sa_family > 0)
    return inet_ntop(addr->sa.sa_family,
                     (addr->sa.sa_family == AF_INET ? (void*)&addr->sin.sin_addr : (void*)&addr->sin6.sin6_addr),
                     buf, size);
  else {
    if (size > 0)
      buf[0] = '\0';
    return buf;
  }
}

// Determines if the given count constitutes an outlier.
static bool rasCountIsOutlier(int count, bool verbose, int totalCount) {
  if (count == 1)
    return true; // A single rank is always considered an outlier...
  if (verbose) {
    return (totalCount != -1 ? count < totalCount * RAS_CLIENT_VERBOSE_OUTLIER_FRACTION : true);
  } else {
    return count <= RAS_CLIENT_DETAIL_THRESHOLD &&
           (totalCount == -1 || count <= totalCount * RAS_CLIENT_OUTLIER_FRACTION);
  }
}
