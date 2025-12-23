/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdarg>
#include <cstddef>
#include <ctime>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "compiler.h"
#include "nccl.h"
#include "utils.h"
#include "ras_internal.h"
#include "compiler.h"

// Outlier count above which we don't print individual details about each of them.
#define RAS_CLIENT_DETAIL_THRESHOLD 10
// Fraction of the count of the total above which we don't consider another set to be an outlier.
#define RAS_CLIENT_OUTLIER_FRACTION 0.25
// Fraction of the count of the total below which a set is considered to be an outlier.
#define RAS_CLIENT_VERBOSE_OUTLIER_FRACTION 0.5

#define STR2(v) #v
#define STR(v) STR2(v)

// Generic auxiliary structure used when processing the results.  Helps with statistics gathering and sorting,
// e.g., for the calculation of the distribution of the number of peers per node, of the number of GPUs per peer,
// of the communicator sizes, or of the counts of collective operations.
struct rasValCount {
  uint64_t value; // The observed value.
  int count; // The number of occurences of this value in the results.
  int firstIdx; // The index of the first occurence of this value in the results.
};

// Communicator status, used in rasAuxComm below.  The values are bitmasks so that they can be combined.
typedef enum {
  RAS_ACS_NOCOMM = 1, // Set if the peer claims not to be a member of a given communicator.
  RAS_ACS_INIT = 2,
  RAS_ACS_RUNNING = 4,
  RAS_ACS_FINALIZE = 8,
  RAS_ACS_ABORT = 16
} rasACStatus;

// Communicator errors, used in rasAuxComm below.  The values are bitmasks so that they can be combined (with the
// exception of RAS_ACE_OK).
typedef enum {
  RAS_ACE_OK = 0,
  RAS_ACE_MISMATCH = 1,
  RAS_ACE_ERROR = 2,
  RAS_ACE_INCOMPLETE = 4
} rasACError;

// Auxiliary structure used when processing the results of the RAS_COLL_COMMS query.  For each communicator, caches
// statistics extracted from the results, such as the number of peers and nodes or the communicator status.  Includes
// a pointer to the communicator data in the results, making it easy to sort the communicators by a different key
// without altering the results buffer, or just to iterate over the communicators, given that the communicator data
// in the resuls is of variable length.
struct rasAuxComm {
  struct rasCollComms::comm* comm; // Points to the results buffer.
  int nPeers;
  int nNodes;
  int ranksPerNodeMin;
  int ranksPerNodeMax;
  unsigned int status; // Bitmask of rasACStatus values.
  unsigned int errors; // Bitmask of rasACError values.
  uint64_t firstCollOpCounts[NCCL_NUM_FUNCTIONS]; // collOpCounts of the first rank, to compare against.
  int nIncompleteRanks; // Number of ranks that we didn't get any response from.
};

// Auxiliary structure used when processing the rasPeerInfo data stored in the global rasPeers array.  Makes it possible
// to extract a subset of peers (e.g., the dead ones), to sort by a different key without altering the original array,
// and also has room for extracted temporary data such as the number of peers per node or the number of GPUs per peer.
struct rasAuxPeerInfo {
  struct rasPeerInfo* peer; // Points to an element in rasPeers.
  int value;
};

// Auxiliary structure used when processing the results of the RAS_COLL_COMMS query, specifically when iterating over
// each communicator's ranks.  Makes it possible to sort by a different key without altering the original array, and
// also has room for extracted temporary data such as the rank's status or a count of collective operations.
struct rasAuxCommRank {
  struct rasCollComms::comm::rank* rank; // Points to the results buffer.
  uint64_t value;
};

// The RAS client listening socket of this RAS thread (normally port 28028).
int rasClientListeningSocket = -1;

// Connected RAS clients.
struct rasClient* rasClientsHead;
struct rasClient* rasClientsTail;

// Minimum byte count to increment the output buffer size by if it's too small.
#define RAS_OUT_INCREMENT 4096

// Internal buffer for storing the formatted results.
static char* rasOutBuffer = nullptr;
static int nRasOutBuffer = 0; // Does _not_ include the terminating '\0' (which _is_ present in the buffer).
static int rasOutBufferSize = 0;

// We use them all over the place; no point in wasting the stack...
static char lineBuf[1024]; // Temporary buffer used for printing at most 10 (RAS_CLIENT_DETAIL_THRESHOLD) rank numbers
                           // or for printing the local GPU devices, which can't be more than 64
                           // small numbers (times two if the NVML mask is different than the CUDA mask).
                           // Still, 1024 should normally be plenty (verbose output may make things more difficult,
                           // but we do check for overflows, so it will just be trimmed).

// CUDA version information - shared across functions.
static int cudaDriverVersion = -1, cudaRuntimeVersion = -1;


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

static int rasAuxPeersValueCompare(const void* e1, const void* e2);
static int ncclSocketsHostCompare(const void* p1, const void* p2);
static int rasValCountsCompareRev(const void* p1, const void* p2);
static int rasAuxCommsCompareRev(const void* p1, const void* p2);
static int rasAuxCommRanksValueCompare(const void* p1, const void* p2);

static const char* rasGpuToString(int cudaDev, int nvmlDev, char* buf, size_t size);
static const char* rasCommRankGpuToString(const struct rasCollComms::comm::rank* rank, char* buf, size_t size);
static const char* ncclErrorToString(ncclResult_t err);
static bool rasCountIsOutlier(int count, bool verbose, int totalCount = -1);

// CUDA version information - shared across functions.
static void rasDumpCommsToJSON(struct rasClient* client, struct rasCollComms* commsData,
                               struct rasCollective* coll, const int* peerIdxConv);
static void jsonWriteHeader(const char* ncclVersion, int cudaRuntime, int cudaDriver,
                            const char* timestamp, int commsCount);
static void jsonStartCommunicator(unsigned long commHash, unsigned long hostHash, unsigned long pidHash,
                                  int commSize, int ranksCount, int missingCount, bool firstComm);
static void jsonWriteRankData(int rank, const char* host, int pid, int cudaDev, int nvmlDev,
                              int initState, int asyncError, bool finalizeCalled, bool destroyFlag,
                              bool abortFlag, const unsigned long* collCounts, bool firstRank);
static void jsonStartMissingRanks();
static void jsonWriteMissingRank(int rank, const char* host, int pid, int cudaDev, int nvmlDev,
                                 bool unresponsive, bool dead, bool firstMissing);
static void jsonEndCommunicator();
static void jsonWriteFooter(double collectionTime, int timeoutsCount);


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

  NCCLCHECK(ncclCalloc(&client, 1));

  client->sock = client->pfd = -1;
  ncclIntruQueueConstruct(&client->sendQ);
  client->timeout =  RAS_COLLECTIVE_LEG_TIMEOUT;
  client->outputFormat = RAS_OUTPUT_TEXT;
  client->monitorMask = 0; // Not in monitor mode.

  if (rasClientsHead) {
    rasClientsTail->next = client;
    client->prev = rasClientsTail;
    rasClientsTail = client;
  } else {
    rasClientsHead = rasClientsTail = client;
  }

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
  if (client->status != RAS_CLIENT_CLOSED && client->status < RAS_CLIENT_FINISHED) {
    ncclIntruQueueEnqueue(&client->sendQ, meta);
    rasPfds[client->pfd].events |= POLLOUT;
  } else {
    INFO(NCCL_RAS, "RAS invalid client status %d -- internal error?", client->status);
  }
}

// Terminates a connection with a RAS client.
static void rasClientTerminate(struct rasClient* client) {
  (void)close(client->sock);
  rasPfds[client->pfd].fd = NCCL_INVALID_SOCKET;
  rasPfds[client->pfd].events = rasPfds[client->pfd].revents = 0;
  while (struct rasMsgMeta* meta = ncclIntruQueueTryDequeue(&client->sendQ)) {
    free(meta);
  }

  if (client == rasClientsHead)
    rasClientsHead = rasClientsHead->next;
  if (client == rasClientsTail)
    rasClientsTail = rasClientsTail->prev;
  if (client->prev)
    client->prev->next = client->next;
  if (client->next)
    client->next->prev = client->prev;
  free(client);
}


//////////////////////////////////////////////////////////////////////
// Functions related to the asynchronous operations of RAS clients. //
//////////////////////////////////////////////////////////////////////

// Invoked when an asynchronous operation that a client was waiting on completes.  Finds the right client and
// reinvokes rasClientRun.
ncclResult_t rasClientResume(struct rasCollective* coll) {
  struct rasClient* client;

  for (client = rasClientsHead; client; client = client->next)
    if (client->coll == coll)
      break;
  if (client == nullptr) {
    INFO(NCCL_RAS, "RAS failed to find a matching client!");
    rasCollFree(coll);
    goto exit;
  }

  NCCLCHECK(rasClientRun(client));
exit:
  return ncclSuccess;
}

// Handles a ready client FD from the main event loop.
void rasClientEventLoop(struct rasClient* client, int pollIdx) {
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
      } else if (strncasecmp(cmd, "set format ", strlen("set format ")) == 0) {
        char* format = cmd + strlen("set format ");
        if (strcasecmp(format, "text") == 0) {
          client->outputFormat = RAS_OUTPUT_TEXT;
          strcpy(rasLine, "OK\n");
        } else if (strcasecmp(format, "json") == 0) {
          client->outputFormat = RAS_OUTPUT_JSON;
          strcpy(rasLine, "OK\n");
        } else {
          snprintf(rasLine, sizeof(rasLine), "ERROR: Invalid format %s\n", format);
        }
        msgLen = strlen(rasLine);
        if (rasClientAllocMsg(&msg, msgLen) != ncclSuccess) {
          rasClientTerminate(client);
          return;
        }
        memcpy(msg, rasLine, msgLen);
        rasClientEnqueueMsg(client, msg, msgLen);
      } else if (strcasecmp(cmd, "status") == 0) {
        client->status = RAS_CLIENT_INIT;
        (void)rasClientRun(client);
      } else if (strcasecmp(cmd, "verbose status") == 0) {
        client->status = RAS_CLIENT_INIT;
        client->verbose = 1;
        (void)rasClientRun(client);
      } else if (strcasecmp(cmd, "monitor") == 0 || strncasecmp(cmd, "monitor ", strlen("monitor ")) == 0) {
        // Parse optional event groups after MONITOR command.
        char* eventSpec = cmd + strlen("monitor");
        while (*eventSpec == ' ') eventSpec++;
        bool parseError = false;
        const char* errorToken = nullptr;
        uint8_t newMask = 0;
        if (*eventSpec) {
          // Event groups specified, parse them.
          char* savePtr = nullptr;
          char* token = strtok_r(eventSpec, ",", &savePtr);
          while (token) {
            if (strcasecmp(token, "lifecycle") == 0) {
              newMask |= RAS_EVENT_LIFECYCLE;
            } else if (strcasecmp(token, "trace") == 0) {
              newMask |= RAS_EVENT_TRACE;
            } else if (strcasecmp(token, "all") == 0) {
              newMask |= RAS_EVENT_ALL;
            } else {
              errorToken = token;
              parseError = true;
              break;
            }
            token = strtok_r(nullptr, ",", &savePtr);
          } // while (token)
        }
        if (parseError) {
          snprintf(rasLine, sizeof(rasLine), "ERROR: Invalid event group '%s'\n", errorToken);
        } else {
          if (newMask != 0) {
            client->monitorMask = newMask;
          } else {
            client->monitorMask = RAS_EVENT_LIFECYCLE;
          }
          strcpy(rasLine, "OK\n");
        }
        msgLen = strlen(rasLine);
        if (rasClientAllocMsg(&msg, msgLen) != ncclSuccess) {
          rasClientTerminate(client);
          return;
        }
        memcpy(msg, rasLine, msgLen);
        rasClientEnqueueMsg(client, msg, msgLen);
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
      NCCLCHECKGOTO(rasClientRunConns(client), ret, exit);
#endif
      client->status = RAS_CLIENT_COMMS;
      if (ret == ncclInProgress) {
        ret = ncclSuccess;
        break;
      }
    case RAS_CLIENT_COMMS:
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
  struct rasAuxPeerInfo* auxRasPeers = nullptr;
  int totalGpus, totalNodes, firstNGpusNode, firstNGpusGlobal, firstNPeersGlobal;
  bool consistentNGpusNode, consistentNGpusGlobal, consistentNPeersGlobal;
  int firstIdx, nPeers;
  struct rasValCount valCounts[NCCL_MAX_LOCAL_RANKS];
  int nValCounts;

  TRACE(NCCL_RAS, "RAS: rasClientRunInit: starting");

  rasOutReset();

  // Get CUDA version info once for all output formats.
  if (cudaRuntimeVersion == -1)
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
  if (cudaDriverVersion == -1)
    cudaDriverGetVersion(&cudaDriverVersion);

  // For structured formats (JSON), skip the initial text output.
  // It will be included in the structured output later.
  if (client->outputFormat == RAS_OUTPUT_TEXT) {
    rasOutAppend("NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX
                 " compiled with CUDA " STR(CUDA_MAJOR) "." STR(CUDA_MINOR) "\n");
    rasOutAppend("CUDA runtime version %d, driver version %d\n\n", cudaRuntimeVersion, cudaDriverVersion);
    msgLen = rasOutLength();
    NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
    rasOutExtract(msg);
    rasClientEnqueueMsg(client, msg, msgLen);
    msg = nullptr;
  }

  totalGpus = totalNodes = 0;
  firstNGpusNode = 0; // #GPUs on the first peer of a node.
  firstNGpusGlobal = 0; // #GPUs on peerIdx 0.
  consistentNGpusNode = true; // Whether #GPUs/peer is consistent between the peers *on any one node*.
  consistentNGpusGlobal = true; // Whether #GPUs/peer is consistent between the peers *on all nodes*.
  consistentNPeersGlobal = true; // Whether #peers/node is consistent between all nodes.
  nPeers = 0; // #peers on a node.
  firstNPeersGlobal = 0; // #peers on the first node.
  for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
    int nGpus = COMPILER_POPCOUNT64(rasPeers[peerIdx].cudaDevs);
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

  TRACE(NCCL_RAS, "RAS: totalNodes %d, nRasPeers %d, totalGpus %d", totalNodes, nRasPeers, totalGpus);
  TRACE(NCCL_RAS, "RAS: consistentNPeersGlobal %d, consistentNGpusGlobal %d, consistentNGpusNode %d",
        consistentNPeersGlobal, consistentNGpusGlobal, consistentNGpusNode);
  TRACE(NCCL_RAS, "RAS: firstNPeersGlobal %d, firstNGpusGlobal %d", firstNPeersGlobal, firstNGpusGlobal);

  // Only output job summary for text format.
  if (client->outputFormat == RAS_OUTPUT_TEXT) {
    rasOutAppend("Job summary\n"
                 "===========\n\n");

    if (consistentNGpusNode && consistentNGpusGlobal && consistentNPeersGlobal) {
      rasOutAppend("  Nodes  Processes         GPUs  Processes     GPUs\n"
                   "(total)   per node  per process    (total)  (total)\n"
                   "%7d"  "  %9d"    "  %11d"     "  %9d"    "  %7d\n",
                   totalNodes, firstNPeersGlobal, firstNGpusGlobal, nRasPeers, totalGpus);
    } else {
    // Gather the stats on the number of processes per node.  However, that number is not a property of a peer,
    // but of a group of peers, so calculating it is more involved.  We store the value in a temporary auxRasPeers
    // array.
    NCCLCHECKGOTO(ncclCalloc(&auxRasPeers, nRasPeers), ret, fail);

    firstIdx = 0;
    nPeers = 0;
    for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
      auxRasPeers[peerIdx].peer = rasPeers+peerIdx;
      if (peerIdx == 0) {
        nPeers = 1;
        firstIdx = 0;
      } else { // peerIdx > 0
        if (!ncclSocketsSameNode(&auxRasPeers[peerIdx].peer->addr, &auxRasPeers[peerIdx-1].peer->addr)) {
          TRACE(NCCL_RAS, "RAS: node %s: nPeers %d",
                ncclSocketToHost(&auxRasPeers[peerIdx].peer->addr, rasLine, sizeof(rasLine)), nPeers);
          for (int i = firstIdx; i < peerIdx; i++) {
            // Go back and update the number of processes of all the elements of that node.
            auxRasPeers[i].value = nPeers;
          }
          nPeers = 1;
          firstIdx = peerIdx;
        } else {
          nPeers++;
        }
      } // peerIdx > 0
      if (peerIdx == nRasPeers-1) {
        // Last iteration of the loop.
        TRACE(NCCL_RAS, "RAS: node %s: nPeers %d",
              ncclSocketToHost(&auxRasPeers[peerIdx].peer->addr, rasLine, sizeof(rasLine)), nPeers);
        for (int i = firstIdx; i < nRasPeers; i++) {
          auxRasPeers[i].value = nPeers;
        }
      }
    } // for (peerIdx)

    // Re-sort it now using the number of processes on the node (value) as the primary key, host IP as the
    // secondary, and process id as the tertiary.
    qsort(auxRasPeers, nRasPeers, sizeof(*auxRasPeers), rasAuxPeersValueCompare);

    // Calculate the distribution of different numbers of peers per node.
    nValCounts = 0;
    for (int peerIdx = 0; peerIdx < nRasPeers;) {
      if (peerIdx == 0 || auxRasPeers[peerIdx].value != auxRasPeers[peerIdx-1].value) {
        valCounts[nValCounts].value = auxRasPeers[peerIdx].value;
        valCounts[nValCounts].count = 1;
        valCounts[nValCounts].firstIdx = peerIdx;
        nValCounts++;
      } else {
        valCounts[nValCounts-1].count++;
      }
      // Advance peerIdx to the next node.
      peerIdx += auxRasPeers[peerIdx].value;
    } // for (peerIdx)
    // valCounts is currently sorted by value (the number of peers per node).  Sort it by the count (most frequent
    // number of peers first).
    qsort(valCounts, nValCounts, sizeof(*valCounts), rasValCountsCompareRev);

    // Print it out, the most frequent peer counts first.
    if (consistentNGpusNode && consistentNGpusGlobal) {
      // consistentNPeersGlobal must be false
      rasOutAppend("  Nodes  Processes         GPUs\n"
                   "          per node  per process\n");
      for (int i = 0; i < nValCounts; i++) {
        struct rasValCount* vc = valCounts+i;
        rasOutAppend("%7d  %9ld  %11d\n",
                     vc->count, vc->value, firstNGpusGlobal);
      }
    } else { // !consistentNGpusNode || !consistentNGpusGlobal
      rasOutAppend("  Nodes  Processes\n"
                   "          per node\n");
      for (int i = 0; i < nValCounts; i++) {
        struct rasValCount* vc = valCounts+i;
        rasOutAppend("%7d  %9ld\n",
                     vc->count, vc->value);
      }

      // We calculate and print the GPUs/process separately.  This is required for !consistentNGpusNode and
      // it also makes our life easier above for !consistentNGpusGlobal (which could require a larger valCounts).

      // Sort peers by the GPU count, to simplify data extraction.  Not sure how fast COMPILER_POPCOUNT64 is so we
      // may just as well cache it...
      for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
        auxRasPeers[peerIdx].value = COMPILER_POPCOUNT64(auxRasPeers[peerIdx].peer->cudaDevs);
        TRACE(NCCL_RAS, "RAS: node %s pid %d: nGpus %d",
              ncclSocketToHost(&auxRasPeers[peerIdx].peer->addr, rasLine, sizeof(rasLine)),
              auxRasPeers[peerIdx].peer->pid, auxRasPeers[peerIdx].value);
      }
      // GPU count is the primary key, host IP is the secondary, and process id is the tertiary.
      qsort(auxRasPeers, nRasPeers, sizeof(*auxRasPeers), rasAuxPeersValueCompare);

      // Calculate the distribution of different numbers of GPUs per peer.
      nValCounts = 0;
      for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++) {
        if (peerIdx == 0 || auxRasPeers[peerIdx].value != auxRasPeers[peerIdx-1].value) {
          valCounts[nValCounts].value = auxRasPeers[peerIdx].value;
          valCounts[nValCounts].count = 1;
          valCounts[nValCounts].firstIdx = peerIdx;
          nValCounts++;
        } else {
          valCounts[nValCounts-1].count++;
        }
      } // for (peerIdx)
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
    } // !consistentNGpusNode || !consistentNGpusGlobal
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
          // auxRasPeers is sorted by the node IP address (not port!) as the secondary key and the pid as
          // the tertiary, which comes in handy when printing...
          for (int peerIdx = vc->firstIdx; peerIdx < vc->count*vc->value + vc->firstIdx; peerIdx += vc->value) {
            lineBuf[0] = '\0';
            for (int j = 0; j < vc->value; j++) {
              snprintf(lineBuf+strlen(lineBuf), sizeof(lineBuf)-strlen(lineBuf), "%s%d",
                       (j > 0 ? "," : ""), auxRasPeers[j].peer->pid);
            }
            rasOutAppend("  Node %s running process%s %s\n",
                         ncclSocketToHost(&auxRasPeers[peerIdx].peer->addr, rasLine, sizeof(rasLine)),
                         (vc->value > 1 ? "es" : ""), lineBuf);
          } // for (peerIdx)
        } // if (rasCountIsOutlier(vc->count))
      } // for (i)
      } // !consistentNPeersGlobal
    } // !consistentNGpusNode || !consistentNGpusGlobal || !consistentNPeersGlobal
  } // TEXT format only

#if 0 // Commented out for now to focus the summary status report on the information most relevant to the users.
      // To be revisited with future extensions to RAS.
  rasOutAppend("\nGathering data about the RAS network (timeout %lds)...", client->timeout / CLOCK_UNITS_PER_SEC);
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;
  {
    struct rasCollRequest collReq = {};
    bool allDone = false;
    rasCollReqInit(&collReq);
    collReq.timeout = client->timeout;
    collReq.type = RAS_COLL_CONNS;
    NCCLCHECKGOTO(rasNetSendCollReq(&collReq, &allDone, &client->coll), ret, fail);
    if (!allDone)
      ret = ncclInProgress; // We need to wait for async. responses.
  }
#endif
  // Only send "Communicators..." message for text format.
  if (client->outputFormat == RAS_OUTPUT_TEXT) {
    rasOutAppend("\nCommunicators...");
    msgLen = rasOutLength();
    NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
    rasOutExtract(msg);
    rasClientEnqueueMsg(client, msg, msgLen);
    msg = nullptr;
  }
  {
    struct rasCollRequest collReq = {};
    bool allDone = false;
    rasCollReqInit(&collReq);
    collReq.timeout = client->timeout;
    collReq.type = RAS_COLL_COMMS;
    NCCLCHECKGOTO(rasNetSendCollReq(&collReq, &allDone, &client->coll), ret, fail);
    if (!allDone)
      ret = ncclInProgress;
  }
  TRACE(NCCL_RAS, "RAS: rasClientRunInit: scheduling RAS_COLL_COMMS and finishing");
exit:
  free(auxRasPeers);
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
  struct rasCollective* coll = client->coll;
  struct rasCollConns* connsData = (struct rasCollConns*)coll->data;
  int expected;
  struct rasPeerInfo* peersBuf = nullptr;

  if (coll == nullptr || coll->nFwdSent != coll->nFwdRecv) {
    INFO(NCCL_RAS, "RAS invalid collective operation status; client status %d -- internal error?", client->status);
    return ncclInternalError;
  }
  client->coll = nullptr;

  rasOutReset();
  rasOutAppend(" obtained a result in %.3fs\n", (clockNano()-coll->startTime)/1e9);
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
                     (COMPILER_POPCOUNT64(peersBuf[peerIdx].cudaDevs) > 1 ? "s" : ""),
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
                 "  Minimum %.3fs, maximum %.3fs, average %.3fs\n",
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
          rasOutAppend("  From node %s process %d to node %s process %d: observed travel time of %.3fs\n",
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
    struct rasCollRequest collReq = {};
    bool allDone = false;
    rasCollReqInit(&collReq);
    collReq.timeout = client->timeout;
    collReq.type = RAS_COLL_COMMS;
    NCCLCHECKGOTO(rasNetSendCollReq(&collReq, &allDone, &client->coll), ret, fail);
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
  struct rasCollective* coll = client->coll;
  struct rasCollComms* commsData = (struct rasCollComms*)coll->data;
  struct rasCollComms::comm* comm;
  struct rasAuxCommRank* auxCommRanks = nullptr;
  struct rasValCount* valCounts = nullptr;
  int nValCounts;
  struct rasValCount* collOpCounts = nullptr;
  struct rasAuxComm* auxComms = nullptr;
  int maxCommSize;
  int* peerIdxConv = nullptr;
  int vcIdx;
  int nPeersMissing;
  uint64_t* peerNvmlDevs = nullptr;
  const char*const statusStr[] = { "NOCOMM", "INIT", "RUNNING", "FINALIZE", "ABORT" };
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

  TRACE(NCCL_RAS, "RAS: rasClientRunComms: starting");
  TRACE(NCCL_RAS, "RAS: coll nLegTimeouts %d, nPeers %d, nData %d; commsData nComms %d",
        coll->nLegTimeouts, coll->nPeers, coll->nData, commsData->nComms);

  if (coll == nullptr || coll->nFwdSent != coll->nFwdRecv) {
    INFO(NCCL_RAS, "RAS invalid collective operation status; client status %d -- internal error?", client->status);
    return ncclInternalError;
  }
  client->coll = nullptr;

  rasOutReset();

  // Calculate the number of missing peers early as we rely on it for other things.
  nPeersMissing = nRasPeers - nRasDeadPeers - coll->nPeers;
  TRACE(NCCL_RAS, "RAS: nRasPeers %d, nRasDeadPeers %d, nPeersMissing %d", nRasPeers, nRasDeadPeers, nPeersMissing);

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
    comm = (struct rasCollComms::comm*)(((char*)(comm+1)) + comm->nRanks * sizeof(*comm->ranks) +
                                        comm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
  }
  NCCLCHECKGOTO(ncclCalloc(&auxCommRanks, maxCommSize), ret, fail);
  TRACE(NCCL_RAS, "RAS: maxCommSize %d", maxCommSize);

  // For convenience, create a translation table from rasCollective's peerIdx to rasPeers peerIdx.
  NCCLCHECKGOTO(ncclCalloc(&peerIdxConv, coll->nPeers), ret, fail);
  for (int peerIdx = 0; peerIdx < coll->nPeers; peerIdx++) {
    peerIdxConv[peerIdx] = rasPeerFind(coll->peers+peerIdx);
    TRACE(NCCL_RAS, "RAS: coll peers[%d] -> rasPeers[%d]", peerIdx, peerIdxConv[peerIdx]);
  }
  // Sort coll->peers to match the ordering of rasPeers -- we may need it later...
  qsort(coll->peers, coll->nPeers, sizeof(*coll->peers), &ncclSocketsCompare);

  // Check output format and call appropriate dump function for JSON.
  if (client->outputFormat == RAS_OUTPUT_JSON) {
    rasDumpCommsToJSON(client, commsData, coll, peerIdxConv);
    msgLen = rasOutLength();
    NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
    rasOutExtract(msg);
    rasClientEnqueueMsg(client, msg, msgLen);
    msg = nullptr;
    client->status = RAS_CLIENT_FINISHED;
    goto exit;
  }

  // Default TEXT format continues below.
  rasOutAppend(" (%.3fs)\n=============\n\n", (clockNano()-coll->startTime)/1e9);

  // Fill in the remaining fields of auxComm's.
  for (int commIdx = 0; commIdx < commsData->nComms; commIdx++) {
    struct rasAuxComm* auxComm = auxComms+commIdx;
    int nRanks = 0;
    comm = auxComm->comm;
    TRACE(NCCL_RAS, "RAS: coll comms[%d]: commId (0x%lx, 0x%lx, 0x%lx), commNRanks %d, nRanks %d, nMissingRanks %d",
          commIdx, comm->commId.commHash, comm->commId.hostHash, comm->commId.pidHash,
          comm->commNRanks, comm->nRanks, comm->nMissingRanks);

    if (comm->nMissingRanks > 0) {
      // There are two possibilities here.  Either we are missing the data on some ranks because the processes are
      // unreachable, or the processes _are_ reachable but didn't report to be part of this communicator (which
      // could definitely happen if some processes have already called ncclCommDestroy or ncclCommAbort).
      if (nPeersMissing == 0 && nRasDeadPeers == 0) {
        // We received data from _all_ processes.  That's an easy case.
        auxComm->errors |= RAS_ACE_MISMATCH;
        auxComm->status |= RAS_ACS_NOCOMM;
      } else {
        // We failed to receive data from some processes but we don't know if that's why we don't have the info about
        // some ranks of this communicator.  We need to check all the missing ranks one-by-one as different ranks may
        // have different reason.
        struct rasCollCommsMissingRank* missingRanks = (struct rasCollCommsMissingRank*)(comm->ranks+comm->nRanks);

        for (int rankIdx = 0; rankIdx < comm->nMissingRanks; rankIdx++) {
          struct rasCollCommsMissingRank* missingRank = missingRanks + rankIdx;
          void* found;
          if ((found = bsearch(&missingRank->addr, coll->peers, coll->nPeers, sizeof(*coll->peers),
                               ncclSocketsCompare)) != nullptr) {
            // We did receive the data from that process, but not about this communicator.
            auxComm->errors |= RAS_ACE_MISMATCH;
            auxComm->status |= RAS_ACS_NOCOMM;
          } else {
            // We failed to receive data from that process.
            auxComm->errors |= RAS_ACE_INCOMPLETE;
            auxComm->nIncompleteRanks++;
          }
          TRACE(NCCL_RAS, "RAS: comm missingRank[%d] commRank %d, addr %td (-> %d), cudaDev %d, nvmlDev %d",
                rankIdx, missingRank->commRank, (found ? ((union ncclSocketAddress*)found) - coll->peers: -1),
                rasPeerFind(&missingRank->addr), missingRank->cudaDev, missingRank->nvmlDev);
        } // for (rankIdx)
      } // nPeersMissing > 0 || nRasDeadPeers > 0
    } // if (comm->nMissingRanks > 0)

    // Initialize auxCommRanks from comm->rank, converting peerIdx to rasPeers, then sort by it -- that way we will
    // have the ranks sorted by node and process, which makes counting easy.
    for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
      struct rasCollComms::comm::rank* rank = comm->ranks+rankIdx;
      auxCommRanks[rankIdx].rank = rank;
      auxCommRanks[rankIdx].value = peerIdxConv[rank->peerIdx];
      TRACE(NCCL_RAS, "RAS: comm rank[%d] commRank %d, peerIdx %d (-> %d), cudaDev %d, nvmlDev %d",
            rankIdx, rank->commRank, rank->peerIdx, peerIdxConv[rank->peerIdx], rank->cudaDev, rank->nvmlDev);
      TRACE(NCCL_RAS, "RAS: comm rank[%d] collOpCounts (%ld, %ld, %ld, %ld, %ld)",
            rankIdx, rank->collOpCounts[0], rank->collOpCounts[1], rank->collOpCounts[2], rank->collOpCounts[3],
            rank->collOpCounts[4]);
      TRACE(NCCL_RAS, "RAS: comm rank[%d] status initState %d, asyncError %d, finalizeCalled %d, destroyFlag %d, "
            "abortFlag %d", rankIdx, rank->status.initState, rank->status.asyncError, rank->status.finalizeCalled,
            rank->status.destroyFlag, rank->status.abortFlag); /**/
    }
    // This also sorts by the commRank, which we don't care about here, but it won't hurt.
    qsort(auxCommRanks, comm->nRanks, sizeof(*auxCommRanks), rasAuxCommRanksValueCompare);

    // Count the peers and nodes, get the status/error indicators.
    for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
      struct rasAuxCommRank* auxRank = auxCommRanks+rankIdx;
      if (rankIdx == 0) {
        auxComm->nPeers = auxComm->nNodes = 1;
        auxComm->ranksPerNodeMin = NCCL_MAX_LOCAL_RANKS;
        auxComm->ranksPerNodeMax = 0;
        memcpy(auxComm->firstCollOpCounts, auxRank->rank->collOpCounts, sizeof(auxComm->firstCollOpCounts));
        nRanks = 1;
      } else { // rankIdx > 0
        if (auxRank->value != auxRank[-1].value) {
          auxComm->nPeers++;
          if (!ncclSocketsSameNode(&rasPeers[auxRank->value].addr, &rasPeers[auxRank[-1].value].addr)) {
            auxComm->nNodes++;
            if (auxComm->ranksPerNodeMin > nRanks)
              auxComm->ranksPerNodeMin = nRanks;
            if (auxComm->ranksPerNodeMax < nRanks)
              auxComm->ranksPerNodeMax = nRanks;
            nRanks = 0;
          }
        } // if (auxRank->value != auxRank[-1].value)
        nRanks++;
      } // rankIdx > 0
      if (rankIdx == comm->nRanks-1) {
        // Last iteration of the loop.
        if (auxComm->ranksPerNodeMin > nRanks)
          auxComm->ranksPerNodeMin = nRanks;
        if (auxComm->ranksPerNodeMax < nRanks)
          auxComm->ranksPerNodeMax = nRanks;
      }

      if (auxRank->rank->status.abortFlag)
        auxComm->status |= RAS_ACS_ABORT;
      else if (auxRank->rank->status.finalizeCalled || auxRank->rank->status.destroyFlag) {
        // destroyFlag is set by ncclCommDestroy and ncclCommAbort.  finalizeCalled appears to be set by
        // ncclCommFinalize only.  According to the docs, ncclCommDestroy *can* be called without calling
        // ncclCommFinalize first.  The code structure here ensures that we attribute destroyFlag properly
        // as a finalize state indicator (and ignore it in case of ncclCommAbort).
        auxComm->status |= RAS_ACS_FINALIZE;
      }
      else if (auxRank->rank->status.initState == ncclSuccess)
        auxComm->status |= RAS_ACS_RUNNING;
      else // auxRank->rank->initState != ncclSuccess
        auxComm->status |= RAS_ACS_INIT;

      for (int collIdx = 0; collIdx < NCCL_NUM_FUNCTIONS && !(auxComm->errors & RAS_ACE_MISMATCH); collIdx++) {
        if (auxRank->rank->collOpCounts[collIdx] != auxComm->firstCollOpCounts[collIdx])
          auxComm->errors |= RAS_ACE_MISMATCH;
      }
      if (auxRank->rank->status.initState != ncclSuccess && auxRank->rank->status.initState != ncclInProgress)
        auxComm->errors |= RAS_ACE_ERROR;
      if (auxRank->rank->status.asyncError != ncclSuccess && auxRank->rank->status.asyncError != ncclInProgress)
        auxComm->errors |= RAS_ACE_ERROR;
    } // for (rankIdx)

    if (COMPILER_POPCOUNT32(auxComm->status) > 1) {
      // We've got a status mismatch between ranks.
      auxComm->errors |= RAS_ACE_MISMATCH;
    }
    TRACE(NCCL_RAS, "RAS: auxComm nPeers %d, nNodes %d, nIncompleteRanks %d",
          auxComm->nPeers, auxComm->nNodes, auxComm->nIncompleteRanks);
    TRACE(NCCL_RAS, "RAS: auxComm ranksPerNodeMin %d, ranksPerNodeMax %d, status 0x%x, errors 0x%x",
          auxComm->ranksPerNodeMin, auxComm->ranksPerNodeMax, auxComm->status, auxComm->errors);
  } // for (commIdx)
  // Sort it by size/nNodes/status/errors/missing ranks.
  if (auxComms)
    qsort(auxComms, commsData->nComms, sizeof(*auxComms), &rasAuxCommsCompareRev);

  // Calculate the distribution of different communicator sizes.
  NCCLCHECKGOTO(ncclCalloc(&valCounts, commsData->nComms), ret, fail);
  nValCounts = 0;
  for (int commIdx = 0; commIdx < commsData->nComms; commIdx++) {
    if (commIdx == 0 ||
        auxComms[commIdx].comm->commNRanks != auxComms[commIdx-1].comm->commNRanks ||
        auxComms[commIdx].nNodes != auxComms[commIdx-1].nNodes ||
        // COMPILER_CLZ returns the number of leading 0-bits, which is a proxy for the index of the highest 1-bit.
        COMPILER_CLZ(auxComms[commIdx].status) != COMPILER_CLZ(auxComms[commIdx-1].status) ||
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

  TRACE(NCCL_RAS, "RAS: rasClientRunComms: done with initial data processing");

  if (commsData->nComms > 0) {
    rasOutAppend("Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors\n"
                 "    #  in group  per comm  per node  per comm  in group\n");
  } else {
    rasOutAppend("No communicator data collected!\n");
  }

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
      ranksTotal += COMPILER_POPCOUNT64(peerNvmlDevs[peerIdx]);
    if (ranksPerNodeMin == ranksPerNodeMax)
      snprintf(rasLine, sizeof(rasLine), "%d", ranksPerNodeMin);
    else
      snprintf(rasLine, sizeof(rasLine), "%d-%d", ranksPerNodeMin, ranksPerNodeMax);
    rasOutAppend("%5d  %8d  %8d  %8s  %8d  %8d  %8s  %6s\n",
                 vcIdx, vc->count, auxComm->nNodes, rasLine, auxComm->comm->commNRanks, ranksTotal,
                 // COMPILER_CLZ returns the number of leading 0-bits.  This makes it possible to translate the
                 // status (which is a bitmask) into an array index.
                 statusStr[(sizeof(unsigned int)*8-1)-COMPILER_CLZ(auxComm->status)], errorStr[auxComm->errors]);
  }
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;

  rasOutAppend("\nErrors\n"
               "======\n\n");

  if (nPeersMissing > 0) {
    rasOutAppend("INCOMPLETE\n"
                 "  Missing communicator data from %d job process%s\n", nPeersMissing, (nPeersMissing > 1 ? "es" : ""));
    if (rasCountIsOutlier(nPeersMissing, client->verbose)) {
      // Extract a list of missing peers.  We don't want to print it right away because it would be sorted
      // by address (including port, which isn't meaningful to end users).
      struct rasAuxPeerInfo* auxPeersBuf = nullptr;
      int nPeersBuf;

      // Both rasPeers and coll->peers are sorted by address (the latter we sorted above) which makes comparing
      // them much easier.
      NCCLCHECKGOTO(ncclCalloc(&auxPeersBuf, nPeersMissing), ret, fail);
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
          bool dead;
          if (!(dead = rasPeerIsDead(&rasPeers[rasPeerIdx].addr))) {
            if (nPeersBuf < nPeersMissing) {
              auxPeersBuf[nPeersBuf++].peer = rasPeers+rasPeerIdx;
            } else {
              INFO(NCCL_RAS, "RAS overflow of auxPeersBuf: nPeersBuf %d, rasPeerIdx %d (%s), collPeerIdx %d -- "
                   "internal error?",
                   nPeersBuf, rasPeerIdx, ncclSocketToString(&rasPeers[rasPeerIdx].addr, rasLine), collPeerIdx);
            }
          }
          TRACE(NCCL_RAS, "RAS rasPeerIdx %d (%s) is missing from coll->peers; dead %d",
                rasPeerIdx, ncclSocketToString(&rasPeers[rasPeerIdx].addr, rasLine), dead);
          rasPeerIdx++;
        } else { // cmp > 0
          // Process not found in rasPeers -- shouldn't happen, unless during a race?
          INFO(NCCL_RAS, "RAS failed to find coll->peer[%d] (%s) in rasPeers -- internal error?",
               collPeerIdx, ncclSocketToString(coll->peers+collPeerIdx, rasLine));
          collPeerIdx++;
        } // cmp > 0
      } // for (rasPeerIdx, collPeerIdx)

      // Sort the output by host and pid.  rasAuxPeersValueCompare uses value as the primary key, which is 0 for
      // all auxPeersBuf elements here, so it will do.
      qsort(auxPeersBuf, nPeersBuf, sizeof(*auxPeersBuf), rasAuxPeersValueCompare);
      for (int peerIdx = 0; peerIdx < nPeersBuf; peerIdx++) {
        struct rasAuxPeerInfo* auxPeer = auxPeersBuf+peerIdx;
        rasOutAppend("  Process %d on node %s managing GPU%s %s\n", auxPeer->peer->pid,
                     ncclSocketToHost(&auxPeer->peer->addr, rasLine, sizeof(rasLine)),
                     (COMPILER_POPCOUNT64(auxPeer->peer->cudaDevs) > 1 ? "s" : ""),
                     rasGpuDevsToString(auxPeer->peer->cudaDevs, auxPeer->peer->nvmlDevs, lineBuf,
                                        sizeof(lineBuf)));
      }
      if (nPeersBuf != nPeersMissing)
        rasOutAppend("  [could not find information on %d process%s]\n",
                     nPeersMissing-nPeersBuf, (nPeersMissing-nPeersBuf > 1 ? "es" : ""));
      free(auxPeersBuf);
    } // if (rasCountIsOutlier(nPeersMissing))
    rasOutAppend("\n");
  }

  if (nRasDeadPeers > 0) {
    rasOutAppend("DEAD\n"
                 "  %d job process%s considered dead (unreachable via the RAS network)\n", nRasDeadPeers,
                 (nRasDeadPeers > 1 ? "es are" : " is"));
    if (rasCountIsOutlier(nRasDeadPeers, client->verbose)) {
      // rasDeadPeers contains only addresses, whereas we want a complete rasPeerInfo, and sorted differently.
      struct rasAuxPeerInfo* auxPeersBuf = nullptr;
      int nPeersBuf = 0;
      NCCLCHECKGOTO(ncclCalloc(&auxPeersBuf, nRasDeadPeers), ret, fail);
      for (int i = 0; i < nRasDeadPeers; i++) {
        int peerIdx = rasPeerFind(rasDeadPeers+i);
        if (peerIdx != -1)
          auxPeersBuf[nPeersBuf++].peer = rasPeers+peerIdx;
      }
      // Sort the output by host and pid, not host and port.  rasAuxPeersValueCompare uses value as the primary key,
      // which is 0 for all auxPeersBuf elements here, so it will do.
      qsort(auxPeersBuf, nPeersBuf, sizeof(*auxPeersBuf), rasAuxPeersValueCompare);
      for (int peerIdx = 0; peerIdx < nPeersBuf; peerIdx++) {
        struct rasAuxPeerInfo* auxPeer = auxPeersBuf+peerIdx;
        rasOutAppend("  Process %d on node %s managing GPU%s %s\n", auxPeer->peer->pid,
                     ncclSocketToHost(&auxPeer->peer->addr, rasLine, sizeof(rasLine)),
                     (COMPILER_POPCOUNT64(auxPeer->peer->cudaDevs) > 1 ? "s" : ""),
                     rasGpuDevsToString(auxPeer->peer->cudaDevs, auxPeer->peer->nvmlDevs, lineBuf,
                                        sizeof(lineBuf)));
      }
      if (nPeersBuf != nRasDeadPeers)
        rasOutAppend("  [could not find information on %d process%s]\n",
                     nRasDeadPeers-nPeersBuf, (nRasDeadPeers-nPeersBuf > 1 ? "es" : ""));
      free(auxPeersBuf);
    } // if (rasCountIsOutlier(nRasDeadPeers)
    rasOutAppend("\n");
  }

  // Continue printing the largest communicators first, as in the summary table.
  for (vcIdx = 0; vcIdx < nValCounts; vcIdx++) {
    struct rasValCount* vc;
    vc = valCounts+vcIdx;
    for (int commIdx = vc->firstIdx; commIdx < vc->count + vc->firstIdx; commIdx++) {
      struct rasAuxComm* auxComm = auxComms+commIdx;
      comm = auxComm->comm;

      if (auxComm->errors & RAS_ACE_INCOMPLETE) {
        rasOutAppend("#%d-%d (%016lx) INCOMPLETE\n"
                     "  Missing communicator data from %d rank%s\n", vcIdx, commIdx - vc->firstIdx,
                     comm->commId.commHash, auxComm->nIncompleteRanks, (auxComm->nIncompleteRanks > 1 ? "s" : ""));
        if (rasCountIsOutlier(auxComm->nIncompleteRanks, client->verbose)) {
          struct rasCollCommsMissingRank* missingRanks = (struct rasCollCommsMissingRank*)(comm->ranks+comm->nRanks);
          for (int rankIdx = 0; rankIdx < comm->nMissingRanks; rankIdx++) {
            struct rasCollCommsMissingRank* missingRank = missingRanks + rankIdx;
            // Filter out ranks that provided a response but not for this communicator.
            if (bsearch(&missingRank->addr, coll->peers, coll->nPeers, sizeof(*coll->peers), ncclSocketsCompare) ==
                nullptr) {
              int peerIdx = rasPeerFind(&missingRank->addr);
              if (peerIdx != -1) {
                rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                             missingRank->commRank,
                             rasGpuToString(missingRank->cudaDev, missingRank->nvmlDev, lineBuf, sizeof(lineBuf)),
                             rasPeers[peerIdx].pid,
                             ncclSocketToHost(&missingRank->addr, rasLine, sizeof(rasLine)));
              } else {
                rasOutAppend("  Rank %d -- [process information not found]\n", missingRank->commRank);
              }
            } // if rank did not respond
          } // for (rankIdx)
        } // if (rasCountIsOutlier(nRanksMissing))
        rasOutAppend("\n");
      } // if (auxComm->errors & RAS_ACE_INCOMPLETE)

      if (auxComm->errors & RAS_ACE_ERROR) {
        int ncclErrors[ncclNumResults];
        int nErrors;
        rasOutAppend("#%d-%d (%016lx) ERROR\n", vcIdx, commIdx - vc->firstIdx, comm->commId.commHash);

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
  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;

  rasOutAppend("Warnings\n"
               "========\n\n");

  if (coll->nLegTimeouts > 0) {
    rasOutAppend("TIMEOUT\n"
                 "  Encountered %d communication timeout%s while gathering communicator data\n\n",
                 coll->nLegTimeouts, (coll->nLegTimeouts > 1 ? "s" : ""));
  }

  // Continue printing the largest communicators first, as in the summary table.
  for (int vcIdx = 0; vcIdx < nValCounts; vcIdx++) {
    struct rasValCount* vc = valCounts+vcIdx;
    for (int commIdx = vc->firstIdx; commIdx < vc->count + vc->firstIdx; commIdx++) {
      struct rasAuxComm* auxComm = auxComms+commIdx;
      comm = auxComm->comm;

      if (auxComm->errors & RAS_ACE_MISMATCH) {
        rasOutAppend("#%d-%d (%016lx) MISMATCH\n", vcIdx, commIdx - vc->firstIdx, comm->commId.commHash);

        if (collOpCounts == nullptr) {
          // Allocating comm->commNRanks elements ensures that we won't need to reallocate, because the valCounts
          // array is reverse-sorted by commNRanks.  On the other hand, for this purpose allocating commNRanks
          // elements may be massively overpessimistic...
          NCCLCHECKGOTO(ncclCalloc(&collOpCounts, comm->commNRanks), ret, fail);
        }

        if (COMPILER_POPCOUNT32(auxComm->status) > 1) {
          rasOutAppend("  Communicator ranks have different status\n");

          // We need to sort the ranks by status.  However, status is normally calculated from other fields.
          // We will store it in the auxCommRanks' value.
          for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
            struct rasCollComms::comm::rank* rank = comm->ranks+rankIdx;
            struct rasAuxCommRank* auxRank = auxCommRanks+rankIdx;
            auxRank->rank = rank;

            if (rank->status.abortFlag)
              auxRank->value = RAS_ACS_ABORT;
            else if (rank->status.finalizeCalled || rank->status.destroyFlag)
              auxRank->value = RAS_ACS_FINALIZE;
            else if (rank->status.initState == ncclSuccess)
              auxRank->value = RAS_ACS_RUNNING;
            else
              auxRank->value = RAS_ACS_INIT;
          }
          qsort(auxCommRanks, comm->nRanks, sizeof(*auxCommRanks), rasAuxCommRanksValueCompare);
          // Calculate the frequency of different status values.
          int nCollOpCounts = 0;
          for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
            if (rankIdx == 0 || auxCommRanks[rankIdx].value != auxCommRanks[rankIdx-1].value) {
              // COMPILER_CLZ returns the number of leading 0-bits.  This makes it possible to translate the
              // status (which is a bitmask) into an array index.  The argument is an unsigned int (there is no
              // 64-bit version seemingly, but we don't actually need one here).
              collOpCounts[nCollOpCounts].value =
                (sizeof(unsigned int)*8-1) - COMPILER_CLZ((unsigned int)auxCommRanks[rankIdx].value);
              collOpCounts[nCollOpCounts].count = 1;
              collOpCounts[nCollOpCounts].firstIdx = rankIdx;
              nCollOpCounts++;
            } else {
              collOpCounts[nCollOpCounts-1].count++;
            }
          }
          if (comm->nMissingRanks - auxComm->nIncompleteRanks > 0) {
            // Add a "fake" element corresponding to the NOCOMM entries, since they are not in the ranks array.
            collOpCounts[nCollOpCounts].value = 0; // The index of "NOCOMM" in statusStr.
            collOpCounts[nCollOpCounts].count = comm->nMissingRanks - auxComm->nIncompleteRanks;
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
                // auxCommRanks is sorted by commRank as the secondary key, which comes in handy when printing...
                for (int rankIdx = vcc->firstIdx; rankIdx < vcc->count+vcc->firstIdx; rankIdx++) {
                  int peerIdx = peerIdxConv[auxCommRanks[rankIdx].rank->peerIdx];
                  if (peerIdx != -1) {
                    if (vcc->count > 1)
                      rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                                   auxCommRanks[rankIdx].rank->commRank,
                                   rasCommRankGpuToString(auxCommRanks[rankIdx].rank, lineBuf, sizeof(lineBuf)),
                                   rasPeers[peerIdx].pid,
                                   ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                    else
                      rasOutAppend("  Rank %d has status %s -- GPU %s managed by process %d on node %s\n",
                                   auxCommRanks[rankIdx].rank->commRank, statusStr[vcc->value],
                                   rasCommRankGpuToString(auxCommRanks[rankIdx].rank, lineBuf, sizeof(lineBuf)),
                                   rasPeers[peerIdx].pid,
                                   ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                  } else { // peerIdx == -1
                    if (vcc->count > 1)
                      rasOutAppend("  Rank %d -- [process information not found]\n",
                                   auxCommRanks[rankIdx].rank->commRank);
                    else
                      rasOutAppend("  Rank %d has status %s -- [process information not found]\n",
                                   auxCommRanks[rankIdx].rank->commRank, statusStr[vcc->value]);
                  } // peerIdx == -1
                } // for (rankIdx)
              } else {
                // NOCOMM ranks are in a different array.
                struct rasCollCommsMissingRank* missingRanks = (struct rasCollCommsMissingRank*)(comm->ranks +
                                                                                                 comm->nRanks);
                for (int rankIdx = 0; rankIdx < comm->nMissingRanks; rankIdx++) {
                  struct rasCollCommsMissingRank* missingRank = missingRanks + rankIdx;
                  // Filter out ranks that did not respond at all.
                  if (bsearch(&missingRank->addr, coll->peers, coll->nPeers, sizeof(*coll->peers),
                              ncclSocketsCompare)) {
                    int peerIdx = rasPeerFind(&missingRank->addr);
                    if (peerIdx != -1) {
                      if (vcc->count > 1) {
                        rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                                     missingRank->commRank, rasGpuToString(missingRank->cudaDev, missingRank->nvmlDev,
                                                                           lineBuf, sizeof(lineBuf)),
                                     rasPeers[peerIdx].pid,
                                     ncclSocketToHost(&missingRank->addr, rasLine, sizeof(rasLine)));
                      } else {
                        rasOutAppend("  Rank %d has status %s -- GPU %s managed by process %d on node %s\n",
                                     missingRank->commRank, statusStr[vcc->value],
                                     rasGpuToString(missingRank->cudaDev, missingRank->nvmlDev,
                                                    lineBuf, sizeof(lineBuf)), rasPeers[peerIdx].pid,
                                     ncclSocketToHost(&missingRank->addr, rasLine, sizeof(rasLine)));
                      }
                    } else { // peerIdx == -1
                      if (vcc->count > 1) {
                        rasOutAppend("  Rank %d -- [process information not found]\n", missingRank->commRank);
                      } else {
                        rasOutAppend("  Rank %d has status %s -- [process information not found]\n",
                                     missingRank->commRank, statusStr[vcc->value]);
                      }
                    } // peerIdx == -1
                  } // if rank responded
                } // for (rankIdx)
              } // vcc->firstIdx == -1
            } // if (rasCountIsOutlier(vcc->count))
          } // for (coc)
        } // if (COMPILER_POPCOUNT32(auxComm->status) > 1)

        for (int collIdx = 0; collIdx < NCCL_NUM_FUNCTIONS; collIdx++) {
          bool inconsistent = false;

          for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
            if (comm->ranks[rankIdx].collOpCounts[collIdx] != auxComm->firstCollOpCounts[collIdx]) {
              inconsistent = true;
              break;
            }
          }

          if (inconsistent) {
            rasOutAppend("  Communicator ranks have different %s operation counts\n", ncclFuncStr[collIdx]);

            // Sort the ranks by collOpCounts[collIdx] and commRank for easy counting.
            for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
              struct rasCollComms::comm::rank* rank = comm->ranks+rankIdx;
              struct rasAuxCommRank* auxRank = auxCommRanks+rankIdx;
              auxRank->rank = rank;
              auxRank->value = rank->collOpCounts[collIdx];
            }
            qsort(auxCommRanks, comm->nRanks, sizeof(*auxCommRanks), rasAuxCommRanksValueCompare);
            // Calculate the frequency of different collOpCounts[collIdx] values.
            int nCollOpCounts = 0;
            for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
              if (rankIdx == 0 || auxCommRanks[rankIdx].value != auxCommRanks[rankIdx-1].value) {
                collOpCounts[nCollOpCounts].value = auxCommRanks[rankIdx].value;
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
              if (vcc->count > 1) {
                if (vcc->value > 0)
                  rasOutAppend("  %d ranks have launched up to operation %ld\n", vcc->count, vcc->value);
                else
                  rasOutAppend("  %d ranks have not launched any operations\n", vcc->count);
              }
              if (rasCountIsOutlier(vcc->count, client->verbose, comm->commNRanks)) {
                // auxCommRanks is sorted by commRank as the secondary key, which comes in handy when printing...
                for (int rankIdx = vcc->firstIdx; rankIdx < vcc->count+vcc->firstIdx; rankIdx++) {
                  int peerIdx = peerIdxConv[auxCommRanks[rankIdx].rank->peerIdx];
                  if (peerIdx != -1) {
                    if (vcc->count > 1) {
                      rasOutAppend("  Rank %d -- GPU %s managed by process %d on node %s\n",
                                   auxCommRanks[rankIdx].rank->commRank,
                                   rasCommRankGpuToString(auxCommRanks[rankIdx].rank, lineBuf, sizeof(lineBuf)),
                                   rasPeers[peerIdx].pid,
                                   ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                    } else {
                      if (vcc->value > 0) {
                        rasOutAppend("  Rank %d has launched up to operation %ld -- GPU %s managed by process %d "
                                     "on node %s\n", auxCommRanks[rankIdx].rank->commRank, vcc->value,
                                     rasCommRankGpuToString(auxCommRanks[rankIdx].rank, lineBuf, sizeof(lineBuf)),
                                     rasPeers[peerIdx].pid,
                                     ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                      } else {
                        rasOutAppend("  Rank %d has not launched any operations -- GPU %s managed by process %d "
                                     "on node %s\n", auxCommRanks[rankIdx].rank->commRank,
                                     rasCommRankGpuToString(auxCommRanks[rankIdx].rank, lineBuf, sizeof(lineBuf)),
                                     rasPeers[peerIdx].pid,
                                     ncclSocketToHost(&rasPeers[peerIdx].addr, rasLine, sizeof(rasLine)));
                      }
                    }
                  } else { // peerIdx == -1
                    if (vcc->count > 1) {
                      rasOutAppend("  Rank %d -- [process information not found]\n",
                                   auxCommRanks[rankIdx].rank->commRank);
                    } else {
                      if (vcc->value > 0)
                        rasOutAppend("  Rank %d has launched up to operation %ld -- [process information not found]\n",
                                     auxCommRanks[rankIdx].rank->commRank, vcc->value);
                      else
                        rasOutAppend("  Rank %d has not launched any operations -- [process information not found]\n",
                                     auxCommRanks[rankIdx].rank->commRank);
                    }
                  } // peerIdx == -1
                } // for (rankIdx)
              } // if (rasCountIsOutlier(vcc->count))
            } // for (coc)
            rasOutAppend("\n");
          } // if (inconsistent)
        } // for (collIdx)
      } // if (auxComm->errors & RAS_ACE_MISMATCH)
    } // for (commIdx)
  } // for (vcIdx)
  rasCollFree(coll);

  msgLen = rasOutLength();
  NCCLCHECKGOTO(rasClientAllocMsg(&msg, msgLen), ret, fail);
  rasOutExtract(msg);
  rasClientEnqueueMsg(client, msg, msgLen);
  msg = nullptr;

  TRACE(NCCL_RAS, "RAS: rasClientRunComms: finishing");
exit:
  free(peerNvmlDevs);
  free(collOpCounts);
  free(valCounts);
  free(peerIdxConv);
  free(auxCommRanks);
  free(auxComms);
  return ret;
fail:
  goto exit;
}

// Generates detailed info about encountered errors, be it initialization ones or asynchronous ones.
static void rasClientBreakDownErrors(struct rasClient* client, struct rasCollComms::comm* comm,
                                     const int* peerIdxConv, int ncclErrors[ncclNumResults], bool isAsync) {
  // Because the number of possible error kinds is finite and small, we don't bother in this case with allocating
  // temporary data structures, counting the errors, sorting arrays, etc.  Instead, in each iteration we pick the most
  // numerous error kind, we iterate through the ranks in search for this error, and immediately add it to the output.
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
  if (nRasOutBuffer >= rasOutBufferSize)
    nRasOutBuffer = rasOutBufferSize - 1; // Should never happen, but just to be extra sure...
exit:
  ;
}

// Copies the output data from an internal buffer to a user-supplied one, including the terminating '\0'.
// The user buffer must already be allocated and be at least rasOutLength() bytes long (which includes
// the terminating '\0').  Resets the output buffer when done.
static void rasOutExtract(char* buffer) {
  if (rasOutBuffer) {
    memcpy(buffer, rasOutBuffer, rasOutLength());
    rasOutReset();
  }
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

// Sorting callback for rasAuxPeerInfo elements.  Sorts by value, with the peers host IP as the secondary key and
// the process id as the tertiary key.
static int rasAuxPeersValueCompare(const void* e1, const void* e2) {
  const struct rasAuxPeerInfo* p1 = (const struct rasAuxPeerInfo*)e1;
  const struct rasAuxPeerInfo* p2 = (const struct rasAuxPeerInfo*)e2;

  if (p1->value == p2->value) {
    // Host IP address is the secondary key.
    int cmp = ncclSocketsHostCompare(&p1->peer->addr, &p2->peer->addr);
    if (cmp == 0) {
      // Process ID is the tertiary key.
      cmp = (p1->peer->pid < p2->peer->pid ? -1 : (p1->peer->pid > p2->peer->pid ? 1 : 0));
    }
    return cmp;
  } else {
    return (p1->value < p2->value ? -1 : 1);
  }
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
    if (family != 0)
      INFO(NCCL_RAS, "RAS invalid address family %d -- internal error?", family);
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
      // COMPILER_CLZ returns the number of leading 0-bits, so in our case the value will be the *smallest*
      // if RAS_ACS_ABORT (8) is set and the *largest* if only RAS_ACS_INIT (1) is set, so we reverse the
      // comparison to get the desired sorting order.
      int s1 = COMPILER_CLZ(c1->status);
      int s2 = COMPILER_CLZ(c2->status);
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

// Sorting callback for rasAuxCommRank elements.  Sorts by value, with rank's commRank as the secondary key.
static int rasAuxCommRanksValueCompare(const void* p1, const void* p2) {
  const struct rasAuxCommRank* r1 = (const struct rasAuxCommRank*)p1;
  const struct rasAuxCommRank* r2 = (const struct rasAuxCommRank*)p2;

  if (r1->value == r2->value) {
    // Use the commRank as the secondary key.
    return (r1->rank->commRank < r2->rank->commRank ? -1 : (r1->rank->commRank > r2->rank->commRank ? 1 : 0));
  } else {
    return (r1->value < r2->value ? -1 : 1);
  }
}


//////////////////////////////////////////////////////////
// JSON utility functions for structured output format. //
//////////////////////////////////////////////////////////

// Writes the JSON document header with metadata and opens the communicators array.
static void jsonWriteHeader(const char* ncclVersion, int cudaRuntime, int cudaDriver,
                            const char* timestamp, int commsCount) {
  rasOutAppend("{\n");
  rasOutAppend("  \"nccl_version\": \"%s\",\n", ncclVersion);
  rasOutAppend("  \"cuda_runtime_version\": %d,\n", cudaRuntime);
  rasOutAppend("  \"cuda_driver_version\": %d,\n", cudaDriver);
  rasOutAppend("  \"timestamp\": \"%s\",\n", timestamp);
  rasOutAppend("  \"communicators_count\": %d,\n", commsCount);
  rasOutAppend("  \"communicators\": [\n");
}

// Starts a new communicator entry in the JSON output.
static void jsonStartCommunicator(unsigned long commHash, unsigned long hostHash, unsigned long pidHash,
                                  int commSize, int ranksCount, int missingCount, bool firstComm) {
  if (!firstComm) rasOutAppend(",\n");
  rasOutAppend("    {\n");
  rasOutAppend("      \"hash\": \"0x%lx\",\n", commHash);
  rasOutAppend("      \"secondary_hash\": \"0x%lx:0x%lx\",\n", hostHash, pidHash);
  rasOutAppend("      \"size\": %d,\n", commSize);
  rasOutAppend("      \"ranks_count\": %d,\n", ranksCount);
  rasOutAppend("      \"missing_ranks_count\": %d,\n", missingCount);
  rasOutAppend("      \"ranks\": [\n");
}

// Writes detailed rank information including status and collective operation counts.
static void jsonWriteRankData(int rank, const char* host, int pid, int cudaDev, int nvmlDev,
                              int initState, int asyncError, bool finalizeCalled, bool destroyFlag,
                              bool abortFlag, const unsigned long* collCounts, bool firstRank) {
  if (!firstRank) rasOutAppend(",\n");
  rasOutAppend("        {\n");
  rasOutAppend("          \"rank\": %d,\n", rank);
  rasOutAppend("          \"host\": \"%s\",\n", host);
  rasOutAppend("          \"pid\": %d,\n", pid);
  rasOutAppend("          \"cuda_dev\": %d,\n", cudaDev);
  rasOutAppend("          \"nvml_dev\": %d,\n", nvmlDev);

  // Status object.
  rasOutAppend("          \"status\": {\n");
  rasOutAppend("            \"init_state\": %d,\n", initState);
  rasOutAppend("            \"async_error\": %d,\n", asyncError);
  rasOutAppend("            \"finalize_called\": %s,\n", finalizeCalled ? "true" : "false");
  rasOutAppend("            \"destroy_flag\": %s,\n", destroyFlag ? "true" : "false");
  rasOutAppend("            \"abort_flag\": %s\n", abortFlag ? "true" : "false");
  rasOutAppend("          },\n");

  // Collective counts object.
  rasOutAppend("          \"collective_counts\": {\n");
  for (int op = 0; op < NCCL_NUM_FUNCTIONS; op++) {
    rasOutAppend("            \"%s\": %lu", ncclFuncToString((ncclFunc_t)op), collCounts[op]);
    if (op < NCCL_NUM_FUNCTIONS - 1) rasOutAppend(",");
    rasOutAppend("\n");
  }
  rasOutAppend("          }\n");
  rasOutAppend("        }");
}

// Closes the ranks array and starts the missing ranks section.
static void jsonStartMissingRanks() {
  rasOutAppend("\n"
               "      ],\n");

  rasOutAppend("      \"missing_ranks\": [\n");
}

// Writes basic information for a missing rank.
static void jsonWriteMissingRank(int rank, const char* host, int pid, int cudaDev, int nvmlDev,
                                 bool unresponsive, bool dead, bool firstMissing) {
  if (!firstMissing) rasOutAppend(",\n");
  rasOutAppend("        {\n");
  rasOutAppend("          \"rank\": %d,\n", rank);
  rasOutAppend("          \"host\": \"%s\",\n", host);
  rasOutAppend("          \"pid\": %d,\n", pid);
  rasOutAppend("          \"cuda_dev\": %d,\n", cudaDev);
  rasOutAppend("          \"nvml_dev\": %d,\n", nvmlDev);

  // Status object.
  rasOutAppend("          \"status\": {\n");
  rasOutAppend("            \"unresponsive\": %s,\n", unresponsive ? "true" : "false");
  rasOutAppend("            \"considered_dead\": %s\n", dead ? "true" : "false");
  rasOutAppend("          }\n");
  rasOutAppend("        }");
}

// Closes the current communicator entry.
static void jsonEndCommunicator() {
  rasOutAppend("\n      ]\n");
  rasOutAppend("    }");
}

// Writes the JSON document footer with RAS metadata and closes the document.
static void jsonWriteFooter(double collectionTime, int timeoutsCount) {
  rasOutAppend("\n  ],\n");
  rasOutAppend("  \"ras\": {\n");
  rasOutAppend("    \"collection_time_sec\": %.3f,\n", collectionTime);
  rasOutAppend("    \"timeouts_count\": %d\n", timeoutsCount);
  rasOutAppend("  }\n");
  rasOutAppend("}\n");
}


////////////////////////////////////////////////////////////
// String formatting functions for various types of data. //
////////////////////////////////////////////////////////////

// Coverts a GPU mask(s) to a string.  If the CUDA mask is different from the NVML mask, both are printed.
const char* rasGpuDevsToString(uint64_t cudaDevs, uint64_t nvmlDevs, char* buf, size_t size) {
  bool first = true;
  buf[0] = '\0';
  for (int i = 0; i < sizeof(cudaDevs)*8; i++)
    if (cudaDevs & (1UL << i)) {
      snprintf(buf+strlen(buf), size-strlen(buf), "%s%d", (first ? "" : ","), i);
      first = false;
    }
  if (cudaDevs != nvmlDevs) {
    snprintf(buf+strlen(buf), size-strlen(buf), " (NVML ");
    first = true;
    for (int i = 0; i < sizeof(nvmlDevs)*8; i++)
      if (nvmlDevs & (1UL << i)) {
        snprintf(buf+strlen(buf), size-strlen(buf), "%s%d", (first ? "" : ","), i);
        first = false;
      }
    snprintf(buf+strlen(buf), size-strlen(buf), ")");
  }
  return buf;
}

// Formats a GPU string based on the CUDA/NVML ids provided.  If the CUDA id is different from the NVML id, both are
// printed.
static const char* rasGpuToString(int cudaDev, int nvmlDev, char* buf, size_t size) {
  snprintf(buf, size, "%d", cudaDev);
  if (cudaDev != nvmlDev) {
    snprintf(buf+strlen(buf), size-strlen(buf), " (NVML %d)", nvmlDev);
  }
  return buf;
}

// Formats a GPU string based on the rasCollComms's rank.  If the CUDA id is different from the NVML id, both are
// printed.
static const char* rasCommRankGpuToString(const struct rasCollComms::comm::rank* rank, char* buf, size_t size) {
  return rasGpuToString(rank->cudaDev, rank->nvmlDev, buf, size);
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
const char* ncclSocketToHost(const union ncclSocketAddress* addr, char* buf, size_t size) {
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

// Dump communicator data to JSON format - build JSON in output buffer.
static void rasDumpCommsToJSON(struct rasClient* client, struct rasCollComms* commsData,
                               struct rasCollective* coll, const int* peerIdxConv) {
  char hostBuf[256], timeBuf[64];

  time_t timestampSec = time(NULL);
  struct tm tmBuffer;
  struct tm* tmInfo = localtime_r(&timestampSec, &tmBuffer);
  strftime(timeBuf, sizeof(timeBuf), "%F %T", tmInfo);

  // Write JSON header with metadata.
  jsonWriteHeader(STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX,
                  cudaRuntimeVersion, cudaDriverVersion, timeBuf, commsData->nComms);

  struct rasCollComms::comm* comm = commsData->comms;

  // Iterate through communicators.
  for (int commIdx = 0; commIdx < commsData->nComms; commIdx++) {
    jsonStartCommunicator(comm->commId.commHash, comm->commId.hostHash, comm->commId.pidHash,
                          comm->commNRanks, comm->nRanks, comm->nMissingRanks, (commIdx == 0));

    // Add each rank.
    for (int rankIdx = 0; rankIdx < comm->nRanks; rankIdx++) {
      struct rasCollComms::comm::rank* rank = comm->ranks + rankIdx;

      // Get host and pid information.
      const char* host = "unknown";
      int pid = -1;
      if (rank->peerIdx >= 0 && peerIdxConv && peerIdxConv[rank->peerIdx] >= 0) {
        int rasPeerIdx = peerIdxConv[rank->peerIdx];
        host = ncclSocketToHost(&rasPeers[rasPeerIdx].addr, hostBuf, sizeof(hostBuf));
        pid = rasPeers[rasPeerIdx].pid;
      }

      jsonWriteRankData(rank->commRank, host, pid, rank->cudaDev, rank->nvmlDev,
                        rank->status.initState, rank->status.asyncError,
                        rank->status.finalizeCalled, rank->status.destroyFlag, rank->status.abortFlag,
                        rank->collOpCounts, (rankIdx == 0));
    }

    // Start missing ranks section.
    jsonStartMissingRanks();

    // Add missing ranks.
    struct rasCollCommsMissingRank* missingRanks = (struct rasCollCommsMissingRank*)(comm->ranks + comm->nRanks);
    for (int missingIdx = 0; missingIdx < comm->nMissingRanks; missingIdx++) {
      struct rasCollCommsMissingRank* missingRank = missingRanks + missingIdx;

      // Get host and pid for missing rank.
      int rasPeerIdx = rasPeerFind(&missingRank->addr);
      const char* host = "unknown";
      int pid = -1;
      bool unresponsive = (bsearch(&missingRank->addr, coll->peers, coll->nPeers, sizeof(*coll->peers),
                                   ncclSocketsCompare) == nullptr);
      bool dead = rasPeerIsDead(&missingRank->addr);
      if (rasPeerIdx >= 0) {
        host = ncclSocketToHost(&rasPeers[rasPeerIdx].addr, hostBuf, sizeof(hostBuf));
        pid = rasPeers[rasPeerIdx].pid;
      }

      jsonWriteMissingRank(missingRank->commRank, host, pid, missingRank->cudaDev, missingRank->nvmlDev,
                           unresponsive, dead, (missingIdx == 0));
    }

    jsonEndCommunicator();

    // Move to the next communicator.
    comm = (struct rasCollComms::comm*)(((char*)(comm+1)) + comm->nRanks * sizeof(*comm->ranks) +
                                        comm->nMissingRanks * sizeof(struct rasCollCommsMissingRank));
  }

  // Write JSON footer with RAS metadata.
  jsonWriteFooter((clockNano()-coll->startTime)/1e9, coll->nLegTimeouts);
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

// Notifies all connected monitoring clients about an event.
// Iterates through the client list and sends formatted notifications to clients
// whose monitor mask matches the event group.
void rasClientsNotifyEvent(rasEventGroup group, const struct rasEventNotification* event) {
  char* msg;
  char formattedNotification[2048];
  time_t currentTime;
  struct tm timeInfo;
  char timeStr[64];
  int msgLen;
  time(&currentTime);
  localtime_r(&currentTime, &timeInfo);
  strftime(timeStr, sizeof(timeStr), "%F %T", &timeInfo);
  for (struct rasClient* client = rasClientsHead; client; client = client->next) {
    if (client->status == RAS_CLIENT_CONNECTED && (client->monitorMask & group)) {
      if (client->outputFormat == RAS_OUTPUT_JSON) {
        const char* groupStr = (group == RAS_EVENT_LIFECYCLE) ? "LIFECYCLE" : "TRACE";
        const struct rasPeerInfo* peer = event->peerInfo;
        if (peer == nullptr && event->peerAddr) {
          int peerIdx = rasPeerFind(event->peerAddr);
          if (peerIdx >= 0) {
            peer = rasPeers + peerIdx;
          }
        }
        if (peer) {
          char hostBuf[SOCKET_NAME_MAXLEN+1];
          char cudaDevs[256], nvmlDevs[256];
          ncclSocketToHost(&peer->addr, hostBuf, sizeof(hostBuf));
          // Build cuda_devs array.
          int offset = 0;
          snprintf(cudaDevs + offset, sizeof(cudaDevs) - offset, "[");
          offset = strlen(cudaDevs);
          for (int i = 0; i < sizeof(peer->cudaDevs)*8; i++) {
            if (peer->cudaDevs & (1ULL << i)) {
              snprintf(cudaDevs + offset, sizeof(cudaDevs) - offset, "%s%d", (offset > 1 ? "," : ""), i);
              offset = strlen(cudaDevs);
            }
          }
          snprintf(cudaDevs + offset, sizeof(cudaDevs) - offset, "]");
          // Build nvml_devs array.
          offset = 0;
          snprintf(nvmlDevs + offset, sizeof(nvmlDevs) - offset, "[");
          offset = strlen(nvmlDevs);
          for (int i = 0; i < sizeof(peer->nvmlDevs)*8; i++) {
            if (peer->nvmlDevs & (1ULL << i)) {
              snprintf(nvmlDevs + offset, sizeof(nvmlDevs) - offset, "%s%d", (offset > 1 ? "," : ""), i);
              offset = strlen(nvmlDevs);
            }
          }
          snprintf(nvmlDevs + offset, sizeof(nvmlDevs) - offset, "]");
          snprintf(formattedNotification, sizeof(formattedNotification),
                   "{\n"
                   "  \"timestamp\": \"%s\",\n"
                   "  \"group\": \"%s\",\n"
                   "  \"event\": \"%s\",\n"
                   "  \"peer\": {\n"
                   "    \"host\": \"%s\",\n"
                   "    \"pid\": %d,\n"
                   "    \"cuda_devs\": %s,\n"
                   "    \"nvml_devs\": %s\n"
                   "  },\n"
                   "  \"details\": \"%s\"\n"
                   "}\n",
                   timeStr, groupStr, event->eventType, hostBuf, peer->pid, cudaDevs, nvmlDevs, event->details);
        } else if (event->peerAddr) {
          // Peer not found, just use address.
          char addrStr[SOCKET_NAME_MAXLEN+1];
          ncclSocketToString(event->peerAddr, addrStr, sizeof(addrStr));
          snprintf(formattedNotification, sizeof(formattedNotification),
                   "{\n"
                   "  \"timestamp\": \"%s\",\n"
                   "  \"group\": \"%s\",\n"
                   "  \"event\": \"%s\",\n"
                   "  \"peer\": {\n"
                   "    \"addr\": \"%s\"\n"
                   "  },\n"
                   "  \"details\": \"%s\"\n"
                   "}\n",
                   timeStr, groupStr, event->eventType, addrStr, event->details);
        } else {
          snprintf(formattedNotification, sizeof(formattedNotification),
                   "{\n"
                   "  \"timestamp\": \"%s\",\n"
                   "  \"group\": \"%s\",\n"
                   "  \"event\": \"%s\",\n"
                   "  \"details\": \"%s\"\n"
                   "}\n",
                   timeStr, groupStr, event->eventType, event->details);
        }
      } else { // client->outputFormat == RAS_OUTPUT_TEXT
        char peerStr[512] = "";
        if (event->peerInfo) {
          rasPeerInfoToString(event->peerInfo, peerStr, sizeof(peerStr));
        } else if (event->peerAddr) {
          rasPeerToString(event->peerAddr, peerStr, sizeof(peerStr));
        }
        if (strlen(peerStr) > 0) {
          snprintf(formattedNotification, sizeof(formattedNotification),
                   "[%s] %s: %s %s\n", timeStr, event->eventType, peerStr, event->details);
        } else {
          snprintf(formattedNotification, sizeof(formattedNotification),
                   "[%s] %s: %s\n", timeStr, event->eventType, event->details);
        }
      }

      msgLen = strlen(formattedNotification);
      if (rasClientAllocMsg(&msg, msgLen) == ncclSuccess) {
        memcpy(msg, formattedNotification, msgLen);
        rasClientEnqueueMsg(client, msg, msgLen);
      }
    }
  }
}

// Invoked during RAS termination to release all the allocated resources.
void rasClientSupportTerminate() {
  (void)close(rasClientListeningSocket);
  rasClientListeningSocket = -1;

  free(rasOutBuffer);
  rasOutBuffer = nullptr;
  nRasOutBuffer = rasOutBufferSize = 0;

  for (struct rasClient* client = rasClientsHead; client;) {
    struct rasClient* clientNext = client->next;
    rasClientTerminate(client);
    client = clientNext;
  }

  // rasClientsHead and rasClientsTail are taken care of by rasClientTerminate().
}
