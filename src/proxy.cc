/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"
#include "shmutils.h"
#define ENABLE_TIMER 0
#include "timer.h"
#include "profiler.h"
#include "transport.h"
#include "cpuset.h"
#include "compiler.h"
#include "os.h"

#include <sys/syscall.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <sched.h>
#include <algorithm>
#include <mutex>

#define NCCL_MAX_PROXY_CONNECTIONS (NCCL_MAX_LOCAL_RANKS+1)

enum { proxyRecv=0, proxySend=1 };
void* ncclProxyServiceUDS(void* _args);

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  /* Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
      /* reduce */ (type == proxyRecv ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

#define PROXYARGS_ALLOCATE_SIZE NCCL_MAX_OPS
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

static void expectedProxyResponseFree(struct ncclProxyState* state) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  struct ncclExpectedProxyResponse* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev->respBuff);
    free(prev);
  }
}

static ncclResult_t expectedProxyResponseStore(struct ncclProxyState* state, void* opId, void* respBuff, int respSize, ncclResult_t res) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  while (elem) {
    if (elem->opId == opId) {
      if (respSize != elem->respSize) {
        WARN("Mismatched response size for opId=%p", opId);
        return ncclInternalError;
      }

      if (elem->done) {
        WARN("Storing response for already completed opId=%p", opId);
        return ncclInternalError;
      }

      if (respSize > 0) {
        memcpy(elem->respBuff, respBuff, respSize);
        free(respBuff);
      }
      elem->done = true;
      elem->res  = res;
      return ncclSuccess;
    }
    elem = elem->next;
  }

  WARN("Proxy response for opId=%p doesn't match any expected response", opId);
  return ncclInternalError;
}

static ncclResult_t expectedProxyResponseEnqueue(struct ncclProxyState* state, void* opId, int respSize) {
  struct ncclExpectedProxyResponse* ex;
  NCCLCHECK(ncclCalloc(&ex, 1));
  ex->opId = opId;

  // Pre-alloc response buffer
  ex->respBuff = malloc(respSize);
  ex->respSize = respSize;
  ex->res      = ncclInternalError;
  ex->done     = false;

  // Enqueue
  struct ncclExpectedProxyResponse* list = state->expectedResponses;
  if (list == NULL) {
    state->expectedResponses = ex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = ex;
  return ncclSuccess;
}

static ncclResult_t expectedProxyResponseDequeue(struct ncclProxyState* state, void* opId, void* respBuff, int* found) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  struct ncclExpectedProxyResponse* prev = NULL;
  *found = 0;
  while (elem) {
    if ((elem->opId == opId) && elem->done) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(respBuff, elem->respBuff, elem->respSize);
      ncclResult_t res = elem->res;
      free(elem->respBuff);
      free(elem);
      *found = 1;
      return res;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static ncclResult_t expectedProxyResponseRemove(struct ncclProxyState* state, void* opId) {
  struct ncclExpectedProxyResponse* elem = state->expectedResponses;
  struct ncclExpectedProxyResponse* prev = NULL;
  while (elem) {
    if (elem->opId == opId) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      free(elem->respBuff);
      free(elem);
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  WARN("Couldn't find opId=%p", opId);
  return ncclInternalError;
}

static ncclResult_t asyncProxyOpEnqueue(struct ncclProxyLocalPeer* peer, ncclProxyAsyncOp* op) {
  ncclProxyAsyncOp* list = peer->asyncOps;
  if (list == NULL) {
    peer->asyncOps = op;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = op;
  return ncclSuccess;
}

static ncclResult_t asyncProxyOpDequeue(struct ncclProxyLocalPeer* peer, ncclProxyAsyncOp* op) {
  struct ncclProxyAsyncOp* elem = peer->asyncOps;
  struct ncclProxyAsyncOp* prev = NULL;
  while (elem) {
    if (elem->opId == op->opId) {
      if (prev == NULL) {
        peer->asyncOps = elem->next;
      } else {
        prev->next = elem->next;
      }

      if (elem->reqBuff) {
        free(elem->reqBuff);
      }
      if (elem->respBuff) {
        free(elem->respBuff);
      }
      free(elem);

      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  if (op) {
    WARN("Attempting to dequeue nonexistent async opId=%p", op->opId);
  } else {
    WARN("Attempting to dequeue null operation");
  }
  return ncclInternalError;
}

static ncclResult_t allocateArgs(struct ncclProxyProgressState* state, struct ncclProxyArgs** argsptr) {
  struct ncclProxyArgs* elem;
  if (state->pool == NULL) {
    // Allocate a new pool of elements. Make sure we allocate the memory close
    // to the network thread
    struct ncclProxyPool* newPool;
    NCCLCHECK(ncclCalloc(&newPool, 1));

    struct ncclProxyArgs* newElems = newPool->elems;
    // Chain newly allocated elements
    for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
      if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
    }
    // Add them all to the pool list
    state->pool = newElems;
    // Save the pool memory block for later resource release
    newPool->next = state->pools;
    state->pools = newPool;
  }
  elem = state->pool;
  state->pool = state->pool->next;
  elem->next = elem->nextPeer = NULL;
  *argsptr = elem;
  return ncclSuccess;
}

//#define DEBUG_PROXY 1
#ifdef DEBUG_PROXY
#define DEBUG_PROXY_PRINT printf
#else
#define DEBUG_PROXY_PRINT(...)
#endif

#define OP_INDEX(op) ((op) ? (op)-state->pools->elems : -1)
#define OP_SEEN 0x100000

ncclResult_t getOpIndex(struct ncclProxyArgs* op, struct ncclProxyProgressState* state, int* poolIndex, int* opIndex) {
  struct ncclProxyPool* pool = state->pools;
  int p = 0;
  while (pool) {
    uint64_t o = op-pool->elems;
    if (o < PROXYARGS_ALLOCATE_SIZE) {
      *opIndex = o;
      *poolIndex = p;
      return ncclSuccess;
    }
    pool = pool->next;
    p++;
  }
  WARN("Could not find pool of op %p", op);
  return ncclInternalError;
}

ncclResult_t printProxyOp(struct ncclProxyArgs* op, int poolIndex, int opIndex) {
  printf("[%d-%d|%ld| %s", poolIndex, opIndex, op->opCount, op->pattern == ncclPatternSend ? "Send" : op->pattern == ncclPatternRecv ? "Recv" : "Coll");
  for (int s=0; s<op->nsubs; s++) {
    struct ncclProxySubArgs* sub = op->subs+s;
    if (op->state == ncclProxyOpProgress) {
      char status = ' ';
      if (op->pattern == ncclPatternRecv) {
        if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) status = 'I'; // Init
        else if (sub->received < sub->posted) status = 'R'; // Receiving
        else if (sub->received < sub->transmitted) status = 'R'; // Receiving
        else if (sub->transmitted < sub->received) status = 'F'; // Flushing
        else if (sub->done < sub->transmitted) status = 'G'; // Waiting on GPU
        else status = 'D'; // Done
      } else if (op->pattern == ncclPatternSend) {
        if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) status = 'I'; // Init
        else if (sub->transmitted < sub->posted) status = 'G'; // Waiting on GPU
        else if (sub->done < sub->transmitted) status = 'S'; // Sending
        else status = 'D'; // Done
      }
      printf(" %d%c/%d", sub->peer, status, sub->channelId);
    } else {
      printf(" %d/%d", sub->peer, sub->channelId);
    }
  }
  printf("]");
  return ncclSuccess;
}
ncclResult_t dumpProxyState(struct ncclProxyProgressState* state) {
  struct ncclProxyArgs* op = state->active;
  int poolIndex, opIndex;
  printf("ACTIVE OPS\n");
  while (op) {
    NCCLCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    NCCLCHECK(printProxyOp(op, poolIndex, opIndex));
    op->state |= OP_SEEN;
    printf("\n");
    struct ncclProxyArgs* nextOp = op->nextPeer;
    while (nextOp) {
      NCCLCHECK(getOpIndex(nextOp, state, &poolIndex, &opIndex));
      if (nextOp->state & OP_SEEN) {
        WARN("List loop at element %d-%d", poolIndex, opIndex);
      }
      printf("| `-> ");
      NCCLCHECK(printProxyOp(nextOp, poolIndex, opIndex));
      nextOp->state |= OP_SEEN;
      printf("\n");
      if (nextOp->next) {
        WARN("Inactive op has next set!");
      }
      nextOp = nextOp->nextPeer;
    }
    if (op->nextPeer == NULL) printf("|\n");
    op = op->next;
    printf("v\n");
  }
  printf("[X]\n");

# if 0
  printf("FREE OPS\n");
  op = state->pool;
  while (op) {
    NCCLCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    NCCLCHECK(printProxyOp(op, poolIndex, opIndex));
    op->state |= OP_SEEN;
    printf("->");
    op = op->next;
  }
  printf("[X]\n");
#else
  op = state->pool;
  while (op) {
    NCCLCHECK(getOpIndex(op, state, &poolIndex, &opIndex));
    if (op->state & OP_SEEN) {
      WARN("List loop at element %d-%d", poolIndex, opIndex);
    }
    op->state |= OP_SEEN;
    op = op->next;
  }
#endif

  struct ncclProxyPool* pool = state->pools;
  poolIndex = 0;
  while (pool) {
    struct ncclProxyArgs* elem = pool->elems;
    for (int e=0; e<PROXYARGS_ALLOCATE_SIZE; e++, elem++) {
      if ((elem->state & OP_SEEN) == 0) {
        printf("Elem %d-%d is not in any list:\n", poolIndex, e);
        NCCLCHECK(printProxyOp(elem, poolIndex, e));
        printf("\n");
      } else {
        elem->state -= OP_SEEN;
      }
    }
    pool = pool->next;
    poolIndex++;
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyOpToArgs(struct ncclProxyOp* op, struct ncclProxyArgs* args, int subIndex) {
  struct ncclProxySubArgs* sub = args->subs+subIndex;
  if (subIndex >= NCCL_PROXY_MAX_SUBS) {
    WARN("Proxy append out of bounds");
    return ncclInternalError;
  }
  //memset(sub, 0, sizeof(struct ncclProxySubArgs));
  sub->connection = op->connection;
  sub->channelId = op->channelId;
  sub->nsteps = op->nsteps;
  sub->nbytes = op->nbytes;
  sub->chunkSize = op->chunkSize;
  sub->offset = 0;
  sub->loopSize = op->loopSize;
  sub->loopOffset = op->loopOffset;
  sub->isOneRPN = op->isOneRPN;
  sub->peer = op->peer;
  sub->reg = op->reg;
  sub->sendMhandle = op->sendMhandle;
  sub->recvMhandle = op->recvMhandle;
  sub->sendbuff = op->sendbuff;
  sub->recvbuff = op->recvbuff;
  sub->eActivationMask = op->eActivationMask;
  sub->taskEventHandle = op->taskEventHandle;
  sub->rank = op->rank;
  sub->pid = op->pid;
  sub->profilerContext = op->profilerContext;
  sub->ringAlgo = op->ringAlgo;
  sub->workCounter = op->workCounter;
  args->nsubs = subIndex+1;
  if (subIndex) {
    args->nChannels = std::min(args->nChannels, op->nChannels);
    args->nPeers = std::min(args->nPeers, op->nPeers);
    if ((args->sliceSteps != op->sliceSteps) ||
        (args->chunkSteps != op->chunkSteps) ||
        (args->protocol != op->protocol) ||
        (args->dtype != op->dtype) ||
        (args->redOp != op->redOp) ||
        (args->coll != op->coll)) {
      WARN("Proxy append mismatch");
      return ncclInternalError;
    }
    if (args->state != ncclProxyOpReady) {
      WARN("Proxy append on running operation");
      return ncclInternalError;
    }
    goto exit;
  }
  //memset(&args->progress, 0, sizeof(struct ncclProxyArgs)-offsetof(struct ncclProxyArgs, progress));
  args->done = 0;
  args->opCount = op->opCount;
  args->sliceSteps = op->sliceSteps;
  args->chunkSteps = op->chunkSteps;
  args->chunkSize = op->chunkSize;
  args->dtype = op->dtype;
  args->redOp = op->redOp;
  args->pattern = op->pattern;
  args->protocol = op->protocol;
  args->coll = op->coll;
  args->collAPI = op->collAPI;
  args->algorithm = op->algorithm;
  args->nChannels = op->nChannels;
  args->nPeers = op->nPeers;
  args->specifics = op->specifics;
  args->state = ncclProxyOpReady;
  args->progress = op->connection->tcomm->proxyProgress;
  args->proxyAppendPtr = op->connection->proxyAppendPtr;
exit:
  if (args->pattern != ncclPatternProfiler) ncclProfilerStartProxyOpEvent(subIndex, args);
  return ncclSuccess;
}

static ncclResult_t ProxyAppend(struct ncclProxyProgressState* state, struct ncclProxyOp* op) {
  struct ncclProxyConnection* connection = op->connection;
  int shared = connection->shared;
  struct ncclProxyArgs* args = *connection->proxyAppendPtr;

  if (args) {
    if (shared && args->opCount == op->opCount) {
      NCCLCHECK(ncclProxyOpToArgs(op, args, args->nsubs));
      DEBUG_PROXY_PRINT("Insert (%d/%5ld/%5ld) as group with %5ld\n", shared, args->opCount, op->opCount, OP_INDEX(args));
    } else {
      struct ncclProxyArgs* prevArgs = args;
      NCCLCHECK(allocateArgs(state, &args));
      NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
      prevArgs->nextPeer = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld\n", OP_INDEX(args), shared, prevArgs->opCount, args->opCount, OP_INDEX(prevArgs));
      *(args->proxyAppendPtr) = args;
    }
  } else {
    // Nothing running for that peer. Add to the list
    NCCLCHECK(allocateArgs(state, &args));
    NCCLCHECK(ncclProxyOpToArgs(op, args, 0));
    if (state->active == NULL) {
      // Create the list
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element\n", OP_INDEX(args), shared, args->opCount);
      state->active = args;
    } else {
      // Append element at the end of the list
      struct ncclProxyArgs* last = state->active;
      while (last->next) last = last->next;
      last->next = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element\n", OP_INDEX(args), shared, args->opCount);
    }
    *(args->proxyAppendPtr) = args;
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyPost(struct ncclProxyOpsPool* pool, int nextOps, int nextOpsEnd) {
  std::lock_guard<std::mutex> lock(pool->mutex);
  if (pool->nextOps == -1) {
    pool->nextOps = nextOps;
    pool->cond.notify_one();
  } else {
    pool->ops[pool->nextOpsEnd].next = nextOps;
  }
  pool->nextOpsEnd = nextOpsEnd;
  return ncclSuccess;
}

static ncclResult_t ncclLocalOpAppend(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, struct ncclProxyOp* proxyOp) {
  int tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
  if (proxyOps == NULL) return ncclInternalError;
  proxyOps += proxyConn->tpLocalRank;
  struct ncclProxyOpsPool* pool = proxyOps->pool;

  TIME_START(0);
  int opIndex = proxyOps->freeOp;
  struct ncclProxyOp* op;
  if (opIndex != -1) {
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  } else {
    // Read the freeOps value and wait for a value different than -1. Once not -1, read the value with acquire and reset -1
    int freeOp = -1;
    while (freeOp == -1) {
      freeOp = COMPILER_ATOMIC_EXCHANGE(&pool->freeOps[tpLocalRank], -1, std::memory_order_acquire);
      if (freeOp == -1) sched_yield();
    }
    opIndex = freeOp;
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  }
  if (op->next != -1) COMPILER_PREFETCH(pool->ops+op->next); // Prefetch next free op
  memcpy(op, proxyOp, sizeof(struct ncclProxyOp));
  if (proxyOp->ringAlgo) proxyOp->ringAlgo->incRefCount();
  op->next = -1;
  op->connection = proxyConn->connection;
  if (proxyOps->nextOps == -1) {
    proxyOps->nextOps = proxyOps->nextOpsEnd = opIndex;
  } else {
    pool->ops[proxyOps->nextOpsEnd].next = opIndex;
    proxyOps->nextOpsEnd = opIndex;
  }
  if (++proxyOps->count == MAX_OPS_PER_PEER) {
    // Post what we have so far to free some ops in the pool
    // Do not post last operations as we could have more coming with the same opCount, and posting
    // them in different batches would break proxyArgs aggregation with subs.
    uint64_t lastOpCount = pool->ops[proxyOps->nextOpsEnd].opCount;
    int lastOp = -1;
    int toSend = 0;
    int ops = 0;
    for (int op= proxyOps->nextOps; op != proxyOps->nextOpsEnd; op=pool->ops[op].next) {
      ops++;
      if (pool->ops[op].opCount != lastOpCount) {
        lastOp = op;
        toSend = ops;
      }
    }
    if (lastOp == -1) {
      WARN("Unable to post incomplete proxy op chain %d..%d (opCount %ld)", proxyOps->nextOps, proxyOps->nextOpsEnd, lastOpCount);
      return ncclInternalError;
    }
    // Cut chain at lastOp
    int nextOps = proxyOps->nextOps;
    proxyOps->nextOps = pool->ops[lastOp].next;
    pool->ops[lastOp].next = -1;
    NCCLCHECK(ncclProxyPost(proxyOps->pool, nextOps, lastOp));
    proxyOps->count -= toSend;
  }
  TIME_STOP(0);
  return ncclSuccess;
}

static void incWorkCounter(struct ncclComm* comm, struct ncclProxyOp* op) {
  op->workCounter = (op->incWorkCounter) ? ++comm->profiler.workCounter[op->channelId] : comm->profiler.workCounter[op->channelId];
}

static ncclResult_t SaveProxyProfiler(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  struct ncclProxyConnector* proxyConn = (op->coll == ncclFuncRecv) ? &comm->profiler.recvProxyConn[op->channelId] : &comm->profiler.sendProxyConn[op->channelId];
  if (justInquire) {
    *justInquire = true;
    if (!comm->planner.persistent) incWorkCounter(comm, op);
  } else {
    op->sendbuff = (uint8_t *)comm->profiler.workStarted;
    op->recvbuff = (uint8_t *)comm->profiler.workCompleted;
    // Ensure that in graph capturing the proxy workCounter is incremented to keep up with kernel workCounter
    if (comm->planner.persistent) incWorkCounter(comm, op);
    NCCLCHECK(ncclLocalOpAppend(comm, proxyConn, op));
  }
  return ncclSuccess;
}

static ncclResult_t SaveProxy(struct ncclComm* comm, struct ncclChannel* channel, int type, int peer, struct ncclProxyOp* op, int connIndex, bool* justInquire) {
  if (peer < 0) return ncclSuccess;

  struct ncclChannelPeer* peerComm = channel->peers[peer];
  struct ncclConnector* connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
  if (connector->transportComm == NULL) {
    WARN("Rank %d has no transport for %s peer %d on channel %d/%d", comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id, connIndex);
    return ncclInternalError;
  }
  if (connector->proxyConn.proxyProgress == NULL) return ncclSuccess;

  if (justInquire) *justInquire = true;
  else {
    op->peer = peer;
    NCCLCHECK(ncclLocalOpAppend(comm, &connector->proxyConn, op));
  }
  return ncclSuccess;
}

// justInquire != nullptr means don't actually do anything, just assertain need of
// ncclProxySaveOp for this op.
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  struct ncclChannel* channel = &comm->channels[op->channelId];
  bool needProxy = false;
  if (justInquire) *justInquire = false;
  switch (op->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo: {
      struct ncclRing* ring = &channel->ring;
      needProxy = NeedProxy(proxyRecv, op->pattern, op->root, ring, comm->nRanks);
      if (op->coll == ncclFuncAllGatherV && op->pattern == ncclPatternRing) {
        op->nsteps = op->specifics.bcast.recvSlices;
        if (op->nsteps == 0) needProxy = false;
      }
      if (needProxy) NCCLCHECK(SaveProxy(comm, channel, proxyRecv, ring->prev, op, 0, justInquire));

      needProxy = NeedProxy(proxySend, op->pattern, op->root, ring, comm->nRanks);
      if (op->coll == ncclFuncAllGatherV && op->pattern == ncclPatternRing) {
        op->nsteps = op->specifics.bcast.sendSlices;
        if (op->nsteps == 0) needProxy = false;
      }
      if (needProxy) NCCLCHECK(SaveProxy(comm, channel, proxySend, ring->next, op, 0, justInquire));
    } break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown: {
      if (op->pattern != ncclPatternTreeDown) { // Tree up
        struct ncclTree* tree = &channel->tree;
        for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
          NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->down[i], op, 0, justInquire));
        }
        NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->up, op, 0, justInquire));
      }
      if (op->pattern != ncclPatternTreeUp) { // Tree down
        struct ncclTree* tree = &channel->tree;
        for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) {
          NCCLCHECK(SaveProxy(comm, channel, proxySend, tree->down[i], op, 0, justInquire));
        }
        NCCLCHECK(SaveProxy(comm, channel, proxyRecv, tree->up, op, 0, justInquire));
      }
    } break;
  case ncclPatternCollnetChain: {
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->collnetChain.up, op, 1, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->collnetChain.up, op, 0, justInquire));
    } break;
  case ncclPatternCollnetDirect: {
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->collnetDirect.out, op, 1, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->collnetDirect.out, op, 0, justInquire));
    } break;
  case ncclPatternNvls: {
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.out, op, 1, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.out, op, 0, justInquire));
    } break;
  case ncclPatternNvlsTree: {
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.treeDown[1], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.treeDown[2], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.treeUp, op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.treeDown[1], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxySend, channel->nvls.treeDown[2], op, 0, justInquire));
      NCCLCHECK(SaveProxy(comm, channel, proxyRecv, channel->nvls.treeUp, op, 0, justInquire));
    } break;
  case ncclPatternPatUp: {
      // Run full algorithm to count the number of steps for each peer.
      ncclResult_t result = ncclSuccess;
      const ssize_t size = op->nbytes/comm->nRanks;
      const int rank = comm->rank, nranks = comm->nRanks;
      int *nstepsSend = NULL, *nstepsRecv = NULL;
      PatRSAlgorithm<char> algo(op->chunkSize, NCCL_STEPS, 16, 0, size, size, op->chunkSize, rank, nranks);
      struct ncclPatStep ps = {0};
      NCCLCHECKGOTO(ncclCalloc(&nstepsSend, log2Up(nranks)), result, exit_pat_up);
      NCCLCHECKGOTO(ncclCalloc(&nstepsRecv, log2Up(nranks)), result, exit_pat_up);

      do {
        algo.getNextOp(&ps);
        if (ps.flags & PatSkipped) continue;
        if (ps.recvDim != -1 && ps.postRecv) nstepsRecv[ps.recvDim]++;
        if (ps.sendDim != -1 && ps.postSend) nstepsSend[ps.sendDim]++;
      } while (ps.last != 2);
      for (int i=0; i<log2Up(nranks); i++) {
        if (nstepsSend[i]) {
          int sendPeer = (rank + (1<<i)) % nranks;
          op->nsteps = nstepsSend[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxySend, sendPeer, op, 0, justInquire), result, exit_pat_up);
        }
        if (nstepsRecv[i]) {
          int recvPeer = (rank - (1<<i) + nranks) % nranks;
          op->nsteps = nstepsRecv[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxyRecv, recvPeer, op, 0, justInquire), result, exit_pat_up);
        }
      }
    exit_pat_up:
      free(nstepsSend);
      free(nstepsRecv);
      NCCLCHECK(result);
    } break;
  case ncclPatternPatDown: {
      // Run full algorithm to count the number of steps for each peer.
      ncclResult_t result = ncclSuccess;
      const ssize_t size = op->nbytes/comm->nRanks;
      const int rank = comm->rank, nranks = comm->nRanks;
      int *nstepsSend = NULL, *nstepsRecv = NULL;
      PatAGAlgorithm<char> algo(op->chunkSize, NCCL_STEPS, 16, 0, size, size, op->chunkSize, rank, nranks);
      struct ncclPatStep ps = {0};
      NCCLCHECKGOTO(ncclCalloc(&nstepsSend, log2Up(nranks)), result, exit_pat_down);
      NCCLCHECKGOTO(ncclCalloc(&nstepsRecv, log2Up(nranks)), result, exit_pat_down);

      do {
        algo.getNextOp(&ps);
        if (ps.flags & PatSkipped) continue;
        if (ps.recvDim != -1 && ps.postRecv) nstepsRecv[ps.recvDim]++;
        if (ps.sendDim != -1 && ps.postSend) nstepsSend[ps.sendDim]++;
      } while (ps.last != 2);
      for (int i=0; i<log2Up(nranks); i++) {
        if (nstepsSend[i]) {
          int sendPeer = (rank - (1<<i) + nranks) % nranks;
          op->nsteps = nstepsSend[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxySend, sendPeer, op, 0, justInquire), result, exit_pat_down);
        }
        if (nstepsRecv[i]) {
          int recvPeer = (rank + (1<<i)) % nranks;
          op->nsteps = nstepsRecv[i];
          NCCLCHECKGOTO(SaveProxy(comm, channel, proxyRecv, recvPeer, op, 0, justInquire), result, exit_pat_down);
        }
      }
    exit_pat_down:
      free(nstepsSend);
      free(nstepsRecv);
      NCCLCHECK(result);
    } break;
  case ncclPatternSend:
  case ncclPatternRecv: {
      if (op->root == comm->rank) return ncclSuccess;
      NCCLCHECK(SaveProxy(comm, channel, op->pattern == ncclPatternSend ? proxySend : proxyRecv, op->root, op, 1, justInquire));
    } break;
  case ncclPatternProfiler: {
      if (ncclProfilerNeedsProxy(comm, op)) NCCLCHECK(SaveProxyProfiler(comm, op, justInquire));
      else incWorkCounter(comm, op);
    } break;
  }
  return ncclSuccess;
}

static ncclResult_t removeOp(struct ncclProxyProgressState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr) {
  struct ncclProxyArgs* freeOp = *opPtr;
  struct ncclProxyArgs* next = freeOp->next;
  DEBUG_PROXY_PRINT("Remove %ld -> %ld -> %ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(freeOp), OP_INDEX(next));
  *opPtr = next;
  if (freeOp->nextPeer) {
    // replace op by nextPeer
    struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = nextPeer;
    } else {
      state->active = nextPeer;
    }
    nextPeer->next = next;
    *(prevOpPtr) = nextPeer;
  } else {
    *(freeOp->proxyAppendPtr) = NULL;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = next;
    } else {
      state->active = next;
    }
  }
  freeOp->next = state->pool;
  state->pool = freeOp;
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld) : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
#ifdef DEBUG_PROXY
  NCCLCHECK(dumpProxyState(state));
#endif
  return ncclSuccess;
}

static ncclResult_t progressOps(struct ncclProxyState* proxyState, struct ncclProxyProgressState* state, struct ncclProxyArgs* opStart, int* idle) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = opStart;
  ncclResult_t status = ncclSuccess;
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    TIME_START(0); TIME_START(1);
    ncclResult_t ret = op->progress(proxyState, op);
    if (op->idle) { TIME_STOP(1); TIME_CANCEL(0); } else { TIME_CANCEL(1); TIME_STOP(0); }
    *idle &= op->idle;
    if (op->state == ncclProxyOpNone || ret != ncclSuccess) {
      //track first error that occured
      if (ret != ncclSuccess && status == ncclSuccess) status = ret;
      TIME_START(2);
      NCCLCHECK(removeOp(state, &op, &prevOp));
      TIME_STOP(2);
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return status;
}

NCCL_PARAM(ProxyAppendBatchSize, "PROXY_APPEND_BATCH_SIZE", 16);

static ncclResult_t ncclProxyGetPostedOps(struct ncclProxyState* proxyState, int* added) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (state->opsPool == NULL) return ncclInternalError;
  struct ncclProxyOpsPool* pool = state->opsPool;

  if (state->nextOps != -1) goto process_nextops;

  void* eHandle;
  // If we have ops to progress, no need to block waiting for something to arrive or even wait for the lock
  // to be available. Exit, continue progress, and come back later.
  {
    std::unique_lock<std::mutex> lock(pool->mutex, std::defer_lock);
    if (state->active != NULL && (pool->nextOps == -1 || !lock.try_lock())) return ncclSuccess;

    if (state->active == NULL) {
      lock.lock();
      if (pool->nextOps == -1 && !state->stop) {
        ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
        ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlSleep);
        pool->cond.wait(lock);
        ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlWakeup);
        ncclProfilerStopProxyCtrlEvent(eHandle);
      }
    }
    state->nextOps = pool->nextOps;
    pool->nextOps = pool->nextOpsEnd = -1;
  }

process_nextops:
  ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
  ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlAppend);
  TIME_START(2);
  int freeOp[NCCL_MAX_PROXY_CONNECTIONS];
  int freeOpEnd[NCCL_MAX_PROXY_CONNECTIONS];
  for (int i = 0; i < proxyState->tpLocalnRanks; i++) freeOp[i] = -1;

  uint64_t lastOpCount = 0;
  int lastPeer = -1;
  int count = 0;
  for (int opIndex = state->nextOps; opIndex != -1;) {
    struct ncclProxyOp* peerOp = pool->ops+opIndex;
    int peer = opIndex / MAX_OPS_PER_PEER;
    if ((lastOpCount && peerOp->opCount != lastOpCount) || ((lastPeer != -1) && peer != lastPeer)) count++;
    if (count == ncclParamProxyAppendBatchSize()+1) break;
    lastOpCount = peerOp->opCount;
    lastPeer = peer;
    if (peerOp->connection == NULL) return ncclInternalError;
    if (peerOp->next != -1) COMPILER_PREFETCH(pool->ops+peerOp->next);
    NCCLCHECK(ProxyAppend(state, peerOp));
    (*added)++;
    int lastOpIndex = opIndex;
    opIndex = peerOp->next;
    // Return op to peer pool
    if (freeOp[peer] == -1) {
      freeOpEnd[peer] = lastOpIndex;
    } else {
      peerOp->next = freeOp[peer];
    }
    freeOp[peer] = lastOpIndex;
    state->nextOps = opIndex;
  }

  for (int i = 0; i < proxyState->tpLocalnRanks; i++) {
    if (freeOp[i] == -1) continue;
    int oldFree = -1, newFree = freeOp[i];
    // prepend the ops freeOp[i]-freeOpEnd[i] in front of the pool->freeOps[i] op
    oldFree = COMPILER_ATOMIC_LOAD(&pool->freeOps[i], std::memory_order_acquire);
    do {
      // Coverity gets confused by the complex code structure here.  The previous "for" loop ensures that freeOpEnd[i]
      // is initialized so long as freeOp[i] is initialized (is not -1).  In the current loop we filter out uninitialized
      // freeOp[i], hence ensuring that freeOpEnd[i] is also initialized.
      // coverity[uninit_use:FALSE]
      pool->ops[freeOpEnd[i]].next = oldFree;
    } while (!COMPILER_ATOMIC_COMPARE_EXCHANGE(&pool->freeOps[i], &oldFree, newFree, /*success=*/std::memory_order_release, /*failure=*/std::memory_order_acquire));
  }
  ncclProfilerRecordProxyCtrlEventState(eHandle, *added, ncclProfilerProxyCtrlAppendEnd);
  ncclProfilerStopProxyCtrlEvent(eHandle);
  TIME_STOP(2);
  return ncclSuccess;
}

#include <signal.h>
static ncclProxyProgressState* ncclLastProxyState;
void ncclDumpProxyState(int signal) {
  dumpProxyState(ncclLastProxyState);
}

NCCL_PARAM(CreateThreadContext, "CREATE_THREAD_CONTEXT", 0);
static int setProxyThreadContext(struct ncclProxyState* proxyState) {
#if CUDART_VERSION >= 11030
  static int createThreadContext = -1;

  if (createThreadContext == -1) {
    createThreadContext = ncclParamCreateThreadContext();
    if (createThreadContext) {
      if (CUPFN(cuCtxCreate) == nullptr || CUPFN(cuCtxDestroy) == nullptr || CUPFN(cuCtxSetCurrent) == nullptr) {
        WARN("Unable to create thread context due to old driver, disabling.");
        createThreadContext = 0;
        goto exit;
      }
    }
  }
  if (createThreadContext) {
    if (proxyState->cudaCtx == NULL) {
      if (CUPFN(cuCtxCreate(&proxyState->cudaCtx,
                            NULL, 0, CU_CTX_SCHED_SPIN|CU_CTX_MAP_HOST, proxyState->cudaDev)) != CUDA_SUCCESS) {
        WARN("Failed to create CUDA context on device %d", proxyState->cudaDev);
        createThreadContext = 0;
        goto exit;
      }
    } else {
      if (CUPFN(cuCtxSetCurrent(proxyState->cudaCtx)) != CUDA_SUCCESS) {
        WARN("Failed to set CUDA context on device %d", proxyState->cudaDev);
        goto exit;
      }
    }
    return 1;
  }
exit:
#endif
  return 0;
}

// Set to SIGUSR1 or SIGUSR2 to help debug proxy state during hangs
NCCL_PARAM(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", -1);
NCCL_PARAM(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);

static ncclAffinity proxyCpuset;
static std::once_flag proxyCpusetOnceFlag;
void proxyCpusetOnceFunc() {
  const char* setEnv = ncclGetEnv("NCCL_PROXY_CPUSET");
  if (setEnv) {
    ncclResult_t res = ncclStrListToCpuset(setEnv, &proxyCpuset);
    if (res != ncclSuccess) {
      INFO(NCCL_ENV, "failed to decode NCCL_PROXY_CPUSET=%s. Ignoring", setEnv);
      goto fail;
    }
    // debug info
    char msg[1024] = {0};
    ncclAffinity currSet;
    ncclOsGetAffinity(&currSet);
    (void)ncclCpusetToStrList(&currSet, msg, sizeof(msg));
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " changed to ");
    (void)ncclCpusetToStrList(&proxyCpuset, msg + strlen(msg), sizeof(msg) - strlen(msg));
    INFO(NCCL_ENV, "NCCL_PROXY_CPUSET = %s: %s", setEnv, msg);
    return;
  }
  // if we arrive here we have either no env or we have failed to decode it
fail:
  ncclOsCpuZero(proxyCpuset);
  return;
}

void* ncclProxyProgress(void *proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;

  // This thread is created by proxyService, therefore setting the affinity is not needed.
  INFO(NCCL_INIT, "[Proxy Progress] Device %d CPU core %d", proxyState->cudaDev, ncclOsGetCpu());

  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Progress] Set CUDA context on device %d", proxyState->cudaDev);
  } else if (!CUDASUCCESS(cudaSetDevice(proxyState->cudaDev))) {
    WARN("[Proxy Progress] Failed to set CUDA device %d", proxyState->cudaDev);
  }

  struct ncclProxyProgressState* state = &proxyState->progressState;
  state->nextOps = -1;
  const int sig = ncclParamProxyDumpSignal();
  if (sig != -1) signal(sig, ncclDumpProxyState);
  ncclLastProxyState = state;
  char threadName[NCCL_THREAD_NAMELEN];
  snprintf(threadName, NCCL_THREAD_NAMELEN, "NCCL Progress%2d", proxyState->cudaDev);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

  int lastIdle = 0;
  /* Too frequent call of ncclProxyGetPostedOps() will result in perf regression for small message
   * communication. proxyOpAppendCounter is a counter that helps us decide if we need to append proxy ops.
   * After each progress, proxyOpAppendCounter will increase by 1 and compare with environment variable
   * ncclParamProgressAppendOpFreq(). If they are equal, we will append proxy ops. This will decrease the
   * frequency of calling ncclProxyGetPostedOps() and reduce the perf impact. */
  int proxyOpAppendCounter = 0;
  do {
    int idle = 1;
    ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
    if (ret != ncclSuccess) {
      COMPILER_ATOMIC_STORE(&proxyState->asyncResult, ret, std::memory_order_release);
      INFO(NCCL_ALL,"%s:%d -> %d [Progress Thread]", __FILE__, __LINE__, ret);
      break;
    }
    if ((lastIdle == 0 && idle == 1) || (lastIdle == 1 && idle == 0)) {
      void* eHandle;
      ncclProfilerStartProxyCtrlEvent(proxyState->profilerContext, &eHandle);
      if (lastIdle == 0 && idle == 1) ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlIdle);
      if (lastIdle == 1 && idle == 0) ncclProfilerRecordProxyCtrlEventState(eHandle, 0, ncclProfilerProxyCtrlActive);
      ncclProfilerStopProxyCtrlEvent(eHandle);
    }
    if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
      int added = 0;
      proxyOpAppendCounter = 0;
      TIME_START(3);
      ret = ncclProxyGetPostedOps(proxyState, &added);
      if (added) { TIME_STOP(3); } else { TIME_CANCEL(3); }
      if (ret != ncclSuccess) {
        COMPILER_ATOMIC_STORE(&proxyState->asyncResult, ret, std::memory_order_release);
        INFO(NCCL_ALL,"%s:%d -> %d [Progress Thread]", __FILE__, __LINE__, ret);
      }
      if (added == 0) {
        std::this_thread::yield(); // No request progressed. Let others run.
      }
    }
    lastIdle = idle;
  } while ((state->stop == 0 || (state->stop == 1 && state->active)) && COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_acquire) == 0);
  return NULL;
}

ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  struct ncclProxyOps* proxyOps = comm->proxyState->proxyOps;
  if (proxyOps == NULL) return ncclSuccess;
  TIME_START(1);
  for (int r = 0; r < comm->sharedRes->tpNLocalRanks; r++) {
    struct ncclProxyOps* ops = proxyOps + r;
    if (ops->pool == NULL || ops->nextOps == -1) continue;
    NCCLCHECK(ncclProxyPost(ops->pool, ops->nextOps, ops->nextOpsEnd));
    ops->nextOps = ops->nextOpsEnd = -1;
    ops->count = 0;
  }
  comm->opCount++;
  TIME_STOP(1);
  return ncclSuccess;
}

static ncclResult_t ncclProxyProgressCreate(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (!state->thread.joinable()) {
    state->thread = std::thread(ncclProxyProgress, proxyState);
    ncclSetThreadName(state->thread, "NCCL Progress%2d", proxyState->tpLocalnRanks);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyProgressDestroy(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;

  // Request the proxy to stop and then wake it
  if (state->opsPool) {
    {
      std::lock_guard<std::mutex> lock(state->opsPool->mutex);
      state->stop = 1;
      state->opsPool->cond.notify_one();
    }
    state->thread.join();
  }

  // Free off any memory allocated for the proxy arg pools
  while (state->pools != NULL) {
    struct ncclProxyPool *next = state->pools->next;
    free(state->pools);
    state->pools = next;
  }

  TIME_PRINT("Proxy");
  return ncclSuccess;
}

#define NCCL_PROXY_CONN_POOL_SIZE_POW2 7
#define NCCL_PROXY_CONN_POOL_SIZE (1<<(NCCL_PROXY_CONN_POOL_SIZE_POW2))
#define NCCL_PROXY_CONN_POOL_MASK ((NCCL_PROXY_CONN_POOL_SIZE)-1)
struct ncclProxyConnectionPool {
  struct ncclProxyConnection** pools;
  int banks;
  int offset;
};

static ncclResult_t ncclProxyNewConnection(struct ncclProxyConnectionPool* pool, int* id) {
  if (pool->offset == NCCL_PROXY_CONN_POOL_SIZE) {
    NCCLCHECK(ncclRealloc(&pool->pools, pool->banks, pool->banks+1));
    NCCLCHECK(ncclCalloc(pool->pools+pool->banks, NCCL_PROXY_CONN_POOL_SIZE));
    pool->banks++;
    pool->offset = 0;
  }
  *id = ((pool->banks-1) << NCCL_PROXY_CONN_POOL_SIZE_POW2) + pool->offset;
  pool->offset++;
  return ncclSuccess;
}

static ncclResult_t ncclProxyGetConnection(struct ncclProxyConnectionPool* pool, int id, struct ncclProxyConnection** conn) {
  int bank = id>>NCCL_PROXY_CONN_POOL_SIZE_POW2;
  int offset = id&NCCL_PROXY_CONN_POOL_MASK;
  if ((pool->pools == NULL) || (bank > pool->banks) || (pool->pools[bank] == NULL)) return ncclInternalError;
  *conn = pool->pools[bank]+offset;
  return ncclSuccess;
}

static ncclResult_t proxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (connection->send) {
    if (ncclTransports[connection->transport]->send.proxyFree) {
      NCCLCHECK(ncclTransports[connection->transport]->send.proxyFree(connection, proxyState));
    }
  } else {
    if (ncclTransports[connection->transport]->recv.proxyFree) {
      NCCLCHECK(ncclTransports[connection->transport]->recv.proxyFree(connection, proxyState));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclProxyFreeConnections(struct ncclProxyConnectionPool* pool, struct ncclProxyState* proxyState) {
  for (int b=0; b<pool->banks; b++) {
    int max = b == pool->banks-1 ? pool->offset : NCCL_PROXY_CONN_POOL_SIZE;
    for (int i=0; i<max; i++) {
      ncclProxyConnection *connection = pool->pools[b]+i;
      if (connection->state != connUninitialized) {
        NCCLCHECK(proxyFree(connection, proxyState));
      }
    }
    free(pool->pools[b]);
  }
  free(pool->pools);
  return ncclSuccess;
}

#include "transport.h"

struct ncclProxyInitReq {
  int transport;
  int send;
  int tpLocalRank;
  int tpRank;
  int sameProcess;
};

struct ncclProxyInitResp {
  ncclProxyConnection* connection;
  char devShmPath[6]; // "XXXXXX" - May or may not be set
};

ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank, struct ncclProxyConnector* proxyConn) {
  struct ncclSocket* sock;
  int ready;
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  int tpProxyRank = comm->topParentRanks[proxyRank];

  proxyConn->sameProcess = ((comm->peerInfo[proxyRank].hostHash == comm->peerInfo[comm->rank].hostHash) &&
                            (comm->peerInfo[proxyRank].pidHash == comm->peerInfo[comm->rank].pidHash)) ? 1 : 0;
  // Keep one connection per local rank
  proxyConn->connection = NULL;
  proxyConn->tpRank = tpProxyRank;
  proxyConn->rank = proxyRank;
  if (sharedProxyState->peerSocks == NULL) {
    NCCLCHECK(ncclCalloc(&sharedProxyState->peerSocks, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->proxyOps, comm->sharedRes->tpNLocalRanks));
    NCCLCHECK(ncclCalloc(&sharedProxyState->sharedDevMems, comm->sharedRes->tpNLocalRanks));
    for (int i = 0; i < comm->sharedRes->tpNLocalRanks; ++i) {
      NCCLCHECK(ncclSocketSetFd(-1, &sharedProxyState->peerSocks[i]));
    }
  }

  proxyConn->tpLocalRank = comm->sharedRes->tpRankToLocalRank[proxyConn->tpRank];
  sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;
  NCCLCHECK(ncclSocketReady(sock, &ready));
  if (!ready) {
    NCCLCHECK(ncclSocketInit(sock, sharedProxyState->peerAddresses+proxyConn->tpRank, comm->sharedRes->magic, ncclSocketTypeProxy, comm->abortFlag));
    NCCLCHECK(ncclSocketConnect(sock));
  }

  struct ncclProxyInitReq req = {0};
  req.transport = transport;
  req.send = send;
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[comm->rank];
  req.sameProcess = proxyConn->sameProcess;

  struct ncclProxyInitResp resp = {0};
  // This usually sends proxyConn->connection to identify which connection this is.
  // However, this is part of the response and therefore is ignored
  NCCLCHECK(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgInit, &req, sizeof(req), &resp, sizeof(resp)));
  proxyConn->connection = resp.connection;

  // If we need proxy progress, map progress ops
  struct ncclTransportComm* tcomm = send ? &ncclTransports[transport]->send : &ncclTransports[transport]->recv;
  if (tcomm->proxyProgress) {
    char poolPath[] = "/dev/shm/nccl-XXXXXX";
    strncpy(poolPath+sizeof("/dev/shm/nccl-")-1, resp.devShmPath, sizeof("XXXXXX")-1);
    struct ncclProxyOps* proxyOps = sharedProxyState->proxyOps + proxyConn->tpLocalRank;
    if (proxyOps->pool == NULL) {
      NCCLCHECK(ncclShmOpen(poolPath, sizeof(poolPath), sizeof(struct ncclProxyOpsPool), (void**)(&proxyOps->pool), NULL, -1, &proxyOps->handle));
      proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
    }
  }
  proxyConn->initialized = true;
  INFO(NCCL_NET|NCCL_PROXY, "Connected to proxy localRank %d -> connection %p", proxyConn->tpLocalRank, proxyConn->connection);
  return ncclSuccess;
}

// UDS support
ncclResult_t ncclProxyCallBlockingUDS(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize, int* reqFd, int *respFd) {
  ncclResult_t res = ncclSuccess;
  struct ncclIpcSocket ipcSock = { 0 };
  void *opId;
  NCCLCHECK(getRandomData(&opId, sizeof(opId)));
  int reqFdtmp = -1;

  int rank = comm->topParentLocalRanks[comm->localRank];
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[proxyConn->tpRank];

  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %p opId %p",
       comm, rank, proxyConn->tpRank, pidHash, reqSize, respSize, respFd, opId);

  // cuMem: Create a UDS socket to receive the response
  NCCLCHECK(ncclIpcSocketInit(&ipcSock, rank, (uint64_t)opId, comm->abortFlag));

  if (reqFd) {
    reqFdtmp = *reqFd;
  } else {
    // give a dummy fd for the other side of UDS socket
    NCCLCHECK(ncclIpcSocketGetFd(&ipcSock, &reqFdtmp));
  }

  ncclIpcHdr hdr;
  memset(&hdr, '\0', sizeof(hdr));
  hdr.type = type;
  hdr.rank = rank;
  hdr.reqSize = reqSize;
  hdr.respSize = respSize;
  hdr.opId = opId;

  assert(reqSize <= sizeof(hdr.data));
  memcpy(&hdr.data, reqBuff, reqSize);
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &hdr, sizeof(hdr), reqFdtmp, proxyConn->tpRank, pidHash), res, error);
  NCCLCHECKGOTO(ncclIpcSocketRecvMsg(&ipcSock, respBuff, respSize, respFd), res, error);
  NCCLCHECKGOTO(ncclIpcSocketClose(&ipcSock), res, error);

  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %d opId %p - DONE",
       comm, rank, proxyConn->tpRank, pidHash, reqSize, respSize, (respFd ? *respFd : -1), opId);

  return res;

error:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  WARN("ncclProxyCallBlockingUDS call to tpRank %d(%lx) failed : %d", proxyConn->tpRank, pidHash, res);
  return res;
}

// cuMem API support
// The request/response is sent out-of-band using ncclIpcSocket for this specific command
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int proxyRank, void *handle, int* convertedFd) {
  ncclResult_t ret = ncclSuccess;

  // Request the allocation of a UDS fd for the handle
  if (comm->gproxyConn[proxyRank].initialized == false) {
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_P2P, 1, proxyRank, &comm->gproxyConn[proxyRank]), ret, error);
  }
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, &comm->gproxyConn[proxyRank], ncclProxyMsgGetFd, handle, sizeof(CUmemGenericAllocationHandle), NULL, 0, NULL, convertedFd), ret, error);

  // We have now received the converted fd over UDS
  INFO(NCCL_PROXY, "UDS: ClientGetFd handle 0x%lx tpRank %d returned fd %d sameProcess %d", *(uint64_t*)handle, comm->topParentRanks[proxyRank], *convertedFd, comm->gproxyConn[proxyRank].sameProcess);

  return ret;

error:
  WARN("ncclProxyClientGetFd call to tpRank %d handle 0x%lx failed : %d", comm->topParentRanks[proxyRank], *(uint64_t*)handle, ret);
  return ret;
}

ncclResult_t ncclProxyClientBatchQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int* localFds, int* rmtFds, int numSegments) {
  ncclResult_t ret = ncclSuccess;
  for (int segment = 0; segment < numSegments; segment++) {
    NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, proxyConn, ncclProxyMsgQueryFd, NULL, 0, (void*) &rmtFds[segment], sizeof(int), &localFds[segment], NULL), ret, fail);
    // We have now received the converted fd for a segment over UDS
    INFO(NCCL_PROXY, "UDS: ClientQueryFdBatch localFd %d tpRank %d remote fd %d sameProcess %d segment %d", localFds[segment], proxyConn->tpRank, rmtFds[segment], proxyConn->sameProcess, segment);
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd, int* rmtFd) {
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, proxyConn, ncclProxyMsgQueryFd, NULL, 0, (void*)rmtFd, sizeof(int), &localFd, NULL), ret, fail);
exit:
  // We have now received the converted fd over UDS
  INFO(NCCL_PROXY, "UDS: ClientQueryFd localFd %d tpRank %d remote fd %d sameProcess %d", localFd, proxyConn->tpRank, *rmtFd, proxyConn->sameProcess);
  return ret;
fail:
  WARN("ncclProxyClientQueryFdBlocking call to tpRank %d localFd %d failed : %d", proxyConn->tpRank, localFd, ret);
  goto exit;
}

const char* ncclProxyMsgTypeStr[] = { "Unknown", "Init", "SharedInit", "Setup", "Connect", "Start", "Close", "Abort", "Stop", "GetFd", "QueryFd", "Register", "Deregister" };
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId) {
  struct ncclSocket* sock;
  ncclResult_t ret = ncclSuccess;
  struct ncclProxyState* sharedProxyState = comm->proxyState;

  if (sharedProxyState->peerSocks == NULL) return ncclInternalError;

  sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;

  NCCLCHECKGOTO(ncclSocketSend(sock, &type, sizeof(int)), ret, error);
  NCCLCHECKGOTO(ncclSocketSend(sock, &proxyConn->connection, sizeof(void*)), ret, error);
  NCCLCHECKGOTO(ncclSocketSend(sock, &reqSize, sizeof(int)), ret, error);
  NCCLCHECKGOTO(ncclSocketSend(sock, &respSize, sizeof(int)), ret, error);
  if (reqSize) NCCLCHECKGOTO(ncclSocketSend(sock, reqBuff, reqSize), ret, error);

  // Send opId to proxy
  NCCLCHECKGOTO(ncclSocketSend(sock, &opId, sizeof(opId)), ret, error);

  // Add proxyOp to expected response queue
  NCCLCHECK(expectedProxyResponseEnqueue(sharedProxyState, opId, respSize));

  return ncclSuccess;
error:
  return ret;
}

ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff, void* opId) {
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  // Receive the connection pointer from the Proxy
  if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
    WARN("Comm %p is in abort state", comm);
    return ncclInternalError;
  }
  if (sharedProxyState->peerSocks == NULL) return ncclInternalError;

  // Check response queue
  int found = 0;
  ncclResult_t res = expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);
  if (found == 0) {
    // Attempt to read in a new response header from the proxy thread
    struct ncclSocket* sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;
    ncclProxyRpcResponseHeader resp = {0};
    int offset = 0;
    if (ncclSuccess != ncclSocketProgress(NCCL_SOCKET_RECV, sock, &resp, sizeof(resp), &offset)) {
      WARN("Socket recv failed while polling for opId=%p", opId);
      return ncclInternalError;
    }

    if (offset == 0) {
      return ncclInProgress;
    // If we've returned a partial response, block to receive the rest of it
    } else if (offset < sizeof(resp)) {
      while (offset < sizeof(resp))
        NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &resp, sizeof(resp), &offset));
    }

    INFO(NCCL_PROXY, "ncclPollProxyResponse Received new opId=%p", resp.opId);

    // If there's a respSize to recv
    if (resp.respSize > 0) {
      if (resp.opId != opId) {
        // Unexpected response, need to buffer the socket data
        respBuff = malloc(resp.respSize);
      }
      assert(respBuff != NULL);
      NCCLCHECK(ncclSocketRecv(sock, respBuff, resp.respSize));
    }

    if (resp.opId == opId) {
      INFO(NCCL_PROXY, "resp.opId=%p matches expected opId=%p", resp.opId, opId);
      NCCLCHECK(expectedProxyResponseRemove(sharedProxyState, resp.opId));
      return resp.res;
    } else {
      INFO(NCCL_PROXY, "Queuing opId=%p respBuff=%p respSize=%d", resp.opId, respBuff, resp.respSize);
      // Store the result and mark response as completed
      NCCLCHECK(expectedProxyResponseStore(sharedProxyState, resp.opId, respBuff, resp.respSize, resp.res));
      return ncclInProgress;
    }
  } else {
    INFO(NCCL_PROXY, "ncclPollProxyResponse Dequeued cached opId=%p", opId);
  }

  return res;
}

ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, void* respBuff, int respSize) {
  // Alloc some memory to act as a handle
  ncclResult_t res = ncclSuccess;
  void* opId = malloc(1);

  NCCLCHECKGOTO(ncclProxyCallAsync(comm, proxyConn, type, reqBuff, reqSize, respSize, opId), res, fail);

  do {
    res = ncclPollProxyResponse(comm, proxyConn, respBuff, opId);
  } while (res == ncclInProgress);

exit:
  free(opId);
  return res;
fail:
  goto exit;
}

static ncclResult_t proxyProgressInit(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (state->opsPool == NULL) {
    int size = sizeof(struct ncclProxyOpsPool);
    struct ncclProxyOpsPool* pool = NULL;

    char shmPath[sizeof("/dev/shm/nccl-XXXXXX")];
    shmPath[0] = '\0';
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, (void**)&pool, NULL, proxyState->tpLocalnRanks, &state->handle));
    // Init pool
    pool->nextOps = -1;

    for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
      pool->freeOps[r] = r*MAX_OPS_PER_PEER;
      for (int i=0; i<MAX_OPS_PER_PEER-1; i++) pool->ops[r*MAX_OPS_PER_PEER+i].next = r*MAX_OPS_PER_PEER+i+1;
      pool->ops[(r+1)*MAX_OPS_PER_PEER-1].next = -1;
    }

    ncclOsSetMutexCondShared(pool->mutex, pool->cond);

    state->opsPool = pool;

    memcpy(state->opsPoolShmSuffix, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof("XXXXXX")-1);

    // All ops structures are created, we can start the progress thread
    NCCLCHECK(ncclProxyProgressCreate(proxyState));
  }
  return ncclSuccess;
}

static void proxyOpsFree(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (ncclShmClose(state->handle) != ncclSuccess) {
    WARN("[Service thread] shm close failed");
  }
}

ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm) {
  struct ncclProxyProgressState* state = &comm->proxyState->progressState;
  if (state->opsPool == NULL) return ncclSuccess;

  if (ncclShmUnlink(state->handle) != ncclSuccess) {
    WARN("[Service thread] proxy ops shm unlink failed");
  }
  return ncclSuccess;
}

static ncclResult_t proxyConnInit(struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, ncclProxyInitReq* req, ncclProxyInitResp* resp, struct ncclProxyConnection** connection) {
  int id;
  NCCLCHECK(ncclProxyNewConnection(connectionPool, &id));
  NCCLCHECK(ncclProxyGetConnection(connectionPool, id, connection));

  (*connection)->sock = &peer->sock;
  (*connection)->transport = req->transport;
  (*connection)->send = req->send;
  (*connection)->tpLocalRank = req->tpLocalRank;
  (*connection)->sameProcess = req->sameProcess;
  peer->tpLocalRank = req->tpLocalRank;
  peer->tpRank = req->tpRank;

  resp->connection = *connection;

  (*connection)->tcomm = (*connection)->send ? &ncclTransports[(*connection)->transport]->send : &ncclTransports[(*connection)->transport]->recv;
  // If we need proxy progress, let's allocate ops and start the thread
  if ((*connection)->tcomm->proxyProgress) {
    NCCLCHECK(proxyProgressInit(proxyState));
    struct ncclProxyProgressState* state = &proxyState->progressState;
    strncpy(resp->devShmPath, state->opsPoolShmSuffix, sizeof(resp->devShmPath));
  }
  INFO(NCCL_NET|NCCL_PROXY, "New proxy %s connection %d from local rank %d, transport %d", (*connection)->send ? "send":"recv", id, (*connection)->tpLocalRank, (*connection)->transport);
  COMPILER_ATOMIC_STORE(&(*connection)->state, connInitialized, std::memory_order_release);
  return ncclSuccess;
}

static ncclResult_t proxyQueryFd(struct ncclProxyState* proxyState, int rank, void *opId, int rmtFd) {
#if CUDART_VERSION >= 11030
  struct ncclIpcSocket ipcSock = { 0 };
  uint64_t hash = (uint64_t) opId;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSock, proxyState->tpRank, hash^1, proxyState->abortFlag), ret, exit);
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &rmtFd, sizeof(int), -1, rank, hash), ret, exit);
exit:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  return ncclSuccess;
#else
  return ncclInternalError;
#endif
}

// cuMem API support
static ncclResult_t proxyGetFd(struct ncclProxyState* proxyState, int rank, void *opId, uint64_t handle) {
#if CUDART_VERSION >= 11030
  // cuMem API support
  ncclResult_t ret = ncclSuccess;
  struct ncclIpcSocket ipcSock = { 0 };
  uint64_t hash = (uint64_t) opId;
  INFO(NCCL_PROXY, "UDS proxyGetFd received handle 0x%lx peer %d opId %lx", handle, rank, hash);

  CUmemAllocationHandleType type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int fd = -1;

  CUCHECK(cuMemExportToShareableHandle(&fd, handle, type, 0));
  // Send back the converted fd using UDS
  NCCLCHECKGOTO(ncclIpcSocketInit(&ipcSock, proxyState->tpRank, hash^1, proxyState->abortFlag), ret, error);
  NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSock, fd, rank, hash), ret, error);
error:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  // We can now safely close the exported fd
  SYSCHECK(close(fd), "close");
  return ret;
#else
  return ncclInternalError;
#endif
}

static ncclResult_t proxyProgressAsync(struct ncclProxyAsyncOp* op, struct ncclProxyState* proxyState, int* asyncOpCount, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool) {
  int done = 1;
  ncclResult_t res = ncclInternalError;
  if (op->type == ncclProxyMsgSetup) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::proxySetup() opId=%p", op->opId);
    res = op->connection->tcomm->proxySetup(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgConnect) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::proxyConnect() opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    res = op->connection->tcomm->proxyConnect(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgSharedInit) {
    int nChannels = (int) *op->reqBuff;
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgSharedInit opId=%p op.reqBuff=%p nChannels=%d", op->opId, op->reqBuff, nChannels);
    if (op->connection->tcomm->proxySharedInit) res = op->connection->tcomm->proxySharedInit(op->connection, proxyState, nChannels);
    COMPILER_ATOMIC_STORE(&op->connection->state, connSharedInitialized, std::memory_order_release);
  }
  else if (op->type == ncclProxyMsgInit) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgInit opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    res = proxyConnInit(peer, connectionPool, proxyState, (ncclProxyInitReq*) op->reqBuff, (ncclProxyInitResp*) op->respBuff, &op->connection);
  } else if (op->type == ncclProxyMsgRegister) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgRegister opId=%p op.reqBuff=%p, op->reqSize=%d, op->respSize=%d", op->opId, op->reqBuff, op->reqSize, op->respSize);
    res = op->connection->tcomm->proxyRegister(op->connection, proxyState, op->reqBuff, op->reqSize, op->respBuff, op->respSize, &done);
  } else if (op->type == ncclProxyMsgDeregister) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgDeregister opId=%p op.reqBuff=%p, op->reqSize=%d, op->respSize=%d", op->opId, op->reqBuff, op->reqSize, op->respSize);
    res = op->connection->tcomm->proxyDeregister(op->connection, proxyState, op->reqBuff, op->reqSize, &done);
  } else return ncclInternalError;

  if (done) {
    INFO(NCCL_PROXY, "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op.respSize=%d done", op->opId, op->type, op->reqBuff, op->respSize);
    if (op->type == ncclProxyMsgSetup)
      COMPILER_ATOMIC_STORE(&op->connection->state, connSetupDone, std::memory_order_release);
    else if (op->type == ncclProxyMsgConnect)
      COMPILER_ATOMIC_STORE(&op->connection->state, connConnected, std::memory_order_release);
    /* if setup or connect is done, we should not return any error at this point since
     * ncclSocketSend might already send the respBuff to the requester. If we still choose
     * to abort and close the connection, it can cause segfault if the requester is using
     * the respBuff. */

    ncclProxyRpcResponseHeader resp = {op->opId, res, op->respSize};

    // Send the opId for referencing async operation
    NCCLCHECK(ncclSocketSend(op->connection->sock, &resp, sizeof(resp)));

    if (op->respSize) {
      // Send the response
      NCCLCHECK(ncclSocketSend(op->connection->sock, op->respBuff, op->respSize));
    }

    asyncProxyOpDequeue(peer, op);
    (*asyncOpCount)--;
    return ncclSuccess;

  } else if (COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_acquire) != 0) {
    return ncclInternalError;
  }

  return ncclInProgress;
}

static ncclResult_t proxyServiceInitOp(int type, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, int* asyncOpCount) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* sock = &peer->sock;
  struct ncclProxyAsyncOp* asyncOp;
  NCCLCHECK(ncclCalloc(&asyncOp, 1));

  asyncOp->type = type;
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->connection, sizeof(void*)), ret, fail);

  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->reqSize, sizeof(int)), ret, fail);
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->respSize, sizeof(int)), ret, fail);
  if (asyncOp->reqSize) {
    NCCLCHECKGOTO(ncclCalloc(&asyncOp->reqBuff, asyncOp->reqSize), ret, fail);
    NCCLCHECKGOTO(ncclSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize), ret, fail);
  }

  // Store opId for completion response
  NCCLCHECKGOTO(ncclSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)), ret, fail);

  if (asyncOp->respSize) NCCLCHECKGOTO(ncclCalloc(&asyncOp->respBuff, asyncOp->respSize), ret, fail);

  asyncProxyOpEnqueue(peer, asyncOp);

  (*asyncOpCount)++;
  NCCLCHECK(proxyProgressAsync(asyncOp, proxyState, asyncOpCount, peer, connectionPool));
exit:
  return ret;
fail:
  if (asyncOp->reqBuff) free(asyncOp->reqBuff);
  if (asyncOp->respBuff) free(asyncOp->respBuff);
  free(asyncOp);
  goto exit;
}

#include <poll.h>

static bool proxyMatchOpType(int type) {
  switch (type) {
    case ncclProxyMsgInit:
    case ncclProxyMsgSharedInit:
    case ncclProxyMsgSetup:
    case ncclProxyMsgConnect:
    case ncclProxyMsgGetFd:
    case ncclProxyMsgRegister:
    case ncclProxyMsgDeregister:
      return true;
    default:
      return false;
  }
}

enum {
  PROXY_RUNNING = 0,
  PROXY_STOP = 1,
  PROXY_ABORT = 2
};

void* ncclProxyService(void* _args) {
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;

  // set the thread affinity before setting the cuda context
  std::call_once(proxyCpusetOnceFlag, proxyCpusetOnceFunc);
  if (ncclOsCpuCount(proxyCpuset)) ncclOsSetAffinity(proxyCpuset);
  INFO(NCCL_INIT, "[Proxy Service] Device %d CPU core %d", proxyState->cudaDev, ncclOsGetCpu());

  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service] Created CUDA context on device %d", proxyState->cudaDev);
  } else if (!CUDASUCCESS(cudaSetDevice(proxyState->cudaDev))) {
    WARN("[Proxy Service] Failed to set CUDA device %d", proxyState->cudaDev);
  }

  // Prepare poll descriptor
  struct ncclProxyConnectionPool connectionPool;
  connectionPool.pools = NULL;
  connectionPool.banks = 0;
  connectionPool.offset = NCCL_PROXY_CONN_POOL_SIZE;

  struct pollfd pollfds[NCCL_MAX_PROXY_CONNECTIONS+1]; // one extra for listenSock fd
  struct ncclProxyLocalPeer peers[NCCL_MAX_PROXY_CONNECTIONS];
  memset(&peers, 0, sizeof(struct ncclProxyLocalPeer)*NCCL_MAX_PROXY_CONNECTIONS);
  for (int s=0; s<NCCL_MAX_PROXY_CONNECTIONS; s++) {
    pollfds[s].fd = NCCL_INVALID_SOCKET;
    pollfds[s].events = POLLHUP|POLLIN;
  }
  if (ncclSocketGetFd(proxyState->listenSock, &pollfds[NCCL_MAX_PROXY_CONNECTIONS].fd) != ncclSuccess) {
    WARN("[Proxy Service] Get listenSock fd fails");
    return NULL;
  };
  pollfds[NCCL_MAX_PROXY_CONNECTIONS].events = POLLIN;

  int maxnpeers = 0;
  int npeers = 0;
  int stop = PROXY_RUNNING;
  int asyncOpCount = 0;
  while (stop == PROXY_RUNNING || npeers > 0) {
    /* Even if local comm aborts, we cannot let proxy thread exit if we still have peer
     * connections. Need to wait until all other related comms call abort and safely exit
     * together, or we could face segmentation fault. */
    if (COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_acquire) != 0) stop = PROXY_ABORT;
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    int ret;
    do {
      // poll all fds including the listenSock
      ret = poll(pollfds, NCCL_MAX_PROXY_CONNECTIONS+1, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service] Poll failed: %s", strerror(errno));
      return NULL;
    }
    if (pollfds[NCCL_MAX_PROXY_CONNECTIONS].revents) {
      // We got an event on the listenSock
      int s = 0;
      while (s < NCCL_MAX_PROXY_CONNECTIONS && pollfds[s].fd >= 0) s++;
      if (s == NCCL_MAX_PROXY_CONNECTIONS) {
        WARN("[Proxy service] Too many connections (%d max)", NCCL_MAX_PROXY_CONNECTIONS);
        return NULL;
      }
      if (maxnpeers < s+1) maxnpeers = s+1;
      if (ncclSocketInit(&peers[s].sock) != ncclSuccess) {
        WARN("[Service thread] Initialize peers[%d].sock fails", s);
        return NULL;
      }
      if (ncclSocketAccept(&peers[s].sock, proxyState->listenSock) != ncclSuccess) {
        WARN("[Service thread] Accept failed %s", strerror(errno));
      } else {
        if (ncclSocketGetFd(&peers[s].sock, &pollfds[s].fd) != ncclSuccess) {
          WARN("[Service thread] Get peers[%d].sock fd fails", s);
          return NULL;
        }
        npeers++;
        peers[s].tpLocalRank = -1;
      }
    }
    for (int s=0; s<maxnpeers; s++) {
      struct ncclProxyLocalPeer* peer = peers+s;
      struct ncclSocket* sock = &peer->sock;
      int closeConn = 0;
      int type = 0;
      ncclResult_t res = ncclSuccess;
      if (pollfds[s].fd == NCCL_INVALID_SOCKET) continue;

      // Progress all ops for this ncclProxyLocalPeer
      if (stop == PROXY_ABORT && ncclCuMemEnable() && ncclCuMemHostEnable() && !proxyState->directMode && COMPILER_ATOMIC_LOAD(&proxyState->stop, std::memory_order_acquire)) closeConn = 1;
      ncclProxyAsyncOp* op = peer->asyncOps;
      while (op != nullptr) {
        ncclProxyAsyncOp* opnext = op->next; /* in case op is freed in proxyProgressAsync */
        type = op->type;
        // Coverity gets confused here by complex code structure.  Yes, connectionPool.pools gets dereferenced, and
        // while calling proxyProgressAsync() connectionPool.pools is NULL, but that changes before it's dereferenced.
        // coverity[var_deref_model:FALSE]
        res = proxyProgressAsync(op, proxyState, &asyncOpCount, peer, &connectionPool);
        if (res == ncclSuccess || res == ncclInProgress) {
          op = opnext;
        } else {
          // Res is a bad result
          closeConn = 1;
          WARN("[Service thread] Error encountered progressing operation=%s, res=%d, closing connection", ncclProxyMsgTypeStr[type], res);
          break;
        }
      }

      // Check for additional ops coming in
      if (pollfds[s].revents & POLLIN) {
        int closed;
        res = ncclSocketTryRecv(sock, &type, sizeof(int), &closed, false /*blocking*/);
        if (res != ncclSuccess && res != ncclInProgress) {
          if (!COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_relaxed))
            WARN("[Service thread] Could not receive type from localRank %d, res=%u, closed=%d", peer->tpLocalRank, res, closed);
          closeConn = 1;
        } else if (closed) {
          INFO(NCCL_INIT|NCCL_NET|NCCL_PROXY, "[Service thread] Connection closed by localRank %d", peer->tpLocalRank);
          closeConn = 1;
        } else if (res == ncclSuccess) { // We received something from the sock
          if (type == ncclProxyMsgStop) {
            stop = PROXY_STOP;
            closeConn = 1;
          } else if (type == ncclProxyMsgClose) {
            closeConn = 1;
          } else if (proxyMatchOpType(type)) {
            res = proxyServiceInitOp(type, peers+s, &connectionPool, proxyState, &asyncOpCount);
          } else {
            WARN("[Service thread] Unknown command %d from localRank %d", type, peer->tpLocalRank);
            closeConn = 1;
          }

          INFO(NCCL_PROXY, "Received and initiated operation=%s res=%d", ncclProxyMsgTypeStr[type], res);
        }
      } else if (pollfds[s].revents & POLLHUP) {
        closeConn = 1;
      }
      if (res != ncclSuccess && res != ncclInProgress) {
        if (!COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_relaxed))
          WARN("[Proxy Service %d] Failed to execute operation %s from rank %d, retcode %d", proxyState->tpRank, ncclProxyMsgTypeStr[type], peer->tpRank, res);
        closeConn = 1;
      }

      if (closeConn) {
        (void)ncclSocketClose(sock);

        if (op != nullptr) {
          asyncProxyOpDequeue(peer, op);
          asyncOpCount--;
        }
        pollfds[s].fd = NCCL_INVALID_SOCKET;
        npeers--;
      }
    }
  }

  // Wait for all operations to complete and stop progress thread before freeing any resource
  if (ncclProxyProgressDestroy(proxyState) != ncclSuccess) {
    WARN("[Proxy Service] proxyDestroy failed");
  }
  for (int s=0; s<maxnpeers; s++) {
    (void)ncclSocketClose(&peers[s].sock);
  }
  ncclProxyFreeConnections(&connectionPool, proxyState);
  (void)ncclSocketClose(proxyState->listenSock);
  free(proxyState->listenSock);
  proxyOpsFree(proxyState);
  return NULL;
}


// Process a request on the UDS socket
static ncclResult_t proxyUDSRecvReq(struct ncclProxyState* proxyState, int reqFd) {
  ncclIpcHdr hdr;
  int rmtFd = -1;

  NCCLCHECK(ncclIpcSocketRecvMsg(&proxyState->ipcSock, &hdr, sizeof(hdr), &rmtFd));
  if (hdr.type == ncclProxyMsgGetFd) {
    // cuMem API support for non-UB case, and rmtFd is not used since UDS proxy thread need to export
    // fd from handle and send it back to the main thread to import the buffer. We just need to close
    // this dummy rmtFd.
    uint64_t handle = *(uint64_t*)hdr.data;
    INFO(NCCL_PROXY, "proxyUDSRecvReq::ncclProxyMsgGetFd rank %d opId %p handle=0x%lx", hdr.rank, hdr.opId, handle);
    close(rmtFd);
    return proxyGetFd(proxyState, hdr.rank, hdr.opId, handle);
  } else if (hdr.type == ncclProxyMsgQueryFd) {
    // remote main thread registers buffer into this rank, it querys rmtFd of this rank through UDS
    // and the rmtFd is returned unchanged back to remote main thread which will use rmtFd to call into
    // proxy service thread for buffer registration.
    INFO(NCCL_PROXY, "proxyUDSRecvReq::proxyQueryFd rank %d opId %p rmtFd %d", hdr.rank, hdr.opId, rmtFd);
    return proxyQueryFd(proxyState, hdr.rank, hdr.opId, rmtFd);
  }

  return ncclInternalError;
}

// UDS fd handle support
void* ncclProxyServiceUDS(void* _args) {
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;
  struct pollfd pollfds[1];

  // set the thread affinity before setting the cuda context
  std::call_once(proxyCpusetOnceFlag, proxyCpusetOnceFunc);
  if (ncclOsCpuCount(proxyCpuset)) ncclOsSetAffinity(proxyCpuset);
  INFO(NCCL_INIT, "[Proxy Service UDS] Device %d CPU core %d", proxyState->cudaDev, ncclOsGetCpu());

  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service UDS] Set CUDA context on device %d", proxyState->cudaDev);
  } else if (!CUDASUCCESS(cudaSetDevice(proxyState->cudaDev))) {
    WARN("[Proxy Service UDS] Failed to set CUDA device %d", proxyState->cudaDev);
  }

  if (ncclIpcSocketGetFd(&proxyState->ipcSock, &pollfds[0].fd) != ncclSuccess) {
    WARN("[Proxy Service UDS] Get listenSock fd fails");
    return NULL;
  };
  pollfds[0].events = POLLIN|POLLHUP;

  while (1) {
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    int ret;
    do {
      ret = poll(pollfds, 1, 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service UDS] Poll failed: %s", strerror(errno));
      return NULL;
    }

    // Check for stop/abort
    if (COMPILER_ATOMIC_LOAD(&proxyState->stop, std::memory_order_acquire) || COMPILER_ATOMIC_LOAD(proxyState->abortFlag, std::memory_order_acquire)) break;

    if (pollfds[0].revents) {
      // A request was seen on the UDS fd
      proxyUDSRecvReq(proxyState, pollfds[0].fd);
    }
  }

  (void)ncclIpcSocketClose(&proxyState->ipcSock);
  INFO(NCCL_PROXY, "[Proxy Service UDS] exit: stop %d abortFlag %d", proxyState->stop, *proxyState->abortFlag);
  return NULL;
}

ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t *peerAddressesUDS) {
  assert(comm->sharedRes->proxyState == nullptr);
  comm->sharedRes->proxyState = new ncclProxyState{};
  comm->proxyState = comm->sharedRes->proxyState;
  comm->proxyState->refCount = 1;
  comm->proxyState->listenSock = sock;
  comm->proxyState->peerAddresses = peerAddresses;
  comm->proxyState->peerAddressesUDS = peerAddressesUDS;
  comm->proxyState->netAttr = NCCL_NET_ATTR_INIT;

  // UDS support
  NCCLCHECK(ncclIpcSocketInit(&comm->proxyState->ipcSock, comm->rank, peerAddressesUDS[comm->rank], comm->abortFlag));
  return ncclSuccess;
}

ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  /* proxyState is shared among parent comm and split comms. comm->proxyState->thread is
   * join()'d by commFree() in init.cc when the refCount reduces down to 0. */
  struct ncclProxyState* proxyState = comm->proxyState;
  if (proxyState->refCount == 1) {
    /* we have to make sure all following fields in comm have been initialized. */
    proxyState->tpRank = comm->rank;
    proxyState->tpnRanks = comm->nRanks;
    proxyState->tpLocalnRanks = comm->localRanks;
    proxyState->cudaDev = comm->cudaDev;
    proxyState->abortFlag = comm->abortFlag;
    proxyState->p2pnChannels = comm->p2pnChannels;
    proxyState->p2pChunkSize = comm->p2pChunkSize;
    proxyState->nChannels = comm->nChannels;
    proxyState->allocP2pNetLLBuffers = comm->allocP2pNetLLBuffers;
    proxyState->dmaBufSupport = comm->dmaBufSupport;
    proxyState->ncclNet = comm->ncclNet;
    proxyState->ncclCollNet = comm->ncclCollNet;
    proxyState->ginState = &comm->sharedRes->ginState;
    proxyState->netContext = comm->netContext;
    proxyState->collNetContext = comm->collNetContext;
    proxyState->profilerContext = comm->profilerContext;
    proxyState->directMode = comm->directMode;
    memcpy(proxyState->buffSizes, comm->buffSizes, sizeof(comm->buffSizes));

    comm->proxyState->thread = std::thread(ncclProxyService, comm->proxyState);
    ncclSetThreadName(comm->proxyState->thread, "NCCL Service %2d", comm->cudaDev);

    // UDS support
    INFO(NCCL_PROXY, "UDS: Creating service thread comm %p rank %d", comm, comm->rank);
    comm->proxyState->threadUDS = std::thread(ncclProxyServiceUDS, comm->proxyState);
    ncclSetThreadName(comm->proxyState->threadUDS, "NCCL UDS Service %2d", comm->cudaDev);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyStop(struct ncclComm* comm) {
  if (comm->proxyState) {
    struct ncclProxyState* sharedProxyState = comm->proxyState;

    if ((comm->proxyRefCountOld = ncclAtomicRefCountDecrement(&sharedProxyState->refCount)) == 0) {
      if (*comm->abortFlag == 0 && sharedProxyState->peerAddresses) {
        // We need to send a ncclProxyMsgStop message to our own proxy
        struct ncclSocket sock;
        int type = ncclProxyMsgStop;
        NCCLCHECK(ncclSocketInit(&sock, sharedProxyState->peerAddresses + comm->topParentRanks[comm->rank], comm->sharedRes->magic, ncclSocketTypeProxy, comm->abortFlag));
        if (ncclSocketConnect(&sock) == ncclSuccess) {
          (void)ncclSocketSend(&sock, &type, sizeof(int));
        }
        (void)ncclSocketClose(&sock);
      }

      if (sharedProxyState->peerSocks) {
        int tplocalRanks = comm->sharedRes->tpNLocalRanks;
        for (int i = 0; i < tplocalRanks; i++) {
          int fd;
          NCCLCHECK(ncclSocketGetFd(sharedProxyState->peerSocks + i, &fd));
          if (fd >= 0) {
            if (sharedProxyState->proxyOps[i].pool) {
              NCCLCHECK(ncclShmClose(sharedProxyState->proxyOps[i].handle));
            }
            if (sharedProxyState->sharedDevMems[i]) {
              if (!ncclCuMemEnable()) {
                CUDACHECK(cudaIpcCloseMemHandle(sharedProxyState->sharedDevMems[i]));
              }
            }
            int type = ncclProxyMsgClose;
            (void)ncclSocketSend(sharedProxyState->peerSocks + i, &type, sizeof(int));
            NCCLCHECK(ncclSocketClose(sharedProxyState->peerSocks + i));
          }
        }
      }
      // Now we notify proxy service and UDS thread to exit.
      COMPILER_ATOMIC_STORE(&comm->proxyState->stop, 1, std::memory_order_release);
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  struct ncclProxyState* sharedProxyState = comm->sharedRes->proxyState;

  if (sharedProxyState) {
    assert(sharedProxyState->refCount == 0);
    free(sharedProxyState->peerAddresses);
    free(sharedProxyState->peerAddressesUDS);
    free(sharedProxyState->peerSocks);
    free(sharedProxyState->proxyOps);
    free(sharedProxyState->sharedDevMems);
    expectedProxyResponseFree(sharedProxyState);
    delete sharedProxyState;
  }
  return ncclSuccess;
}
