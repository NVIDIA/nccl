/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "socket.h"
#include "shm.h"
#include "profiler.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <sys/syscall.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

enum { proxyRecv=0, proxySend=1 };

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

      memcpy(elem->respBuff, respBuff, respSize);
      free(respBuff);
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
  sub->peer = op->root;
  sub->reg = op->reg;
  sub->buffer = op->buffer;
  args->nsubs = subIndex+1;
  if (subIndex) {
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
    return ncclSuccess;
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
  args->specifics = op->specifics;
  args->state = ncclProxyOpReady;
  args->progress = op->connection->tcomm->proxyProgress;
  args->proxyAppendPtr = op->connection->proxyAppendPtr;
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
  pthread_mutex_lock(&pool->mutex);
  if (pool->nextOps == -1) {
    pool->nextOps = nextOps;
    pthread_cond_signal(&pool->cond);
  } else {
    pool->ops[pool->nextOpsEnd].next = nextOps;
  }
  pool->nextOpsEnd = nextOpsEnd;
  pthread_mutex_unlock(&pool->mutex);
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
    int freeOp;
    while ((freeOp = pool->freeOps[tpLocalRank]) == -1) sched_yield();
    int freeOpNew;
    while ((freeOpNew = __sync_val_compare_and_swap(pool->freeOps+tpLocalRank, freeOp, -1)) != freeOp) freeOp = freeOpNew;
    opIndex = freeOp;
    op = pool->ops+opIndex;
    proxyOps->freeOp = op->next;
  }
  if (op->next != -1) __builtin_prefetch(pool->ops+op->next); // Prefetch next free op
  memcpy(op, proxyOp, sizeof(struct ncclProxyOp));
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
    NCCLCHECK(ncclLocalOpAppend(comm, &connector->proxyConn, op));
  }
  return ncclSuccess;
}

// justInquire != nullptr means don't actually do anything, just assertain need of
// ncclProxySaveOp for this op.
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire) {
  struct ncclChannel* channel = &comm->channels[op->channelId];
  if (justInquire) *justInquire = false;
  switch (op->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo: {
      struct ncclRing* ring = &channel->ring;
      if (NeedProxy(proxyRecv, op->pattern, op->root, ring, comm->nRanks)) {
        NCCLCHECK(SaveProxy(comm, channel, proxyRecv, ring->prev, op, 0, justInquire));
      }
      if (NeedProxy(proxySend, op->pattern, op->root, ring, comm->nRanks)) {
        NCCLCHECK(SaveProxy(comm, channel, proxySend, ring->next, op, 0, justInquire));
      }
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
  case ncclPatternSend:
  case ncclPatternRecv: {
      if (op->root == comm->rank) return ncclSuccess;
      NCCLCHECK(SaveProxy(comm, channel, op->pattern == ncclPatternSend ? proxySend : proxyRecv, op->root, op, 1, justInquire));
    } break;
  }
  return ncclSuccess;
}

NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);

ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyOp* op, int reg) {
  memset(op, 0, sizeof(struct ncclProxyOp));
  int channelId = info->channelId;
  struct ncclChannel* channel = info->comm->channels+channelId;
  op->channelId = channelId;
  op->sliceSteps = 1;
  op->chunkSteps = 1;
  op->dtype = info->datatype;
  op->protocol = info->protocol;

  int stepSize = info->comm->buffSizes[op->protocol]/NCCL_STEPS;

  if (op->protocol == NCCL_PROTO_SIMPLE) stepSize = info->comm->p2pChunkSize;
  info->chunkSize = stepSize;
  op->root = info->root;

  struct ncclChannelPeer* peer = channel->peers[op->root];
  if (info->coll == ncclFuncSend) {
    op->pattern = ncclPatternSend;
    if (op->root != info->comm->rank && peer->send[1].transportComm == &netTransport.send) {
      // Tune chunk size for the network
      if (info->protocol == NCCL_PROTO_SIMPLE && info->count < stepSize) info->chunkSize /= 4;
      else if (info->count < 8*stepSize) info->chunkSize /= 2;
      if (info->protocol == NCCL_PROTO_SIMPLE && peer->send[1].proxyConn.sameProcess) op->reg = reg;
    }
  } else if (info->coll == ncclFuncRecv) {
    op->pattern = ncclPatternRecv;
    if (op->root != info->comm->rank && peer->recv[1].transportComm == &netTransport.recv) {
      // Tune chunk size for the network
      if (info->protocol == NCCL_PROTO_SIMPLE && info->count < stepSize) info->chunkSize /= 4;
      else if (info->count < 8*stepSize) info->chunkSize /= 2;
      if (info->protocol == NCCL_PROTO_SIMPLE && peer->recv[1].proxyConn.sameProcess) op->reg = reg;
    }
  } else {
    WARN("P2p operation is neither send or recv");
    return ncclInternalError;
  }
  if (ncclParamChunkSize() != 0) {
    info->chunkSize = ncclParamChunkSize();
  }
  op->buffer = op->reg ? info->recvbuff : NULL;
  op->chunkSize = info->chunkSize;
  op->nbytes = info->count;

  // Compute nSteps for proxies
  int chunkEffectiveSize = op->chunkSize;
  if (op->protocol == NCCL_PROTO_LL) {
    chunkEffectiveSize /= 2;
    op->nbytes *= 2;
    op->nbytes = DIVUP(op->nbytes, sizeof(union ncclLLFifoLine)) * sizeof(union ncclLLFifoLine);
  }

  if (!op->reg) op->nbytes = std::min(op->nbytes, (ssize_t)info->chunkSize);
  op->nsteps = DIVUP(info->count, chunkEffectiveSize);
  if (op->nsteps == 0 || op->reg) op->nsteps = 1;

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
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    TIME_START(0); TIME_START(1);
    NCCLCHECK(op->progress(proxyState, op));
    if (op->idle) { TIME_STOP(1); TIME_CANCEL(0); } else { TIME_CANCEL(1); TIME_STOP(0); }
    *idle &= op->idle;
    if (op->state == ncclProxyOpNone) {
      TIME_START(2);
      NCCLCHECK(removeOp(state, &op, &prevOp));
      TIME_STOP(2);
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(ProxyAppendBatchSize, "PROXY_APPEND_BATCH_SIZE", 16);

static ncclResult_t ncclProxyGetPostedOps(struct ncclProxyState* proxyState, int* added) {
  struct ncclProxyProgressState* state = &proxyState->progressState;
  if (state->opsPool == NULL) return ncclInternalError;
  struct ncclProxyOpsPool* pool = state->opsPool;

  struct ncclProxyArgs profArgs; // Only used for profiling purposes
  if (state->nextOps != -1) goto process_nextops;

  // If we have ops to progress, no need to block waiting for something to arrive or even wait for the lock
  // to be available. Exit, continue progress, and come back later.
  if (state->active != NULL && (pool->nextOps == -1 || pthread_mutex_trylock(&pool->mutex) != 0)) return ncclSuccess;

  if (state->active == NULL) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->nextOps == -1 && !state->stop) {
      struct ncclProxyArgs profArgs; // Only used for profiling purposes
      ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileSleep);
      pthread_cond_wait(&pool->cond, &pool->mutex);
      ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileWakeup);
    }
    if (state->stop) { // We might have been woken up to stop.
      pthread_mutex_unlock(&pool->mutex);
      return ncclSuccess;
    }
  }

  state->nextOps = pool->nextOps;
  pool->nextOps = pool->nextOpsEnd = -1;
  pthread_mutex_unlock(&pool->mutex);
  if (state->nextOps == -1) return ncclInternalError;

process_nextops:
  ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileAppend);
  TIME_START(2);
  int freeOp[NCCL_MAX_LOCAL_RANKS];
  int freeOpEnd[NCCL_MAX_LOCAL_RANKS];
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
    if (peerOp->next != -1) __builtin_prefetch(pool->ops+peerOp->next);
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
    int newFree = freeOp[i];
    int oldFree = pool->freeOps[i];
    pool->ops[freeOpEnd[i]].next = oldFree;
    if (oldFree == -1) {
      // Nothing for the main thread to consume, we can set it.
      pool->freeOps[i] = newFree;
    } else {
      // The main thread may recycle free ops at any time, replace the freeOps value atomically and check it worked.
      int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);
      if (swap != oldFree) {
        if (swap != -1) return ncclInternalError;
        // Ops were recycled while we were trying to swap, just set the value directly now.
        pool->ops[freeOpEnd[i]].next = -1;
        pool->freeOps[i] = newFree;
      }
    }
  }
  profArgs.opCount = *added;
  ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileAppendEnd);
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
      }
    }
  }
  if (createThreadContext) {
    if (proxyState->cudaCtx == NULL) {
      if (CUPFN(cuCtxCreate(&proxyState->cudaCtx,
                                  CU_CTX_SCHED_SPIN|CU_CTX_MAP_HOST, proxyState->cudaDev)) != CUDA_SUCCESS) {
        WARN("Failed to create CUDA context on device %d", proxyState->cudaDev);
        createThreadContext = 0;
      }
    } else {
      if (CUPFN(cuCtxSetCurrent(proxyState->cudaCtx)) != CUDA_SUCCESS) {
        WARN("Failed to set CUDA context on device %d", proxyState->cudaDev);
        return 0;
      }
      return 1;
    }
  }
#endif
  return 0;
}

// Set to SIGUSR1 or SIGUSR2 to help debug proxy state during hangs
NCCL_PARAM(ProxyDumpSignal, "PROXY_DUMP_SIGNAL", -1);
NCCL_PARAM(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);

void* ncclProxyProgress(void *proxyState_) {
  struct ncclProxyState* proxyState = (struct ncclProxyState*)proxyState_;
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Progress] Created CUDA context on device %d", proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) {
    WARN("[Proxy Progress] Failed to set CUDA device %d", proxyState->cudaDev);
  }
  // if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);

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
  struct ncclProxyArgs profArgs; // Only used for profiling purposes
  while ((state->stop == 0 || (state->stop == 1 && state->active)) && *proxyState->abortFlag == 0) {
    int idle = 1;
    ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);
    if (ret != ncclSuccess) {
      __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
      INFO(NCCL_ALL,"%s:%d -> %d [Progress Thread]", __FILE__, __LINE__, ret);
      continue;
    }
    if (lastIdle == 0 && idle == 1) ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileIdle);
    if (lastIdle == 1 && idle == 0) ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileActive);
    if (idle || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
      int added = 0;
      proxyOpAppendCounter = 0;
      TIME_START(3);
      if (state->stop == 0)
        ret = ncclProxyGetPostedOps(proxyState, &added);
      if (added) { TIME_STOP(3); } else { TIME_CANCEL(3); }
      if (ret != ncclSuccess) {
        __atomic_store_n(&proxyState->asyncResult, ret, __ATOMIC_RELEASE);
        INFO(NCCL_ALL,"%s:%d -> %d [Progress Thread]", __FILE__, __LINE__, ret);
      }
      if (added == 0) {
        sched_yield(); // No request progressed. Let others run.
      }
    }
    lastIdle = idle;
  }
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
  if (!state->thread) {
    pthread_create(&state->thread, NULL, ncclProxyProgress, proxyState);
    ncclSetThreadName(state->thread, "NCCL Progress%2d", proxyState->tpLocalnRanks);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyProgressDestroy(struct ncclProxyState* proxyState) {
  struct ncclProxyProgressState* state = &proxyState->progressState;

  // Request the proxy to stop and then wake it
  if (state->opsPool) {
    pthread_mutex_lock(&state->opsPool->mutex);
    state->stop = 1;
    pthread_cond_signal(&state->opsPool->cond);
    pthread_mutex_unlock(&state->opsPool->mutex);
    pthread_join(state->thread, NULL);
  }

  // Free off any memory allocated for the proxy arg pools
  while (state->pools != NULL) {
    struct ncclProxyPool *next = state->pools->next;
    free(state->pools);
    state->pools = next;
  }

  ncclProfilingDump();
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

ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int tpProxyRank, struct ncclProxyConnector* proxyConn) {
  struct ncclSocket* sock;
  int ready, proxyRank = -1;
  struct ncclProxyState* sharedProxyState = comm->proxyState;

  // Keep one connection per local rank
  for (int i = 0; i < comm->localRanks; ++i) {
    /* find the proxy rank in comm. */
    if (comm->topParentRanks[comm->localRankToRank[i]] == tpProxyRank) {
      proxyRank = comm->localRankToRank[i];
      break;
    }
  }
  proxyConn->sameProcess = comm->peerInfo[proxyRank].pidHash == comm->peerInfo[comm->rank].pidHash ? 1 : 0;
  // Keep one connection per local rank
  proxyConn->connection = NULL;
  proxyConn->tpRank = tpProxyRank;
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
      NCCLCHECK(ncclShmOpen(poolPath, sizeof(struct ncclProxyOpsPool), (void**)(&proxyOps->pool), NULL, 0, &proxyOps->handle));
      proxyOps->nextOps = proxyOps->nextOpsEnd = proxyOps->freeOp = -1;
    }
  }
  INFO(NCCL_NET|NCCL_PROXY, "Connected to proxy localRank %d -> connection %p", proxyConn->tpLocalRank, proxyConn->connection);
  return ncclSuccess;
}

// UDS support
ncclResult_t ncclProxyCallBlockingUDS(struct ncclComm* comm, int tpRank, int type, void* reqBuff, int reqSize, void* respBuff, int respSize, int *respFd) {
  ncclResult_t res = ncclSuccess;
  struct ncclIpcSocket ipcSock = { 0 };
  void *opId = (void*)((((uintptr_t)random()) << 32) | random());

  int rank = comm->topParentLocalRanks[comm->localRank];
  struct ncclProxyState* sharedProxyState = comm->proxyState;
  uint64_t pidHash = sharedProxyState->peerAddressesUDS[tpRank];

  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %p opId %p",
       comm, rank, tpRank, pidHash, reqSize, respSize, respFd, opId);

  // cuMem: Create a UDS socket to receive the response
  NCCLCHECK(ncclIpcSocketInit(&ipcSock, rank, (uint64_t)opId, comm->abortFlag));

  ncclIpcHdr hdr;
  hdr.type = type;
  hdr.rank = rank;
  hdr.reqSize = reqSize;
  hdr.respSize = respSize;
  hdr.opId = opId;
  assert(reqSize <= sizeof(hdr.data));
  memcpy(&hdr.data, reqBuff, reqSize);
  NCCLCHECKGOTO(ncclIpcSocketSendMsg(&ipcSock, &hdr, sizeof(hdr), -1, tpRank, pidHash), res, error);
  NCCLCHECKGOTO(ncclIpcSocketRecvMsg(&ipcSock, respBuff, respSize, respFd), res, error);
  NCCLCHECKGOTO(ncclIpcSocketClose(&ipcSock), res, error);

  INFO(NCCL_PROXY, "ProxyCall UDS comm %p rank %d tpRank %d(%lx) reqSize %d respSize %d respFd %d opId %p - DONE",
       comm, rank, tpRank, pidHash, reqSize, respSize, (respFd ? *respFd : -1), opId);

  return res;

error:
  NCCLCHECK(ncclIpcSocketClose(&ipcSock));
  WARN("ncclProxyCallBlockingUDS call to tpRank %d(%lx) failed : %d", tpRank, pidHash, res);
  return res;
}

// cuMem API support
// The request/response is sent out-of-band using ncclIpcSocket for this specific command
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int tpRank, void *handle, int* convertedFd) {
  ncclResult_t ret = ncclSuccess;

  // Request the allocation of a UDS fd for the handle
  NCCLCHECKGOTO(ncclProxyCallBlockingUDS(comm, tpRank, ncclProxyMsgGetFd, handle, sizeof(CUmemGenericAllocationHandle), NULL, 0, convertedFd), ret, error);

  // We have now received the converted fd over UDS
  INFO(NCCL_PROXY, "UDS: ClientGetFd handle 0x%lx tpRank %d returned fd %d", *(uint64_t*)handle, tpRank, *convertedFd);

  return ret;

error:
  WARN("ncclProxyClientGetFd call to tpRank %d handle 0x%lx failed : %d", tpRank, *(uint64_t*)handle, ret);
  return ret;
}

const char* ncclProxyMsgTypeStr[] = { "Unknown", "Init", "SharedInit", "Setup", "Connect", "Start", "Close", "Abort", "Stop", "GetFd" };
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff, int reqSize, int respSize, void* opId) {
  struct ncclSocket* sock;
  ncclResult_t ret = ncclSuccess;
  struct ncclProxyState* sharedProxyState = comm->proxyState;

  if (sharedProxyState->peerSocks == NULL) return ncclInternalError;

  sock = sharedProxyState->peerSocks + proxyConn->tpLocalRank;
  if (sock == NULL) return ncclInternalError;

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
  if (__atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED)) {
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
    NCCLCHECK(ncclShmOpen(shmPath, size, (void**)&pool, NULL, proxyState->tpLocalnRanks + 1, &state->handle));
    // Init pool
    pool->nextOps = -1;

    for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
      pool->freeOps[r] = r*MAX_OPS_PER_PEER;
      for (int i=0; i<MAX_OPS_PER_PEER-1; i++) pool->ops[r*MAX_OPS_PER_PEER+i].next = r*MAX_OPS_PER_PEER+i+1;
      pool->ops[(r+1)*MAX_OPS_PER_PEER-1].next = -1;
    }

    // Setup mutex/cond to work inter-process
    pthread_mutexattr_t mutexAttr;
    pthread_mutexattr_init(&mutexAttr);
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&pool->mutex, &mutexAttr);
    pthread_condattr_t condAttr;
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&pool->cond, &condAttr);
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
  __atomic_store_n(&(*connection)->state, connInitialized, __ATOMIC_RELEASE);
  return ncclSuccess;
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
  (void) close(fd);
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
    __atomic_store_n(&op->connection->state, connSharedInitialized, __ATOMIC_RELEASE);
  }
  else if (op->type == ncclProxyMsgInit) {
    TRACE(NCCL_PROXY, "proxyProgressAsync::ncclProxyMsgInit opId=%p op.reqBuff=%p", op->opId, op->reqBuff);
    res = proxyConnInit(peer, connectionPool, proxyState, (ncclProxyInitReq*) op->reqBuff, (ncclProxyInitResp*) op->respBuff, &op->connection);
  } else return ncclInternalError;

  if (done) {
    INFO(NCCL_PROXY, "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op.respSize=%d done", op->opId, op->type, op->reqBuff, op->respSize);
    if (op->type == ncclProxyMsgSetup)
      __atomic_store_n(&op->connection->state, connSetupDone, __ATOMIC_RELEASE);
    else if (op->type == ncclProxyMsgConnect)
      __atomic_store_n(&op->connection->state, connConnected, __ATOMIC_RELEASE);
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

  } else if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_RELAXED) != 0) {
    return ncclInternalError;
  }

  return ncclInProgress;
}

static ncclResult_t proxyServiceInitOp(int type, struct ncclProxyLocalPeer* peer, struct ncclProxyConnectionPool* connectionPool, struct ncclProxyState* proxyState, int* asyncOpCount) {
  struct ncclSocket* sock = &peer->sock;
  struct ncclProxyAsyncOp* asyncOp;
  NCCLCHECK(ncclCalloc(&asyncOp, 1));

  asyncOp->type = type;
  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->connection, sizeof(void*)));

  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->reqSize, sizeof(int)));
  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->respSize, sizeof(int)));
  if (asyncOp->reqSize) {
    NCCLCHECK(ncclCalloc(&asyncOp->reqBuff, asyncOp->reqSize));
    NCCLCHECK(ncclSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize));
  }

  // Store opId for completion response
  NCCLCHECK(ncclSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)));

  if (asyncOp->respSize) NCCLCHECK(ncclCalloc(&asyncOp->respBuff, asyncOp->respSize));

  asyncProxyOpEnqueue(peer, asyncOp);

  (*asyncOpCount)++;
  NCCLCHECK(proxyProgressAsync(asyncOp, proxyState, asyncOpCount, peer, connectionPool));
  return ncclSuccess;
}

#include <poll.h>

static bool proxyMatchOpType(int type) {
  switch (type) {
    case ncclProxyMsgInit:
    case ncclProxyMsgSharedInit:
    case ncclProxyMsgSetup:
    case ncclProxyMsgConnect:
    case ncclProxyMsgGetFd:
      return true;
    default:
      return false;
  }
}

void* ncclProxyService(void* _args) {
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;
  // if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service] Created CUDA context on device %d", proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) {
    WARN("[Proxy Service] Failed to set CUDA device %d", proxyState->cudaDev);
  }
  // if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);

  // Prepare poll descriptor
  struct ncclProxyConnectionPool connectionPool;
  connectionPool.pools = NULL;
  connectionPool.banks = 0;
  connectionPool.offset = NCCL_PROXY_CONN_POOL_SIZE;

  struct pollfd pollfds[NCCL_MAX_LOCAL_RANKS+1];
  struct ncclProxyLocalPeer peers[NCCL_MAX_LOCAL_RANKS];
  memset(&peers, 0, sizeof(struct ncclProxyLocalPeer)*NCCL_MAX_LOCAL_RANKS);
  for (int s=0; s<NCCL_MAX_LOCAL_RANKS; s++) {
    pollfds[s].fd = -1;
    pollfds[s].events = POLLHUP|POLLIN;
  }
  if (ncclSocketGetFd(proxyState->listenSock, &pollfds[NCCL_MAX_LOCAL_RANKS].fd) != ncclSuccess) {
    WARN("[Proxy Service] Get listenSock fd fails");
    return NULL;
  };
  pollfds[NCCL_MAX_LOCAL_RANKS].events = POLLIN;

  int maxnpeers = 0;
  int npeers = 0;
  int stop = 0;
  int asyncOpCount = 0;
  while (stop == 0 || (stop == 1 && npeers > 0)) {
    /* Even if local comm aborts, we cannot let proxy thread exit if we still have peer
     * connections. Need to wait until all other related comms call abort and safely exit
     * together, or we could face segmentation fault. */
    if (__atomic_load_n(proxyState->abortFlag, __ATOMIC_RELAXED) != 0) stop = 1;
    /* never let proxy service thread blocks in poll, or it cannot receive abortFlag. */
    int ret;
    do {
      ret = poll(pollfds, NCCL_MAX_LOCAL_RANKS+1, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service] Poll failed: %s", strerror(errno));
      return NULL;
    }
    if (pollfds[NCCL_MAX_LOCAL_RANKS].revents) {
      int s = 0;
      while (s < NCCL_MAX_LOCAL_RANKS && pollfds[s].fd >= 0) s++;
      if (s == NCCL_MAX_LOCAL_RANKS) {
        WARN("[Proxy service] Too many connections (%d max)", NCCL_MAX_LOCAL_RANKS);
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
      if (pollfds[s].fd == -1) continue;

      // Progress all ops for this ncclProxyLocalPeer
      ncclProxyAsyncOp* op = peer->asyncOps;
      while (op != nullptr) {
        ncclProxyAsyncOp* opnext = op->next; /* in case op is freed in proxyProgressAsync */
        type = op->type;
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
          WARN("[Service thread] Could not receive type from localRank %d, res=%u, closed=%d", peer->tpLocalRank, res, closed);
          closeConn = 1;
        } else if (closed) {
          INFO(NCCL_INIT|NCCL_NET|NCCL_PROXY, "[Service thread] Connection closed by localRank %d", peer->tpLocalRank);
          closeConn = 1;
        } else if (res == ncclSuccess) { // We received something from the sock
          if (type == ncclProxyMsgStop) {
            stop = 1;
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
        WARN("[Proxy Service %d] Failed to execute operation %s from rank %d, retcode %d", proxyState->tpRank, ncclProxyMsgTypeStr[type], peer->tpRank, res);
        closeConn = 1;
      }

      if (closeConn) {
        ncclSocketClose(sock);

        if (op != nullptr) {
          asyncProxyOpDequeue(peer, op);
          asyncOpCount--;
        }
        pollfds[s].fd = -1;
        npeers--;
      }
    }
  }

  // Wait for all operations to complete and stop progress thread before freeing any resource
  if (ncclProxyProgressDestroy(proxyState) != ncclSuccess) {
    WARN("[Proxy Service] proxyDestroy failed");
  }
  for (int s=0; s<maxnpeers; s++) {
    ncclSocketClose(&peers[s].sock);
  }
  ncclProxyFreeConnections(&connectionPool, proxyState);
  ncclSocketClose(proxyState->listenSock);
  free(proxyState->listenSock);
  proxyOpsFree(proxyState);
  return NULL;
}


// Process a request on the UDS socket
static ncclResult_t proxyUDSRecvReq(struct ncclProxyState* proxyState, int reqFd) {
  ncclIpcHdr hdr;
  NCCLCHECK(ncclIpcSocketRecvMsg(&proxyState->ipcSock, &hdr, sizeof(hdr), NULL));
  if (hdr.type == ncclProxyMsgGetFd) {
    // cuMem API support
    uint64_t handle = *(uint64_t*)hdr.data;
    INFO(NCCL_PROXY, "proxyUDSRecvReq::ncclProxyMsgGetFd rank %d opId %p handle=0x%lx", hdr.rank, hdr.opId, handle);
    return proxyGetFd(proxyState, hdr.rank, hdr.opId, handle);
  }

  return ncclInternalError;
}

// UDS fd handle support
void* ncclProxyServiceUDS(void* _args) {
  struct ncclProxyState* proxyState =  (struct ncclProxyState*) _args;
  struct pollfd pollfds[1];

  if (setProxyThreadContext(proxyState)) {
    INFO(NCCL_INIT, "[Proxy Service UDS] Created CUDA context on device %d", proxyState->cudaDev);
  } else if (cudaSetDevice(proxyState->cudaDev) != cudaSuccess) {
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
    if (proxyState->stop || *proxyState->abortFlag) break;

    if (pollfds[0].revents) {
      // A request was seen on the UDS fd
      proxyUDSRecvReq(proxyState, pollfds[0].fd);
    }
  }

  ncclIpcSocketClose(&proxyState->ipcSock);
  INFO(NCCL_PROXY, "[Proxy Service UDS] exit: stop %d abortFlag %d", proxyState->stop, *proxyState->abortFlag);
  return NULL;
}

ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses, uint64_t *peerAddressesUDS) {
  assert(comm->sharedRes->proxyState == NULL);
  NCCLCHECK(ncclCalloc(&comm->sharedRes->proxyState, 1));
  comm->proxyState = comm->sharedRes->proxyState;
  comm->proxyState->refCount = 1;
  comm->proxyState->listenSock = sock;
  comm->proxyState->peerAddresses = peerAddresses;
  comm->proxyState->peerAddressesUDS = peerAddressesUDS;

  // UDS support
  NCCLCHECK(ncclIpcSocketInit(&comm->proxyState->ipcSock, comm->rank, peerAddressesUDS[comm->rank], comm->abortFlag));
  // Seed the random number generator for UDS filename generation
  struct timeval time;
  gettimeofday(&time,NULL);
  unsigned int seed = time.tv_sec*time.tv_usec;
  seed ^= getpid();
  srandom(seed);
  return ncclSuccess;
}

ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  /* proxyState is shared among parent comm and split comms. comm->proxyState->thread is
   * pthread_join()'d by commFree() in init.cc when the refCount reduces down to 0. */
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
    memcpy(proxyState->buffSizes, comm->buffSizes, sizeof(comm->buffSizes));

    pthread_create(&comm->proxyState->thread, NULL, ncclProxyService, comm->proxyState);
    ncclSetThreadName(comm->proxyState->thread, "NCCL Service %2d", comm->cudaDev);

    // UDS support
    INFO(NCCL_PROXY, "UDS: Creating service thread comm %p rank %d", comm, comm->rank);
    pthread_create(&comm->proxyState->threadUDS, NULL, ncclProxyServiceUDS, comm->proxyState);
    ncclSetThreadName(comm->proxyState->threadUDS, "NCCL UDS Service %2d", comm->cudaDev);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyStop(struct ncclComm* comm) {
  if (comm->proxyState) {
    struct ncclProxyState* sharedProxyState = comm->proxyState;

    if ((comm->proxyRefCountOld = ncclAtomicRefCountDecrement(&sharedProxyState->refCount)) == 0) {
      if (comm->proxyState->threadUDS) {
        // UDS support
        comm->proxyState->stop = 1;
      }

      if (sharedProxyState->peerAddresses) {
        if (__atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED) == 0) {
          struct ncclSocket sock;
          int type = ncclProxyMsgStop;
          NCCLCHECK(ncclSocketInit(&sock, sharedProxyState->peerAddresses + comm->topParentRanks[comm->rank], comm->sharedRes->magic, ncclSocketTypeProxy, comm->abortFlag));
          NCCLCHECK(ncclSocketConnect(&sock));
          NCCLCHECK(ncclSocketSend(&sock, &type, sizeof(int)));
          NCCLCHECK(ncclSocketClose(&sock));
        }
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
            if (__atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED) == 0) NCCLCHECK(ncclSocketSend(sharedProxyState->peerSocks + i, &type, sizeof(int)));
            NCCLCHECK(ncclSocketClose(sharedProxyState->peerSocks + i));
          }
        }
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  struct ncclProxyState* sharedProxyState = comm->sharedRes->proxyState;

  assert(sharedProxyState->refCount == 0);
  free(sharedProxyState->peerAddresses);
  free(sharedProxyState->peerAddressesUDS);
  free(sharedProxyState->peerSocks);
  free(sharedProxyState->proxyOps);
  free(sharedProxyState->sharedDevMems);
  expectedProxyResponseFree(sharedProxyState);
  free(sharedProxyState);
  return ncclSuccess;
}
