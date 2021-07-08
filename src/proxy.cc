/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"

enum { proxyRecv=0, proxySend=1 };

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
      /* reduce */ (type == proxyRecv ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

#define PROXYARGS_ALLOCATE_SIZE 128
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

static ncclResult_t allocateArgs(struct ncclComm* comm, struct ncclProxyArgs** argsptr) {
  struct ncclProxyState* state = &comm->proxyState;
  struct ncclProxyArgs* elem;
  if (state->pool == NULL) {
    // Check whether there are freed elements
    if (state->poolReturned) {
      pthread_mutex_lock(&state->poolMutex);
      state->pool = state->poolReturned;
      state->poolReturned = NULL;
      pthread_mutex_unlock(&state->poolMutex);
    } else {
      // Allocate a new pool of elements. Make sure we allocate the memory close
      // to the network thread
      struct ncclProxyPool* newPool;
      cpu_set_t affinitySave;
      if (CPU_COUNT(&comm->cpuAffinity)) {
        sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
        sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
      }
      NCCLCHECK(ncclCalloc(&newPool, 1));
      if (CPU_COUNT(&comm->cpuAffinity)) {
        sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
      }

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
ncclResult_t dumpProxyState(struct ncclProxyState* state) {
#ifdef DEBUG_PROXY
  struct ncclProxyArgs* op = state->ops;
  while (op) {
    if (op->idle & OP_SEEN) {
      WARN("Active list loop at element %ld", OP_INDEX(op));
    }
    op->idle |= OP_SEEN;
    printf("[%ld(%ld/%d)]", OP_INDEX(op), op->opCount, op->nsubs);
    if (op->nextPeer) {
      printf("(%ld)", OP_INDEX(op->nextPeer));
      struct ncclProxyArgs* n = op->nextPeer;
      n->idle |= OP_SEEN;
      while (n->nextPeer) {
        n = n->nextPeer;
        n->idle |= OP_SEEN;
      }
    }
    printf("->");
    op = op->next;
  }
  printf("[X]\n");

  struct ncclProxyArgs* free = state->pool;
  while (free) {
    if (free->idle & OP_SEEN) {
      WARN("Free list loop at element %ld", OP_INDEX(free));
    }
    free->idle |= OP_SEEN;
    free = free->next;
  }

  struct ncclProxyPool* p = state->pools;
  int i = 0;
  while (p) {
    for (int e=0; e<PROXYARGS_ALLOCATE_SIZE; e++) {
      if ((p->elems[e].idle & OP_SEEN) == 0) {
        WARN("Element %d of pool %d has been lost", e, i);
        struct ncclProxyArgs* free = state->pool;
        printf("Free list ");
        while (free) {
          printf("--> %ld ", OP_INDEX(free));
          free = free->next;
        }
        printf("\n");
        return ncclInternalError;
      }
      p->elems[e].idle -= OP_SEEN;
    }
    p = p->next;
    i++;
  }
#endif
  return ncclSuccess;
}

static ncclResult_t ProxyAppend(struct ncclProxyState* state, struct ncclProxyArgs* args) {
  struct ncclProxyArgs* proxyAppend = *args->proxyAppendPtr;
  int shared = args->subs[0].connector->conn.shared;
  if (proxyAppend) {
    if (shared && proxyAppend->opCount == args->opCount) {
      if ((proxyAppend->sliceSteps != args->sliceSteps) ||
          (proxyAppend->chunkSteps != args->chunkSteps) ||
          (proxyAppend->protocol != args->protocol) ||
          (proxyAppend->dtype != args->dtype) ||
          (proxyAppend->redOp != args->redOp)) {
        WARN("Proxy append mismatch");
        return ncclInternalError;
      }
      if (proxyAppend->nsubs >= NCCL_PROXY_MAX_SUBS) {
        WARN("Proxy append out of bound");
        return ncclInternalError;
      }
      memcpy(proxyAppend->subs+proxyAppend->nsubs, args->subs, sizeof(struct ncclProxySubArgs));
      proxyAppend->nsubs++;
      args->next = proxyAppend->next;
      // Free args as we merged them
      args->next = state->poolFreed;
      state->poolFreed = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as group with %5ld\n", OP_INDEX(args), shared, proxyAppend->opCount, args->opCount, OP_INDEX(proxyAppend));
    } else {
      proxyAppend->nextPeer = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld\n", OP_INDEX(args), shared, proxyAppend->opCount, args->opCount, OP_INDEX(proxyAppend));
      *(args->proxyAppendPtr) = args;
    }
  } else {
    // Nothing running for that peer. Add to the list
    if (state->ops == NULL) {
      // Create the list
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element\n", OP_INDEX(args), shared, args->opCount);
      state->ops = args;
    } else {
      // Append element at the end of the list
      struct ncclProxyArgs* last = state->ops;
      while (last->next) last = last->next;
      last->next = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element\n", OP_INDEX(args),shared, args->opCount);
    }
    *(args->proxyAppendPtr) = args;
  }
  return ncclSuccess;
}

static ncclResult_t SaveProxy(int type, int peer, struct ncclProxyArgs* args, int connIndex) {
  if (peer < 0) return ncclSuccess;

  struct ncclChannel* channel = args->subs[0].channel;
  struct ncclPeer* peerComm = channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? peerComm->recv+connIndex : peerComm->send+connIndex;
  if (connector->transportComm == NULL) {
    WARN("Rank %d has no transport for %s peer %d on channel %d", connector->comm->rank,
        type == proxyRecv ? "recv" : "send", peer, channel->id);
    return ncclInternalError;
  }
  if (connector->transportComm->proxy == NULL) return ncclSuccess;

  struct ncclProxyState* state = &connector->comm->proxyState;
  struct ncclProxyArgs* op;
  NCCLCHECK(allocateArgs(connector->comm, &op));
  memcpy(op, args, sizeof(struct ncclProxyArgs));
  op->subs[0].connector = connector;
  op->progress = connector->transportComm->proxy;
  op->state = ncclProxyOpReady;
  op->proxyAppendPtr = connector->proxyAppendPtr;

  if (state->nextOps == NULL) state->nextOps = op;
  else state->nextOpsEnd->next = op;
  state->nextOpsEnd = op;
  return ncclSuccess;
}

ncclResult_t ncclProxySaveColl(struct ncclProxyArgs* args, int nranks) {
  struct ncclChannel* channel = args->subs[0].channel;
  int pattern = args->pattern;
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &channel->ring;
    if (NeedProxy(proxyRecv, pattern, args->root, ring, nranks)) NCCLCHECK(SaveProxy(proxyRecv, ring->prev, args, 0));
    if (NeedProxy(proxySend, pattern, args->root, ring, nranks)) NCCLCHECK(SaveProxy(proxySend, ring->next, args, 0));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxyRecv, tree->down[i], args, 0));
    NCCLCHECK(SaveProxy(proxySend, tree->up, args, 0));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxySend, tree->down[i], args, 0));
    NCCLCHECK(SaveProxy(proxyRecv, tree->up, args, 0));
  }
  if (pattern == ncclPatternCollTreeUpDown) {
    // CollTree up
    NCCLCHECK(SaveProxy(proxySend, channel->collTree.out, args, 1));  // For CollTree up, we are using push
    // CollTree down
    NCCLCHECK(SaveProxy(proxyRecv, channel->collTree.out, args, 0));
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyComputeP2p(struct ncclInfo* info, struct ncclProxyArgs* args) {
  memset(args, 0, sizeof(struct ncclProxyArgs));
  int channelId = info->channelId;
  args->nsubs = 1;
  struct ncclProxySubArgs* sub = args->subs;

  struct ncclChannel* channel = info->comm->channels+channelId;
  sub->channel = channel;
  args->sliceSteps = 1;
  args->chunkSteps = 1;
  args->protocol = NCCL_PROTO_SIMPLE;
  args->dtype = info->datatype;
  sub->delta = info->delta;
  sub->recvbytes = info->recvbytes;
  sub->sendbytes = info->sendbytes;

  int stepSize = info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR;
  info->recvChunkSize = stepSize;
  info->sendChunkSize = stepSize;

  if (info->delta > 0 && info->recvbytes >= 0) {
    int peerrecv = (info->comm->nRanks+info->comm->rank-info->delta)%info->comm->nRanks;
    if (channel->peers[peerrecv].recv[0].transportComm && channel->peers[peerrecv].recv[0].transportComm->proxy) {
      // Tune chunk size for the network
      if (info->recvbytes < stepSize) info->recvChunkSize /= 4;
      else if (info->recvbytes < 8*stepSize) info->recvChunkSize /= 2;
    }
    sub->recvChunkSize = info->recvChunkSize;
  }
  if (info->delta > 0 && info->sendbytes >= 0) {
    int peersend = (info->comm->rank+info->delta)%info->comm->nRanks;
    if (channel->peers[peersend].send[0].transportComm && channel->peers[peersend].send[0].transportComm->proxy) {
      // Tune chunk size for the network
      if (info->sendbytes < stepSize) info->sendChunkSize /= 4;
      else if (info->sendbytes < 8*stepSize) info->sendChunkSize /= 2;
    }
    sub->sendChunkSize = info->sendChunkSize;
  }
  return ncclSuccess;
}

ncclResult_t ncclProxySaveP2p(struct ncclComm* comm, struct ncclProxyArgs* args) {
  struct ncclProxySubArgs* sub = args->subs;
  struct ncclChannel* channel = sub->channel;
  args->opCount = channel->workFifoTail-1;
  args->commOpCount = comm->opCount;
  const ssize_t recvbytesOrig = sub->recvbytes;
  const ssize_t sendbytesOrig = sub->sendbytes;
  if (sub->delta > 0 && recvbytesOrig >= ssize_t(0)) {
    int peerrecv = (comm->nRanks+comm->rank-sub->delta)%comm->nRanks;
    sub->recvbytes = recvbytesOrig;
    sub->sendbytes = 0;
    sub->nsteps = DIVUP(sub->recvbytes, sub->recvChunkSize);
    if (sub->nsteps == 0) sub->nsteps = 1;
    NCCLCHECK(SaveProxy(proxyRecv, peerrecv, args, 0));
  }
  if (sub->delta > 0 && sendbytesOrig >= ssize_t(0)) {
    int peersend = (comm->rank+sub->delta)%comm->nRanks;
    sub->sendbytes = sendbytesOrig;
    sub->recvbytes = 0;
    sub->nsteps = DIVUP(sub->sendbytes, sub->sendChunkSize);
    if (sub->nsteps == 0) sub->nsteps = 1;
    NCCLCHECK(SaveProxy(proxySend, peersend, args, 0));
  }
  // Reset proxy args for potentially multiple cuda graph launches
  // It is safe as long as SaveProxy copies contents of args to op
  sub->recvbytes = recvbytesOrig;
  sub->sendbytes = sendbytesOrig;
  return ncclSuccess;
}

static ncclResult_t removeOp(struct ncclProxyState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr) {
  struct ncclProxyArgs* freeOp = *opPtr;
  DEBUG_PROXY_PRINT("Remove %ld -> %ld -> %ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(freeOp), OP_INDEX(freeOp->next));
  struct ncclProxyArgs* next = freeOp->next;
  *opPtr = next;
  if (freeOp->nextPeer) {
    // replace op by nextPeer
    struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = nextPeer;
    } else {
      state->ops = nextPeer;
    }
    nextPeer->next = next;
    *(prevOpPtr) = nextPeer;
  } else {
    *(freeOp->proxyAppendPtr) = NULL;
    if (*prevOpPtr) {
      (*prevOpPtr)->next = next;
    } else {
      state->ops = next;
    }
  }
  freeOp->next = state->poolFreed;
  state->poolFreed = freeOp;
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld)                                               : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
  NCCLCHECK(dumpProxyState(state));
  return ncclSuccess;
}

static ncclResult_t progressOps(struct ncclProxyState* state, struct ncclProxyArgs** opsPtr, int* idle, struct ncclComm* comm) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* op = *opsPtr;
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    NCCLCHECK(op->progress(op));
    *idle &= op->idle;
    if (op->state == ncclProxyOpNone) {
      NCCLCHECK(removeOp(state, &op, &prevOp));
    } else {
      prevOp = op;
      op = op->next;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyAppendPosted(struct ncclProxyState* state) {
  // Return any freed element first
  if (state->poolFreed) {
    struct ncclProxyArgs* end = state->poolFreed;
    while (end->next) end = end->next;
    pthread_mutex_lock(&state->poolMutex);
    end->next = state->poolReturned;
    state->poolReturned = state->poolFreed;
    pthread_mutex_unlock(&state->poolMutex);
    state->poolFreed = NULL;
  }

  // Then wait until we have new work to do
  pthread_mutex_lock(&state->opsMutex);
  while (state->postedOps == NULL) {
    if (state->stop) return ncclSuccess;
    pthread_cond_wait(&state->cond, &state->opsMutex);
  }

  // Sort operations as we append them : collectives and
  // receives first, then sends.

  struct ncclProxyArgs* next, *prev = NULL, *op = state->postedOps;
  int commOpCount = op->commOpCount;
  while (op && op->commOpCount == commOpCount) {
    next = op->next;
    if (op->subs[0].sendbytes) {
      if (prev) prev->next = next;
      else state->postedOps = next;
      op->next = NULL;
      NCCLCHECK(ProxyAppend(state, op));
    } else prev = op;
    op = next;
  }
  op = state->postedOps;
  while (op && op->commOpCount == commOpCount) {
    next = op->next;
    op->next = NULL;
    NCCLCHECK(ProxyAppend(state, op));
    op = next;
  }
  state->postedOps = op;
  if (op == NULL) state->postedOpsEnd = NULL;
  NCCLCHECK(dumpProxyState(state));
  pthread_mutex_unlock(&state->opsMutex);

  if (state->poolFreed) {
    struct ncclProxyArgs* end = state->poolFreed;
    while (end->next) end = end->next;
    pthread_mutex_lock(&state->poolMutex);
    end->next = state->poolReturned;
    state->poolReturned = state->poolFreed;
    pthread_mutex_unlock(&state->poolMutex);
    state->poolFreed = NULL;
  }

  return ncclSuccess;
}


void* persistentThread(void *comm_) {
  struct ncclComm* comm = (struct ncclComm*)comm_;
  struct ncclProxyState* state = &comm->proxyState;
  char threadName[16];
  sprintf(threadName, "NCCLproxy %5d", comm->rank);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

  struct ncclProxyArgs** opsPtr = &state->ops;
  while (1) {
    if (*comm->abortFlag) {
      return NULL;
    }

    while (*opsPtr == NULL) {
      if (state->stop) {
        // No more commands to process and proxy has been requested to stop
        return NULL;
      }
      ncclResult_t ret = ncclProxyAppendPosted(state);
      if (ret != ncclSuccess) {
        comm->fatalError = ret;
        INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
        return NULL;
      }
    }
    int idle = 1;
    ncclResult_t ret = progressOps(state, opsPtr, &idle, comm);
    if (ret != ncclSuccess) {
      comm->fatalError = ret;
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
      return NULL;
    }
    if (idle) {
      sched_yield(); // No request progressed. Let others run.
    }
  }
}

ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;
  if (state->nextOps == NULL) return ncclSuccess;
  pthread_mutex_lock(&state->opsMutex);
  if (state->postedOps) state->postedOpsEnd->next = state->nextOps;
  else state->postedOps = state->nextOps;
  state->postedOpsEnd = state->nextOpsEnd;
  state->nextOps = state->nextOpsEnd = NULL;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->opsMutex);
  comm->opCount++;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersInit(struct ncclComm* comm, int cuda, int* size, char** ptr) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  if (state->size == 0) {
    int p2pnChannels = 1;
    while (p2pnChannels < comm->nChannels) p2pnChannels *= 2;
    int p2pSize = 2*p2pnChannels*NCCL_MAX_WORK_ELEMENTS*comm->buffSizes[NCCL_PROTO_SIMPLE]/SENDRECV_SLICEFACTOR;
    int collNetSize = 2*comm->nChannels*comm->buffSizes[NCCL_PROTO_SIMPLE];
    state->size = std::max(p2pSize, collNetSize);
  }

  *size = state->size;

  if (cuda && state->cudaBuff == NULL) {
    NCCLCHECK(ncclCudaCalloc(&state->cudaBuff, *size));
  } else if (state->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&state->hostBuff, *size));
  }
  *ptr = cuda ? state->cudaBuff : state->hostBuff;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersGetP2p(struct ncclComm* comm, int cuda, int type, int channel, int slot, int index, char** ptr) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  // Use different pools for separate send/recv.
  char* buff = cuda ? state->cudaBuff : state->hostBuff;
  int slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(NCCL_STEPS*SENDRECV_SLICEFACTOR);
  int globalSlot = (((type*comm->p2pnChannels+channel)*NCCL_STEPS)+slot)*NCCL_MAX_WORK_ELEMENTS+index;
  *ptr = buff + slotSize * globalSlot;
  return ncclSuccess;
}
ncclResult_t ncclProxySharedBuffersGetCollNet(struct ncclComm* comm, int cuda, int type, int slot, int channel, char** ptr) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  // Use different pools for different channels.
  char* buff = cuda ? state->cudaBuff : state->hostBuff;
  int slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  int globalSlot = (type*NCCL_STEPS+slot)*comm->nChannels+channel;
  *ptr = buff + slotSize * globalSlot;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersDestroy(struct ncclComm* comm) {
  struct ncclProxySharedBuffers* state = &comm->proxyState.sharedBuffs;
  CUDACHECK(cudaFree(state->cudaBuff));
  NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  return ncclSuccess;
}

ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  if (!comm->proxyThread) {
    comm->proxyState.cond = PTHREAD_COND_INITIALIZER;
    comm->proxyState.opsMutex = PTHREAD_MUTEX_INITIALIZER;
    comm->proxyState.poolMutex = PTHREAD_MUTEX_INITIALIZER;
    comm->proxyState.ops = NULL;
    pthread_create(&comm->proxyThread, NULL, persistentThread, comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;

  // Request the proxy to stop and then wake it
  pthread_mutex_lock(&state->opsMutex);
  state->stop = true;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->opsMutex);
  if (comm->proxyThread) pthread_join(comm->proxyThread, NULL);

  // Free off any memory allocated for the proxy arg pools
  pthread_mutex_lock(&state->poolMutex);
  struct ncclProxyState* proxyState = &comm->proxyState;
  while (proxyState->pools != NULL) {
    struct ncclProxyPool *next = proxyState->pools->next;
    free(proxyState->pools);
    proxyState->pools = next;
  }
  pthread_mutex_unlock(&state->poolMutex);

  NCCLCHECK(ncclProxySharedBuffersDestroy(comm));

  return ncclSuccess;
}
