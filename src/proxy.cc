/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "graph.h"
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
  pthread_mutex_lock(&state->poolMutex);
  if (state->pool == NULL) {
    // Allocate a new pool of elements
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
  pthread_mutex_unlock(&state->poolMutex);
  elem->next = elem->nextPeer = elem->nextGroup = NULL;
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
    printf("[%ld]", OP_INDEX(op));
    if (op->nextPeer) {
      printf("(%ld)", OP_INDEX(op->nextPeer));
      struct ncclProxyArgs* n = op->nextPeer;
      n->idle |= OP_SEEN;
      while (n->nextGroup || n->nextPeer) {
        n = n->nextGroup ? n->nextGroup : n->nextPeer;
        n->idle |= OP_SEEN;
      }
    }
    if (op->nextGroup)  {
      printf("--G->");
      op = op->nextGroup;
    } else {
      printf("--N->");
      op = op->next;
    }
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

static ncclResult_t ProxyAppend(struct ncclProxyState* state, struct ncclProxyArgs* args, int shared) {
  struct ncclProxyArgs* proxyAppend = *args->proxyAppendPtr;
  if (proxyAppend) {
    if (shared && proxyAppend->opCount == args->opCount) {
      args->next = proxyAppend->next;
      proxyAppend->next = NULL;
      proxyAppend->nextGroup = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as group, prevGroup %5ld, next %5ld : \n", OP_INDEX(args), shared, proxyAppend->opCount, args->opCount, OP_INDEX(proxyAppend), OP_INDEX(args->next));
    } else {
      proxyAppend->nextPeer = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld/%5ld) as nextPeer of %5ld                  : \n", OP_INDEX(args), shared, proxyAppend->opCount, args->opCount, OP_INDEX(proxyAppend));
    }
  } else {
    // Nothing running for that peer. Add to the list
    if (state->ops == NULL) {
      // Create the list
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as first element                            : \n", OP_INDEX(args), shared, args->opCount);
      state->ops = args;
    } else {
      // Append element at the end of the list
      struct ncclProxyArgs* last = state->ops;
      while (last->nextGroup || last->next) last = last->nextGroup ? last->nextGroup : last->next;
      last->next = args;
      DEBUG_PROXY_PRINT("Insert  %5ld (%d/%5ld) as last element                             : \n", OP_INDEX(args),shared, args->opCount);
    }
  }
  *(args->proxyAppendPtr) = args;
  return ncclSuccess;
}

static ncclResult_t SaveProxy(int type, int peer, struct ncclProxyArgs* args) {
  if (peer < 0) return ncclSuccess;

  struct ncclPeer* peerComm = args->channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? &peerComm->recv : &peerComm->send;
  if (connector->transportComm == NULL) {
    WARN("[%d] Error no transport for %s peer %d on channel %d", connector->comm->rank,
        type == proxyRecv ? "recv" : "send", peer, args->channel->id);
    return ncclInternalError;
  }
  if (connector->transportComm->proxy == NULL) return ncclSuccess;

  struct ncclProxyState* state = &connector->comm->proxyState;
  struct ncclProxyArgs* op;
  NCCLCHECK(allocateArgs(connector->comm, &op));
  memcpy(op, args, sizeof(struct ncclProxyArgs));
  op->connector = connector;
  op->progress = connector->transportComm->proxy;
  op->state = ncclProxyOpReady;

  op->proxyAppendPtr =
    connector->conn.shared ?
    state->sharedBuffs->proxyAppend+2*args->channel->id+type : // Shared buffers
    &connector->proxyAppend;  // Dedicated buffers

  if (state->nextOps == NULL) state->nextOps = op;
  else state->nextOpsEnd->next = op;
  state->nextOpsEnd = op;
  return ncclSuccess;
}

ncclResult_t ncclProxySaveColl(struct ncclProxyArgs* args, int pattern, int root, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &args->channel->ring;
    if (NeedProxy(proxyRecv, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy(proxyRecv, ring->prev, args));
    if (NeedProxy(proxySend, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy(proxySend, ring->next, args));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &args->channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxyRecv, tree->down[i], args));
    NCCLCHECK(SaveProxy(proxySend, tree->up, args));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &args->channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxySend, tree->down[i], args));
    NCCLCHECK(SaveProxy(proxyRecv, tree->up, args));
  }
  if (pattern == ncclPatternCollTreeUp) {
    // CollTree up
    struct ncclTree* tree = &args->channel->collTree;
    NCCLCHECK(SaveProxy(proxyRecv, tree->down[0], args));
    NCCLCHECK(SaveProxy(proxySend, tree->up, args));
  }
  if (pattern == ncclPatternCollTreeDown) {
    // CollTree down
    struct ncclTree* tree = &args->channel->collTree;
    NCCLCHECK(SaveProxy(proxySend, tree->down[0], args));
    NCCLCHECK(SaveProxy(proxyRecv, tree->up, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclProxySaveP2p(struct ncclInfo* info, struct ncclChannel* channel, int segment) {
  struct ncclProxyArgs args;
  memset(&args, 0, sizeof(struct ncclProxyArgs));
  args.channel = channel;
  args.sliceSteps = 1;
  args.chunkSteps = 1;
  args.protocol = NCCL_PROTO_SIMPLE;
  args.segment = segment;
  args.opCount = channel->workFifoTail-1;
  args.dtype = info->datatype;
  if (info->delta > 0 && info->recvbytes >= 0) {
    int peerrecv = (info->comm->nRanks+info->comm->rank-info->delta)%info->comm->nRanks;
    args.nsteps = DIVUP(info->recvbytes, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR);
    if (args.nsteps == 0) args.nsteps = 1;
    args.recvbytes = info->recvbytes;
    args.sendbytes = 0;
    NCCLCHECK(SaveProxy(proxyRecv, peerrecv, &args));
  }
  if (info->delta > 0 && info->sendbytes >= 0) {
    int peersend = (info->comm->rank+info->delta)%info->comm->nRanks;
    args.nsteps = DIVUP(info->sendbytes, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR);
    if (args.nsteps == 0) args.nsteps = 1;
    args.sendbytes = info->sendbytes;
    args.recvbytes = 0;
    NCCLCHECK(SaveProxy(proxySend, peersend, &args));
  }
  return ncclSuccess;
}

static ncclResult_t removeOp(struct ncclProxyState* state, struct ncclProxyArgs** opPtr, struct ncclProxyArgs** prevOpPtr, struct ncclProxyArgs** prevGroupPtr) {
  struct ncclProxyArgs* freeOp = *opPtr;
  DEBUG_PROXY_PRINT("Remove %ld/%ld -> %ld -> %ld/%ld\n", OP_INDEX(*prevOpPtr), OP_INDEX(*prevGroupPtr), OP_INDEX(freeOp), OP_INDEX(freeOp->next), OP_INDEX(freeOp->nextGroup));
  if (*prevGroupPtr && *prevOpPtr) return ncclInternalError;
  if (freeOp->nextGroup) {
    // Part of a group : remove the element
    struct ncclProxyArgs* next = freeOp->nextGroup;
    *opPtr = next;
    if (*prevGroupPtr) {
      (*prevGroupPtr)->nextGroup = next;
    } else if (*prevOpPtr) {
      (*prevOpPtr)->next = next;
    } else {
      state->ops = next;
    }
  } else {
    struct ncclProxyArgs* next = freeOp->next;
    *opPtr = next;
    if ((*prevGroupPtr)) {
      (*prevGroupPtr)->next = next;
      (*prevGroupPtr)->nextGroup = NULL;
      (*prevGroupPtr)->nextPeer = freeOp->nextPeer;
      if (*(freeOp->proxyAppendPtr) == freeOp) *(freeOp->proxyAppendPtr) = *prevGroupPtr;
      (*prevOpPtr) = *prevGroupPtr;
      (*prevGroupPtr) = NULL;
    } else {
      if (freeOp->nextPeer) {
        // replace op by nextPeer
        struct ncclProxyArgs* nextPeer = freeOp->nextPeer;
        if (*prevOpPtr) {
          (*prevOpPtr)->next = nextPeer;
        } else {
          state->ops = nextPeer;
        }
        struct ncclProxyArgs* lastGroup = nextPeer;
        while (lastGroup->nextGroup) lastGroup = lastGroup->nextGroup;
        lastGroup->next = next;
        *(prevOpPtr) = lastGroup;
      } else {
        *(freeOp->proxyAppendPtr) = NULL;
        if (*prevOpPtr) {
          (*prevOpPtr)->next = next;
        } else {
          state->ops = next;
        }
      }
    }
  }
  pthread_mutex_lock(&state->poolMutex);
  freeOp->next = state->pool;
  state->pool = freeOp;
  pthread_mutex_unlock(&state->poolMutex);
  DEBUG_PROXY_PRINT("Removed %5ld (%5ld)                                               : ", OP_INDEX(freeOp), OP_INDEX(*freeOp->proxyAppendPtr));
  NCCLCHECK(dumpProxyState(state));
  return ncclSuccess;
}

static ncclResult_t progressOps(struct ncclProxyState* state, struct ncclProxyArgs** opsPtr, int* idle, struct ncclComm* comm) {
  struct ncclProxyArgs* prevOp = NULL;
  struct ncclProxyArgs* prevGroup = NULL;
  struct ncclProxyArgs* op = *opsPtr;
  while (op) {
    if (op->state == ncclProxyOpNone) return ncclInternalError;
    // opCount >= lastOpCount are part of an ongoing GroupStart/GroupEnd that hasn't started
    // yet and might be cancelled before they even start. Hold on on those.
    if (op->opCount < comm->lastOpCount) {
      NCCLCHECK(op->progress(op));
      *idle &= op->idle;
    }
    if (op->state == ncclProxyOpNone) {
      NCCLCHECK(removeOp(state, &op, &prevOp, &prevGroup));
    } else {
      if (op->nextGroup) {
        prevGroup = op;
        prevOp = NULL;
        op = op->nextGroup;
      } else {
        prevOp = op;
        prevGroup = NULL;
        op = op->next;
      }
    }
  }
  return ncclSuccess;
}

void* persistentThread(void *comm_) {
  struct ncclComm* comm = (struct ncclComm*)comm_;
  struct ncclProxyState* state = &comm->proxyState;
  char threadName[16];
  sprintf(threadName, "NCCLproxy %5d", comm->rank);
  nvtxNameOsThreadA(syscall(SYS_gettid), threadName);

  pthread_mutex_lock(&state->opsMutex);
  struct ncclProxyArgs** opsPtr = &state->ops;
  while (1) {
    if (*comm->abortFlag) {
      pthread_mutex_unlock(&state->opsMutex);
      return NULL;
    }

    while (*opsPtr == NULL) {
      if (state->stop) {
        // No more commands to process and proxy has been requested to stop
        pthread_mutex_unlock(&state->opsMutex);
        return NULL;
      }
      pthread_cond_wait(&state->cond, &state->opsMutex);
    }
    int idle = 1;
    ncclResult_t ret = progressOps(state, opsPtr, &idle, comm);
    if (ret != ncclSuccess) {
      comm->fatalError = ret;
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
      pthread_mutex_unlock(&state->opsMutex);
      return NULL;
    }
    if (idle) {
      pthread_mutex_unlock(&state->opsMutex);
      sched_yield(); // No request progressed. Let others run.
      pthread_mutex_lock(&state->opsMutex);
    }
  }
}

ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;
  pthread_mutex_lock(&state->opsMutex);

  // Sort operations as we append them : collectives and
  // receives first, then sends.
  ncclProxyArgs* next, *prev = NULL, *op = state->nextOps;
  while (op) {
    next = op->next;
    if (op->sendbytes) {
      if (prev) prev->next = next;
      else state->nextOps = next;
      op->next = NULL;
      NCCLCHECK(ProxyAppend(state, op, op->connector->conn.shared));
    } else prev = op;
    op = next;
  }
  op = state->nextOps;
  while (op) {
    next = op->next;
    op->next = NULL;
    NCCLCHECK(ProxyAppend(state, op, op->connector->conn.shared));
    op = next;
  }
  state->nextOps = state->nextOpsEnd = NULL;
  NCCLCHECK(dumpProxyState(state));

  if (state->ops != NULL)
    pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->opsMutex);
  return ncclSuccess;
}

NCCL_PARAM(ProxySharedBuffersCount, "SHARED_BUFF_COUNT", -2);

ncclResult_t ncclProxySharedBuffersInit(struct ncclComm* comm, int cuda, int* size, char** ptr) {
  struct ncclProxySharedBuffers* state = comm->proxyState.sharedBuffs;
  if (state == NULL) {
    NCCLCHECK(ncclCalloc(&state, 1));
    comm->proxyState.sharedBuffs = state;
    state->nslots = ncclParamProxySharedBuffersCount();
    if (state->nslots == -2)  {
      state->nslots = NCCL_STEPS*NCCL_MAX_WORK_ELEMENTS;
    }
    state->slotSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(NCCL_STEPS*SENDRECV_SLICEFACTOR);
  }

  char* buff;
  int* used;
  *size = 2*comm->p2pnChannels*state->slotSize*state->nslots;

  if (cuda && state->cudaBuff[0] == NULL) {
    NCCLCHECK(ncclCudaCalloc(&buff, *size));
    NCCLCHECK(ncclCalloc(&used, 2*comm->p2pnChannels*state->nslots));
    for (int i=0; i<2*comm->p2pnChannels; i++) {
      state->cudaBuff[i] = buff + state->nslots*state->slotSize*i;
      state->cudaUsed[i] = used + state->nslots*i;
    }
  } else if (state->hostBuff[0] == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&buff, *size));
    NCCLCHECK(ncclCalloc(&used, 2*comm->p2pnChannels*state->nslots));
    for (int i=0; i<2*comm->p2pnChannels; i++) {
      state->hostBuff[i] = buff + state->nslots*state->slotSize*i;
      state->hostUsed[i] = used + state->nslots*i;
    }
  }
  buff = cuda ? state->cudaBuff[0] : state->hostBuff[0];

  *ptr = buff;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersAlloc(struct ncclComm* comm, int cuda, int type, int channel, int size, char** ptr) {
  struct ncclProxySharedBuffers* state = comm->proxyState.sharedBuffs;
  // Use different pools for different channels and also separate send/recv.
  int p = 2*channel+type;
  int* used = cuda ? state->cudaUsed[p] : state->hostUsed[p];
  char* buff = cuda ? state->cudaBuff[p] : state->hostBuff[p];
  if (buff == NULL) return ncclInternalError;
  int nslots = 1;
  while (nslots*state->slotSize < size) nslots *= 2;
  for (int s=0; s<state->nslots; s+=nslots) {
    int u = 0;
    for (int i=0; i<nslots; i++) u += used[s+i];
    if (u == 0) {
      for (int i=0; i<nslots; i++) used[s+i] = 1;
      *ptr = buff+state->slotSize*s;
      return ncclSuccess;
    }
  }
  *ptr = NULL;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersFree(struct ncclComm* comm, int cuda, int type, int channel, int size, char* ptr) {
  struct ncclProxySharedBuffers* state = comm->proxyState.sharedBuffs;
  int p = 2*channel+type;
  int* used = cuda ? state->cudaUsed[p] : state->hostUsed[p];
  char* buff = cuda ? state->cudaBuff[p] : state->hostBuff[p];
  if (buff == NULL) return ncclInternalError;
  int nslots = 1;
  while (nslots*state->slotSize < size) nslots *= 2;
  int s = (ptr-buff)/state->slotSize;
  if (s < 0 || s+nslots > state->nslots) {
    WARN("Error freeing shared buffer : freeing ptr %p size %d (start %p slot size %d nslots %d)", ptr, size, buff, state->slotSize, state->nslots);
    return ncclInternalError;
  }
  for (int i=0; i<nslots; i++) used[s+i] = 0;
  return ncclSuccess;
}

ncclResult_t ncclProxySharedBuffersDestroy(struct ncclComm* comm) {
  struct ncclProxySharedBuffers* state = comm->proxyState.sharedBuffs;
  if (state) {
    CUDACHECK(cudaFree(state->cudaBuff[0]));
    free(state->cudaUsed[0]);
    NCCLCHECK(ncclCudaHostFree(state->hostBuff[0]));
    free(state->hostUsed[0]);
    free(state);
  }
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
