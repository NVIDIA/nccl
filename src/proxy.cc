/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "collectives.h"

#define RECV 0
#define SEND 1

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == RECV ?   myrank : nextrank ):
      /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

enum { proxyRecv=0, proxySend=1 };

#define PROXYARGS_ALLOCATE_SIZE 32
struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};

static ncclResult_t allocateArgs(struct ncclComm* comm, struct ncclProxyArgs** argsptr) {
  struct ncclProxyState* state = &comm->proxyState;
  struct ncclProxyArgs* elem;
  pthread_mutex_lock(&state->mutex);
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
  pthread_mutex_unlock(&state->mutex);
  elem->next = elem->nextPeer = NULL;
  *argsptr = elem;
  return ncclSuccess;
}

static void ProxyAppend(struct ncclConnector* connector, struct ncclProxyArgs* args) {
  struct ncclComm* comm = connector->comm;
  struct ncclProxyState* state = &comm->proxyState;
  pthread_mutex_lock(&state->mutex);
  if (connector->proxyAppend == NULL) {
    // Nothing running for that peer. Add to the circular list
    if (state->ops == NULL) {
      // Create the list
      args->next = args;
      state->ops = args;
    } else {
      // Insert element in the list
      args->next = state->ops->next;
      state->ops->next = args;
    }
    connector->proxyAppend = args;
  } else {
    // There is an active operation already for that peer.
    // Add it to the per-peer list
    connector->proxyAppend->nextPeer = args;
    connector->proxyAppend = args;
  }
  pthread_mutex_unlock(&state->mutex);
}

template <int type>
static ncclResult_t SaveProxy(int peer, struct ncclProxyArgs* args) {
  if (peer < 0) return ncclSuccess;

  struct ncclPeer* peerComm = args->channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? &peerComm->recv : &peerComm->send;
  if (connector->transportComm == NULL) {
    WARN("[%d] Error no transport for %s peer %d on channel %d\n", connector->comm->rank,
        type == proxyRecv ? "recv" : "send", peer, args->channel->id);
    return ncclInternalError;
  }
  if (connector->transportComm->proxy == NULL) return ncclSuccess;

  struct ncclProxyArgs* op;
  NCCLCHECK(allocateArgs(connector->comm, &op));
  memcpy(op, args, sizeof(struct ncclProxyArgs));
  op->connector = connector;
  op->progress = connector->transportComm->proxy;
  op->state = ncclProxyOpReady;
  ProxyAppend(connector, op);
  return ncclSuccess;
}

ncclResult_t ncclProxySaveColl(struct ncclProxyArgs* args, int pattern, int root, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &args->channel->ring;
    if (NeedProxy(RECV, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxyRecv>(ring->prev, args));
    if (NeedProxy(SEND, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxySend>(ring->next, args));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &args->channel->treeUp;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy<proxyRecv>(tree->down[i], args));
    NCCLCHECK(SaveProxy<proxySend>(tree->up, args));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &args->channel->treeDn;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy<proxySend>(tree->down[i], args));
    NCCLCHECK(SaveProxy<proxyRecv>(tree->up, args));
  }
  if (pattern == ncclPatternCollTreeUp) {
    // CollTree up
    struct ncclTree* tree = &args->channel->collTreeUp;
    NCCLCHECK(SaveProxy<proxyRecv>(tree->down[0], args));
    NCCLCHECK(SaveProxy<proxySend>(tree->up, args));
  }
  if (pattern == ncclPatternCollTreeDown) {
    // CollTree down
    struct ncclTree* tree = &args->channel->collTreeDn;
    NCCLCHECK(SaveProxy<proxySend>(tree->down[0], args));
    NCCLCHECK(SaveProxy<proxyRecv>(tree->up, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclProxySaveP2p(struct ncclInfo* info, struct ncclChannel* channel) {
  struct ncclProxyArgs args;
  memset(&args, 0, sizeof(struct ncclProxyArgs));
  args.channel = channel;
  args.sliceSteps = 1;
  args.chunkSteps = 1;
  args.protocol = NCCL_PROTO_SIMPLE;
  args.opCount = info->comm->opCount;
  args.dtype = info->datatype;
  if (info->delta > 0 && info->sendbytes >= 0) {
    int peersend = (info->comm->rank+info->delta)%info->comm->nRanks;
    args.nsteps = DIVUP(info->sendbytes, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR);
    if (args.nsteps == 0) args.nsteps = 1;
    NCCLCHECK(SaveProxy<proxySend>(peersend, &args));
  }
  if (info->delta > 0 && info->recvbytes >= 0) {
    int peerrecv = (info->comm->nRanks+info->comm->rank-info->delta)%info->comm->nRanks;
    args.nsteps = DIVUP(info->recvbytes, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/SENDRECV_SLICEFACTOR);
    if (args.nsteps == 0) args.nsteps = 1;
    NCCLCHECK(SaveProxy<proxyRecv>(peerrecv, &args));
  }
  return ncclSuccess;
}

void* persistentThread(void *comm_) {
  struct ncclComm* comm = (struct ncclComm*)comm_;
  struct ncclProxyState* state = &comm->proxyState;
  struct ncclProxyArgs* op = NULL;
  ncclResult_t ret = ncclSuccess;
  int idle = 1;
  int idleSpin = 0;
  while (1) {
    do {
      if (*comm->abortFlag) return NULL;
      if (op == NULL) {
        pthread_mutex_lock(&state->mutex);
        op = state->ops;
        if (op == NULL) {
          if (state->stop) {
            // No more commands to process and proxy has been requested to stop
            pthread_mutex_unlock(&state->mutex);
            return NULL;
          }
          pthread_cond_wait(&state->cond, &state->mutex);
        }
        pthread_mutex_unlock(&state->mutex);
      }
    } while (op == NULL);
    op->idle = 0;
    // opCount >= lastOpCount are part of an ongoing GroupStart/GroupEnd that hasn't started
    // yet and might be cancelled before they even start. Hold on on those.
    if (op->state != ncclProxyOpNone && op->opCount < comm->lastOpCount) ret = op->progress(op);
    if (ret != ncclSuccess) {
      comm->fatalError = ret;
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
      return NULL;
    }
    idle &= op->idle;
    pthread_mutex_lock(&state->mutex);
    if (!idle) idleSpin = 0;
    struct ncclProxyArgs *next = op->next;
    if (next->state == ncclProxyOpNone) {
      struct ncclProxyArgs *freeOp = next;
      if (next->nextPeer) {
        // Replace next by its next per-peer element.
        next = next->nextPeer;
        if (op != freeOp) {
          next->next = freeOp->next;
          op->next = next;
        } else {
          next->next = next;
        }
      } else {
        // Remove next from circular list
        next->connector->proxyAppend = NULL;
        if (op != freeOp) {
          next = next->next;
          op->next = next;
        } else {
          next = NULL;
        }
      }
      if (freeOp == state->ops) state->ops = next;
      freeOp->next = state->pool;
      state->pool = freeOp;
    }
    op = next;
    if (op == state->ops) {
      if (idle == 1) {
        if (++idleSpin == 10) {
          sched_yield();
          idleSpin = 0;
        }
      }
      idle = 1;
    }
    pthread_mutex_unlock(&state->mutex);
  }
}

ncclResult_t ncclProxyStart(struct ncclComm* comm) {
  pthread_mutex_lock(&comm->proxyState.mutex);
  if (comm->proxyState.ops != NULL)
    pthread_cond_signal(&comm->proxyState.cond);
  pthread_mutex_unlock(&comm->proxyState.mutex);
  return ncclSuccess;
}

ncclResult_t ncclProxyCreate(struct ncclComm* comm) {
  if (!comm->proxyThread) {
    comm->proxyState.cond = PTHREAD_COND_INITIALIZER;
    comm->proxyState.mutex = PTHREAD_MUTEX_INITIALIZER;
    comm->proxyState.ops = NULL;
    pthread_create(&comm->proxyThread, NULL, persistentThread, comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclProxyDestroy(struct ncclComm* comm) {
  struct ncclProxyState* state = &comm->proxyState;

  // Request the proxy to stop and then wake it
  pthread_mutex_lock(&state->mutex);
  state->stop = true;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->mutex);
  if (comm->proxyThread) pthread_join(comm->proxyThread, NULL);

  // Free off any memory allocated for the proxy arg pools
  pthread_mutex_lock(&state->mutex);
  struct ncclProxyState* proxyState = &comm->proxyState;
  while (proxyState->pools != NULL) {
    struct ncclProxyPool *next = proxyState->pools->next;
    free(proxyState->pools);
    proxyState->pools = next;
  }
  pthread_mutex_unlock(&state->mutex);

  return ncclSuccess;
}
