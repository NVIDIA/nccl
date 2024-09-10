#include "comm.h"
#include "transport.h"

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all rings");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
