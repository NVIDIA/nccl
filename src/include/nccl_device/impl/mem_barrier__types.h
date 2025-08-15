#ifndef _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#define _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#include "../mem_barrier.h"
#include "core__types.h"

struct ncclLsaBarrierHandle {
  ncclDevResourceHandle bufHandle;
  int nBarriers;
};

#if __CUDACC__
template<typename Coop>
struct ncclLsaBarrierSession_internal {
  Coop coop;
  ncclDevComm const& comm;
  ncclTeam team;
  ncclLsaBarrierHandle handle;
  int index;
  bool multimem;
  ncclMultimemHandle mmHandle;
  uint32_t epoch;

  NCCL_DEVICE_INLINE uint32_t* mcInbox(bool multimem) {
    uint32_t* state;
    if (multimem) { // multicast
      state = (uint32_t*)ncclGetResourceBufferMultimemPointer(comm, handle.bufHandle, mmHandle);
    } else { // unicast
      state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    }
    return state + 2*handle.nBarriers + index;
  }

  NCCL_DEVICE_INLINE uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = (uint32_t*)ncclGetResourceBufferPeerPointer(comm, handle.bufHandle, team, owner);
    return state + 3*handle.nBarriers + index*team.nRanks + peer;
  }
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
