#ifndef _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_CORE_H_
#include <nccl.h>
#include "coop.h"
#include "utility.h"

struct ncclDevComm;
struct ncclTeam;
// typedef struct ncclWindow_vidmem* ncclWindow_t; // in nccl.h
struct ncclMultimemHandle;

typedef uint32_t ncclDevResourceHandle;

struct ncclLsaBarrierHandle;
struct ncclLLA2AHandle;

struct ncclTeam {
  int nRanks, rank, stride;
};

template<typename T> struct ncclSymPtr;

struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};

struct ncclDevCommRequirements {
  struct ncclDevResourceRequirements* resourceRequirementsList;
  struct ncclTeamRequirements* teamRequirementsList;

  bool multimem; // Enable multimem on lsa team

  int lsaBarrierCount;
  ncclLsaBarrierHandle* outLsaBarrierHandle; // If non-null, target assigned during ncclDevCommCreate.

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;
  ncclLLA2AHandle* outLsaLLA2AHandle; // If non-null, target assigned during ncclDevCommCreate.
};
struct ncclDevResourceRequirements {
  struct ncclDevResourceRequirements* next;
  size_t bufferSize, bufferAlign;
  ncclDevResourceHandle* outBufferHandle; // If non-null, target assigned during ncclDevCommCreate.
};
struct ncclTeamRequirements {
  struct ncclTeamRequirements* next;
  struct ncclTeam team;
  bool multimem;
  ncclMultimemHandle* outMultimemHandle; // If non-null, target assigned during ncclDevCommCreate.
};

__host__ ncclResult_t ncclDevCommCreate(ncclComm_t, ncclDevCommRequirements const*, ncclDevComm* outDevComm);
__host__ ncclResult_t ncclDevCommDestroy(ncclComm_t, ncclDevComm const* devComm);

////////////////////////////////////////////////////////////////////////////////
// Team API:

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const&);
__host__ ncclTeam ncclTeamWorld(ncclComm_t);

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const&);
__host__ ncclTeam ncclTeamLsa(ncclComm_t);

NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam a, ncclTeam b, int bPeer);
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam a, ncclTeam b, int bPeer);

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
__host__ int ncclTeamRankToWorld(ncclComm_t, ncclTeam, int rank);

NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
__host__ int ncclTeamRankToLsa(ncclComm_t, ncclTeam, int rank);

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamInnerFactor(ncclTeam parent, int innerSize);
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamOuterFactor(ncclTeam parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam parent, ncclTeam subset, int index);

// Equivalent to ncclTeamOuterFactor of lsa team.
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const&);
__host__ ncclTeam ncclTeamRail(ncclComm_t);

// Get offset of resource buffer within `comm.resourceWindow`.
NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle);

#if __CUDACC__
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const&, ncclDevResourceHandle);
#endif

////////////////////////////////////////////////////////////////////////////////
// Window API:

#if __CUDACC__
template<typename Coop>
NCCL_DEVICE_INLINE ncclWindow_t ncclFindWindow(Coop, ncclDevComm const&, void const *ptr);

NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle);
NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const&);
#endif

#if __CUDACC__
// Convenience for combining ncclGet***Pointer() with resource handle.
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const&, ncclDevResourceHandle);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const&, ncclDevResourceHandle, int peer);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const&, ncclDevResourceHandle, ncclTeam, int peer);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(ncclDevComm const&, ncclDevResourceHandle, ncclMultimemHandle);
NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(ncclDevComm const&, ncclDevResourceHandle);
#endif

#endif // _NCCL_DEVICE_CORE_H_
