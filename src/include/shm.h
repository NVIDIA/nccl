#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include "comm.h"

struct shmLegacyIpc {
  char shmSuffix[7];
  ncclShmHandle_t handle;
  size_t shmSize;
};

struct shmCuIpc {
  union {
    CUmemFabricHandle handle;
    CUmemGenericAllocationHandle data;
  };
  int tpProxyRank;
  void *ptr;
  size_t size;
};

struct shmIpcDesc {
  union
  {
    struct shmLegacyIpc shmli;
    struct shmCuIpc shmci;
  };
  bool legacy;
};

typedef struct shmIpcDesc ncclShmIpcDesc_t;

ncclResult_t ncclShmAllocateShareableBuffer(int tpProxyRank, size_t size, bool legacy, ncclShmIpcDesc_t *descOut, void **hptr, void **dptr);
ncclResult_t ncclShmImportShareableBuffer(struct ncclComm *comm, ncclShmIpcDesc_t *desc, void **hptr, void **dptr, ncclShmIpcDesc_t *descOut);
ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t *desc);

#endif
