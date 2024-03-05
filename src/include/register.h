#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08
};

struct ncclReg {
  // common attributes
  size_t pages;
  int refs;
  uintptr_t addr;
  uint32_t state;
  // net reg
  int nDevs;
  int devs[MAXCHANNELS];
  void** handles;
  // nvls reg
  uintptr_t baseAddr;
  size_t baseSize;
  CUdeviceptr regAddr;
  size_t regSize;
  int dev;
  CUmemGenericAllocationHandle mcHandle;
  uintptr_t caddrs[NCCL_MAX_LOCAL_RANKS]; /* use to check if NVLS buffers match among intra-node ranks */
};

struct ncclRegCache {
  struct ncclReg **slots;
  int capacity, population;
  uintptr_t pageSize;
  void* sComms[MAXCHANNELS];
  void* rComms[MAXCHANNELS];
};

ncclResult_t ncclRegCleanup(struct ncclComm* comm);
ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** reg);

#endif
