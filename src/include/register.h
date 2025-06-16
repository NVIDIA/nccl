#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

#include "device.h"

#include <cuda.h>
#include <stdint.h>

int64_t ncclParamLocalRegister();
int64_t ncclParamGraphRegister();

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08,
  COLLNET_REG_COMPLETE = 0x10,
  IPC_REG_COMPLETE = 0x20
};

struct ncclPeerRegIpcAddr {
  uintptr_t* devPeerRmtAddrs;
  uintptr_t* hostPeerRmtAddrs;
};

struct ncclRegNetHandles {
  void* handle;
  struct ncclProxyConnector* proxyConn;
  struct ncclRegNetHandles* next;
};

struct ncclSymRegTask {
  struct ncclSymRegTask *next;
  void* buff;
  size_t baseSize;
  CUmemGenericAllocationHandle memHandle;
  struct ncclReg* regHandle;
  size_t alignment;
};

struct ncclReg {
  // common attributes
  uintptr_t begAddr, endAddr; // page aligned
  int localRefs;
  int graphRefs;
  uint32_t state;
  // net reg
  struct ncclRegNetHandles* netHandleHead;
  // nvls reg
  CUdeviceptr regAddr;
  size_t regUCSize, regMCSize;
  int dev;
  CUmemGenericAllocationHandle mcHandle;
  uintptr_t caddrs[NCCL_MAX_LOCAL_RANKS]; /* use to check if NVLS buffers match among intra-node ranks */
  // collnet reg
  void* collnetHandle;
  struct ncclProxyConnector* collnetProxyconn;
  // general ipc reg
  struct ncclPeerRegIpcAddr regIpcAddrs;
  struct ncclIpcRegInfo* ipcInfos[NCCL_MAX_LOCAL_RANKS];
  // symmetric reg
  void* baseSymPtr;
  size_t symSize;
  int winFlags;
};

struct ncclRegCache {
  struct ncclReg **slots;
  int capacity, population;
  uintptr_t pageSize;
};

struct ncclWindow {
  struct ncclReg* handle;
};

ncclResult_t ncclRegCleanup(struct ncclComm* comm);
ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg *handle);
ncclResult_t ncclRegLocalIsValid(struct ncclReg *reg, bool *isValid);
ncclResult_t ncclCommSymmetricRegisterInternal(struct ncclComm* comm, void* buff, size_t baseSize, size_t alignment, CUmemGenericAllocationHandle memHandle, struct ncclReg* regHandle);

#endif
