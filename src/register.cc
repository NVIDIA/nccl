/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"

ncclResult_t ncclNetDeregister(struct ncclComm* comm, struct ncclReg* reg) {
  struct ncclRegCache* cache = &comm->regCache;
  ncclDebugNoWarn = NCCL_NET;
  for (int d=0; d<reg->nDevs; d++) {
    if (reg->handles[d] != NULL) NCCLCHECK(comm->ncclNet->deregMr(cache->sComms[reg->devs[d]], reg->handles[d]));
  }
  reg->nDevs = 0;
  free(reg->handles);
  reg->handles = NULL;
  ncclDebugNoWarn = 0;
  return ncclSuccess;
}

ncclResult_t ncclNetRegister(struct ncclComm* comm, void* addr, size_t size, struct ncclReg* reg) {
  struct ncclRegCache* cache = &comm->regCache;
  int netCount;
  NCCLCHECK(ncclTopoGetNetCount(comm->topo, &netCount));
  if (netCount == 0) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;

  // Find local devices for p2p operations
  for (int c=0; c<comm->p2pnChannels; c++) {
    int dev;
    if (ncclTopoGetLocalNet(comm->topo, comm->rank, c, &dev) != ncclSuccess) goto end; // No local net
    ncclNetProperties_t props;
    NCCLCHECKGOTO(comm->ncclNet->getProperties(dev, &props), ret, end);
    if (props.regIsGlobal == 0) { // We need to be sure all NICs support global registration.
      reg->nDevs = 0;
      break;
    }
    int found = 0;
    for (int d=0; d<reg->nDevs; d++) if (reg->devs[d] == dev) found = 1;
    if (!found) reg->devs[reg->nDevs++] = dev;
  }

  NCCLCHECKGOTO(ncclCalloc(&reg->handles, reg->nDevs), ret, end);

  ncclDebugNoWarn = NCCL_NET;
  for (int d=0; d<reg->nDevs; d++) {
    int dev = reg->devs[d];
    reg->handles[d] = NULL;

    if (cache->sComms[dev] == NULL) {
      // Create a loopback network comm object for that device to register the buffers.
      void *lComm = NULL;
      ncclNetHandle_t netHandle;
      bool connected = false;
      NCCLCHECKGOTO(comm->ncclNet->listen(dev, &netHandle, &lComm), ret, end);
      while (!connected) {
        if (*comm->abortFlag) {
          goto end;
        }
        if (cache->sComms[dev] == NULL)
          NCCLCHECKGOTO(comm->ncclNet->connect(dev, &netHandle, cache->sComms+dev, NULL), ret, end);
        if (cache->rComms[dev] == NULL)
          NCCLCHECKGOTO(comm->ncclNet->accept(lComm, cache->rComms+dev, NULL), ret, end);
        connected = (cache->rComms[dev] != NULL) && (cache->sComms[dev] != NULL);
      }
      NCCLCHECK(comm->ncclNet->closeListen(lComm));
    }
    if (comm->ncclNet->regMr(cache->sComms[dev], addr, size, NCCL_PTR_CUDA, reg->handles+d) != ncclSuccess) {
      reg->handles[d] = NULL;
      NCCLCHECK(ncclNetDeregister(comm, reg));
      reg->nDevs = 0;
      goto end;
    }
  }
end:
  ncclDebugNoWarn = 0;
  if (ret != ncclSuccess) NCCLCHECK(ncclNetDeregister(comm, reg));
  return ret;
}

ncclResult_t ncclRegFind(struct ncclComm* comm, const void* data, size_t size, struct ncclReg** reg) {
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;

  *reg = NULL;
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot]->addr) return ncclSuccess;
    if ((addr >= cache->slots[slot]->addr) &&
        ((addr-cache->slots[slot]->addr)/pageSize+pages) <= cache->slots[slot]->pages) {
      *reg = cache->slots[slot];
      return ncclSuccess;
    }
  }
}
NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, void** handle) {
  if (!ncclParamLocalRegister()) return ncclSuccess;
  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  for (int slot=0; /*true*/; slot++) {
    if ((slot == cache->population) || (addr < cache->slots[slot]->addr)) {
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclReg*));
      NCCLCHECK(ncclCalloc(cache->slots+slot, 1));
      struct ncclReg* regSlot = cache->slots[slot];
      regSlot->addr = addr;
      regSlot->pages = pages;
      regSlot->refs = 1;
      NCCLCHECK(ncclNetRegister(comm, (void*)addr, pages*pageSize, regSlot));
      regSlot->state |= NET_REG_COMPLETE;
      cache->population += 1;
      *handle = regSlot;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot]->addr) &&
        ((addr-cache->slots[slot]->addr)/pageSize+pages) <= cache->slots[slot]->pages) {
      cache->slots[slot]->refs++;
      *handle = cache->slots[slot];
      return ncclSuccess;
    }
  }
}

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  for (int i=0; i<cache->population; i++) {
    INFO(NCCL_INIT, "Cleanup buffer %p pages %lx", (void*)cache->slots[i]->addr, cache->slots[i]->pages);
    NCCLCHECK(ncclNetDeregister(comm, cache->slots[i]));
    if (cache->slots[i]->state & NVLS_REG_COMPLETE) NCCLCHECK(ncclNvlsDeregBuffer(&cache->slots[i]->mcHandle, cache->slots[i]->regAddr, cache->slots[i]->dev, cache->slots[i]->regSize));
    free(cache->slots[i]);
  }
  free(cache->slots);
  for (int d=0; d<MAXCHANNELS; d++) {
    if (cache->sComms[d]) NCCLCHECK(comm->ncclNet->closeSend(cache->sComms[d]));
    if (cache->rComms[d]) NCCLCHECK(comm->ncclNet->closeRecv(cache->rComms[d]));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NCCLCHECK(PtrCheck(comm, "ncclCommRegister", "comm"));
  if (comm->checkPointers) NCCLCHECK(CudaPtrCheck(buff, comm, "buff", "ncclCommRegister"));
  NCCLCHECK(ncclRegister(comm, buff, size, handle));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  NCCLCHECK(PtrCheck(comm, "ncclCommRegister", "comm"));
  struct ncclReg* reg = (struct ncclReg*)handle;
  struct ncclRegCache* cache = &comm->regCache;
  int slot;
  for (slot=0; slot<cache->population && cache->slots[slot] != reg; slot++);
  if (slot == cache->population) {
    WARN("Deregister: Could not find handle");
    return ncclInvalidUsage;
  }
  if (--reg->refs) return ncclSuccess;
  NCCLCHECK(ncclNetDeregister(comm, reg));
  if (reg->state & NVLS_REG_COMPLETE) {
    NCCLCHECK(ncclNvlsDeregBuffer(&reg->mcHandle, reg->regAddr, reg->dev, reg->regSize));
    reg->regAddr = (CUdeviceptr)NULL;
  }
  free(reg);
  memmove(cache->slots+slot, cache->slots+slot+1, (cache->population-slot-1)*sizeof(struct ncclReg*));
  cache->population -= 1;
  return ncclSuccess;
}
