/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

struct ncclP2Pinfo {
  void* buff;
  ssize_t nbytes;
};

typedef ncclRecyclableList<struct ncclP2Pinfo> ncclP2Plist;

static ncclResult_t ncclSaveP2pInfo(ncclP2Plist* &p2p, void* buff, ssize_t nBytes) {
  if (p2p == NULL) p2p = new ncclP2Plist();
  struct ncclP2Pinfo* next;
  NCCLCHECK(p2p->getNewElem(&next));
  next->buff = buff;
  next->nbytes = nBytes;
  return ncclSuccess;
}
#endif
