/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

struct ncclP2Pinfo {
 const void* sendbuff;
  void* recvbuff;
  ssize_t sendbytes;
  ssize_t recvbytes;
};

struct ncclP2PConnect {
  int nrecv[MAXCHANNELS];
  int nsend[MAXCHANNELS];
  int* recv;
  int* send;
};

struct ncclP2Plist {
  struct ncclP2Pinfo *peerlist;
  int count;
  struct ncclP2PConnect connect;
};

#endif
