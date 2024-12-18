/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RAS_H_
#define NCCL_RAS_H_

#include "socket.h"

// Structure used to communicate data about NCCL ranks from NCCL threads to RAS.
struct rasRankInit {
  union ncclSocketAddress addr;
  pid_t pid;
  int cudaDev;
  int nvmlDev;
};

ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank);
ncclResult_t ncclRasCommFini(const struct ncclComm* comm);
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks);

#endif // !NCCL_RAS_H_
