/*************************************************************************
 * Copyright (c) 2019, SPLab. All rights reserved.
 ************************************************************************/

#ifndef STAT_COLLECTOR_H_
#define STAT_COLLECTOR_H_

#include "nccl.h"
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <unistd.h>
#include <iostream>
#include <assert.h>

inline void create_comm_stat(ncclProf_t* nccl_prof, commType_t comm_type,
    int from_rank, int to_rank, uint64_t start_micros, uint64_t end_micros, int comm_bytes) {
  pthread_mutex_lock(&nccl_prof->mu_);
  commStat_t* comm_stat = (commStat_t*) malloc(sizeof(commStat_t));
  comm_stat->comm_type = comm_type;
  comm_stat->from_rank = from_rank;
  comm_stat->to_rank = to_rank;
  comm_stat->start_micros = start_micros;
  comm_stat->end_micros = end_micros;
  comm_stat->comm_bytes = comm_bytes;
  nccl_prof->stat_vector->push_back(comm_stat);
  pthread_mutex_unlock(&nccl_prof->mu_);
}

#endif // end include guard
