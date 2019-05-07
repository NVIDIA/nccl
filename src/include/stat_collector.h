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

inline const std::string CommTypeToString(commType_t comm_type) {
  switch (comm_type) {
    case P2P_SEND:  return std::string("P2P_SEND");
    case P2P_RECV:  return std::string("P2P_RECV");
    case SHM_SEND:  return std::string("SHM_SEND");
    case SHM_RECV:  return std::string("SHM_RECV");
    case NET_SEND:  return std::string("NET_SEND");
    case NET_RECV:  return std::string("NET_RECV");
    default:        return std::string("UndefinedType");
  }
}

inline commStat_t* create_comm_stat(commType_t comm_type, int from_rank, int to_rank, 
    uint64_t start_micros, uint64_t end_micros, int comm_bytes) {
  commStat_t* comm_stat = (commStat_t*) malloc(sizeof(commStat_t));
  comm_stat->comm_type = comm_type;
  comm_stat->from_rank = from_rank;
  comm_stat->to_rank = to_rank;
  comm_stat->start_micros = start_micros;
  comm_stat->end_micros = end_micros;
  comm_stat->comm_bytes = comm_bytes;
  return comm_stat;
}

inline void enqueue_stat(ncclProf_t* nccl_prof, commStat_t* comm_stat) {
  pthread_mutex_lock(&nccl_prof->mu_);
  if (nccl_prof->stat_vector == nullptr) {
    nccl_prof->stat_vector = new std::vector<commStat_t*>();
  }
  nccl_prof->stat_vector->push_back(comm_stat);
  pthread_mutex_unlock(&nccl_prof->mu_);
}

inline const std::string ncclprof_tostring(ncclProf_t* nccl_prof) {
  std::string ret = "";
  std::string tensor_name(nccl_prof->tensor_name);
  ret += std::string("{\n  \"Step\": ") + std::to_string(nccl_prof->step) + std::string(",\n  \"Tensor\": \"") + tensor_name + std::string("\",\n  \"WorkerID\": ") + std::to_string(nccl_prof->worker_id);
  ret += std::string(",\n  \"CommStatList\": [\n");
  std::vector<commStat_t*>::iterator iter;
  for (iter = nccl_prof->stat_vector->begin(); iter != nccl_prof->stat_vector->end(); iter++) {
    if (iter != nccl_prof->stat_vector->begin()) {
      ret += std::string(",\n");
    }
    ret += std::string("    { \"CommType\": \"") + CommTypeToString((*iter)->comm_type) + std::string("\"");
    ret += std::string(", \"FromRank\": ") + std::to_string((*iter)->from_rank);
    ret += std::string(", \"ToRank\": ") + std::to_string((*iter)->to_rank);
    ret += std::string(", \"StartMicros\": ") + std::to_string((*iter)->start_micros);
    ret += std::string(", \"EndMicros\": ") + std::to_string((*iter)->end_micros);
    ret += std::string(", \"CommBytes\": ") + std::to_string((*iter)->comm_bytes) + std::string(" }");
  }
  ret += std::string("\n  ]\n");
  ret += std::string("},\n");
  return ret;
}

/* Class for CommStat Collector */
class StatCollector {
public:
  StatCollector() : worker_id(-1), mu_(PTHREAD_MUTEX_INITIALIZER), \
                    saved_in_file(0), started_save_in_file_thread(0) {
  }

  ~StatCollector() {
    std::unordered_map<int64_t, NcclProfVector*>::iterator iter;
    for (iter = step_stats_.begin(); iter != step_stats_.end(); iter++) {
      NcclProfVector::iterator iter_;
      for (iter_ = iter->second->begin(); iter_ != iter->second->end(); iter_++) {
        std::vector<commStat_t*>::iterator iter__;
        for (iter__ = (*iter_)->stat_vector->begin(); iter__ != (*iter_)->stat_vector->end(); iter__++) {
          free(*iter__);
        }
        (*iter_)->stat_vector->clear();
        free((char*)(*iter_)->tensor_name);
      }
      iter->second->clear();
    }
    step_stats_.clear();
  }

  void save(ncclProf_t* nccl_prof) {
    pthread_mutex_lock(&mu_);
    if (!nccl_prof->saved) {
      std::unordered_map<int64_t, NcclProfVector*>::iterator iter = step_stats_.find(nccl_prof->step);
      if (iter == step_stats_.end()) {
        NcclProfVector* nccl_prof_vector = new NcclProfVector();
        step_stats_.insert(std::pair<int64_t, NcclProfVector*>(nccl_prof->step, nccl_prof_vector));
      }
      step_stats_[nccl_prof->step]->push_back(nccl_prof);
      nccl_prof->saved = 1;
    }
    if (worker_id == -1) {
      worker_id = nccl_prof->worker_id;
    } else {
      assert(worker_id == nccl_prof->worker_id);
    }
    pthread_mutex_unlock(&mu_);
  }

  ncclResult_t save_metadata() {
    FILE *fp;
    std::string homedir(std::getenv("HOME"));
    char _hostname[HOST_NAME_MAX];
    gethostname(_hostname, HOST_NAME_MAX);
    std::string hostname(_hostname);
    std::string save_filedir = std::string(std::getenv("HOME")) + std::string("/prof_dir/") + hostname + \
                               std::string("/worker:") + std::to_string(worker_id);
    system(("mkdir -p " + save_filedir).c_str());
    std::unordered_map<int64_t, NcclProfVector*>::iterator iter;
    for (iter = step_stats_.begin(); iter != step_stats_.end(); iter++) {
      std::string save_filename = save_filedir + std::string("/nccl_meta_") + std::to_string(iter->first);
      if ((fp = fopen(save_filename.c_str(), "w")) == NULL) {
        std::cout << "Cannot open file " << save_filename << std::endl;
        return ncclSystemError;
      }
      NcclProfVector::iterator iter_;
      fputs("[\n", fp);
      for (iter_ = iter->second->begin(); iter_ != iter->second->end(); iter_++) {
        const std::string ncclprof_str = ncclprof_tostring(*iter_);
        fputs(ncclprof_str.c_str(), fp);
      }
      fputs("]\n", fp);
      fclose(fp);
    }
    return ncclSuccess;
  }

  void save_in_file() {
    pthread_mutex_lock(&mu_);
    if (started_save_in_file_thread) {
      pthread_mutex_unlock(&mu_);
      return;
    }
    started_save_in_file_thread = 1;
    pthread_mutex_unlock(&mu_);
    while(!saved_in_file) {
      sleep(1);
    }
    save_metadata();
  }

  void set_saved_in_file() {
    pthread_mutex_lock(&mu_);
    saved_in_file = 1;
    pthread_mutex_unlock(&mu_);
  }

private:
  typedef std::vector<ncclProf_t*> NcclProfVector;
  pthread_mutex_t mu_;
  std::unordered_map<int64_t, NcclProfVector*> step_stats_;
  int worker_id;
  int started_save_in_file_thread;
  int saved_in_file; // Whether all the profiled data is saved in file.
                     // Assume no more data is profiled after saved_in_file became 1.
};

static pthread_mutex_t scmu_ = PTHREAD_MUTEX_INITIALIZER;
inline StatCollector* GetStatCollector() {
  pthread_mutex_lock(&scmu_);
  static StatCollector* stat_collector = new StatCollector();
  pthread_mutex_unlock(&scmu_);
  return stat_collector;
}

inline void* StatCollectorSaveInFileThread(void *unused) {
  StatCollector* stat_collector = GetStatCollector();
  stat_collector->save_in_file();
  static const int ok_return = 1;
  return (void*) &ok_return;
}

inline ncclResult_t StartStatCollector() {
  pthread_t thread_id;
  pthread_create(&thread_id, NULL, StatCollectorSaveInFileThread, NULL);
  return ncclSuccess;
}

#endif // end include guard
