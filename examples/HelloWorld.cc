// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <utils.h>

int main(int argc, char* argv[]) {
  int rank = 0, nproc = 1, localRank = 0;
  int count = 1024 * 1024;

  ncclComm_t comm;
  cudaStream_t stream;
  int* userBuff = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.commDesc = "example_pg";

  ncclUniqueId ncclId;
  NCCLCHECK(ncclGetUniqueId(&ncclId));

  printf("Hello world. NCCL_VERSION %d-%s\n", NCCL_VERSION_CODE, NCCL_SUFFIX);

  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRankConfig(&comm, nproc, ncclId, rank, &config));

  CUDACHECK(cudaMalloc(&userBuff, count * sizeof(int)));
  NCCLCHECK(ncclAllReduce(
      (const void*)userBuff, userBuff, count, ncclInt, ncclSum, comm, stream));

#ifdef NCCL_COMM_GET_UNIQUE_HASH
  uint64_t ncclCommHash;
  NCCLCHECK(ncclCommGetUniqueHash(comm, &ncclCommHash));
  printf("ncclCommHash %lx\n", ncclCommHash);
#endif

#ifdef NCCL_COMM_DESCRIPTION
  char commDesc[10];
  NCCLCHECK(ncclCommGetDesc(comm, commDesc));
  printf("nccl commDesc %s\n", commDesc);
#endif

#ifdef NCCL_COMM_DUMP
  std::unordered_map<std::string, std::string> dump;
  NCCLCHECK(ncclCommDump(comm, dump));
  for (auto& it : dump) {
    printf(
        "Dump from comm %p %s: %s\n",
        comm,
        it.first.c_str(),
        it.second.c_str());
  }
#endif

#if defined(NCCL_COMM_SPLIT_TYPE_SUPPORTED)
  ncclComm_t splitComm;
  NCCLCHECK(
      ncclCommSplitType(comm, NCCL_SPLIT_TYPE_NODE, 0, &splitComm, nullptr));
  int splitRank;
  NCCLCHECK(ncclCommUserRank(splitComm, &splitRank));
  printf(
      "Created NCCL_SPLIT_TYPE_NODE type splitComm %p, rank %d\n",
      splitComm,
      splitRank);
  NCCLCHECK(ncclCommDestroy(splitComm));
#endif

  CUDACHECK(cudaFree(userBuff));
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));

  return EXIT_SUCCESS;
}
