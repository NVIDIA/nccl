#include "argcheck.h"
#include "comm.h"
#include "nccl.h"

NCCL_API(ncclResult_t, ncclCommGetDesc, ncclComm_t comm, char* desc);

ncclResult_t ncclCommGetDesc(ncclComm_t comm, char* desc) {
  NCCLCHECK(PtrCheck(comm, "ncclCommGetDesc", "comm"));

  strcpy(desc, comm->config.commDesc);
  return ncclSuccess;
}

