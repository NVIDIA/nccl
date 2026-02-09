/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

void dumpLine(int* values, int nranks, const char* prefix) {
  constexpr int line_length = 128;
  char line[line_length];
  int num_width = snprintf(nullptr, 0, "%d", nranks-1);  // safe as per "man snprintf"
  int n = snprintf(line, line_length, "%s", prefix);
  for (int i = 0; i < nranks && n < line_length-1; i++) {
    n += snprintf(line + n, line_length - n, " %*d", num_width, values[i]);
    // At this point n may be more than line_length-1, so don't use it
    // for indexing into "line".
  }
  if (n >= line_length) {
    // Sprintf wanted to write more than would fit in the buffer. Assume
    // line_length is at least 4 and replace the end with "..." to
    // indicate that it was truncated.
    snprintf(line+line_length-4, 4, "...");
  }
  INFO(NCCL_INIT, "%s", line);
}

ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  ncclResult_t ret = ncclSuccess;
  uint64_t* rankFound;
  int rankFoundSize = DIVUP(nranks, 64);
  NCCLCHECK(ncclCalloc(&rankFound, rankFoundSize));

  for (int r=0; r<nrings; r++) {
    char prefix[40];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rankFound[current/64] |= (1UL<<(current%64));
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    snprintf(prefix, sizeof(prefix), "Channel %02d/%02d :", r, nrings);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      ret = ncclInternalError;
      goto end;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      uint64_t bits = rankFound[i/64], mask = 1<<(i%64);
      // Fast check 64 ranks at a time
      if (mask == 1 && bits == 0xffffffffffffffff) { i += 63; continue; }
      if ((bits & mask) == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        ret = ncclInternalError;
        goto end;
      }
    }
    memset(rankFound, 0, rankFoundSize*sizeof(uint64_t));
  }
end:
  free(rankFound);
  return ret;
}
