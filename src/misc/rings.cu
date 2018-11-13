/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "param.h"

#define NCCL_MAX_SCORE 7

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|0 2 3 1|1 3 2 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0"
 * Rings with a non-matching number of ranks are ignored so we can provide
 * rings for multiple cases.
 */
#define MAX_ENV_RANKS 512
static ncclResult_t parseRings(const char* str, int* nringsRet, int nranks, int* prev, int* next) {
  int ranks[MAX_ENV_RANKS];
  int nrings = 0;
  int rank = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number
  do {
    int digit = str[offset] - '0';
    if (digit >= 0 && digit <= 9) {
      if (status == 0) {
        ranks[rank] = digit;
        status = 1;
      } else {
        ranks[rank] = ranks[rank]*10+digit;
      }
    } else {
      if (status == 1) {
        rank++;
        if (rank == MAX_ENV_RANKS) goto end;
      }
      status = 0;
      if (str[offset] == '|' || str[offset] == '\0') {
        int prevRank = ranks[rank-1];
        // Ignore rings if nranks doesn't match
        if (rank != nranks) goto newring;

        for (int r=0; r<nranks; r++) {
          int rank = ranks[r];
          // Ignore rings with ranks out of bounds
          if (rank < 0 || rank >= nranks) goto newring;
          // Ignore rings with duplicate ranks
          for (int i=0; i<r; i++)
            if (ranks[i] == rank) goto newring;

          next[nrings*nranks+prevRank] = rank;
          prev[nrings*nranks+rank] = prevRank;
          prevRank = rank;
        }
        nrings++;
newring:
        rank = 0;
      }
    }
  } while (str[offset++] != 0);
end:
  *nringsRet = nrings;
  return ncclSuccess;
}

/*
 * Ring creation algorithm
 *
 * First, we establish hierarchical coordinates depending on the way ranks can
 * communicate. After fillCoords, we have for each rank a unique 3-int array
 * {   node, pci_domain,   rank } corresponding to the three transports :
 * { 2[NET],     1[SHM], 0[P2P] }.
 * Also, we renumber ranks (to indexes) based on their growing coordinates.
 *
 * Then, we ask transports to connect groups together. We start with net, then
 * shm, then p2p. We maintain two arrays, prev and next, where values are equal
 * to -1 when ranks are not yet connected, and a rank otherwise. We never
 * connect ranks outside our group, meaning that on 4 nodes of 2 sockets of 4
 * ranks, if we are rank 13, we should see something like (provided we have a
 * single net interface, hence a single ring) :
 *
 * Connecting all nodes                                <13>
 * 2[NET] : prev 31 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 23 -1 -1 -1 -1 -1 -1 -1
 *          next -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1 -1 -1 -1 -1 16 -1 -1 -1 -1 -1 -1 -1 24 -1 -1 -1 -1 -1 -1 -1  0
 *
 * Connecting P2P domains with shared memory           <13>
 * 1[SHM] : prev 31 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 11 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 23 -1 -1 -1 -1 -1 -1 -1
 *          next -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1 12 -1 -1 -1 16 -1 -1 -1 -1 -1 -1 -1 24 -1 -1 -1 -1 -1 -1 -1  0
 *
 * Connecting ranks (only inside the P2P domain)       <13>
 * 0[P2P] : prev 31 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 11 12 13 14 15 -1 -1 -1 -1 -1 -1 -1 23 -1 -1 -1 -1 -1 -1 -1
 *          next -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1 12 13 14 15 16 -1 -1 -1 -1 -1 -1 -1 24 -1 -1 -1 -1 -1 -1 -1  0
 *
 * Hence, when we ask a transport to connect groups, we provide it with a subview of the ranks (except for net
 * which always sees the full world). That way, P2P can bruteforce all combinations inside the node without
 * risking to explode in terms of combinations, and we scale better.
 *
 * Finally, we loop over Network scores to try to create rings with high scores (=locality) and decrease until
 * we get at least one ring.
 */

static void recIsConnected(int rank, int* connected, int nranks, int* matrix, int transport) {
  connected[rank] = 1;
  for (int r=0; r<nranks; r++) {
    if (connected[r] == 0 && matrix[rank*nranks+r] == transport) {
      recIsConnected(r, connected, nranks, matrix, transport);
    }
  }
}

static void isConnected(int rank, int* connected, int nranks, int* matrix, int transport) {
  for (int r=0; r<nranks; r++) connected[r] = 0;
  recIsConnected(rank, connected, nranks, matrix, transport);
}

#define NEW_IDX(rank) do { \
  rankToIdx[rank] = idx; \
  idxToRank[idx] = rank; \
  for (int t=0; t<NTRANSPORTS; t++) coords[rank*NTRANSPORTS+t] = current[t]; \
  idx++; \
} while (0)

int findConnected(int rank, int* matrix, int nranks, int transport, int* coords) {
  for (int r=0; r<nranks; r++) {
    if (coords[r*NTRANSPORTS] == -1 && matrix[rank*nranks+r] == transport) return r;
  }
  return -1;
}

static ncclResult_t fillCoords(int nranks, int* matrix, int* coords, int* rankToIdx, int* idxToRank) {
  int current[NTRANSPORTS];
  int* p2pConnected;
  NCCLCHECK(ncclCalloc(&p2pConnected, nranks));
  for (int i=0; i<NTRANSPORTS; i++) current[i] = 0;
  int curRank = 0, idx = 0;
  while (1) {
    // P2P is handled separately as there is no level below it and we need to
    // cover the case of being connected to another GPU indirectly.
    // So we detect all GPUs in the same P2P domain once and add them all at
    // once.
    isConnected(curRank, p2pConnected, nranks, matrix, 0);
    for (int r=0; r<nranks; r++) {
      if (p2pConnected[r]) {
        NEW_IDX(r);
        curRank = r;
        current[0]++;
      }
    }
    current[0] = 0;

    if (idx == nranks) {
      free(p2pConnected);
      return ncclSuccess;
    }

    // Find next group, either connected through SHM or NET.
    int rank;
    int transport = 1;
    while ((rank = findConnected(curRank, matrix, nranks, transport, coords)) == -1) {
      current[transport] = 0;
      transport++;
      if (transport == NTRANSPORTS) { free(p2pConnected); return ncclInternalError; }
    }
    curRank = rank;
    current[transport]++;
  }
}

NCCL_PARAM(MinNrings, "MIN_NRINGS", 0);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", 0);

/* Users can force the number of threads with an environment variable */
NCCL_PARAM(Nthreads, "NTHREADS", -2);
ncclResult_t getEnvThreads(int* nthreads) {
  int64_t nt = ncclParamNthreads();
  if (nt != -2)
    *nthreads = nt;
  return ncclSuccess;
}

/* Main ring creation function */
ncclResult_t ncclGetRings(int* nrings, int* nthreads, int rank, int nranks, int* transports, ncclTvalue_t* values, int* prev, int* next) {
  *nrings = 0;

  if (nranks == 1) return ncclSuccess;

  char* str = getenv("NCCL_RINGS");
  if (str && strlen(str)>0) {
    int ret = parseRings(str, nrings, nranks, prev, next);
    if (ret == ncclSuccess && *nrings > 0) {
      if (rank == 0) INFO(NCCL_INIT,"%d ring(s) set by environment", *nrings);
      NCCLCHECK(getEnvThreads(nthreads));
      return ncclSuccess;
    }
    if (rank == 0) INFO(NCCL_INIT,"No valid ring found in environment, ignoring");
    *nrings = 0;
  }

  // Compute hierarchical topology groups, indexes, and rank<->index tables
  int* coords, *globalIdxToRank, *globalRankToIdx;
  NCCLCHECK(ncclCalloc(&coords, nranks*NTRANSPORTS));
  for (int i=0; i<nranks*NTRANSPORTS; i++) coords[i] = -1;
  NCCLCHECK(ncclCalloc(&globalIdxToRank, nranks));
  NCCLCHECK(ncclCalloc(&globalRankToIdx, nranks));

  NCCLCHECK(fillCoords(nranks, transports, coords, globalRankToIdx, globalIdxToRank));

  // Start with a high score, then decrease until we find rings
  int minScore = NCCL_MAX_SCORE;
  int nringsTmp;
  int *prevTmp, *nextTmp, *idxToRank, *rankToIdx, *groups, *subgroups;
  NCCLCHECK(ncclCalloc(&prevTmp, nranks*MAXRINGS));
  NCCLCHECK(ncclCalloc(&nextTmp, nranks*MAXRINGS));
  NCCLCHECK(ncclCalloc(&idxToRank, nranks));
  NCCLCHECK(ncclCalloc(&rankToIdx, nranks));
  NCCLCHECK(ncclCalloc(&groups, nranks));
  NCCLCHECK(ncclCalloc(&subgroups, nranks));

  int nThreads;
  do {
    nThreads = *nthreads;
    for (int i=0; i<nranks*MAXRINGS; i++) prevTmp[i] = nextTmp[i] = -1;
    nringsTmp = MAXRINGS;
    // Loop over transports to connect groups
    for (int t=NTRANSPORTS-1; t>=0; t--) {
      for (int i=0; i<nranks; i++) idxToRank[i] = rankToIdx[i] = -1;

      int nidx = 0;
      for (int i=0; i<nranks; i++) {
        // Extract only ranks in the same local area as rank
        // We need to extract them in the topological order, hence we iterate over indexes, not ranks
        int r = globalIdxToRank[i];
        int sameLocal = 1;
        for (int tr = NTRANSPORTS-1; tr > t; tr--) if (coords[r*NTRANSPORTS+tr] != coords[rank*NTRANSPORTS+tr]) sameLocal = 0;
        if (!sameLocal) continue;

        groups[nidx] = coords[r*NTRANSPORTS+t];
        subgroups[nidx] = t ? coords[r*NTRANSPORTS+t-1] : nidx;
        rankToIdx[r] = nidx;
        idxToRank[nidx] = r;
        nidx++;
      }

      int ngroups = groups[nidx-1] + 1; // Coords should be ordered

      ncclTvalue_t* subvalues;
      int *subprev, *subnext;
      NCCLCHECK(ncclCalloc(&subvalues, nidx*nidx));
      NCCLCHECK(ncclCalloc(&subprev, nidx*nringsTmp));
      NCCLCHECK(ncclCalloc(&subnext, nidx*nringsTmp));
      if (ngroups > 1) {
        /* Extract subvalues */
        for (int i=0; i<nidx; i++) {
          for (int j=0; j<nidx; j++) {
            if (transports[idxToRank[i]*nranks+idxToRank[j]] == t)
              subvalues[i*nidx+j] = values[idxToRank[i]*nranks+idxToRank[j]];
            else
              subvalues[i*nidx+j] = 0;
          }
        }
        /* Extract subprev/subnext */
        for (int i=0; i<nidx*nringsTmp; i++) {
          subprev[i] = subnext[i] = -1;
        }
        for (int r=0; r<nringsTmp; r++) {
          int start = -1, end = -1;
          for (int i=0; i<nranks; i++) {
            if (rankToIdx[i] == -1) continue;
            if (prevTmp[r*nranks+i] != -1) start = i;
            if (nextTmp[r*nranks+i] != -1) end = i;
          }
          if (start != -1 && end != -1) {
            subprev[r*nidx+rankToIdx[start]] = rankToIdx[end];
            subnext[r*nidx+rankToIdx[end]] = rankToIdx[start];
          }
        }
        /* Get rings */
        NCCLCHECK(ncclTransports[t].getRings(nidx, groups, subgroups, subvalues, &nringsTmp, subprev, subnext, minScore, &nThreads));
        /* Merge subprev/subnext into prev/next */
        for (int r=0; r<nringsTmp; r++) {
          for (int i=0; i<nidx; i++) {
            if ((prevTmp[r*nranks+idxToRank[i]] == -1) && (subprev[r*nidx+i] != -1)) prevTmp[r*nranks+idxToRank[i]] = idxToRank[subprev[r*nidx+i]];
            if ((nextTmp[r*nranks+idxToRank[i]] == -1) && (subnext[r*nidx+i] != -1)) nextTmp[r*nranks+idxToRank[i]] = idxToRank[subnext[r*nidx+i]];
          }
        }
        //for (int r=0; r<nringsTmp; r++) {
        //printf("[%d] [%d] [%d] [%d] Prev ", rank, minScore, t, r); for (int i=0; i<nranks; i++) printf("%d ", prevTmp[r*nranks+i]); printf("\n");
        //printf("[%d] [%d] [%d] [%d] Next ", rank, minScore, t, r); for (int i=0; i<nranks; i++) printf("%d ", nextTmp[r*nranks+i]); printf("\n");
        //}
      }
      free(subvalues);
      free(subprev);
      free(subnext);
      if (nringsTmp == 0) break;
    }
    minScore--;
    if (nringsTmp > *nrings) {
      *nrings = nringsTmp;
      for (int i=0; i<nranks*(*nrings); i++) {
        prev[i] = prevTmp[i];
        next[i] = nextTmp[i];
      }
    }
  } while (nringsTmp == 0 && minScore);

  free(coords);
  free(globalRankToIdx);
  free(globalIdxToRank);
  free(prevTmp);
  free(nextTmp);
  free(idxToRank);
  free(rankToIdx);
  free(groups);
  free(subgroups);

  *nthreads = nThreads;

  if (*nrings == 0) {
    WARN("Could not create rings, falling back on simple ring");
    *nrings = 1;
    prev[rank] = (rank-1+nranks) % nranks;
    next[rank] = (rank+1)%nranks;
  }

  int maxNrings = ncclParamMaxNrings();
  int minNrings = ncclParamMinNrings();
  if (maxNrings > 0 && minNrings > maxNrings) {
    if (rank == 0) WARN("NCCL_MIN_NRINGS set to a value greater than NCCL_MAX_NRINGS, ignoring NCCL_MIN_NRINGS");
    minNrings = 0;
  }
  if (minNrings > MAXRINGS) {
    if (rank == 0) WARN("NCCL_MIN_NRINGS set to a value greater than the maximum number of rings supported (%d), limiting it to %d", MAXRINGS, MAXRINGS);
    minNrings = MAXRINGS;
  }
  if (maxNrings > 0 && maxNrings <= *nrings) {
    if (rank == 0) INFO(NCCL_INIT,"Limiting to %d rings per user request.", maxNrings);
    *nrings = maxNrings;
  } else {
    int defaultMinNrings = ncclCudaCompCap() == 3 ? 2 : 1;
    if (minNrings < defaultMinNrings) minNrings = defaultMinNrings;
    if (minNrings > 0 && minNrings > *nrings) {
      if (rank == 0 && minNrings > defaultMinNrings) INFO(NCCL_INIT,"Duplicating rings to %d per user request.", minNrings);
      for (int r=*nrings; r<MAXRINGS && r <minNrings; r++) {
        for (int i=0; i<nranks; i++) {
          prev[r*nranks+i] = prev[(r-*nrings)*nranks+i];
          next[r*nranks+i] = next[(r-*nrings)*nranks+i];
        }
      }
      *nrings = minNrings;
    }
  }

  NCCLCHECK(getEnvThreads(nthreads));
  return ncclSuccess;
}
