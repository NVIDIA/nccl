/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "comm.h"
#include "nccl_device/impl/core__funcs.h"

NCCL_API(ncclTeam_t, ncclTeamWorld, ncclComm_t comm);
ncclTeam_t ncclTeamWorld(ncclComm_t comm) {
  ncclTeam_t ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}

NCCL_API(ncclTeam_t, ncclTeamLsa, ncclComm_t comm);
ncclTeam_t ncclTeamLsa(ncclComm_t comm) {
  // Ignoring errors since if it fails ncclDevrInitOnce will try again.
  // The returned team will be junk and the next "interesting" API call that
  // needs ncclDevrInitOnce will report the error.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};

  ncclTeam_t ans;
  ans.nRanks = comm->devrState.lsaSize;
  ans.rank = comm->devrState.lsaSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API(ncclTeam_t, ncclTeamCft, ncclComm_t comm, ncclCftTeamMode_t mode);
ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode) {
  // Ignoring errors as in ncclTeamLsa().
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};

  ncclTeam_t flatTeam;
  flatTeam.nRanks = comm->devrState.cftSize;
  flatTeam.rank = comm->devrState.cftSelf;
  flatTeam.stride = 1;
  if (mode == NCCL_CFT_TEAM_FLAT) return flatTeam;

  int innerSize;
  if (mode == NCCL_CFT_TEAM_HIER_MULTIMEM) {
    innerSize = comm->devrState.cftMcSize;
  } else if (mode == NCCL_CFT_TEAM_HIER_LSA) {
    innerSize = comm->devrState.lsaSize;
  } else {
    return ncclTeam_t{};
  }
  return ncclTeamOuterFactor(flatTeam, innerSize);
}

NCCL_API(ncclTeam_t, ncclTeamCftMultimem, ncclComm_t comm);
ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm) {
  // Ignoring errors as in ncclTeamLsa().
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};

  ncclTeam_t ans;
  ans.nRanks = comm->devrState.cftMcSize;
  ans.rank = comm->devrState.cftMcSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API(ncclTeam_t, ncclTeamRail, ncclComm_t comm);
ncclTeam_t ncclTeamRail(ncclComm_t comm) {
  // Ignoring errors as above.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return ncclTeam_t{};

  ncclTeam_t ans;
  ans.nRanks = comm->nRanks / comm->devrState.lsaSize;
  ans.rank = comm->rank / comm->devrState.lsaSize;
  ans.stride = comm->devrState.lsaSize;
  return ans;
}

NCCL_API(int, ncclTeamRankToWorld, ncclComm_t comm, ncclTeam_t team, int rank);
int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank) {
  return comm->rank + (rank - team.rank) * team.stride;
}

NCCL_API(int, ncclTeamRankToLsa, ncclComm_t comm, ncclTeam_t team, int rank);
int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank) {
  // Ignoring errors as above.
  if (ncclSuccess != ncclDevrInitOnce(comm)) return -1;

  return comm->devrState.lsaSelf + (rank - team.rank) * team.stride;
}
