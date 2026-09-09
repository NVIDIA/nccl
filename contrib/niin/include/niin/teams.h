/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NVSHMEM team management: split, translate, destroy.
//
// Teams are stored as {start, stride, size} triples in a fixed-size table.
// IDs 0-2 are reserved (WORLD, SHARED, NODE). User teams get IDs >= 3.

#ifndef NIIN_TEAMS_H_
#define NIIN_TEAMS_H_

#include "niin/types.h"
#include <cstdio>

// Team config struct (matching NVSHMEM)
typedef struct {
  int num_contexts;
} nvshmem_team_config_t;

#define NVSHMEM_TEAM_CONFIG_INITIALIZER { 0 }

#define NIIN_MAX_TEAMS 64
#define NIIN_FIRST_USER_TEAM 6  // 0-5 are predefined (WORLD, SHARED, NODE, SAME_MYPE_NODE, SAME_GPU, GPU_LEADERS)

namespace niin {
namespace teams {

struct TeamInfo {
  bool valid;
  int start;    // first PE in parent (world rank)
  int stride;   // stride between PEs in world rank space
  int size;     // number of PEs
  int myRank;   // this PE's rank within the team, or -1 if not a member
};

struct TeamTable {
  TeamInfo teams[NIIN_MAX_TEAMS];
  int nextId;
  bool initialized;
};

inline TeamTable& table() {
  static TeamTable t = {};
  return t;
}

// Called by niin_host_init_common after global state is set up
inline void initPredefined(int worldRank, int worldSize, int lsaRank, int lsaSize) {
  auto& t = table();

  // TEAM_WORLD (id=0): all PEs
  t.teams[NVSHMEM_TEAM_WORLD] = {true, 0, 1, worldSize, worldRank};

  // TEAM_SHARED (id=1): same node (LSA team)
  int lsaStart = worldRank - lsaRank;
  t.teams[NVSHMEM_TEAM_SHARED] = {true, lsaStart, 1, lsaSize, lsaRank};

  // NVSHMEMX_TEAM_NODE (id=2): alias for SHARED
  t.teams[NVSHMEMX_TEAM_NODE] = {true, lsaStart, 1, lsaSize, lsaRank};

  // NVSHMEMX_TEAM_SAME_MYPE_NODE (id=3): same local rank across nodes (rail team)
  // E.g., local rank 0 on all nodes. stride=lsaSize, start=lsaRank.
  int railSize = (worldSize + lsaSize - 1) / lsaSize;  // number of nodes
  int railRank = worldRank / lsaSize;
  t.teams[NVSHMEMX_TEAM_SAME_MYPE_NODE] = {true, lsaRank, lsaSize, railSize, railRank};

  // NVSHMEMI_TEAM_SAME_GPU (id=4): PEs sharing same GPU
  // In NCCL's 1-PE-per-GPU model, this is always just this PE alone.
  t.teams[NVSHMEMI_TEAM_SAME_GPU] = {true, worldRank, 1, 1, 0};

  // NVSHMEMI_TEAM_GPU_LEADERS (id=5): one PE per GPU
  // In NCCL's 1-PE-per-GPU model, this is the same as TEAM_WORLD.
  t.teams[NVSHMEMI_TEAM_GPU_LEADERS] = {true, 0, 1, worldSize, worldRank};

  t.nextId = NIIN_FIRST_USER_TEAM;
  t.initialized = true;
}

// Convert a team-local rank to a world rank
inline int teamRankToWorld(int teamId, int teamRank) {
  auto& t = table();
  if (teamId < 0 || teamId >= NIIN_MAX_TEAMS || !t.teams[teamId].valid) return -1;
  auto& tm = t.teams[teamId];
  return tm.start + teamRank * tm.stride;
}

// Convert a world rank to a team-local rank (-1 if not member)
inline int worldRankToTeam(int teamId, int worldRank) {
  auto& t = table();
  if (teamId < 0 || teamId >= NIIN_MAX_TEAMS || !t.teams[teamId].valid) return -1;
  auto& tm = t.teams[teamId];
  int delta = worldRank - tm.start;
  if (delta < 0 || tm.stride == 0) return -1;
  if (delta % tm.stride != 0) return -1;
  int rank = delta / tm.stride;
  if (rank < 0 || rank >= tm.size) return -1;
  return rank;
}

} // namespace teams
} // namespace niin

// ---------------------------------------------------------------------------
// nvshmem_team_split_strided
// ---------------------------------------------------------------------------
inline int nvshmem_team_split_strided(nvshmem_team_t parent_team,
                                       int PE_start, int PE_stride, int PE_size,
                                       const nvshmem_team_config_t* config,
                                       long config_mask,
                                       nvshmem_team_t* new_team) {
  (void)config; (void)config_mask;
  auto& t = niin::teams::table();
  if (!t.initialized) { *new_team = NVSHMEM_TEAM_INVALID; return -1; }
  if (t.nextId >= NIIN_MAX_TEAMS) { *new_team = NVSHMEM_TEAM_INVALID; return -1; }

  // Get parent team info
  if (parent_team < 0 || parent_team >= NIIN_MAX_TEAMS || !t.teams[parent_team].valid) {
    *new_team = NVSHMEM_TEAM_INVALID;
    return -1;
  }
  auto& parent = t.teams[parent_team];

  // Compute world-space start and stride
  int worldStart = parent.start + PE_start * parent.stride;
  int worldStride = PE_stride * parent.stride;

  // Check if this PE is a member
  int myWorldRank = t.teams[NVSHMEM_TEAM_WORLD].myRank;
  int delta = myWorldRank - worldStart;
  int myTeamRank = -1;
  if (delta >= 0 && worldStride > 0 && delta % worldStride == 0) {
    int r = delta / worldStride;
    if (r >= 0 && r < PE_size) myTeamRank = r;
  }

  int id = t.nextId++;
  t.teams[id] = {true, worldStart, worldStride, PE_size, myTeamRank};

  *new_team = (myTeamRank >= 0) ? id : NVSHMEM_TEAM_INVALID;
  return 0;
}

// ---------------------------------------------------------------------------
// nvshmem_team_split_2d
// ---------------------------------------------------------------------------
inline int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange,
                                  const nvshmem_team_config_t* xaxis_config,
                                  long xaxis_mask,
                                  nvshmem_team_t* xaxis_team,
                                  const nvshmem_team_config_t* yaxis_config,
                                  long yaxis_mask,
                                  nvshmem_team_t* yaxis_team) {
  auto& t = niin::teams::table();
  if (!t.initialized || parent_team < 0 || parent_team >= NIIN_MAX_TEAMS ||
      !t.teams[parent_team].valid) {
    *xaxis_team = NVSHMEM_TEAM_INVALID;
    *yaxis_team = NVSHMEM_TEAM_INVALID;
    return -1;
  }
  auto& parent = t.teams[parent_team];
  int parentSize = parent.size;
  int myParentRank = parent.myRank;

  if (myParentRank < 0) {
    *xaxis_team = NVSHMEM_TEAM_INVALID;
    *yaxis_team = NVSHMEM_TEAM_INVALID;
    return 0;
  }

  int yrange = (parentSize + xrange - 1) / xrange;
  int myRow = myParentRank / xrange;
  int myCol = myParentRank % xrange;

  // X-axis team: PEs in the same row
  int xSize = (myRow == yrange - 1) ? (parentSize - myRow * xrange) : xrange;
  int xStart = myRow * xrange;
  nvshmem_team_split_strided(parent_team, xStart, 1, xSize, xaxis_config, xaxis_mask, xaxis_team);

  // Y-axis team: PEs in the same column
  int ySize = yrange;
  // Last column might be shorter
  if (myCol >= (parentSize % xrange) && (parentSize % xrange) != 0) {
    ySize = yrange - 1;
  }
  nvshmem_team_split_strided(parent_team, myCol, xrange, ySize, yaxis_config, yaxis_mask, yaxis_team);

  return 0;
}

// ---------------------------------------------------------------------------
// nvshmem_team_destroy
// ---------------------------------------------------------------------------
inline void nvshmem_team_destroy(nvshmem_team_t team) {
  auto& t = niin::teams::table();
  if (team >= NIIN_FIRST_USER_TEAM && team < NIIN_MAX_TEAMS) {
    t.teams[team].valid = false;
  }
}

// ---------------------------------------------------------------------------
// nvshmem_team_translate_pe
// ---------------------------------------------------------------------------
inline int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe,
                                      nvshmem_team_t dest_team) {
  int worldRank = niin::teams::teamRankToWorld(src_team, src_pe);
  if (worldRank < 0) return -1;
  return niin::teams::worldRankToTeam(dest_team, worldRank);
}

// ---------------------------------------------------------------------------
// nvshmem_team_get_config
// ---------------------------------------------------------------------------
inline void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t* config) {
  if (config) config->num_contexts = 0;
}

// ---------------------------------------------------------------------------
// Host-side team query helpers (called by query.h trampolines)
// ---------------------------------------------------------------------------
inline int niin_teams_my_pe(int team) {
  auto& t = niin::teams::table();
  if (team < 0 || team >= NIIN_MAX_TEAMS || !t.teams[team].valid) return -1;
  return t.teams[team].myRank;
}

inline int niin_teams_n_pes(int team) {
  auto& t = niin::teams::table();
  if (team < 0 || team >= NIIN_MAX_TEAMS || !t.teams[team].valid) return -1;
  return t.teams[team].size;
}

#endif // NIIN_TEAMS_H_
