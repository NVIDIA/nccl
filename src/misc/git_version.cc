/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_git_version.h"

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define NCCL_GIT_VERSION_STRING "NCCL git version " NCCL_GIT_BRANCH " " NCCL_GIT_COMMIT_HASH
const char * ncclGetGitVersion(void) {
  return NCCL_GIT_VERSION_STRING;
}
