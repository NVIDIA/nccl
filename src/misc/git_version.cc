/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "git_version.h"

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define GIT_VERSION_STRING "NCCL git version " GIT_BRANCH " " GIT_COMMIT_HASH
const char * ncclGetGitVersion(void) {
  return GIT_VERSION_STRING;
}
