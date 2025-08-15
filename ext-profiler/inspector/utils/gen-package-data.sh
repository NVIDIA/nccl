#!/bin/bash

CONTAINER_NAME=$1
CU_ARCH_TARGETS=$2
DEPLOYMENT=$3

TAG=$(git describe --abbrev=0 --tags HEAD)

# Get the current git branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Get the current commit hash
COMMIT_HASH=$(git rev-parse --short HEAD)

set -euo pipefail
# Check if the worktree is dirty
if [ -n "$(git status --porcelain)" ]; then
    DIRTY_BOOL="true"
else
    DIRTY_BOOL="false"
fi

HOSTTRIPLE=$(gcc -dumpmachine)

jq -n \
  --arg target "${HOSTTRIPLE}" \
  --arg version "${TAG}" \
  --arg deployment "${DEPLOYMENT}" \
  --arg commit "${COMMIT_HASH}" \
  --arg dirty_repo "${DIRTY_BOOL}" \
  --arg container_name "${CONTAINER_NAME}" \
  --arg branch_name "${BRANCH}" \
  --arg cu_arch_targets "${CU_ARCH_TARGETS}" \
  '{
     tags: [],
     comment: "",
     deployment: $deployment,
     nccl: {
      version: $version,
      commit: $commit,
      branch_name: $branch_name,
      dirty_repo: $dirty_repo,
      },
     build_env: {
       target: $target,
       cu_arch_targets: $cu_arch_targets,
       pytorch: {
         image: $container_name
       },
     },
   }'
