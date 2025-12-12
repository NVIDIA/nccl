#!/bin/sh

# Get the current git branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Get the current commit hash
COMMIT_HASH=$(git rev-parse --short HEAD)

# Check if the worktree is dirty
if [ -n "$(git status --porcelain)" ]; then
    DIRTY="-dirty"
else
    DIRTY=""
fi

# Combine them into a single string
GIT_INFO="${COMMIT_HASH}${DIRTY}"

# Output the git information
echo "${GIT_INFO}"
