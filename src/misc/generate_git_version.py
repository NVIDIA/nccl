#!/usr/bin/env python3

#
# Generates the NCCL git_version.h header file
# but only replaces it if it has changed
#

import subprocess
import os
import sys

from pathlib import Path
def write_if_changed(path, new_content, encoding="utf-8"):
    path = Path(path)
    if path.exists():
        old_content = path.read_text(encoding=encoding)
        if old_content == new_content:
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding=encoding)
    return True

def run_git(cmd, fallback="unknown"):
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return result if result else fallback
    except Exception:
        return fallback

def main(output_path):
    git_hash = run_git(["git", "describe", "--dirty", "--always"])
    git_branch = run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    content = f"""\
/* Auto-generated. Do not edit. */
#ifndef NCCL_GIT_VERSION_H
#define NCCL_GIT_VERSION_H

#ifndef GIT_BRANCH
#define GIT_BRANCH "{git_branch}"
#endif
#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "{git_hash}"
#endif

#endif /* NCCL_GIT_VERSION_H */
"""

    # Replace only if content differs
    if write_if_changed(output_path, content):
        print("Updated ", output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_git_version.py <output_header_path>")
        sys.exit(1)
    main(sys.argv[1])

