#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# run-clang-format.sh — Apply clang-format to all NCCL source files.
#
# The default is reformatting src/**.
# Clang-format will ignore any path listed in the top-level .clang-format-ignore file.
#
# Usage:
#   ./run-clang-format.sh [path]            # Apply formatting in-place
#   ./run-clang-format.sh --list [path]     # List files that would be reformatted, plus files ignored via .clang-format-ignore
#   ./run-clang-format.sh --diff [path]     # Show unified diff of changes without applying

set -euo pipefail

MIN_VERSION=22

usage() {
  echo "Usage: $0 [--list | --diff] [path]" >&2
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE=""
TARGET_PATH="$PWD/src"
TARGET_SET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list | --diff)
      if [[ -n "$MODE" ]]; then
        usage
        exit 1
      fi
      MODE="$1"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --*)
      usage
      exit 1
      ;;
    *)
      if [[ "$TARGET_SET" -eq 1 ]]; then
        usage
        exit 1
      fi
      TARGET_PATH="$1"
      TARGET_SET=1
      ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Locate clang-format and verify version
# ---------------------------------------------------------------------------
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

if ! command -v "$CLANG_FORMAT" &>/dev/null; then
  echo "ERROR: clang-format not found. Install clang-format >= $MIN_VERSION." >&2
  exit 2
fi

VERSION=$("$CLANG_FORMAT" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)
if [[ -z "$VERSION" || "$VERSION" -lt "$MIN_VERSION" ]]; then
  echo "ERROR: clang-format >= $MIN_VERSION required, found: $("$CLANG_FORMAT" --version)" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Collect files, excluding vendored directories
# Top-level .clang-format-ignore controls which files won't be reformatted
# ---------------------------------------------------------------------------
FILES=()
if [[ -f "$TARGET_PATH" ]]; then
  FILES+=("$TARGET_PATH")
elif [[ -d "$TARGET_PATH" ]]; then
  while IFS= read -r f; do
    FILES+=("$f")
  done < <(
    find "$TARGET_PATH" \
      -type f \
      \( -name '*.cc' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' \) \
      | sort
  )
else
  echo "ERROR: path not found: $TARGET_PATH" >&2
  exit 1
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No source files found under $TARGET_PATH." >&2
  exit 0
fi

# ---------------------------------------------------------------------------
# Process files
#
# A single pass classifies every file as ignored, needing formatting, or
# already formatted, performing the per-mode side effect (print diff / format
# in-place) inline. clang-format emits no output to stdout for files matched
# by .clang-format-ignore; an empty probe result means the file was skipped,
# not emptied, so it is recorded as ignored rather than diffed or written.
# ---------------------------------------------------------------------------
if [[ "$MODE" != "--list" ]]; then
  echo "Checking format for ${#FILES[@]} files under $TARGET_PATH..."
fi

NEED=()
IGNORED=()
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
for f in "${FILES[@]}"; do
  "$CLANG_FORMAT" --style=file "$f" >"$TMP"
  if [[ ! -s "$TMP" ]]; then
    IGNORED+=("$f")
    continue
  fi
  if diff -q "$f" "$TMP" >/dev/null; then
    continue
  fi
  NEED+=("$f")
  case "$MODE" in
    --diff)
      echo "--- $f"
      diff -u "$f" "$TMP" || true
      ;;
    "")
      "$CLANG_FORMAT" -i --style=file "$f"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Per-mode summary
# ---------------------------------------------------------------------------
case "$MODE" in
  --list)
    echo "Files under $TARGET_PATH that would be reformatted (${#NEED[@]} of ${#FILES[@]}):"
    for f in ${NEED[@]+"${NEED[@]}"}; do
      echo "  $f"
    done
    echo "Files ignored via top-level .clang-format-ignore (${#IGNORED[@]} of ${#FILES[@]}):"
    for f in ${IGNORED[@]+"${IGNORED[@]}"}; do
      echo "  $f"
    done
    ;;

  --diff)
    if [[ ${#NEED[@]} -eq 0 ]]; then
      echo "All files already formatted."
    fi
    echo "Ignored ${#IGNORED[@]} of ${#FILES[@]} files via top-level .clang-format-ignore."
    ;;

  "")
    echo "Ignored ${#IGNORED[@]} of ${#FILES[@]} files via top-level .clang-format-ignore."
    echo "Done."
    ;;
esac
