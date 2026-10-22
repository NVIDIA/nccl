#!/usr/bin/env bash
# Load the patched NCCL without overriding protocol or buffer selection.
# Copy streams and host buffers live in an isolated CUDA context.
# Keep the application's default asynchronous Work.wait behavior.
set -euo pipefail
: "${NCCL_LIBRARY:?Set NCCL_LIBRARY to libnccl.so.2 from a full build}"
if [[ $# -eq 0 || ! -f "$NCCL_LIBRARY" ]]; then
  echo "Usage: NCCL_LIBRARY=/absolute/path/libnccl.so.2 $0 COMMAND [ARGS...]" >&2
  exit 2
fi
lib_dir=$(cd -- "$(dirname -- "$NCCL_LIBRARY")" && pwd)
export SGLANG_NCCL_SO_PATH="$lib_dir/$(basename -- "$NCCL_LIBRARY")"
export LD_PRELOAD="$SGLANG_NCCL_SO_PATH${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
