#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-archive.txz|tar.gz>"
    exit 1
fi

ARCHIVE_PATH="$1"
FILENAME=$(basename "$ARCHIVE_PATH")

# ==========================================
# 1. INFER METADATA FROM FILENAME
# ==========================================
MODULE_NAME=$(echo "$FILENAME" | cut -d'_' -f1)
VERSION=$(echo "$FILENAME" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
CUDA_VER=$(echo "$FILENAME" | grep -oE 'cuda[0-9]+' | sed 's/cuda/cu/')
FULL_VERSION="${VERSION}+${CUDA_VER}"

if [[ "$FILENAME" == *"aarch64"* ]]; then
    ARCH="aarch64"
elif [[ "$FILENAME" == *"x86_64"* ]]; then
    ARCH="x86_64"
else
    echo "Error: Could not infer architecture."
    exit 1
fi

# manylinux policy to assert in strict mode (override env var if needed).
EXPECTED_POLICY="${EXPECTED_POLICY:-manylinux_2_28_${ARCH}}"

# STRICT_MANYLINUX=1 asserts exactly EXPECTED_POLICY and fails if it isn't met.
# Otherwise auditwheel auto-detects the best-fitting policy (its --plat default,
# never fails on policy), so builds on any distro keep working.
STRICT_MANYLINUX="${STRICT_MANYLINUX:-0}"
if [ "$STRICT_MANYLINUX" -eq 1 ]; then
    PLAT_ARGS=(--plat "$EXPECTED_POLICY" --only-plat)
else
    PLAT_ARGS=()
fi

# Don't graft libs provided at runtime (CUDA driver) or shipped in this wheel.
EXCLUDE_ARGS=(
    --exclude "libcuda.so.1"
    --exclude "libnccl.so.2"
)

echo "📦 Package: nvidia/$MODULE_NAME | Version: $FULL_VERSION | Arch: $ARCH"

# ==========================================
# 2. SET UP WORKSPACE & EXTRACT FILES
# ==========================================
BUILD_DIR=$(mktemp -d)

# NEW: Create the nested namespace structure (nvidia/nccl_custom)
mkdir -p "$BUILD_DIR/nvidia/$MODULE_NAME"

# Extract binaries directly into the nested folder
tar -xf "$ARCHIVE_PATH" -C "$BUILD_DIR/nvidia/$MODULE_NAME" --strip-components=1

# ==========================================
# 3. GENERATE pyproject.toml
# ==========================================
cat << EOF > "$BUILD_DIR/pyproject.toml"
[build-system]
requires = ["hatchling>=1.18"]
build-backend = "hatchling.build"

[project]
name = "$MODULE_NAME"
version = "$FULL_VERSION"
requires-python = ">=3.8"

[tool.hatch.build.targets.wheel]
packages = ["nvidia"]
EOF

# ==========================================
# 4. CREATE EMPTY PYTHON PACKAGE
# ==========================================
touch "$BUILD_DIR/nvidia/$MODULE_NAME/__init__.py"

# ==========================================
# 5. ENVIRONMENT SETUP & BUILD
# ==========================================
echo "🚀 Compiling with uv..."
OUTPUT_DIR="$PWD"
cd "$BUILD_DIR"

uv venv .venv
source .venv/bin/activate
uv pip install "build>=1.0"
# auditwheel checks/repairs/tags the wheel below. Pinned >=6.5.1 to include
# known fixes for policy selection and symbol detection.
# It needs the patchelf binary on PATH (>=0.14, per auditwheel)
uv pip install "auditwheel>=6.5.1" "patchelf>=0.14"

uv build --wheel

# ==========================================
# 6. AUDIT, REPAIR (verified manylinux tag) & CLEANUP
# ==========================================
ANY_WHEEL=$(ls dist/*any.whl)

echo "🔎 auditwheel show (pre-repair inspection):"
auditwheel show "$ANY_WHEEL"

WHEELHOUSE="$BUILD_DIR/wheelhouse"
echo "🏷️ Repairing & tagging wheel (strict=$STRICT_MANYLINUX) ..."
# Checks/repairs/tags the wheel; in strict mode fails if it can't meet the policy.
auditwheel repair \
    "${PLAT_ARGS[@]}" \
    "${EXCLUDE_ARGS[@]}" \
    -w "$WHEELHOUSE" \
    "$ANY_WHEEL"

FINAL_WHEEL=$(ls "$WHEELHOUSE"/*.whl)
mv "$FINAL_WHEEL" "$OUTPUT_DIR/"

deactivate
rm -rf "$BUILD_DIR"

echo "✅ Success! Built: $(basename "$FINAL_WHEEL")"
