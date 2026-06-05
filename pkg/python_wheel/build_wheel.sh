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
    PLATFORM_TAG="manylinux_2_18_aarch64"
elif [[ "$FILENAME" == *"x86_64"* ]]; then
    PLATFORM_TAG="manylinux_2_18_x86_64"
else
    echo "Error: Could not infer architecture."
    exit 1
fi

echo "📦 Package: nvidia/$MODULE_NAME | Version: $FULL_VERSION | Target: $PLATFORM_TAG"

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

uv build --wheel

# ==========================================
# 6. RETAG & CLEANUP
# ==========================================
echo "🏷️ Re-tagging wheel for $PLATFORM_TAG..."
ANY_WHEEL=$(ls dist/*any.whl)

wheel tags --platform-tag "$PLATFORM_TAG" "$ANY_WHEEL"
rm "$ANY_WHEEL"

FINAL_WHEEL=$(ls dist/*.whl)
mv "$FINAL_WHEEL" "$OUTPUT_DIR/"

deactivate
rm -rf "$BUILD_DIR"

echo "✅ Success! Built: $(basename "$FINAL_WHEEL")"
