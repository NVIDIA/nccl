#!/bin/bash
# Script to build NCCL documentation with automatic Python venv setup
# Usage: docs/userguide/build_docs.sh [sphinx-target]
#   Examples:
#     docs/userguide/build_docs.sh html
#     docs/userguide/build_docs.sh linkcheck

set -e

# Determine script and repository root directories
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
DOC_DIR="$REPO_ROOT/docs/userguide"
VENV_DIR="$DOC_DIR/.venv"
REQUIREMENTS="$DOC_DIR/requirements.txt"
# Determine build target (default to pkg.doc.build)
BUILD_TARGET="${1:-pkg.doc.build}"

# Check if doc directory exists
if [ ! -d "$DOC_DIR" ]; then
    echo "Error: Documentation directory not found at $DOC_DIR"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: requirements.txt not found at $REQUIREMENTS"
    exit 1
fi

echo "========================================"
echo "NCCL Documentation Build"
echo "========================================"
echo "Venv directory: $VENV_DIR"
echo "Build target: $BUILD_TARGET"
echo ""

# Create venv if it doesn't exist or if requirements have changed
VENV_NEEDS_UPDATE=0
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating new environment..."
    VENV_NEEDS_UPDATE=1
elif [ "$REQUIREMENTS" -nt "$VENV_DIR/bin/activate" ]; then
    echo "Requirements file is newer than venv. Updating environment..."
    VENV_NEEDS_UPDATE=1
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

if [ $VENV_NEEDS_UPDATE -eq 1 ]; then
    # Remove old venv if it exists
    if [ -d "$VENV_DIR" ]; then
        echo "Removing old virtual environment..."
        rm -rf "$VENV_DIR"
    fi

    echo "Creating Python virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"

    # Activate venv
    source "$VENV_DIR/bin/activate"

    # Upgrade pip and install requirements
    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing dependencies from $REQUIREMENTS..."
    pip install -r "$REQUIREMENTS"

    echo "Virtual environment setup complete!"
else
    # Just activate the existing venv
    source "$VENV_DIR/bin/activate"
fi

echo ""
echo "========================================"
echo "Building Documentation"
echo "========================================"
echo "Python: $(which python)"
echo "Sphinx: $(which sphinx-build)"
echo "Sphinx version: $(sphinx-build --version)"
echo ""

MAKE_DIR="$DOC_DIR"
echo "$BUILD_TARGET" | grep \\. && MAKE_DIR="$REPO_ROOT"

# Build the documentation
echo "Running: make -C $MAKE_DIR $BUILD_TARGET"
make -C "$MAKE_DIR" "$BUILD_TARGET"

# Check if build was successful
BUILD_EXIT_CODE=$?

# Deactivate venv
deactivate

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Documentation build complete!"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Documentation build FAILED!"
    echo "========================================"
    exit $BUILD_EXIT_CODE
fi

