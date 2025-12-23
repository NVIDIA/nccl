UV ?= uv
# Ensure only UV-managed Python versions are used
export UV_PYTHON_PREFERENCE := only-managed
export UV_PYTHON := 3.13

# Root Makefile passes BUILDDIR; default to ../build if not set
MKFILE_DIR           := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILDDIR             ?= $(abspath $(MKFILE_DIR)/../build)
NCCL4PY_DIR          := $(abspath $(MKFILE_DIR))
NCCL4PY_BINDINGS_DIR := $(NCCL4PY_DIR)/nccl/bindings
DIST_DIR             := $(BUILDDIR)/dist

ifdef CUDA_HOME
CUDA_VERSION := $(shell $(CUDA_HOME)/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
CUDA_MAJOR := $(firstword $(subst ., ,$(CUDA_VERSION)))
CUDA_MINOR := $(word 2,$(subst ., ,$(CUDA_VERSION)))
PYTORCH_INDEX_URL := https://download.pytorch.org/whl/cu$(CUDA_MAJOR)$(CUDA_MINOR)
endif

# Supported CUDA major versions
SUPPORTED_CUDA_VERSIONS := 12 13

.DEFAULT_GOAL := build
.PHONY: all dev build clean check-cuda

all: build

check-cuda:
ifndef CUDA_HOME
	$(error CUDA_HOME is not set. Please set CUDA_HOME to your CUDA installation directory, e.g. export CUDA_HOME=/usr/local/cuda)
endif
ifeq ($(CUDA_VERSION),)
	$(error nvcc not found at $(CUDA_HOME)/bin/nvcc or unable to detect CUDA version)
endif
ifeq ($(filter $(CUDA_MAJOR),$(SUPPORTED_CUDA_VERSIONS)),)
	$(error Only CUDA $(SUPPORTED_CUDA_VERSIONS) are supported. Found CUDA $(CUDA_MAJOR))
endif
	@echo "Using CUDA $(CUDA_VERSION) at $(CUDA_HOME)"

dev: check-cuda
	@echo "Installing nccl4py with cu$(CUDA_MAJOR) extra and test dependencies..."
	cd $(NCCL4PY_DIR) && $(UV) sync --group test --extra cu$(CUDA_MAJOR)
	cd $(NCCL4PY_DIR) && $(UV) pip install -e .[cu$(CUDA_MAJOR)]
	cd $(NCCL4PY_DIR) && $(UV) pip install -i $(PYTORCH_INDEX_URL) torch
	cd $(NCCL4PY_DIR) && $(UV) pip install cupy-cuda$(CUDA_MAJOR)x

build: check-cuda
	@echo "Using Python: $$($(UV) run python --version)"
	$(UV) build --out-dir "$(DIST_DIR)" "$(NCCL4PY_DIR)"
	@echo "Wheel built successfully in $(DIST_DIR)"

clean:
	@rm -rf $(DIST_DIR) \
			$(NCCL4PY_DIR)/*.egg-info \
			$(NCCL4PY_DIR)/build \
			$(NCCL4PY_DIR)/dist
	@find $(NCCL4PY_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find $(NCCL4PY_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	@find $(NCCL4PY_BINDINGS_DIR) -type f \( -name "*.so" -o -name "*.cpp" -o -name "*.c" \) -delete 2>/dev/null || true
