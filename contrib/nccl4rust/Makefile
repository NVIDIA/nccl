# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

NCCL_ROOT ?= ../..
NCCL_INCLUDE_DIR ?= $(abspath $(NCCL_ROOT)/build/include)
NCCL_LIB_DIR ?= $(abspath $(NCCL_ROOT)/build/lib)
CUDA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc))))
NVCC ?= $(CUDA_HOME)/bin/nvcc
CARGO ?= cargo
ARCH ?= 90
BUILDDIR ?= build
SMOKE_DIR := tests/cuda-oxide-smoke
RUNTIME_SMOKE_DIR := tests/runtime-smoke

DEVICE_LTOIR := $(BUILDDIR)/libnccl4rust_device_sm_$(ARCH).ltoir
DEVICE_SRC := shim/nccl4rust_device.cu
DEVICE_HDR := shim/nccl4rust_device.h

.PHONY: all device check clean FORCE

all: check device

check:
	NCCL_INCLUDE_DIR="$(NCCL_INCLUDE_DIR)" \
	NCCL_LIB_DIR="$(NCCL_LIB_DIR)" \
	CUDA_HOME="$(CUDA_HOME)" \
	$(CARGO) check --workspace

device: $(DEVICE_LTOIR)

# Always rebuild: this shim inlines NCCL's versioned public device headers, and
# its output must never be reused after switching NCCL trees or architectures.
$(DEVICE_LTOIR): $(DEVICE_SRC) $(DEVICE_HDR) FORCE
	@mkdir -p $(BUILDDIR)
	$(NVCC) -std=c++17 -arch=sm_$(ARCH) -dlto -ltoir -x cu --extended-lambda \
		-I$(NCCL_INCLUDE_DIR) -Ishim \
		$(DEVICE_SRC) -o $@

FORCE:

clean:
	rm -rf $(BUILDDIR) target .oxide-artifacts \
		$(SMOKE_DIR)/target $(SMOKE_DIR)/.oxide-artifacts \
		$(RUNTIME_SMOKE_DIR)/target
	rm -f nccl_device.ll crates/nccl-device/nccl_device.ll \
		$(SMOKE_DIR)/nccl4rust_cuda_oxide_smoke.ll \
		$(SMOKE_DIR)/nccl4rust_cuda_oxide_smoke.ptx \
		$(SMOKE_DIR)/nccl4rust_cuda_oxide_smoke_sm_*.cubin
