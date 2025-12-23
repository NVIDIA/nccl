#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../makefiles/common.mk
# LLVM IR generation

ifneq ($(EMIT_LLVM_IR), 0)
# Source and target files
LLVM_SRC := nccl_device_wrapper__impl.h
BUILDDIR ?= $(abspath ../build)
OBJDIR := $(BUILDDIR)/obj/llvm_ir
LIBDIR := $(BUILDDIR)/lib

# LLVM IR output files (all aux files in obj/llvm_ir)
UNOPTIMIZED_BC := $(OBJDIR)/libnccl_device.bc.unoptimized
OPTIMIZED_BC := $(OBJDIR)/libnccl_device.bc.optimized
LLVM_IR_FILE := $(OBJDIR)/libnccl_device.ll
FINAL_BC := $(LIBDIR)/libnccl_device.bc

# Build configuration
ifeq ($(shell test "0$(CUDA_MAJOR)" -lt 12; echo $$?),0)
	BITCODE_LIB_ARCH ?= sm_70
else
	BITCODE_LIB_ARCH ?= sm_90
endif

# Notes:
# Device API code follows C++17
# Error in GIN code while compiling with c++17
BITCODE_CXX_STD ?= gnu++17

CLANG ?= clang
OPT ?= opt
LLVM_DIS ?= llvm-dis
LLVM_AS ?= llvm-as
PYTHON ?= python3

# Include paths
NCCL_INCLUDES := -I$(BUILDDIR)/include -I../src/include -I../src/include/nccl_device -I../src/device
CUDA_INCLUDES := -I$(CUDA_INC) -I$(CUDA_INC)/cccl

# Clang flags for LLVM IR generation
COMMON_CLANG_FLAGS := -std=$(BITCODE_CXX_STD) -x cuda \
               		  --cuda-path=$(CUDA_HOME) --cuda-device-only \
               		  --cuda-gpu-arch=$(BITCODE_LIB_ARCH) \
               		  $(NCCL_INCLUDES) $(CUDA_INCLUDES) \
               		  -D__clang_llvm_bitcode_lib__ \
               		  -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR)
CLANG_FLAGS := -c -emit-llvm -O1 $(COMMON_CLANG_FLAGS)

# Optimization passes
OPT_PASSES := --passes='internalize,inline,globaldce' \
              -internalize-public-api-list='*nccl*'

# Build rules
WRAPPER_HEADER := $(BUILDDIR)/include/nccl_device_wrapper.h

$(WRAPPER_HEADER): nccl_device_wrapper.h
	@mkdir -p $(BUILDDIR)/include
	@echo "Copying nccl_device_wrapper.h to build include..."
	cp -f $< $@

$(UNOPTIMIZED_BC): $(LLVM_SRC)
	@mkdir -p $(OBJDIR)
	@echo "Generating unoptimized LLVM bitcode..."
	$(CLANG) $(CLANG_FLAGS) $(LLVM_SRC) -o $@

$(OPTIMIZED_BC): $(UNOPTIMIZED_BC)
	@echo "Optimizing LLVM bitcode..."
	$(OPT) $(OPT_PASSES) $< -o $@

# The LLVM bc file by default forces flush subnormal to 1.
# Removing the flag so that the subnormal handling is preserved.
$(LLVM_IR_FILE): $(OPTIMIZED_BC)
	@echo "Generating LLVM IR..."
	$(LLVM_DIS) $< -o $@.tmp
	@echo "Cleaning LLVM IR (removing nvvm-reflect-ftz)..."
	@myVar="$$(cat $@.tmp | grep -E '!([0-9]+) = !\{[^"]*"nvvm-reflect-ftz"' | cut -d ' ' -f 1)"; \
	awk '!/nvvm-reflect-ftz/' $@.tmp | sed "/^!llvm\.module\.flags = /s/$$myVar, //" > $@
	@rm -f $@.tmp

$(FINAL_BC): $(LLVM_IR_FILE)
	@mkdir -p $(LIBDIR)
	@echo "Generating final bitcode from cleaned LLVM IR..."
	$(LLVM_AS) $< -o $@

llvm_ir: $(FINAL_BC) $(OPTIMIZED_BC) $(WRAPPER_HEADER)
	@echo "LLVM IR and bitcode generated successfully:"
	@echo "  C++ Standard: $(BITCODE_CXX_STD)"
	@echo "  GPU Architecture: $(BITCODE_LIB_ARCH)"
	@echo "  Unoptimized: $(UNOPTIMIZED_BC)"
	@echo "  Optimized: $(OPTIMIZED_BC)"
	@echo "  LLVM IR (Human-readable): $(LLVM_IR_FILE)"
	@echo "  Final BC: $(FINAL_BC)"
	@echo "  Wrapper Header: $(WRAPPER_HEADER)"
else
llvm_ir:
	@echo "LLVM IR generation disabled (EMIT_LLVM_IR=0)"

endif

build: llvm_ir

clean:
	rm -rf $(OBJDIR) $(FINAL_BC) $(WRAPPER_HEADER)

.PHONY: llvm_ir build clean
