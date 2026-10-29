#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# nccl IR ABI tests
# 	To add a new test:
#		1. Add a kernel that calls the ncclIr* functions (nccl_device_ir.h) as devapi_<name>_ir.cu in bindings/ir/tests.
#		2. Add a host program devapi_<name>.cu in bindings/ir/tests. With -DUSE_IR it loads devapi_<name>_ir.cubin; with -DUSE_LTO_IR it loads devapi_<name>.ltoir.
#		3. Add <name> to TESTS
#
# 	To run all tests:
#   	make src.build ir.build NCCL_EMIT_LLVM_IR=1 NCCL_EMIT_LTO_IR=1   	once, from the repo root to build libnccl_device.bc and libnccl_device.ltoir
#   	make -C bindings/ir/tests            							build devapi_<name>_ir and devapi_<name>_ltoir executable for each name
#   	make -C bindings/ir/tests run        							mpirun -np 2 each of those programs

TESTS := barrier barrier_lsa data_ring data_ring2 reduce_copy signal_ring

ARCH         ?= sm_90
NVCC_GENCODE ?= -gencode=arch=compute_$(ARCH:sm_%=%),code=$(ARCH)
include ../../../makefiles/common.mk
ROOT      := $(abspath ../../..)
BUILDDIR  ?= $(ROOT)/build
DST       := $(BUILDDIR)/test/ir
MPI_HOME  ?= $(abspath $(dir $(shell which mpicc))..)
CLANG     ?= clang
LLVM_LINK ?= llvm-link
OPT       ?= opt
LLC       ?= llc
PTX       ?= ptx86
PTXAS     ?= ptxas
NVLINK    ?= $(CUDA_HOME)/bin/nvlink
OPT_PASSES := default<O3>,wholeprogramdevirt,inline,mem2reg,function-attrs,ipsccp,globalopt,globaldce,instcombine,loop-unroll,simplifycfg

# Relevant headers and libraries to use when compiling tests
HDRS := ../nccl_device_ir.h $(wildcard *.h)
DEV_INC   := -I.. -I$(BUILDDIR)/include -I$(CUDA_INC) -I$(CUDA_INC)/cccl
HOST_INC  := -I.. -I$(BUILDDIR)/include -I$(ROOT)/src/include -I$(ROOT)/test/unit -I$(MPI_HOME)/include
HOST_LIBS := -L$(BUILDDIR)/lib -lnccl $(NVLDFLAGS) -L$(MPI_HOME)/lib -lmpi -lcuda

# ---- actual make commands

# For every test, we want to build $(DST)/devapi_<name>_ir and $(DST)/devapi_<name>_ltoir
all: $(foreach test,$(TESTS),$(DST)/devapi_$(test)_ir $(DST)/devapi_$(test)_ltoir)

# Each devapi_* test is executed with `mpirun -np 2`
run: all
	@set -e; for name in $(TESTS); do \
	  echo "=== $$name"; \
	  mpirun -np 2 $(DST)/devapi_$${name}_ir; \
	  mpirun -np 2 $(DST)/devapi_$${name}_ltoir; \
	done

clean:
	rm -rf $(DST)

.SECONDARY:		# Keeps devapi_<name>_ir.cubin and devapi_<name>.ltoir alive at runtime

# LLVM output: Turns devapi_<name>_ir.cu + libnccl_device.bc -> devapi_<name>_ir.cubin
# 	clang emits LLVM bitcode
# 	llvm link with libnccl_device.bc
#	optimization pass
# 	nvlink to get the final cubin
$(DST)/%_ir.cubin: %_ir.cu $(HDRS) $(BUILDDIR)/lib/libnccl_device.bc
	mkdir -p $(DST)
	$(CLANG) -c -emit-llvm -std=c++17 -x cuda --cuda-path=$(CUDA_HOME) --cuda-device-only \
	  --cuda-gpu-arch=$(ARCH) $(DEV_INC) $< -o $(DST)/$*_ir.bc
	$(LLVM_LINK) --only-needed $(DST)/$*_ir.bc $(BUILDDIR)/lib/libnccl_device.bc -o $(DST)/$*_ir.linked.bc
	$(OPT) -passes='$(OPT_PASSES)' $(DST)/$*_ir.linked.bc -S -o $(DST)/$*_ir.ll
	$(LLC) -O3 --disable-tail-calls -mcpu=$(ARCH) -mattr=$(PTX) $(DST)/$*_ir.ll -o $(DST)/$*_ir.ptx
	$(PTXAS) -c -arch=$(ARCH) $(DST)/$*_ir.ptx -o $(DST)/$*_ir.unlinked.cubin
	$(NVLINK) -arch=$(ARCH) $(DST)/$*_ir.unlinked.cubin -o $@

# LTOIR output: devapi_<name>_ir.cu -> devapi_<name>.ltoir
# 	nvcc emits LTOIR
$(DST)/%.ltoir: %_ir.cu $(HDRS)
	mkdir -p $(DST)
	$(NVCC) -std=c++17 -arch=$(ARCH) -dlto --ltoir $(DEV_INC) $< -o $@
# 	copy libnccl_device.ltoir to the right place
$(DST)/libnccl_device.ltoir: $(BUILDDIR)/lib/libnccl_device.ltoir
	mkdir -p $(DST)
	cp -f $< $@

# nvcc compiles host program devapi_<name>.cu -> devapi_<name>_ir and devapi_<name>_ltoir
$(DST)/devapi_%_ir: devapi_%.cu $(HDRS) $(DST)/devapi_%_ir.cubin
	$(NVCC) $(NVCUFLAGS) $(HOST_INC) -DUSE_IR $< -o $@ $(HOST_LIBS)

$(DST)/devapi_%_ltoir: devapi_%.cu $(HDRS) $(DST)/devapi_%.ltoir $(DST)/libnccl_device.ltoir
	$(NVCC) $(NVCUFLAGS) $(HOST_INC) -DUSE_LTO_IR -DLTO_IR_ARCH=\"$(ARCH)\" $< -o $@ $(HOST_LIBS) -lnvJitLink

.PHONY: all run clean
