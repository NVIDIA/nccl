#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.DEFAULT_GOAL: build
include ../../../makefiles/common.mk
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)
DST_DIR  := $(BUILDDIR)

# Host sources build with $(CXX); the device-hook TU (plugin_dev.cu) builds with
# nvcc. It needs relocatable device code so the __device__ hook has a real,
# callable global address for NCCL's kernel to invoke it; the final link goes
# through nvcc to perform the device link.
CC_SRC := $(wildcard *.cc)
CU_SRC := $(wildcard *.cu)
CC_OBJ := $(CC_SRC:%.cc=${DST_DIR}/%.o)
CU_OBJ := $(CU_SRC:%.cu=${DST_DIR}/%.o)

build: ${DST_DIR}/libnccl-profiler-example.so

${DST_DIR}/%.o: %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) -Inccl -I${CUDA_INC} -fPIC -c -o $@ $<

${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) $(NVCC_GENCODE) --relocatable-device-code=true -std=c++17 -Inccl -I${CUDA_INC} -Xcompiler -fPIC -c -o $@ $<

${DST_DIR}/libnccl-profiler-example.so: ${CC_OBJ} ${CU_OBJ}
	@printf "Linking    %-35s\n" $@
	$(NVCC) $(NVCC_GENCODE) --relocatable-device-code=true -Xcompiler -fPIC -shared -o $@ $^

clean:
	rm -f ${DST_DIR}/*.o ${DST_DIR}/libnccl-profiler-example.so
