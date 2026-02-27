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

SRC_FILES := $(wildcard *.cc)
DST_DIR   := $(BUILDDIR)
OBJ_FILES := $(SRC_FILES:%.cc=${DST_DIR}/%.o)
DEP_FILES := $(OBJ_FILES:%.o=%.dep)

build: ${DST_DIR}/libnccl-profiler-example.so

${DST_DIR}/libnccl-profiler-example.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) -Inccl -I${CUDA_INC} -fPIC -shared -o $@ $^

clean:
	rm -f ${DST_DIR}/libnccl-profiler-example.so
