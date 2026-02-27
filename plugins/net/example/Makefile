#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.DEFAULT_GOAL: build
include ../../../makefiles/common.mk
SRCDIR   ?= $(abspath ../../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.c)

build: ${BUILDDIR}/libnccl-net-example.so

${BUILDDIR}/libnccl-net-example.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -Inccl -fPIC -shared -o $@ $^

clean:
	rm -f ${BUILDDIR}/libnccl-net-example.so
