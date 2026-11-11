#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.PHONY : all clean

# debian.build and txz.build both recurse into the top-level src.build; running
# them concurrently makes two sub-makes compile the same objects at once.
.NOTPARALLEL:

default : build
build : debian.build txz.build

BUILDDIR ?= $(abspath ../build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := debian txz
all:   ${TARGETS:%=%.build}
prep:  ${TARGETS:%=%.prep}
build: ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.prep:
	${MAKE} -C $* prep BUILDDIR=${ABSBUILDDIR}

%.build:
	${MAKE} -C $* build BUILDDIR=${ABSBUILDDIR}

%.clean:
	${MAKE} -C $* clean
