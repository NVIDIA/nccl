#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.PHONY : all clean

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
