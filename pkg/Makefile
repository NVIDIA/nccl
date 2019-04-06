#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
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
