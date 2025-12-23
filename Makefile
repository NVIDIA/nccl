#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.PHONY: all clean

default: src.build
default: ir.build

install: src.install
BUILDDIR ?= $(abspath ./build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src pkg nccl4py ir
clean: ${TARGETS:%=%.clean}
examples.build: src.build
ir.build: src.build
LICENSE_FILES := LICENSE.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${BUILDDIR}
	install -m 644 $< $@

src.%:
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR}

examples: src.build
	${MAKE} -C examples NCCL_HOME=${ABSBUILDDIR}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

nccl4py.%:
	${MAKE} -C nccl4py $* BUILDDIR=${ABSBUILDDIR}

# IR generation requires src.build first
ir.%:
	${MAKE} -C ir $* BUILDDIR=${ABSBUILDDIR}

pkg.debian.prep: lic
pkg.txz.prep: lic
