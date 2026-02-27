#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
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
	${MAKE} -C docs/examples NCCL_HOME=${ABSBUILDDIR}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

nccl4py.%:
	${MAKE} -C bindings/nccl4py $* BUILDDIR=${ABSBUILDDIR}

# IR generation requires src.build first
ir.%:
	${MAKE} -C bindings/ir $* BUILDDIR=${ABSBUILDDIR}

pkg.debian.prep: lic
pkg.txz.prep: lic
