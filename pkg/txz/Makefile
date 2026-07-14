#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
TXZPREPDIR  := $(BUILDDIR)/txz
PKGDIR  := $(BUILDDIR)/pkg/txz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_ARCH   := $(shell uname -m)

prep: $(TXZTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build: prep
	$(MAKE) -C ../.. BUILDDIR=$(BUILDDIR)
	@printf "Building tar.xz package\n"
	(cd $(BUILDDIR); bash txz/create_txz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../nccl*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/@NCCL_MAJOR@/$(NCCL_MAJOR)/g" \
	    -e "s/@NCCL_MINOR@/$(NCCL_MINOR)/g" \
	    -e "s/@NCCL_PATCH@/$(NCCL_PATCH)/g" \
	    -e "s/@NCCL_SUFFIX@/$(NCCL_SUFFIX)/g" \
	    -e "s/@CUDA_MAJOR@/$(CUDA_MAJOR)/g" \
	    -e "s/@CUDA_MINOR@/$(CUDA_MINOR)/g" \
	    -e "s/@PKG_REVISION@/$(PKG_REVISION)/g" \
	    -e "s/@PKG_ARCH@/$(PKG_ARCH)/g" \
	    $< > $@
