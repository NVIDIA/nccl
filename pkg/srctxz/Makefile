#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
TXZPREPDIR  := $(BUILDDIR)/srctxz
PKGDIR  := $(BUILDDIR)/pkg/srctxz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_REVISION   ?= 3
PKG_ARCH       := $(shell uname -m)

prep: $(TXZTARGETS)

build: prep
	$(MAKE) -C ../../src clean
	@printf "Building source tar.xz package\n"
	(cd $(BUILDDIR); SRCTXZ_APITESTS=$(SRCTXZ_APITESTS) bash srctxz/create_srctxz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../../nccl-src*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    $< > $@
