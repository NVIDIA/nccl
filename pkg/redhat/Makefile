#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
RPMPREPDIR := $(BUILDDIR)/redhat
PKGDIR  := $(BUILDDIR)/pkg/rpm/

RPMGEN_IN  := $(wildcard *.in)
RPMGEN     := $(RPMGEN_IN:.in=)
RPMFILES   := $(RPMGEN)
RPMTARGETS := $(patsubst %, $(RPMPREPDIR)/%, $(RPMFILES))

PKG_ARCH       ?= $(shell uname -m)

prep : $(RPMTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build : prep
	$(MAKE) -C ../.. src.build BUILDDIR=$(BUILDDIR)
	$(MAKE) -C ../txz build BUILDDIR=$(BUILDDIR)
	@printf "Building Redhat package\n"
	mkdir -p $(PKGDIR)
	rpmbuild --define "_sourcedir $(BUILDDIR)/pkg/txz" \
                 --define "_rpmdir $(PKGDIR)" \
                 --define "_builddir $(PKGDIR)/build/" \
                 --define "_buildrootdir $(PKGDIR)/buildroot/" \
                 -bb $(BUILDDIR)/redhat/nccl.spec

clean:
	rm -Rf $(RPMPREPDIR) $(PKGDIR)

$(RPMPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	sed -e "s/@NCCL_MAJOR@/$(NCCL_MAJOR)/g" \
	    -e "s/@NCCL_MINOR@/$(NCCL_MINOR)/g" \
	    -e "s/@NCCL_PATCH@/$(NCCL_PATCH)/g" \
	    -e "s/@NCCL_SUFFIX@/$(NCCL_SUFFIX)/g" \
	    -e "s/@CUDA_MAJOR@/$(CUDA_MAJOR)/g" \
	    -e "s/@CUDA_MINOR@/$(CUDA_MINOR)/g" \
	    -e "s/@PKG_REVISION@/$(PKG_REVISION)/g" \
	    -e "s/@PKG_ARCH@/$(PKG_ARCH)/g" \
	    $< > $@

$(RPMPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	cp -f $< $@
