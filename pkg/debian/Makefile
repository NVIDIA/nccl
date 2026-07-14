#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
DEBPREPDIR := $(BUILDDIR)/debian
PKGDIR  := $(BUILDDIR)/pkg/deb/

DEBGEN_IN  := $(wildcard *.in)
DEBGEN     := $(DEBGEN_IN:.in=)
DEBFILES   := compat copyright libnccl-dev.install rules $(DEBGEN)
DEBTARGETS := $(patsubst %, $(DEBPREPDIR)/%, $(DEBFILES))

PKG_TIMESTAMP  := $(shell date -R)
PKG_ARCH       ?= $(shell dpkg-architecture -qDEB_HOST_ARCH)
PKG_MULTIARCH  ?= $(shell dpkg-architecture -qDEB_HOST_MULTIARCH)

prep : $(DEBTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build : prep
	$(MAKE) -C ../.. src.build BUILDDIR=$(BUILDDIR)
	@printf "Building Debian package\n"
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b -Zxz)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../libnccl*.deb $(PKGDIR)/

clean:
	rm -Rf $(DEBPREPDIR) $(PKGDIR)

$(DEBPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	sed -e "s/@NCCL_MAJOR@/$(NCCL_MAJOR)/g" \
	    -e "s/@NCCL_MINOR@/$(NCCL_MINOR)/g" \
	    -e "s/@NCCL_PATCH@/$(NCCL_PATCH)/g" \
	    -e "s/@NCCL_SUFFIX@/$(NCCL_SUFFIX)/g" \
	    -e "s/@CUDA_MAJOR@/$(CUDA_MAJOR)/g" \
	    -e "s/@CUDA_MINOR@/$(CUDA_MINOR)/g" \
	    -e "s/@PKG_REVISION@/$(PKG_REVISION)/g" \
	    -e "s/@PKG_TIMESTAMP@/$(PKG_TIMESTAMP)/g" \
	    -e "s/@PKG_ARCH@/$(PKG_ARCH)/g" \
	    -e "s/@PKG_MULTIARCH@/$(PKG_MULTIARCH)/g" \
	    $< > $@

$(DEBPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	cp -f $< $@
