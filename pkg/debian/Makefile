#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
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
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../libnccl*.deb $(PKGDIR)/

clean:
	rm -Rf $(DEBPREPDIR) $(PKGDIR)

$(DEBPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Timestamp}/$(PKG_TIMESTAMP)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    -e "s/\$${pkg:MultiArch}/$(PKG_MULTIARCH)/g" \
	    $< > $@

$(DEBPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	cp -f $< $@
