#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
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

PKG_TIMESTAMP  := $(shell date -R)
ARCH           := $(shell uname -m)
PKG_ARCH       ?= $(shell uname -m)
PKG_MULTIARCH  ?= $(shell $(CXX) -print-multiarch)
ifeq ($(PKG_MULTIARCH),)
# Hardwire the PKG_MULTIARCH directory as the RHEL6 distribution agnostic compiler (gcc 4.8.3) doesn't set it
PKG_MULTIARCH  := $(ARCH)-linux-gnu
endif

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

$(RPMPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(RPMPREPDIR)
	cp -f $< $@
