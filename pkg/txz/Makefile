#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
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
	$(MAKE) -C ../.. src.build BUILDDIR=$(BUILDDIR)
	@printf "Building tar.xz package\n"
	(cd $(BUILDDIR); bash txz/create_txz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../nccl*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@
