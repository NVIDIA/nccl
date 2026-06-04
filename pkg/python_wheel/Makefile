#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk

BUILDDIR     ?= $(abspath ../../build)
PKGDIR       := $(BUILDDIR)/pkg/python_wheel
TXZPKGDIR    := $(BUILDDIR)/pkg/txz
WHEEL_SCRIPT := $(abspath ./build_wheel.sh)

.PHONY: default build prep clean

default: build

prep:
	$(MAKE) -C ../.. pkg.txz.build BUILDDIR=$(BUILDDIR)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build: prep
	@mkdir -p $(PKGDIR)
	@TXZ=$$(ls $(TXZPKGDIR)/nccl*.txz 2>/dev/null | head -n 1); \
	  if [ -z "$$TXZ" ]; then \
	    echo "ERROR: no nccl*.txz found in $(TXZPKGDIR)"; exit 1; \
	  fi; \
	  printf "Building Python wheel from %s\n" "$$TXZ"; \
	  cd $(PKGDIR) && bash $(WHEEL_SCRIPT) "$$TXZ"

clean:
	rm -Rf $(PKGDIR)
