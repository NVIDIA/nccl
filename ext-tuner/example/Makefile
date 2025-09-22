#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

.DEFAULT_GOAL: build
PLUGIN_SO:=libnccl-tuner-example.so
include ../../makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.c)
DST_DIR   := $(BUILDDIR)/test/unit/plugins

default: ${BUILDDIR}/$(PLUGIN_SO)

build: ${BUILDDIR}/$(PLUGIN_SO)

${BUILDDIR}/$(PLUGIN_SO): plugin.c
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -Inccl $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

# Test targets - delegate to test directory
test:
	$(MAKE) -C test test TEST_CASE=$(TEST_CASE)

test-verbose:
	$(MAKE) -C test test-verbose TEST_CASE=$(TEST_CASE)

# Build tests
test-build:
	$(MAKE) -C test all

# Optimize configurations from performance data
optimize-config:
	@if [ -z "$(CSV_FILE)" ]; then \
		echo "Usage: make optimize-config CSV_FILE=path/to/data.csv [OUTPUT=config.conf] [METRIC=latency_us]"; \
		echo "Example: make optimize-config CSV_FILE=scripts/sample_performance_data.csv"; \
		exit 1; \
	fi
	python3 scripts/optimize_config.py $(CSV_FILE) \
		$(if $(OUTPUT),-o $(OUTPUT)) \
		$(if $(METRIC),-m $(METRIC)) \
		$(if $(SIZE_RANGES),--size-ranges $(SIZE_RANGES)) \
		$(if $(DRY_RUN),--dry-run) \
		$(if $(NO_HEADER),--no-header)

clean:
	rm -f ${BUILDDIR}/$(PLUGIN_SO)
	$(MAKE) -C test clean

.PHONY: test test-verbose test-build optimize-config clean
