#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../../makefiles/common.mk

EXAMPLES = 01_fault_detection

all: $(EXAMPLES)

$(EXAMPLES):
	$(MAKE) -C $@/c

clean:
	for example in $(EXAMPLES); do \
		$(MAKE) -C $$example/c clean; \
	done

test: all
	for example in $(EXAMPLES); do \
		echo "Testing $$example..."; \
		$(MAKE) -C $$example/c test || exit 1; \
	done

install: all
	for example in $(EXAMPLES); do \
		echo "Installing $$example..."; \
		$(MAKE) -C $$example/c install || exit 1; \
	done

help:
	@echo "NCCL RAS Examples"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all examples"
	@echo "  clean   - Clean all build artifacts"
	@echo "  test    - Test all examples"
	@echo "  install - Install all examples"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  01_fault_detection - Interactive RAS fault-detection walkthrough"
	@echo ""
	@echo "All RAS examples require MPI."
	@echo ""
	@echo "To build/run an individual example:"
	@echo "  make -C 01_fault_detection/c"

.PHONY: all clean test install help $(EXAMPLES)
