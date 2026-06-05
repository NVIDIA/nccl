#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# NCCL Fundamental Examples
EXAMPLES = 01_ring_pattern

# Default target
all: $(EXAMPLES)

# Build individual examples
$(EXAMPLES):
	$(MAKE) -C $@/c

# Clean all build artifacts
clean:
	for example in $(EXAMPLES); do \
		$(MAKE) -C $$example/c clean; \
	done

# Test all examples
test: all
	for example in $(EXAMPLES); do \
		echo "Testing $$example..."; \
		$(MAKE) -C $$example/c test || exit 1; \
	done

# Install all examples
install: all
	for example in $(EXAMPLES); do \
		echo "Installing $$example..."; \
		$(MAKE) -C $$example/c install || exit 1; \
	done

# Help
help:
	@echo "NCCL Point to Point Examples"
	@echo "============================"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all examples"
	@echo "  clean   - Clean all build artifacts"
	@echo "  test    - Test all examples"
	@echo "  install - Install all examples"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  01_ring_pattern - Use send and receive operations to form a ring pattern"
	@echo ""
	@echo "To build/run individual examples:"
	@echo "  make -C 01_ring_pattern/c"

.PHONY: all clean test install help $(EXAMPLES)
