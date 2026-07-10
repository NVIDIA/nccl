#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# NCCL Device API Examples
EXAMPLES = 01_allreduce_lsa 02_alltoall_gin 03_alltoall_hybrid

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
	@echo "NCCL Device API Examples"
	@echo "========================"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all examples"
	@echo "  clean   - Clean all build artifacts"
	@echo "  test    - Test all examples"
	@echo "  install - Install all examples"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  01_allreduce_lsa          - AllReduce collective operation"
	@echo "  02_alltoall_gin           - Pure GIN AlltoAll (network-only)"
	@echo "  03_alltoall_hybrid        - Hybrid GIN+LSA AlltoAll (optimized)"
	@echo ""
	@echo "To build/run individual examples:"
	@echo "  make -C 01_allreduce_lsa/c"
	@echo "  make -C 02_alltoall_gin/c"
	@echo "  make -C 03_alltoall_hybrid/c"

.PHONY: all clean test install help $(EXAMPLES)
