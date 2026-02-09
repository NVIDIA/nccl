#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# NCCL Device API Examples
EXAMPLES = 01_allreduce_lsa 02_alltoall_gin 03_alltoall_hybrid

# Default target
all: $(EXAMPLES)

# Build individual examples
$(EXAMPLES):
	$(MAKE) -C $@

# Clean all build artifacts
clean:
	for example in $(EXAMPLES); do \
		$(MAKE) -C $$example clean; \
	done

# Test all examples
test: all
	for example in $(EXAMPLES); do \
		echo "Testing $$example..."; \
		$(MAKE) -C $$example test; \
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
	@echo "  help    - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  01_allreduce_lsa          - AllReduce collective operation"
	@echo "  02_alltoall_gin           - Pure GIN AlltoAll (network-only)"
	@echo "  03_alltoall_hybrid        - Hybrid GIN+LSA AlltoAll (optimized)"
	@echo ""
	@echo "To build/run individual examples:"
	@echo "  make -C 01_allreduce_lsa"
	@echo "  make -C 02_alltoall_gin"
	@echo "  make -C 03_alltoall_hybrid"

.PHONY: all clean test help $(EXAMPLES)
