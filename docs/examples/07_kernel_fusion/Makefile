#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# NCCL Device API Fusion of Computation/Communication Examples
EXAMPLES = 01_rmsnorm_lsa 02_rmsnorm_multimem 03_rmsnorm_gin 04_rmsnorm_hybrid

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
	@echo "  01_rmsnorm_lsa           - Pure LSA RMSNorm"
	@echo "  02_rmsnorm_multimem      - Multimem RMSNorm (SM 9.0+)"
	@echo "  03_rmsnorm_gin           - Pure GIN RMSNorm (network-only)"
	@echo "  04_rmsnorm_hybrid        - Hybrid LSA/GIN RMSNorm"
	@echo ""
	@echo "To build/run individual examples:"
	@echo "  make -C 01_rmsnorm_lsa/c"
	@echo "  make -C 02_rmsnorm_multimem/c"
	@echo "  make -C 03_rmsnorm_gin/c"
	@echo "  make -C 04_rmsnorm_hybrid/c"

.PHONY: all clean test install help $(EXAMPLES)
