#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# NCCL Examples Main Makefile

# Define all category directories
CATEGORIES := 01_communicators 02_point_to_point 03_collectives 04_user_buffer_registration 05_symmetric_memory 06_device_api

# Default target
all: $(CATEGORIES)

# Build all categories
$(CATEGORIES):
	@echo "Building $@..."
	@$(MAKE) -C $@

# Clean all categories
clean:
	@echo "Cleaning all examples..."
	@for category in $(CATEGORIES); do \
		$(MAKE) -C $$category clean; \
	done

# Test all examples
test: all
	@echo "Testing all examples..."
	@for category in $(CATEGORIES); do \
		$(MAKE) -C $$category test; \
	done

# Install all examples
install: all
	@echo "Installing all examples..."
	@for category in $(CATEGORIES); do \
		$(MAKE) -C $$category install; \
	done

# Help target
help:
	@echo "NCCL Examples Main Makefile"
	@echo "==========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build all examples"
	@echo "  clean    - Clean all build artifacts"
	@echo "  test     - Build and test all examples"
	@echo "  install  - Install all examples"
	@echo "  help     - Show this help message"
	@echo ""

.PHONY: all clean test install help $(CATEGORIES)
