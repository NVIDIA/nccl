#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Include common build rules
include ../../makefiles/common.mk

# NCCL Fundamental Examples
EXAMPLES = 01_multiple_devices_single_process 02_one_device_per_pthread

ifeq ($(MPI), 1)
EXAMPLES += 03_one_device_per_process_mpi
endif

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
ifneq ($(MPI),1)
		$(MAKE) -C 03_one_device_per_process_mpi clean
endif

# Test all examples
test: all
	for example in $(EXAMPLES); do \
		echo "Testing $$example..."; \
		$(MAKE) -C $$example test || exit 1; \
	done

# Help
help:
	@echo "NCCL Communicator Init Examples"
	@echo "==============================="
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all examples"
	@echo "  clean   - Clean all build artifacts"
	@echo "  test    - Test all examples"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  01_multiple_devices_single_process - Create communicators using multiple GPUs in a single thread"
	@echo "  02_one_device_per_pthread - Create communicators using one GPU per thread"
	@echo "  03_one_device_per_process_mpi - Create communicators using one GPU per MPI process"
	@echo ""
	@echo "To build/run individual examples:"
	@echo "  make -C 01_multiple_devices_single_process"

.PHONY: all clean test help $(EXAMPLES)
