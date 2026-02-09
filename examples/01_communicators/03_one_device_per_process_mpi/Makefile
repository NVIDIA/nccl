#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# This examples needs to be built with MPI support
MPI = 1

# Include common build rules
include ../../../makefiles/common.mk
include ../../../makefiles/examples.mk

# Target executable
TARGET = one_device_per_process_mpi

# Source files
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

# Compile source files
%.o: %.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: $(TARGET)
	@echo "Testing $(TARGET)..."
	@echo "Running with 1 process"
	$(MPIRUN) -np 1 ./$(TARGET)
	@echo ""
	@echo "Running with 2 processes"
	$(MPIRUN) -np 2 ./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "NCCL Example: One Device per Process (MPI)"
	@echo "=========================================="
	@echo ""
	@echo "This example shows how to use ncclCommInitRank to create"
	@echo "communicators for multiple GPUs using pthreads."
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build the example (default)"
	@echo "  test          - Build and run tests with different process counts"
	@echo "  clean         - Remove build artifacts"
	@echo "  install       - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help          - Show this help"

.PHONY: all test clean install help
