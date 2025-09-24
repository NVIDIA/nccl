#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Include common build rules
include ../../../makefiles/common.mk
include ../../../makefiles/examples.mk

# Target executable
TARGET = multiple_devices_single_process

# Source files
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: $(TARGET)
	@echo "Testing $(TARGET)..."
	@echo "Running with all available GPUs"
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "NCCL Example: Multiple Devices Single Process"
	@echo "=============================================="
	@echo ""
	@echo "This example shows how to use ncclCommInitAll to create"
	@echo "communicators for multiple GPUs in a single process."
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test with all GPUs"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help      - Show this help"

.PHONY: all test clean install help
