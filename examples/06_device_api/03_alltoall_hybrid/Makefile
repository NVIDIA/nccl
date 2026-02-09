#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Include common build rules
include ../../../makefiles/common.mk
include ../../../makefiles/examples.mk

# Target executable
TARGET = alltoall_hybrid

# Common utilities
COMMON_INC = ../../common/include
COMMON_SRC = ../../common/src

# Build configuration
INCLUDES += -I$(COMMON_INC)

# Source files
SOURCES = main.cu $(COMMON_SRC)/utils.cc
OBJECTS = $(SOURCES:.cu=.o)
OBJECTS := $(OBJECTS:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
else
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -lpthread -o $@
endif
	@echo "Built target $@"

# Compile source files
%.o: %.cu
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cc
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
else
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
endif

# Test target
test: $(TARGET)
	@echo "Testing $(TARGET)..."
ifeq ($(MPI),1)
	@echo "Running with 2 processes"
	$(MPIRUN) -np 2 ./$(TARGET)
else
	@echo "Running with all available GPUs"
	./$(TARGET)
endif

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "NCCL Example: Hybrid AlltoAll Device API"
	@echo "========================================="
	@echo ""
	@echo "This example demonstrates hybrid communication combining"
	@echo "GPU-Initiated Networking (GIN) for remote peers with"
	@echo "Load Store Access (LSA) for local peers."
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test with all GPUs"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help      - Show this help"

.PHONY: all test clean install help
