#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Include common build rules
include ../../../../../makefiles/common.mk
include ../../../../../makefiles/examples.mk

# Target executable
TARGET = rmsnorm_lsa

# Common utilities
COMMON_INC = ../../../common/include
COMMON_SRC = ../../../common/src
RMSNORM_INC = ../../include

# Build configuration
INCLUDES += -I$(COMMON_INC) -I$(RMSNORM_INC)

# Source files
SOURCES = main.cu utils.cc
# utils.cc is a shared helper in common/src, compiled per-example into a local
# utils.o so parallel example builds do not race on one shared object.
# vpath NAME DIR tells make to search DIR for the prerequisite NAME, so the
# %.o: %.cc rule below finds utils.cc in $(COMMON_SRC) and builds a local utils.o.
vpath utils.cc $(COMMON_SRC)
OBJECTS = $(SOURCES:.cu=.o)
OBJECTS := $(OBJECTS:.cc=.o)

# Header dependencies (rmsnorm_utils.cuh needed for main.cu)
CU_HEADERS = $(RMSNORM_INC)/rmsnorm_utils.cuh

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

# Compile CUDA files with rmsnorm header dependency
%.o: %.cu $(CU_HEADERS)
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ files
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
	@echo "NCCL Example (C): Pure LSA RMSNorm Device API"
	@echo "============================================="
	@echo ""
	@echo "This example demonstrates fused computation and communication"
	@echo "using Load Store Accessible (LSA) for RMSNorm operations."
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA 12.2 or above"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test with all GPUs"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help      - Show this help"

.PHONY: all test clean install help
