#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Include common build rules
include ../../../makefiles/common.mk
include ../../../makefiles/examples.mk

# Target executable
TARGET = allreduce_ub

# Common utilities
COMMON_INC = ../../common/include
COMMON_SRC = ../../common/src

# Build configuration
INCLUDES += -I$(COMMON_INC)

# Source files
SOURCES = main.cc $(COMMON_SRC)/utils.cc
OBJECTS = $(SOURCES:.cc=.o)

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
	@echo "NCCL Example: User Buffer Registration Allreduce"
	@echo "=============================================="
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test with all GPUs"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help      - Show this help"

.PHONY: all test clean install help
