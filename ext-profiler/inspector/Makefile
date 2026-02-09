# Standalone Makefile for NCCL Inspector (independent from main NCCL)
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Compiler settings
CXX ?= g++
NVCC ?= nvcc

# Build options
DEBUG ?= 0
ASAN ?= 0
UBSAN ?= 0

# CUDA paths - default to /usr/local/cuda or require explicit CUDA_HOME
CUDA_HOME ?= $(shell \
  if [ -d /usr/local/cuda ]; then echo /usr/local/cuda; \
  else echo "ERROR: CUDA installation not found. Please set CUDA_HOME." >&2; exit 1; fi)
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64

# Detect CUDA version
NVCC := $(CUDA_HOME)/bin/nvcc
CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)

# CUDA 13.0 requires c++17
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
  CXXSTD ?= -std=c++17
else
  CXXSTD ?= -std=c++14
endif

# Build directories
OBJDIR := obj
SRCDIR := .

# Compiler flags
CXXFLAGS := $(CXXSTD) -fPIC -Wall -Wextra -O2
INCLUDES := -I$(CUDA_INC) -I$(SRCDIR) -I$(SRCDIR)/nccl
LDFLAGS := -shared -L$(CUDA_LIB)
LIBS := -lcudart -ldl -lpthread

# Debug flags
ifneq ($(DEBUG), 0)
CXXFLAGS += -g -DDEBUG
endif

# Sanitizer flags
ifneq ($(ASAN), 0)
CXXFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address -static-libasan
endif

ifneq ($(UBSAN), 0)
CXXFLAGS += -fsanitize=undefined
LDFLAGS += -fsanitize=undefined -static-libubsan
endif

# Source files
SRCS := inspector.cc \
	inspector_prom.cc \
	inspector_plugin.cc \
	inspector_cudawrap.cc \
	json.cc \
	version.cc

OBJS := $(SRCS:%.cc=$(OBJDIR)/%.o)

# Target library
TARGET := libnccl-profiler-inspector.so

# Build rules
.PHONY: all clean help check-cuda

all: check-cuda $(TARGET)

# Check that CUDA installation exists
check-cuda:
	@if [ ! -d "$(CUDA_HOME)" ]; then \
		echo "ERROR: CUDA installation not found at $(CUDA_HOME)"; \
		echo "Please install CUDA or set CUDA_HOME to the correct path"; \
		echo "Example: make CUDA_HOME=/path/to/cuda"; \
		exit 1; \
	fi
	@if [ ! -f "$(CUDA_INC)/cuda.h" ]; then \
		echo "ERROR: cuda.h not found at $(CUDA_INC)/cuda.h"; \
		echo "Please check your CUDA installation"; \
		exit 1; \
	fi
	@if [ ! -d "$(CUDA_LIB)" ]; then \
		echo "ERROR: CUDA library directory not found at $(CUDA_LIB)"; \
		echo "Please check your CUDA installation"; \
		exit 1; \
	fi
	@if ! ls $(CUDA_LIB)/libcudart.so* >/dev/null 2>&1; then \
		echo "ERROR: libcudart.so not found in $(CUDA_LIB)"; \
		echo "Please check your CUDA installation"; \
		exit 1; \
	fi
	@echo "Using CUDA installation at: $(CUDA_HOME)"

$(TARGET): $(OBJDIR) $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
	@echo "Built standalone NCCL Inspector: $@"

$(OBJDIR)/%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/version.o: version.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

version.cc:
	@echo "// Auto-generated version info" > version.cc
	@echo "#ifdef __cplusplus" >> version.cc
	@echo "extern \"C\" {" >> version.cc
	@echo "#endif" >> version.cc
	@echo "const char* get_git_version_info() {" >> version.cc
	@if command -v git >/dev/null 2>&1; then \
		echo "  return \"$(shell git describe --always --dirty 2>/dev/null || echo 'unknown')\"; " >> version.cc; \
	else \
		echo "  return \"standalone-build\"; " >> version.cc; \
	fi
	@echo "}" >> version.cc
	@echo "#ifdef __cplusplus" >> version.cc
	@echo "}" >> version.cc
	@echo "#endif" >> version.cc

$(OBJDIR):
	mkdir -p $(OBJDIR)


clean:
	rm -rf $(OBJDIR) $(TARGET) version.cc

# Help target
help:
	@echo "Standalone NCCL Inspector Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the inspector library (default)"
	@echo "  clean   - Clean build artifacts"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1   - Enable debug build"
	@echo "  ASAN=1    - Enable address sanitizer"
	@echo "  UBSAN=1   - Enable undefined behavior sanitizer"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_HOME - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  CXX       - C++ compiler (default: g++)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make DEBUG=1"
	@echo "  make CUDA_HOME=/path/to/cuda"
