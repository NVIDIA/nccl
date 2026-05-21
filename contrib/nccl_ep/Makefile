# NCCL_EP: Expert Parallelism extension for NCCL
#
# Build with: make -C contrib/nccl_ep
# Requires NCCL to be built first: make src.build

# std::optional and other C++17 features used in nccl_moe.cc
CXXSTD := -std=c++17
include ../../makefiles/common.mk

# Check that NVCC_GENCODE does not include compute capabilities below sm_90
# Extract all sm_XX numbers and check if any is < 90
SM_NUMBERS := $(shell echo '$(NVCC_GENCODE)' | grep -oE 'sm_([0-9]+)' | sed 's/sm_//' | sort -u)
HAS_SM_BELOW_90 := $(shell for sm in $(SM_NUMBERS); do if [ $$sm -lt 90 ]; then echo yes; break; fi; done)
ifeq ($(HAS_SM_BELOW_90),yes)
  $(error ERROR: NVCC_GENCODE for NCCL_EP must include at least sm_90 (compute capability 9.0). Found incompatible compute capability. Current value: $(NVCC_GENCODE))
endif

.PHONY: all clean lib ep_test ep_bench

BUILDDIR ?= $(abspath ../../build)
OBJDIR := $(BUILDDIR)/obj/nccl_ep
LIBDIR := $(BUILDDIR)/lib
INCDIR := $(BUILDDIR)/include
DST_DIR := $(BUILDDIR)/test/nccl_ep
JIT_DIR := $(INCDIR)/nccl_ep
JIT_CONFIG_HEADER := $(JIT_DIR)/config.h

# NVCC JIT support is required for HT mode.
NCCL_EP_JIT_HEADERS := \
    device/hybrid_ep.cuh \
    device/device_primitives.cuh \
    device/hybridep_configs.cuh \
    device/cuda_compat_shims.cuh \
    device/scan_kernel.cuh \
    device/macros.cuh
NCCL_EP_JIT_COMMON_HEADER := $(JIT_DIR)/common.hpp
NCCL_EP_JIT_HEADER_TARGETS := $(NCCL_EP_JIT_HEADERS:%=$(JIT_DIR)/%) $(NCCL_EP_JIT_COMMON_HEADER)
NCCL_EP_JIT_DEFS := -include $(abspath $(JIT_CONFIG_HEADER))

# Include paths - add current directory so common.hpp can find device/ep headers
# Include NCCL source headers for GIN support
INCFLAGS = -I. -I./include -I./device -I$(BUILDDIR)/include -I$(BUILDDIR)/include/nccl_device -I../../src/include

# GIN (Global Input Network) support - adds multinode RDMA capabilities


# Rebuild NVCUFLAGS with our NVCC_GENCODE (after INCFLAGS is complete)
# Add --expt-relaxed-constexpr for std::max, std::min, etc. in device code
# CUDA code MUST use -fvisibility=hidden for kernel registration to work correctly.
# The ncclEp* API is in nccl_ep.cc (compiled with CXXFLAGS), so it still gets exported.
NVCUFLAGS := $(NVCC_GENCODE) --extended-lambda $(INCFLAGS) --expt-relaxed-constexpr --compiler-options "-fPIC -fvisibility=hidden" -ccbin $(CXX) $(CXXSTD)
CXXFLAGS += $(INCFLAGS) $(NCCL_EP_JIT_DEFS) -I$(CUDA_INC) -I$(CUDA_INC)/cccl -fPIC -fvisibility=default

# Optional warp-level timing instrumentation (adds cudaMalloc + cudaMemcpy per kernel launch).
# Usage: make WARP_TIMING=1
# Define on both the device build (NVCUFLAGS) and the host JIT compiler driver
# (CXXFLAGS), since the latter forwards the macro to JIT-compiled device code.
ifeq ($(WARP_TIMING),1)
NVCUFLAGS += -DHYBRIDEP_ENABLE_WARP_TIMING
CXXFLAGS  += -DHYBRIDEP_ENABLE_WARP_TIMING
endif

# Dependency generation flags for automatic header tracking
DEPFLAGS = -MMD -MP

# Source files for library
HOST_SRC := nccl_ep.cc
DEVICE_SRC := device/low_latency.cu \
              device/hybridep_adapter.cu
HOST_SRC += device/jit/jit_utils.cc \
            device/jit/jit_cache.cc \
            device/jit/nvcc_compiler.cc \
            device/jit/jit_runtime.cc
# Object files
HOST_OBJ := $(HOST_SRC:%.cc=$(OBJDIR)/%.o)
DEVICE_OBJ := $(DEVICE_SRC:%.cu=$(OBJDIR)/%.o)
ALL_OBJ := $(HOST_OBJ) $(DEVICE_OBJ)

# Device glue object
DEVGLUE_OBJ := $(OBJDIR)/device_glue.o

# Library version (single source of truth: include/nccl_ep.h).
# Use `.` instead of `#` to dodge both make's comment scanner and gawk's
# `\#` warning; the trailing space anchors the match tightly enough.
NCCL_EP_MAJOR := $(shell awk '/^.define NCCL_EP_MAJOR / {print $$3}' include/nccl_ep.h)
NCCL_EP_MINOR := $(shell awk '/^.define NCCL_EP_MINOR / {print $$3}' include/nccl_ep.h)
NCCL_EP_PATCH := $(shell awk '/^.define NCCL_EP_PATCH / {print $$3}' include/nccl_ep.h)

# Library output
LIBNAME := libnccl_ep.a
LIBTARGET := $(LIBDIR)/$(LIBNAME)

# Shared library output:
#   libnccl_ep.so.MAJOR.MINOR.PATCH (actual file)
#   libnccl_ep.so.MAJOR             (SONAME; linker symlink)
#   libnccl_ep.so                   (unversioned symlink for build/runtime loading)
SOLIBNAME    := libnccl_ep.so
SOLIBSONAME  := $(SOLIBNAME).$(NCCL_EP_MAJOR)
SOLIBVERSION := $(SOLIBNAME).$(NCCL_EP_MAJOR).$(NCCL_EP_MINOR).$(NCCL_EP_PATCH)
SOLIBTARGET  := $(LIBDIR)/$(SOLIBVERSION)
SOLIBLINKS   := $(LIBDIR)/$(SOLIBSONAME) $(LIBDIR)/$(SOLIBNAME)

# Header installation
HEADERS := include/nccl_ep.h include/ep_enums.h
HEADER_TARGETS := $(HEADERS:include/%=$(INCDIR)/%)

# Build targets
all: lib ep_test ep_bench

lib: $(LIBTARGET) $(SOLIBTARGET) $(SOLIBLINKS) $(HEADER_TARGETS) $(NCCL_EP_JIT_HEADER_TARGETS)

# Compile host code
$(OBJDIR)/%.o: %.cc $(JIT_CONFIG_HEADER)
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# Compile device code (with automatic header dependency generation)
$(OBJDIR)/%.o: %.cu
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) $(DEPFLAGS) -dc $< -o $@

# Device link
$(DEVGLUE_OBJ): $(DEVICE_OBJ)
	@echo "Device linking"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@ $(LDFLAGS)

# Create static library
$(LIBTARGET): $(ALL_OBJ) $(DEVGLUE_OBJ) | $(NCCL_EP_JIT_HEADER_TARGETS)
	@echo "Creating library $@"
	@mkdir -p $(dir $@)
	ar rcs $@ $^

# Create shared library (for Python ctypes)
# Links against libnccl.so for NCCL symbols (ncclCommCuDevice, etc.)
# Uses rpath so the library can find libnccl.so at runtime
# Embeds SONAME=libnccl_ep.so.MAJOR so consumers record the major-versioned name.
$(SOLIBTARGET): $(ALL_OBJ) $(DEVGLUE_OBJ) | $(NCCL_EP_JIT_HEADER_TARGETS)
	@echo "Creating shared library $@"
	@mkdir -p $(dir $@)
	$(NVCC) -shared $(NVCC_GENCODE) -Xlinker -soname=$(SOLIBSONAME) -o $@ $^ -L$(LIBDIR) -lnccl -lcuda -Xlinker -rpath -Xlinker $(LIBDIR) $(LDFLAGS)

# Versioned-soname and unversioned symlinks (libnccl_ep.so.MAJOR, libnccl_ep.so).
$(LIBDIR)/$(SOLIBSONAME): $(SOLIBTARGET)
	@ln -sf $(SOLIBVERSION) $@
$(LIBDIR)/$(SOLIBNAME): $(LIBDIR)/$(SOLIBSONAME)
	@ln -sf $(SOLIBSONAME) $@

# Install headers
$(INCDIR)/%.h: include/%.h
	@echo "Installing $<"
	@mkdir -p $(dir $@)
	cp $< $@

$(INCDIR)/%.hpp: include/%.hpp
	@echo "Installing $<"
	@mkdir -p $(dir $@)
	cp $< $@

$(JIT_CONFIG_HEADER):
	@echo "Generating $@"
	@mkdir -p $(dir $@)
	@printf '%s\n' '#pragma once' > $@
	@printf '#define NCCL_EP_JIT_SOURCE_DIR "%s"\n' "$(abspath $(JIT_DIR))" >> $@
	@printf '#define NCCL_EP_JIT_BUILD_INCLUDE_DIR "%s"\n' "$(abspath $(INCDIR))" >> $@
	@printf '#define NCCL_EP_JIT_CUDA_INCLUDE_DIR "%s"\n' "$(abspath $(CUDA_INC))" >> $@

$(JIT_DIR)/common.hpp: include/common.hpp
	@echo "Installing JIT header $<"
	@mkdir -p $(dir $@)
	cp $< $@

$(JIT_DIR)/%: %
	@echo "Installing JIT header $<"
	@mkdir -p $(dir $@)
	cp $< $@

# Include generated dependency files (the - prefix silences errors if files don't exist yet)
-include $(ALL_OBJ:.o=.d)

# Detect CUPTI headers at compile time (optional; enables kernel-level GPU timing)
CUPTI_AVAILABLE := $(shell echo '\#include <cupti.h>' | $(CXX) -x c++ -E - >/dev/null 2>&1 && echo 1)
ifeq ($(CUPTI_AVAILABLE),1)
  CUPTI_CFLAGS  := -DHAVE_CUPTI
  CUPTI_LDFLAGS := -lcupti
else
  $(info NOTE: CUPTI headers not found — ep_bench will build without kernel-level timing.)
  $(info       Ensure CUDA Toolkit with CUPTI headers is installed, or set CPATH accordingly.)
  CUPTI_CFLAGS  :=
  CUPTI_LDFLAGS :=
endif

# Build ep_test and ep_bench (require MPI)
ifeq ($(MPI), 1)
MPI_CFLAGS := -DMPI_SUPPORT -I$(MPI_HOME)/include
MPI_LDFLAGS := -L$(MPI_HOME)/lib -lmpi

EP_TEST_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_TEST_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda

# ep_bench: performance benchmark (NVTX for profiling, optional CUPTI for kernel timing)
EP_BENCH_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) $(CUPTI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_BENCH_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda $(CUPTI_LDFLAGS)
ep_test: lib $(DST_DIR)/ep_test
ep_bench: lib $(DST_DIR)/ep_bench

$(DST_DIR)/ep_test.o: ep_test.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_TEST_NVCUFLAGS) -c $<

$(DST_DIR)/ep_test: $(DST_DIR)/ep_test.o $(SOLIBTARGET)
	@printf "Linking    %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_TEST_NVCUFLAGS) $< $(EP_TEST_NVLDFLAGS)

$(DST_DIR)/ep_bench.o: ep_bench.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_BENCH_NVCUFLAGS) -c $<

$(DST_DIR)/ep_bench: $(DST_DIR)/ep_bench.o $(SOLIBTARGET)
	@printf "Linking    %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_BENCH_NVCUFLAGS) $< $(EP_BENCH_NVLDFLAGS)
else
ifeq ($(MPI_IBM),1)
MPI_CFLAGS := -DMPI_SUPPORT
MPI_LDFLAGS := -lmpi_ibm

EP_TEST_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_TEST_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda
EP_BENCH_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) $(CUPTI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_BENCH_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda $(CUPTI_LDFLAGS)
ep_test: lib $(DST_DIR)/ep_test
ep_bench: lib $(DST_DIR)/ep_bench

$(DST_DIR)/ep_test.o: ep_test.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_TEST_NVCUFLAGS) -c $<

$(DST_DIR)/ep_test: $(DST_DIR)/ep_test.o $(SOLIBTARGET)
	@printf "Linking    %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_TEST_NVCUFLAGS) $< $(EP_TEST_NVLDFLAGS)

$(DST_DIR)/ep_bench.o: ep_bench.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_BENCH_NVCUFLAGS) -c $<

$(DST_DIR)/ep_bench: $(DST_DIR)/ep_bench.o $(SOLIBTARGET)
	@printf "Linking    %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(EP_BENCH_NVCUFLAGS) $< $(EP_BENCH_NVLDFLAGS)
else
ep_test ep_bench:
	@echo "Skipping ep_test/ep_bench (MPI not enabled). Use MPI=1 to build."
endif
endif

clean:
	rm -rf $(OBJDIR) $(LIBTARGET) $(SOLIBTARGET) $(SOLIBLINKS) $(HEADER_TARGETS) $(JIT_DIR) $(DST_DIR)
