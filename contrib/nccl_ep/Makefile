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

# Include paths - add current directory so common.hpp can find device/ep headers
# Include NCCL source headers for GIN support
INCFLAGS = -I. -I./include -I./device -I$(BUILDDIR)/include -I$(BUILDDIR)/include/nccl_device -I../../src/include

# GIN (Global Input Network) support - adds multinode RDMA capabilities


# Rebuild NVCUFLAGS with our NVCC_GENCODE (after INCFLAGS is complete)
# Add --expt-relaxed-constexpr for std::max, std::min, etc. in device code
# CUDA code MUST use -fvisibility=hidden for kernel registration to work correctly.
# The ncclEp* API is in nccl_ep.cc (compiled with CXXFLAGS), so it still gets exported.
NVCUFLAGS := $(NVCC_GENCODE) $(INCFLAGS) --expt-relaxed-constexpr --compiler-options "-fPIC -fvisibility=hidden" -ccbin $(CXX) $(CXXSTD)
CXXFLAGS += $(INCFLAGS) -fPIC -fvisibility=default

# Dependency generation flags for automatic header tracking
DEPFLAGS = -MMD -MP

# Source files for library
HOST_SRC := nccl_ep.cc
DEVICE_SRC := device/low_latency.cu \
              device/hybridep_adapter.cu
# Object files
HOST_OBJ := $(HOST_SRC:%.cc=$(OBJDIR)/%.o)
DEVICE_OBJ := $(DEVICE_SRC:%.cu=$(OBJDIR)/%.o)
ALL_OBJ := $(HOST_OBJ) $(DEVICE_OBJ)

# Device glue object
DEVGLUE_OBJ := $(OBJDIR)/device_glue.o

# Library output
LIBNAME := libnccl_ep.a
LIBTARGET := $(LIBDIR)/$(LIBNAME)

# Shared library output
SOLIBNAME := libnccl_ep.so
SOLIBTARGET := $(LIBDIR)/$(SOLIBNAME)

# Header installation
HEADERS := include/nccl_ep.h include/common.hpp
HEADER_TARGETS := $(HEADERS:include/%=$(INCDIR)/%)

# Build targets
all: lib ep_test ep_bench

lib: $(LIBTARGET) $(SOLIBTARGET) $(HEADER_TARGETS)

# Compile host code
$(OBJDIR)/%.o: %.cc
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
$(LIBTARGET): $(ALL_OBJ) $(DEVGLUE_OBJ)
	@echo "Creating library $@"
	@mkdir -p $(dir $@)
	ar rcs $@ $^

# Create shared library (for Python ctypes)
# Links against libnccl.so for NCCL symbols (ncclCommCuDevice, etc.)
# Uses rpath so the library can find libnccl.so at runtime
$(SOLIBTARGET): $(ALL_OBJ) $(DEVGLUE_OBJ)
	@echo "Creating shared library $@"
	@mkdir -p $(dir $@)
	$(NVCC) -shared $(NVCC_GENCODE) -o $@ $^ -L$(LIBDIR) -lnccl -Xlinker -rpath -Xlinker $(LIBDIR) $(LDFLAGS)

# Install headers
$(INCDIR)/%.h: include/%.h
	@echo "Installing $<"
	@mkdir -p $(dir $@)
	cp $< $@

$(INCDIR)/%.hpp: include/%.hpp
	@echo "Installing $<"
	@mkdir -p $(dir $@)
	cp $< $@

# Include generated dependency files (the - prefix silences errors if files don't exist yet)
-include $(ALL_OBJ:.o=.d)

# Build ep_test and ep_bench (require MPI)
ifeq ($(MPI), 1)
MPI_CFLAGS := -DMPI_SUPPORT -I$(MPI_HOME)/include
MPI_LDFLAGS := -L$(MPI_HOME)/lib -lmpi

EP_TEST_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_TEST_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda

# ep_bench: performance benchmark (NVTX for profiling)
EP_BENCH_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_BENCH_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda
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
EP_BENCH_NVCUFLAGS := $(NVCC_GENCODE) -I$(INCDIR) $(MPI_CFLAGS) --compiler-options "-fPIC" -ccbin $(CXX) $(CXXSTD)
EP_BENCH_NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda
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
	rm -rf $(OBJDIR) $(LIBTARGET) $(SOLIBTARGET) $(HEADER_TARGETS) $(DST_DIR)
