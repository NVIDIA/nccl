# Unit tests for nccl_ep output layout (GPU-major vs expert-major).
#
# Build:  make -C contrib/nccl_ep/tests [BUILDDIR=<path>]
# Run:    bash run_tests.sh
#   or:   NCCL_HOME=<build> bash run_tests.sh

BUILDDIR ?= $(abspath ../../../build)
LIBDIR   := $(BUILDDIR)/lib
INCDIR   := $(BUILDDIR)/include
DST_DIR  := $(BUILDDIR)/test/nccl_ep

GTEST_DIR  := $(abspath ../../../test/googletest/googletest)
GTEST_LIBS := $(BUILDDIR)/test/gtest.a $(BUILDDIR)/test/gtest_main.a

NVCC         ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
NVCC_GENCODE ?= -gencode=arch=compute_90,code=sm_90
CXXSTD       := -std=c++17

NVCUFLAGS := $(NVCC_GENCODE) \
             -I$(INCDIR) \
             -I$(GTEST_DIR)/include \
             --expt-relaxed-constexpr \
             --compiler-options "-fPIC -pthread" \
             -ccbin $(CXX) $(CXXSTD)

NVLDFLAGS := -L$(LIBDIR) -lnccl_ep -lnccl -lpthread -lrt -ldl -lcuda

TEST_BINARIES := $(DST_DIR)/test_output_layout $(DST_DIR)/test_handle_maps $(DST_DIR)/test_lifecycle $(DST_DIR)/test_ht_bwd

.PHONY: all clean
all: $(TEST_BINARIES)

$(BUILDDIR)/test/gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	@mkdir -p $(BUILDDIR)/test
	$(CXX) $(CXXSTD) -pthread -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -c $< -o $@

$(BUILDDIR)/test/gtest_main.o: $(GTEST_DIR)/src/gtest_main.cc
	@mkdir -p $(BUILDDIR)/test
	$(CXX) $(CXXSTD) -pthread -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -c $< -o $@

$(BUILDDIR)/test/gtest.a: $(BUILDDIR)/test/gtest-all.o
	$(AR) rcs $@ $^

$(BUILDDIR)/test/gtest_main.a: $(BUILDDIR)/test/gtest-all.o $(BUILDDIR)/test/gtest_main.o
	$(AR) rcs $@ $^

$(DST_DIR):
	@mkdir -p $@

INTERNAL_HEADERS := $(abspath ../nccl_ep_test_internal.h) test_common.h

$(DST_DIR)/%.o: %.cu $(INTERNAL_HEADERS) | $(DST_DIR)
	@printf "Compiling  %-45s > %s\n" $< $@
	$(NVCC) $(NVCUFLAGS) -c $< -o $@

.PRECIOUS: $(DST_DIR)/%.o

$(DST_DIR)/%: $(DST_DIR)/%.o $(LIBDIR)/libnccl_ep.so $(GTEST_LIBS)
	@printf "Linking    %-45s > %s\n" $< $@
	$(NVCC) $(NVCUFLAGS) $< $(NVLDFLAGS) $(GTEST_LIBS) -o $@

clean:
	rm -f $(TEST_BINARIES) $(patsubst %,%.o,$(TEST_BINARIES))
