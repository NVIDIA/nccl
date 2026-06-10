# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Makefile for the basic_api correctness suite.
#
# Mirrors an external pytest reference matrix at the C level. Two binaries
# share a single descriptor table in basic_api_test_core.h:
#
# TU split (matches MR !34's host-cc-refactor convention):
#   test_helpers.cu    — kernels + launchers (the only file that needs nvcc)
#   *.cc / *.h         — host-only test driver code; nvcc forwards .cc to g++
#
#   basic_api_test_mpi   — MPI bootstrap (multi-host)
#   basic_api_test_local — single-process pthreads via ncclCommInitAll
#
# This Makefile is fully self-contained: it can be built standalone (`make`
# from inside tests/) or invoked from the project root via the delegation
# rule there (`make tests` at the project root).
#
# Both Makefiles include ../Makefile.common to pick up the same NCCL/CUDA/MPI
# toolchain config.
#
# Usage (standalone):
#   make                        - Build all public test binaries
#   make basic_api_test_mpi
#   make basic_api_test_local
#   make clean                  - Remove only the test binaries
#   make help                   - Show this help
#
# Usage (from project root):
#   make tests                  - Equivalent to `make -C tests`
#   make basic_api_test_mpi     - Delegates to `make -C tests basic_api_test_mpi`
#   make tests-clean            - Delegates to `make -C tests clean`
#
# Required env: NCCL_HOME (see ../Makefile.common).

include ../Makefile.common
include gtest.mk

# Test sources / headers (paths absolute via $(TESTSDIR) from common.mk).
# test_helpers.cu is the only nvcc-required TU; both binaries link it in.
TEST_HDRS := $(TESTSDIR)/test_helpers.h $(TESTSDIR)/basic_api_test_core.h
TEST_HELPERS_SRC := $(TESTSDIR)/test_helpers.cu
TEST_MPI_SRCS    := $(TESTSDIR)/basic_api_test_mpi.cc   $(TEST_HELPERS_SRC)
TEST_LOCAL_SRCS  := $(TESTSDIR)/basic_api_test_local.cc $(TEST_HELPERS_SRC)

.PHONY: all tests basic_api_test_mpi basic_api_test_local gtest gtest-clean clean help

all: tests

ifeq ($(GTEST_AVAILABLE),0)

# No gtest source available — emit a friendly no-op for every gtest-
# dependent target. Library and benchmarks build fine without these;
# point GTEST_DIR at a googletest source tree to build the tests.
tests basic_api_test_mpi basic_api_test_local gtest:
	@echo "tests/Makefile: googletest not found — skipping basic_api test build."
	@echo "  Looked for vendored copy at \$$(TESTSDIR)/googletest."
	@echo "  To build the tests, point GTEST_DIR at a googletest source tree:"
	@echo "    make tests GTEST_DIR=/path/to/googletest"

gtest-clean:
	rm -rf $(GTEST_OBJDIR)

clean: gtest-clean
	@printf "Cleaning test binaries...\n"
	rm -f $(BINDIR)/basic_api_test_mpi $(BINDIR)/basic_api_test_local

else

tests: $(BINDIR)/basic_api_test_mpi $(BINDIR)/basic_api_test_local

basic_api_test_mpi:   $(BINDIR)/basic_api_test_mpi   ; @:
basic_api_test_local: $(BINDIR)/basic_api_test_local ; @:

$(BINDIR)/basic_api_test_mpi: $(TEST_MPI_SRCS) $(GTEST_A) \
                              $(M2N_LIBTARGET) $(LIB_HDRS) $(TEST_HDRS)
	@printf "Building basic_api_test_mpi (gtest, MPI, GTEST_DIR=$(GTEST_DIR))...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(MPI_CFLAGS) -I$(TESTSDIR) -I$(GTEST_INC) \
	    $(TEST_MPI_SRCS) $(GTEST_A) $(M2N_CLIENT_LDFLAGS) $(BENCH_LDFLAGS) -lpthread
	@printf "Done! Test binary at: $@\n"

$(BINDIR)/basic_api_test_local: $(TEST_LOCAL_SRCS) $(GTEST_A) \
                                $(M2N_LIBTARGET) $(LIB_HDRS) $(TEST_HDRS)
	@printf "Building basic_api_test_local (gtest, no MPI, GTEST_DIR=$(GTEST_DIR))...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -I$(TESTSDIR) -I$(GTEST_INC) \
	    $(TEST_LOCAL_SRCS) $(GTEST_A) $(M2N_CLIENT_LDFLAGS) $(LIB_LDFLAGS) -lpthread
	@printf "Done! Test binary at: $@\n"

clean:
	@printf "Cleaning test binaries...\n"
	rm -f $(BINDIR)/basic_api_test_mpi $(BINDIR)/basic_api_test_local
	rm -rf $(GTEST_OBJDIR)

endif

# Convenience: building the lib triggers the parent Makefile, so the test
# rules above resolve $(M2N_LIBTARGET) via the implicit chain.
$(M2N_LIBTARGET):
	@$(MAKE) --no-print-directory -C $(NCCL_RESHARD_ROOT) lib

help:
	@echo "tests/Makefile — public basic_api correctness suite"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)         - Build gtest basic_api binaries"
	@echo "  tests                 - Same as all"
	@echo "  basic_api_test_mpi    - Build only the MPI gtest matrix binary"
	@echo "  basic_api_test_local  - Build only the local pthread gtest matrix binary"
	@echo "  gtest                 - Build gtest.a / gtest_main.a (from tests/googletest/)"
	@echo "  clean                 - Remove all test binaries + gtest objects"
	@echo "  help                  - Show this help"
	@echo ""
	@echo "Configuration is shared with the parent Makefile via ../Makefile.common"
	@echo "(NCCL_HOME, CUDA_HOME, MPI_HOME, NVCC_GENCODE, DEBUG, BUILDDIR)."
	@echo ""
	@echo "From the project root, equivalent invocations are:"
	@echo "  make tests"
	@echo "  make basic_api_test_mpi"
	@echo "  make basic_api_test_local"
	@echo "  make tests-clean"
