# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Makefile for the NCCL M2N library — reshard release.
#
# Shared toolchain config lives in Makefile.common and is also pulled in by
# benchmarks/Makefile and tests/Makefile so all binaries pick up the same
# flags. The `bench` and `tests` groups are delegated to their respective
# subdirectory Makefiles; this file owns the shared library and install/clean.
#
# Usage:
#   make              - Build the shared library (default)
#   make lib          - Build shared library (libnccl_m2n.so)
#   make bench        - Delegate to benchmarks/Makefile (all benches)
#   make bench <name> - Delegate to benchmarks/Makefile, named picker. <name> is:
#                       reshard, reshard_batch_user_window, reshard_model
#                       (e.g. `make bench reshard_model`)
#   make tests        - Delegate to tests/Makefile (basic_api tests)
#   make install      - Install lib and headers to PREFIX
#   make clean        - Remove all build artifacts
#   make bench-clean  - Delegate to benchmarks/Makefile (clean bench binaries)
#   make tests-clean  - Delegate to tests/Makefile (clean test binaries)
#   make help         - Show all targets and options
#
# Required environment variables:
#   NCCL_HOME         - Path to NCCL build directory (must contain internal
#                       headers like nccl_device.h)
#
# Optional environment variables:
#   CUDA_HOME         - Path to CUDA installation (default: /usr/local/cuda)
#   MPI_HOME          - Path to MPI installation (for benchmarks only)
#   PREFIX            - Install prefix (default: /usr/local)
#   NVCC_GENCODE      - GPU arch flags (default: sm_80, sm_90, sm_100)
#   DEBUG             - 1=lineinfo, full=device debug
#   BUILDDIR          - Override build/ output directory

include Makefile.common

# Names of the bench picker targets — declared here so the root Makefile can
# treat them as phony (each delegates to benchmarks/Makefile preserving its
# own picker semantics).
BENCH_PICK_NAMES := reshard reshard_batch_user_window reshard_model

.PHONY: all lib bench bench-clean tests basic_api_test_mpi basic_api_test_local \
        clean tests-clean install help $(BENCH_PICK_NAMES)

all: lib

# ============================================================================
# Library
# ============================================================================
lib: $(M2N_LIBTARGET) $(M2N_HEADER_TARGET)

$(M2N_LIBTARGET): $(LIB_SRCS) $(LIB_HDRS)
	@printf "Building $(M2N_LIBNAME)...\n"
	@mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $(NVCUFLAGS) $(LIB_SRCS) $(M2N_LIB_LDFLAGS)
	@printf "Done! Library at: $@\n"

$(M2N_HEADER_TARGET): $(SRCDIR)/nccl_m2n.h
	@mkdir -p $(INCDIR)
	cp $< $@

# ============================================================================
# Benchmarks — fully delegated to benchmarks/Makefile.
# ============================================================================
# `make bench`              → builds all benches
# `make bench reshard_model` → picker, forwarded as-is
# `make reshard_model` etc.  → standalone alias, forwarded one-by-one
bench: lib
	@$(MAKE) --no-print-directory -C $(BENCHDIR) bench $(filter $(BENCH_PICK_NAMES),$(MAKECMDGOALS))

# When `bench` is also a goal it has already picked the named bench(es), so
# the alias targets become no-ops to avoid a redundant sub-make. When invoked
# standalone (`make reshard`), they delegate normally.
ifneq ($(filter bench,$(MAKECMDGOALS)),)
$(BENCH_PICK_NAMES): ; @:
else
$(BENCH_PICK_NAMES): lib
	@$(MAKE) --no-print-directory -C $(BENCHDIR) $@
endif

bench-clean:
	@$(MAKE) --no-print-directory -C $(BENCHDIR) clean

# ============================================================================
# Tests — fully delegated to tests/Makefile.
# ============================================================================
tests basic_api_test_mpi basic_api_test_local: lib
	@$(MAKE) --no-print-directory -C $(TESTSDIR) $@

tests-clean:
	@$(MAKE) --no-print-directory -C $(TESTSDIR) clean

# ============================================================================
# Install
# ============================================================================
PREFIX ?= /usr/local
install: lib
	@printf "Installing to $(PREFIX)...\n"
	@mkdir -p $(PREFIX)/lib $(PREFIX)/include
	cp $(M2N_LIBTARGET) $(PREFIX)/lib/
	cp $(M2N_HEADER_TARGET) $(PREFIX)/include/
	@printf "Done!\n"

# ============================================================================
# Clean / help
# ============================================================================
clean:
	rm -rf $(BUILDDIR)

help:
	@echo "NCCL M2N Library Makefile — reshard release"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)         - Build shared library"
	@echo "  lib                   - Build shared library (libnccl_m2n.so)"
	@echo "  bench                 - Build all benchmarks (delegates to benchmarks/)"
	@echo "  bench <name>          - Build only named bench(es). <name> is one of:"
	@echo "                          reshard, reshard_batch_user_window"
	@echo "  tests                 - Build basic_api_test_{mpi,local}"
	@echo "  basic_api_test_mpi    - Build only the MPI test binary (delegates to tests/)"
	@echo "  basic_api_test_local  - Build only the no-MPI test binary (delegates to tests/)"
	@echo "  install               - Install to PREFIX (default: /usr/local)"
	@echo "  clean                 - Remove all build artifacts"
	@echo "  bench-clean           - Remove only the bench binaries (delegates to benchmarks/)"
	@echo "  tests-clean           - Remove only the test binaries (delegates to tests/)"
	@echo "  help                  - Show this help"
	@echo ""
	@echo "Required:  NCCL_HOME    - Path to NCCL build directory"
	@echo "Optional:  CUDA_HOME    - Path to CUDA (default: /usr/local/cuda)"
	@echo "           MPI_HOME     - Path to MPI (default: system MPI)"
	@echo "           PREFIX       - Install prefix (default: /usr/local)"
	@echo "           NVCC_GENCODE - GPU arch flags (default: sm_80, sm_90, sm_100)"
	@echo "           DEBUG        - 1=lineinfo, full=device debug"
	@echo "           BUILDDIR     - Override build/ output directory"
	@echo ""
	@echo "See benchmarks/Makefile and tests/Makefile for the bench and test"
	@echo "build rules (each cross-references Makefile.common via"
	@echo "  include ../Makefile.common)."
