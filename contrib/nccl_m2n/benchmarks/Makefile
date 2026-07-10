# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Makefile for the NCCL M2N benchmarks (reshard release).
#
# This Makefile is fully self-contained: it can be built standalone (`make`
# from inside benchmarks/) or invoked from the project root via the delegation
# rule there (`make bench` at the project root).
#
# Both Makefiles include ../Makefile.common to pick up the same NCCL/CUDA/MPI
# toolchain config.
#
# Usage (standalone):
#   make                              - Build all benchmark executables
#   make bench                        - Same as default
#   make bench <name>                 - Build only the named bench(es). <name> is one of:
#                                       reshard, reshard_batch_user_window,
#                                       reshard_model
#                                       (e.g. `make bench reshard_model`)
#   make reshard |                   - Aliases for the three benches.
#        reshard_batch_user_window |
#        reshard_model
#   make clean                        - Remove only the bench binaries
#   make help                         - Show this help
#
# Usage (from project root):
#   make bench                        - Equivalent to `make -C benchmarks bench`
#   make bench reshard_model          - Same picker, equivalent to
#                                       `make -C benchmarks bench reshard_model`
#   make bench-clean                  - Delegates to `make -C benchmarks clean`
#
# Required env: NCCL_HOME (see ../Makefile.common).

include ../Makefile.common

# ============================================================================
# Bench source files
# ============================================================================
# Shared validation kernels + their host launchers. Compiled once and
# link-included into the benchmarks that need validation helpers.
BENCH_KERNELS_SRC := $(BENCHDIR)/bench_common_kernels.cu
BENCH_KERNELS_HDR := $(BENCHDIR)/bench_common_kernels.h

# Host-only drivers (.cc — nvcc forwards to the host compiler).
BENCH_SRCS := $(BENCHDIR)/reshard_bench.cc $(BENCH_KERNELS_SRC)

# CUDA TUs (define their own __global__ kernels).
BATCH_BENCH_SRCS := $(BENCHDIR)/reshard_batch_bench_user_window.cu
MODEL_BENCH_SRCS := $(BENCHDIR)/reshard_model_bench.cu $(BENCH_KERNELS_SRC)

# ============================================================================
# Bench picker — `make bench` builds all benches; `make bench <name>...`
# (e.g. `make bench reshard_model`) builds only the named benches.
# ============================================================================
BENCH_PICK_NAMES := reshard reshard_batch_user_window reshard_model
bench_bin_reshard                     := reshard_bench
bench_bin_reshard_batch_user_window   := reshard_batch_bench_user_window
bench_bin_reshard_model               := reshard_model_bench

BENCH_PICKED := $(filter $(BENCH_PICK_NAMES),$(MAKECMDGOALS))

.PHONY: all bench clean help $(BENCH_PICK_NAMES)

all: bench

ifeq ($(BENCH_PICKED),)
bench: $(BINDIR)/reshard_bench $(BINDIR)/reshard_batch_bench_user_window $(BINDIR)/reshard_model_bench
else
bench: $(foreach n,$(BENCH_PICKED),$(BINDIR)/$(bench_bin_$(n)))
endif

# Sub-name aliases — these also work standalone (e.g. `make reshard_model`).
reshard:                     $(BINDIR)/reshard_bench                     ; @:
reshard_batch_user_window:   $(BINDIR)/reshard_batch_bench_user_window   ; @:
reshard_model:               $(BINDIR)/reshard_model_bench               ; @:

# Single-layer benchmark
$(BINDIR)/reshard_bench: $(BENCH_SRCS) $(BENCH_KERNELS_HDR) $(M2N_LIBTARGET) $(LIB_HDRS)
	@printf "Building reshard_bench...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(MPI_CFLAGS) $(BENCH_SRCS) $(M2N_CLIENT_LDFLAGS) $(BENCH_LDFLAGS)
	@printf "Done! Benchmark binary at: $@\n"

# Batched user-window benchmark
$(BINDIR)/reshard_batch_bench_user_window: $(BATCH_BENCH_SRCS) $(M2N_LIBTARGET) $(LIB_HDRS)
	@printf "Building reshard_batch_bench_user_window...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(MPI_CFLAGS) $(BATCH_BENCH_SRCS) $(M2N_CLIENT_LDFLAGS) $(BENCH_LDFLAGS)
	@printf "Done! Batch benchmark binary at: $@\n"

# Model transfer benchmark (config-driven, real model/system JSON files)
$(BINDIR)/reshard_model_bench: $(MODEL_BENCH_SRCS) $(BENCH_KERNELS_HDR) $(M2N_LIBTARGET) $(LIB_HDRS)
	@printf "Building reshard_model_bench...\n"
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $(MPI_CFLAGS) $(MODEL_BENCH_SRCS) $(M2N_CLIENT_LDFLAGS) $(BENCH_LDFLAGS)
	@printf "Done! Model benchmark binary at: $@\n"

# Convenience: building the lib triggers the parent Makefile, so the bench
# rules above resolve $(M2N_LIBTARGET) via the implicit chain.
$(M2N_LIBTARGET):
	@$(MAKE) --no-print-directory -C $(NCCL_RESHARD_ROOT) lib

clean:
	@printf "Cleaning bench binaries...\n"
	rm -f $(BINDIR)/reshard_bench $(BINDIR)/reshard_batch_bench_user_window \
	      $(BINDIR)/reshard_model_bench

help:
	@echo "benchmarks/Makefile — NCCL M2N benchmarks (reshard release)"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)                    - Build all benches (same as bench)"
	@echo "  bench                            - Build all benches"
	@echo "  bench <name>                     - Build only named bench(es). <name> is one of:"
	@echo "                                     reshard, reshard_batch_user_window,"
	@echo "                                     reshard_model"
	@echo "  reshard |"
	@echo "    reshard_batch_user_window |"
	@echo "    reshard_model                  - Aliases for the three benches"
	@echo "  clean                            - Remove only the bench binaries"
	@echo "  help                             - Show this help"
	@echo ""
	@echo "Configuration is shared with the parent Makefile via ../Makefile.common"
	@echo "(NCCL_HOME, CUDA_HOME, MPI_HOME, NVCC_GENCODE, DEBUG, BUILDDIR)."
	@echo ""
	@echo "From the project root, equivalent invocations are:"
	@echo "  make bench"
	@echo "  make bench reshard_model"
	@echo "  make bench-clean"
