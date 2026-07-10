# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Common gtest build rules for NCCL M2N tests.
#
# googletest is an optional dependency. The Makefile builds the tests when
# a googletest source tree is available, and silently skips them when it
# is not — the library and benchmarks build either way.
#
# Source resolution (in order):
#   1. Caller-supplied $(GTEST_DIR), passed on the make command line or in
#      the environment.
#   2. Vendored copy at $(TESTSDIR)/googletest.
#
# When neither resolves, $(GTEST_AVAILABLE) stays at 0 (no $(error)) and
# tests/Makefile turns gtest-dependent targets into a friendly no-op.
# Pass GTEST_DIR=/path/to/googletest to build the tests against any
# checkout of https://github.com/google/googletest.
#
# Include this file from tests/Makefile after Makefile.common.

GTEST_VENDORED := $(TESTSDIR)/googletest
GTEST_OBJDIR := $(BUILDDIR)/gtest

# GTEST_DIR may already be set by the caller. Honor that path as-is.
ifeq ($(origin GTEST_DIR),undefined)
    ifneq ($(wildcard $(GTEST_VENDORED)/include/gtest/gtest.h),)
        GTEST_DIR := $(GTEST_VENDORED)
    endif
endif

ifeq ($(origin GTEST_DIR),undefined)
    GTEST_AVAILABLE := 0
else ifeq ($(wildcard $(GTEST_DIR)/include/gtest/gtest.h),)
    GTEST_AVAILABLE := 0
else
    GTEST_AVAILABLE := 1
endif

ifeq ($(GTEST_AVAILABLE),1)
GTEST_INC := $(GTEST_DIR)/include
GTEST_SRC := $(GTEST_DIR)/src

GTEST_CXXFLAGS := -std=c++17 -pthread -O2 -Wall -Wno-uninitialized \
                  -isystem $(GTEST_INC) -I$(GTEST_DIR)

GTEST_HDRS := $(wildcard $(GTEST_DIR)/include/gtest/*.h) \
              $(wildcard $(GTEST_DIR)/include/gtest/internal/*.h) \
              $(wildcard $(GTEST_SRC)/*.h)

GTEST_A := $(GTEST_OBJDIR)/gtest.a
GTEST_MAIN_A := $(GTEST_OBJDIR)/gtest_main.a

$(GTEST_OBJDIR)/gtest-all.o: $(GTEST_SRC)/gtest-all.cc $(GTEST_HDRS)
	@mkdir -p $(GTEST_OBJDIR)
	$(CXX) $(GTEST_CXXFLAGS) -c $< -o $@

$(GTEST_OBJDIR)/gtest_main.o: $(GTEST_SRC)/gtest_main.cc $(GTEST_HDRS)
	@mkdir -p $(GTEST_OBJDIR)
	$(CXX) $(GTEST_CXXFLAGS) -c $< -o $@

$(GTEST_A): $(GTEST_OBJDIR)/gtest-all.o
	$(AR) rcs $@ $^

$(GTEST_MAIN_A): $(GTEST_OBJDIR)/gtest-all.o $(GTEST_OBJDIR)/gtest_main.o
	$(AR) rcs $@ $^

.PHONY: gtest gtest-clean
gtest: $(GTEST_A) $(GTEST_MAIN_A)

gtest-clean:
	rm -rf $(GTEST_OBJDIR)
endif
