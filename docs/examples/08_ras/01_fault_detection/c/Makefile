#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# This example requires MPI
MPI = 1

include ../../../../../makefiles/common.mk
include ../../../../../makefiles/examples.mk

TARGET    = ras_fault_detection
COMMON_INC = ../../../common/include
COMMON_SRC = ../../../common/src

INCLUDES += -I$(COMMON_INC)
SOURCES   = main.cc utils.cc
# utils.cc is a shared helper in common/src, compiled per-example into a local
# utils.o so parallel example builds do not race on one shared object.
# vpath NAME DIR tells make to search DIR for the prerequisite NAME, so the
# %.o: %.cc rule below finds utils.cc in $(COMMON_SRC) and builds a local utils.o.
vpath utils.cc $(COMMON_SRC)
OBJECTS   = $(SOURCES:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

%.o: %.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# This example induces faults and is meant for interactive submission: run it,
# then query RAS from another terminal (nc / ncclras) and watch the
# RUNNING -> INCOMPLETE -> DEAD (or MISMATCH -> OK) transitions yourself.
# It is intentionally NOT driven automatically -- anticipating the injected
# faults in an automated test would require brittle, timing-dependent parsing
# of the RAS report.  Build it and run it by hand (see README.md).
test:
	@echo "08_ras/01_fault_detection: interactive example -- not run automatically."
	@echo "Build with 'make', then run under mpirun/srun by hand (see README.md)."

clean:
	rm -f $(OBJECTS) $(TARGET)

install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

help:
	@echo "NCCL Example: RAS Fault Detection"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the example"
	@echo "  test    - (interactive example; not run automatically -- see README.md)"
	@echo "  clean   - Remove build artifacts"
	@echo "  install - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo ""
	@echo "Run options:"
	@echo "  mpirun -np N ./ras_fault_detection [--type exit|suspend|sleep]"
	@echo "                                     [--ranks R,...] [--wait S] [--observe S]"

.PHONY: all test clean install help
