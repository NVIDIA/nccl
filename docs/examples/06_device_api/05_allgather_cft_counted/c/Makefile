#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

include ../../../../../makefiles/common.mk
include ../../../../../makefiles/examples.mk

TARGET = allgather_cft_counted

COMMON_INC = ../../../common/include
COMMON_SRC = ../../../common/src

INCLUDES += -I$(COMMON_INC)

SOURCES = main.cu $(COMMON_SRC)/utils.cc
OBJECTS = $(SOURCES:.cu=.o)
OBJECTS := $(OBJECTS:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
else
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -lpthread -o $@
endif
	@echo "Built target $@"

%.o: %.cu
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cc
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
else
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
endif

test: $(TARGET)
ifeq ($(MPI),1)
	$(MPIRUN) -np 2 ./$(TARGET)
else
	./$(TARGET)
endif

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all test clean
