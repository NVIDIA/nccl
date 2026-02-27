#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

#
# Makefile for NCCL Tuner Plugin Unit Tests
#

CC := gcc
CFLAGS := -Wall -Wextra -g -std=c99 -fPIC
INC := -I. -I../nccl
TARGET := test_plugin
SOURCES := test_plugin.c

# Default target
all: $(TARGET)

# Build the test executable
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INC) -o $(TARGET) $(SOURCES)

# Run the tests
test: $(TARGET)
	./$(TARGET) $(TEST_CASE)

# Run tests with verbose output
test-verbose: $(TARGET)
	NCCL_DEBUG=INFO ./$(TARGET) $(TEST_CASE)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.gcov *.gcda *.gcno test_*.conf

.PHONY: all test test-verbose clean
