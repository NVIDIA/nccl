# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror

BUILD_DIR ?= ../../build/test/plugin_lifecycle
TEST_BIN := $(BUILD_DIR)/plugin_lifecycle_test

.PHONY: all run clean

all: $(TEST_BIN)

$(TEST_BIN): plugin_lifecycle_test.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
