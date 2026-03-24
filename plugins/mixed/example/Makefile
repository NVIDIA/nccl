#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.PHONY=test

NCCL_PLUGINS:=../..
NCCL_HOME:=../../../build
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I$(NCCL_PLUGINS)/net/example/nccl -I$(NCCL_PLUGINS)/tuner/example/nccl

ifeq ($(shell uname -s), Linux)
  CFLAGS += -DNCCL_OS_LINUX
else ifeq ($(OS), Windows_NT)
  CFLAGS += -DNCCL_OS_WINDOWS
endif

PLUGIN_SO:=libnccl-mixed.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	$(CC) $(CFLAGS) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

test: $(PLUGIN_SO)
	@bash test.sh

clean:
	rm -f $(PLUGIN_SO)
