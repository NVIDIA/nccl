#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/diagnostics/device

MANIFEST := $(OBJDIR)/manifest
DIAG_DEVGLUE_OBJ := $(OBJDIR)/diagnostics_device_glue.o

INCFLAGS = -I. -I.. -I../.. -I../../device -I$(BUILDDIR)/include -I../../include -I../../include/plugin
NVCUFLAGS += $(INCFLAGS) --compiler-options "-fPIC -fvisibility=hidden"

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(suffix $2),$1,$2)
endef

DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -MM -dc $1
define DEPENDS
@$(SAY) "Dependencies" $2;\
 mkdir -p $(dir $1);\
 mk=$$($(call DEPENDS$(suffix $2),$2));\
 [[ $$mk =~ ^[^:]*:(.*)$$ ]];\
 files=$${BASH_REMATCH[1]};\
 files=$$(for x in $$files; do case "$$x" in '\'|$$'\t') ;; *) echo "$$x"; esac; done);\
 files=$$(for x in $$files; do [[ "$$(realpath "$$x")" == "$$(realpath "$(NCCLDIR)")"* ]] && echo "$$x"; done);\
 echo "$(patsubst %.d,%.o,$1) $1: " $$files > $1
endef

all: $(MANIFEST)

SRCS = p2p.cu
LIB_OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS))

$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(DIAG_DEVGLUE_OBJ): $(LIB_OBJS)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

$(MANIFEST): $(LIB_OBJS) $(DIAG_DEVGLUE_OBJ)
	@echo $^ > $@

-include $(wildcard $(OBJDIR)/*.d)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)
