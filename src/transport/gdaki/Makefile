#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../build)
OBJDIR := $(BUILDDIR)/obj/gdaki

MANIFEST := $(OBJDIR)/manifest

INCFLAGS  = -I. -I../include -I$(CUDA_INC)
CXXFLAGS  += $(INCFLAGS)

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cpp = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(or $3,$(suffix $2)),$1,$2)
endef

DEPENDS.cpp = $(CXX) $(CXXFLAGS) -M -c $1
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

SRCS = $(wildcard doca-gpunetio/src/*.cpp)

LIB_OBJS = $(SRCS:doca-gpunetio/src/%.cpp=doca-gpunetio/%.o)

$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(MANIFEST): $(LIB_OBJS)
	mkdir -p $(OBJDIR)
	@echo $^ > $@

-include $(wildcard $(OBJDIR)/*.d)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)
