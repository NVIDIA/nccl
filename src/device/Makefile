#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../build)
OBJDIR := $(BUILDDIR)/obj/device

MANIFEST := $(OBJDIR)/manifest
DEVGLUE_OBJ  := $(OBJDIR)/device_glue.o

INCFLAGS  = -I. -I.. -I$(BUILDDIR)/include -I../include -I../include/plugin
NVCUFLAGS += $(INCFLAGS) --compiler-options "-fPIC -fvisibility=hidden"
CXXFLAGS  += $(INCFLAGS)

NVCUFLAGS_SYM += -ccbin $(CXX) $(CXXSTD) --expt-extended-lambda -Xptxas -maxrregcount=128 -Xfatbin -compress-all
NVCUFLAGS_SYM += $(INCFLAGS) --compiler-options "-fPIC -fvisibility=hidden"

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
COMPILE.kernel = $(NVCC) $(NVCUFLAGS) -dw $2 -o $1
COMPILE.cc = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(or $3,$(suffix $2)),$1,$2)
endef

ifeq ($(shell echo "$$((1000*$(CUDA_MAJOR) + 10*$(CUDA_MINOR) >= 12090))"),1)
	NVCC_GENCODE_LDMC_FP8 = -gencode=arch=compute_100f,code=sm_100f
else ifeq ($(shell echo "$$((1000*$(CUDA_MAJOR) + 10*$(CUDA_MINOR) >= 12070))"),1)
  NVCC_GENCODE_LDMC_FP8 = -gencode=arch=compute_100a,code=sm_100a
else
	NVCC_GENCODE_LDMC_FP8 =
endif

define COMPILE_SYM
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 if [[ -n "$3" ]]; then\
 $(NVCC) $(NVCUFLAGS_SYM) $3 -dw $2 -o $1;\
 else\
 touch $2.empty.cu; $(NVCC) $(NVCUFLAGS_SYM) -dw $2.empty.cu -o $1; rm $2.empty.cu;\
 fi
endef

DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -M -dc $1
DEPENDS.cc = $(CXX) $(CXXFLAGS) -M -c $1
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

$(OBJDIR)/gensrc: generate.py
	@mkdir -p $@
	(which python3 >/dev/null || \
	  (bar='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'; \
	   printf "\n$${bar}\nERROR: Building NCCL requires a Python 3 installation invokable as 'python3'.\n$${bar}\n\n" 1>&2; \
	   exit 1)) \
	&& ./generate.py $@ "$(ONLY_FUNCS)"

$(OBJDIR)/gensrc/symmetric: $(OBJDIR)/gensrc symmetric/generate.py
	@mkdir -p $@
	./symmetric/generate.py $@

# The trailing ";" is necessary to make this an "empty recipe":
# https://www.gnu.org/software/make/manual/html_node/Empty-Recipes.html
$(OBJDIR)/gensrc/rules.mk: $(OBJDIR)/gensrc ;

$(OBJDIR)/gensrc/symmetric/rules.mk: $(OBJDIR)/gensrc/symmetric ;

-include $(OBJDIR)/gensrc/rules.mk
# "gensrc/rules.mk" populates $(LIB_OBJS_GEN)

-include $(OBJDIR)/gensrc/symmetric/rules.mk
# "gensrc/symmetric/rules.mk" populates $(LIB_OBJS_SYM_GEN)

SRCS = common.cu onerank.cu

LIB_OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS)) $(LIB_OBJS_GEN) $(LIB_OBJS_SYM_GEN)

$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/genobj/%.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/%.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/$*)

$(OBJDIR)/genobj/symmetric/%.o: $(OBJDIR)/gensrc/symmetric $(OBJDIR)/genobj/symmetric/%.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/symmetric/$*)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(OBJDIR)/genobj/%.d: $(OBJDIR)/gensrc/%
	$(call DEPENDS,$@,$<)

$(OBJDIR)/genobj/symmetric/%.d: $(OBJDIR)/gensrc/symmetric/%
	$(call DEPENDS,$@,$<)

$(DEVGLUE_OBJ): $(LIB_OBJS)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

$(MANIFEST): $(LIB_OBJS) $(DEVGLUE_OBJ)
	@echo $^ > $@

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(OBJDIR)/genobj/*.d)
-include $(wildcard $(OBJDIR)/genobj/symmetric/*.d)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)
