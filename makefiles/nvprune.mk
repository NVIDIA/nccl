#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
# Included by src/device/Makefile and src/diagnostics/device/Makefile after their
# device link, with NVPRUNE_GLUE set to the device link object.
#
NVPRUNE_RULE_FILE := $(lastword $(MAKEFILE_LIST))
# libnccl.so links copies of the device objects without the pre-link relocatable
# cubins in __nv_relfatbin, made by src/device/nvprune_copy.sh after the device
# link $(NVPRUNE_GLUE); a copy that is the same as its object (nothing removed) is
# not used. libnccl_static.a keeps the original objects. Needs CUDA
# 11.3 or later (nvprune --no-relocatable-elf); otherwise, or with
# NCCL_PRUNE_SHARED_RELOCATABLE=0, libnccl.so links the original objects.
NCCL_PRUNE_SHARED_RELOCATABLE ?= 1
NVPRUNE ?= $(CUDA_HOME)/bin/nvprune
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump
READELF ?= readelf
NVPRUNE_SCRIPT := $(NCCLDIR)/src/device/nvprune_copy.sh
# The file with these rules (set where they are included), so that a change to it remakes the copies.
NVPRUNE_RULE_FILE ?=
NVPRUNE_NO_RELOC := $(if $(filter 1,$(NCCL_PRUNE_SHARED_RELOCATABLE)),$(shell test "0$(CUDA_MAJOR)" -gt 11 -o "0$(CUDA_MAJOR)" -eq 11 -a "0$(CUDA_MINOR)" -ge 3 && echo yes))
ifeq ($(NVPRUNE_NO_RELOC),yes)
SHARED_OBJS := $(patsubst $(OBJDIR)/%,$(OBJDIR)/shared/%,$(LIB_OBJS))
else
SHARED_OBJS := $(LIB_OBJS)
endif

# Rewritten only when the choice above or the tools change, so the copies and
# manifest.shared follow such a change in the same BUILDDIR.
NVPRUNE_STAMP := $(OBJDIR)/shared/options
$(NVPRUNE_STAMP): NVPRUNE_CHECK
	@mkdir -p $(dir $@)
	@echo '$(firstword $(SHARED_OBJS)) $(NVPRUNE) $(CUOBJDUMP) $(READELF)' > $@.tmp || exit 1; \
	if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(OBJDIR)/shared/%.o: $(OBJDIR)/%.o $(NVPRUNE_GLUE) $(NVPRUNE_SCRIPT) $(NVPRUNE_RULE_FILE) $(NVPRUNE_STAMP)
	@mkdir -p $(dir $@)
	NVPRUNE='$(NVPRUNE)' CUOBJDUMP='$(CUOBJDUMP)' READELF='$(READELF)' sh '$(NVPRUNE_SCRIPT)' '$<' '$(NVPRUNE_GLUE)' '$@'

# Built by "$(MAKE) -C <this dir> shared" from src/Makefile, for libnccl.so only.
# .SECONDARY above would let make skip a missing copy, so a missing copy
# forces this rule (and with it the copy) to run.
shared: $(SHARED_MANIFEST)
$(SHARED_MANIFEST): $(SHARED_OBJS) $(NVPRUNE_GLUE) $(NVPRUNE_STAMP) $(if $(filter-out $(wildcard $(SHARED_OBJS)),$(SHARED_OBJS)),NVPRUNE_CHECK)
	@for o in $(SHARED_OBJS); do \
	  case "$$o" in $(OBJDIR)/shared/*) orig=$(OBJDIR)/$${o#$(OBJDIR)/shared/}; cmp -s "$$o" "$$orig" && o=$$orig ;; esac; \
	  printf '%s ' "$$o"; \
	done > $@.tmp && echo $(NVPRUNE_GLUE) >> $@.tmp && mv -f $@.tmp $@

.PHONY: shared NVPRUNE_CHECK
