#
# Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives/device

LIBSRCFILES := all_reduce.cu broadcast.cu reduce.cu all_gather.cu reduce_scatter.cu

LIBOBJ     := $(patsubst %.cu,$(OBJDIR)/%_sum.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_prod.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_min.o, $(LIBSRCFILES)) \
              $(patsubst %.cu,$(OBJDIR)/%_max.o, $(LIBSRCFILES)) \
              $(OBJDIR)/functions.o

LIBSRCFILES += functions.cu

DEPFILES   := $(patsubst %.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
DEPENDFILES := $(DEPFILES:%.d=%.dep)
STATICLIB  := $(OBJDIR)/colldevice.a
DEVOBJ     := $(OBJDIR)/devlink.o

NVCUFLAGS  += -I. -I.. -I$(BUILDDIR)/include -I../../include --compiler-options "-fPIC -fvisibility=hidden"


all: $(STATICLIB)

# Dummy rule so that the extra dependency (%.dep) files are preserved by make
all_deps: $(DEPENDFILES)

-include $(DEPFILES)

$(STATICLIB): $(LIBOBJ) $(DEVOBJ)
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

# We do not want make to build *.d when running make clean.
# So we only provide targets for .dep which will produce .dep and .d,
# with only .d being included, and .dep keeping track of what needs to
# be regenerated.
$(OBJDIR)/%.dep : %.cu
	@mkdir -p $(OBJDIR)
	@$(NVCC) $(NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

# Compiled kernels and collectives with relocatable device code ...
$(OBJDIR)/functions.o : functions.cu $(OBJDIR)/functions.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_sum.o : %.cu $(OBJDIR)/%.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=0 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_prod.o : %.cu $(OBJDIR)/%.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=1 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_min.o : %.cu $(OBJDIR)/%.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=2 $(NVCUFLAGS) -dc $< -o $@

$(OBJDIR)/%_max.o : %.cu $(OBJDIR)/%.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -DNCCL_OP=3 $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(STATICLIB) test
