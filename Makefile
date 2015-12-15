#
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0

CUDACODE := -gencode=arch=compute_35,code=sm_35 \
            -gencode=arch=compute_50,code=sm_50 \
            -gencode=arch=compute_52,code=sm_52

BUILDDIR := build

NVCC       := $(CUDA_HOME)/bin/nvcc
GPP        := g++
CPPFLAGS   := -I$(CUDA_HOME)/include
CXXFLAGS   := -O3 -fPIC -fvisibility=hidden
NVCUFLAGS  := $(CUDACODE) -O3 -lineinfo -std=c++11 -maxrregcount 96

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra
CXXFLAGS  += -Wall -Wextra
endif

LDFLAGS    := -L$(CUDA_HOME)/lib64 -lcudart
MPIFLAGS   := -I$(MPI_HOME)/include -L$(MPI_HOME)/lib -lmpi

.PHONY : lib clean test mpitest install
.DEFAULT : lib

INCEXPORTS  := nccl.h
LIBSRCFILES := libwrap.cu core.cu all_gather.cu all_reduce.cu broadcast.cu reduce.cu reduce_scatter.cu
LIBNAME     := libnccl.so
APIVER      := 1
TESTS       := all_gather_test all_reduce_test broadcast_test reduce_test reduce_scatter_test
MPITESTS    := mpi_test

INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
TSTDIR := $(BUILDDIR)/test
MPITSTDIR := $(BUILDDIR)/mpitest

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))
LIBTARGET  := $(patsubst %, $(LIBDIR)/%.$(APIVER), $(LIBNAME))
LIBLINK    := $(patsubst lib%.so, -l%, $(LIBNAME))
LIBOBJ     := $(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(LIBSRCFILES)))
TESTBINS   := $(patsubst %, $(TSTDIR)/%, $(TESTS))
MPITESTBINS:= $(patsubst %, $(MPITSTDIR)/%, $(MPITESTS))
DEPFILES   := $(patsubst %.o, %.d, $(LIBOBJ)) $(patsubst %, %.d, $(TESTBINS)) $(patsubst %, %.d, $(MPITESTBINS))

lib : $(INCTARGETS) $(LIBTARGET)

-include $(DEPFILES)

$(LIBTARGET) : $(LIBOBJ)
	@printf "Linking   %-25s\n" $@
	@mkdir -p $(LIBDIR)
	@$(GPP) $(CPPFLAGS) $(CXXFLAGS) -shared -Wl,-soname,$(LIBNAME).$(APIVER) -o $@ $(LDFLAGS) $(LIBOBJ)
	@ln -sf $(LIBNAME).$(APIVER) $(LIBDIR)/$(LIBNAME)

$(INCDIR)/%.h : src/%.h
	@printf "Grabbing  %-25s > %-25s\n" $< $@
	@mkdir -p $(INCDIR)
	@cp -f $< $@

$(OBJDIR)/%.o : src/%.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	@mkdir -p $(OBJDIR)
	@$(NVCC) -c $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< -o $@
	@$(NVCC) -M $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf build

test : lib $(TESTBINS)

$(TSTDIR)/% : src/%.cu lib
	@printf "Building  %-25s > %-24s\n" $< $@
	@mkdir -p $(TSTDIR)
	@$(NVCC) -Ibuild/include $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@ $< -Lbuild/lib $(LIBLINK) $(LDFLAGS) -lcuda -lcurand -lnvToolsExt
	@$(NVCC) -M -Ibuild/include $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< -Lbuild/lib $(LIBLINK) $(LDFLAGS) -lcuda -lcurand -lnvToolsExt > $(@:%=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%=%.d.tmp) > $(@:%=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%=%.d)
	@rm -f $(@:%=%.d.tmp)

mpitest : lib $(MPITESTBINS)

$(MPITSTDIR)/% : src/%.cu lib
	@printf "Building  %-25s > %-24s\n" $< $@
	@mkdir -p $(MPITSTDIR)
	@$(NVCC) $(MPIFLAGS) -Ibuild/include $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@ $< -Lbuild/lib $(LIBLINK) $(LDFLAGS)
	@$(NVCC) $(MPIFLAGS) -M -Ibuild/include $(CPPFLAGS) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< -Lbuild/lib $(LIBLINK) $(LDFLAGS) > $(@:%=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%=%.d.tmp) > $(@:%=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%=%.d)
	@rm -f $(@:%=%.d.tmp)

install : lib
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	cp -P -v build/lib/* $(PREFIX)/lib/
	cp -v build/include/* $(PREFIX)/include/
