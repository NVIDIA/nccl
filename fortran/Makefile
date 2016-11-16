FC := gfortran
FCNAME := $(notdir $(FC))

BUILDDIR ?= ../build
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj

LIBNAME    := libncclfor.so
LIBSONAME  := $(patsubst %,%.$(NCCL_MAJOR),$(LIBNAME))
LIBTARGET  := $(patsubst %,%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH),$(LIBNAME))
LIBLINK    += $(patsubst lib%.so,-l%,$(LIBNAME))

LIBCUDAFOR := libcudafor.so

ifeq ($(FCNAME), pgf90)
FCMODFLAGS  := -module $(INCDIR)
FCPREFLAGS  := -Mpreprocess
FCCUDAFLAGS := -Mcuda,cuda$(CUDA_MAJOR).$(CUDA_MINOR)
FCFLAGS     := -fast -O3
else
# non-pgf90 do not have CUDA support, compile our own CUDA lib
CUDAFORDEP  := $(LIBDIR)/$(LIBCUDAFOR) 
CUDALINK    := -L$(CUDA_LIB) -lcudart
CUDAFORLINK := -lcudafor
ifeq ($(FCNAME), gfortran)
FCMODFLAGS  := -J$(INCDIR)
FCPREFLAGS  += -cpp
FCFLAGS     += -ffree-line-length-none
else ifeq ($(FC), ifort)
FCMODFLAGS  := -module $(INCDIR)
FCPREFLAGS  += -fpp
endif
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

lib: $(CUDAFORDEP)
	make $(LIBDIR)/$(LIBTARGET)

$(LIBDIR)/$(LIBTARGET): $(OBJDIR)/ncclfor.o
	@printf "Linking   %-25s\n" $@
	$(FC) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) $< -o $(LIBDIR)/$(LIBTARGET)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(LIBCUDAFOR): $(OBJDIR)/cudafor.o
	@printf "Linking   %-25s\n" $@
	$(FC) -shared -Wl,--no-as-needed -Wl,-soname,libcudafor.so $< -o $(LIBDIR)/$(LIBCUDAFOR)

$(OBJDIR)/%.o: src/%.f90 
	@printf "Building  %-25s > %-24s\n" $< $@
	$(FC) -c $(FCMODFLAGS) $(FCPREFLAGS) -fPIC $(FCCUDAFLAGS) $(FCFLAGS) $< -o $@

TESTS := reduce_ptr_out allreduce_ptr_out reducescatter_ptr_out broadcast_ptr allgather_ptr_out
ifeq ($(FCNAME), pgf90)
TESTS += reduce_arr_out allreduce_arr_out reducescatter_arr_out broadcast_arr allgather_arr_out
endif

TESTDIR  := $(BUILDDIR)/test/fortran
TESTBINS := $(patsubst %,$(TESTDIR)/%,$(TESTS))

test: lib $(TESTBINS)

$(TESTDIR)/%: test/%.f90 lib
	@printf "Building  %-25s > %-24s\n" $< $@
	@mkdir -p $(TESTDIR)
	$(FC) $(FCCUDAFLAGS) $(FCFLAGS) $< $(CUDALINK) -I$(INCDIR) -L$(LIBDIR) $(CUDAFORLINK) $(LIBLINK) -o $@

clean:
	rm -f $(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(LIBCUDAFOR) $(OBJDIR)/*for.o $(INCDIR)/*.mod
	rm -rf $(TESTDIR)/

