.PHONY: build-CoMMA

all: build-CoMMA

build-CoMMA: clone-CoMMA
	cd CoMMA && cargo build

clone-CoMMA:
	@if [ ! -d CoMMA ] ; then \
		git clone https://github.com/google/CoMMA.git; \
		ln -s $(PWD)/.. CoMMA/third_party/nccl/ext-profiler; \
	fi

clean:
	@if [ -d CoMMA ] ; then \
		cd CoMMA && cargo clean; \
	fi

delete:
	@if [ -d CoMMA ] ; then \
		rm -rf CoMMA; \
	fi
