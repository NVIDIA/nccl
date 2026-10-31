# CPU-only test for the ib_write_bw hostname check. No CUDA or GPU needed.
# Needs GoogleTest (pkg-config: gtest, gtest_main).

CXX ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -g
GTEST_CFLAGS := $(shell pkg-config --cflags gtest_main)
GTEST_LIBS := $(shell pkg-config --libs gtest_main)

test_hostname_safe: test_hostname_safe.cpp ../hostname.h
	$(CXX) $(CXXFLAGS) $(GTEST_CFLAGS) $< $(GTEST_LIBS) -pthread -o $@

.PHONY: check clean
check: test_hostname_safe
	./test_hostname_safe

clean:
	rm -f test_hostname_safe
