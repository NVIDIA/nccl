CC := gcc
CXX := g++
CFLAGS := -Wall -Wextra -Wno-format-truncation -g -std=c99 -fPIC -D_GNU_SOURCE
CXXFLAGS := -Wall -Wextra -g -std=c++14 -fPIC -D_GNU_SOURCE
LDFLAGS := -lpthread

C_SOURCES := $(wildcard test_*.c)
CXX_SOURCES := $(wildcard test_*.cc)
C_TARGETS := $(C_SOURCES:.c=)
CXX_TARGETS := $(CXX_SOURCES:.cc=)
TARGETS := $(C_TARGETS) $(CXX_TARGETS)

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test: $(TARGETS)
	@for t in $(TARGETS); do echo "=== $$t ==="; ./$$t; echo; done

clean:
	rm -f $(TARGETS)

.PHONY: all test clean
