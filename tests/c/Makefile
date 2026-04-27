CC := gcc
CFLAGS := -Wall -Wextra -Wno-format-truncation -g -std=c99 -fPIC -D_GNU_SOURCE
LDFLAGS := -lpthread

SOURCES := $(wildcard test_*.c)
TARGETS := $(SOURCES:.c=)

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: $(TARGETS)
	@for t in $(TARGETS); do echo "=== $$t ==="; ./$$t; echo; done

clean:
	rm -f $(TARGETS)

.PHONY: all test clean
