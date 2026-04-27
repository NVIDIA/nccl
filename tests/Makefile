SUBDIRS := $(shell ls -d c python 2>/dev/null)

test:
	@for d in $(SUBDIRS); do $(MAKE) -C $$d test; done

.PHONY: test
