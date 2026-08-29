TESTS := $(wildcard test_*.py)

test:
	@for t in $(TESTS); do echo "=== $$t ==="; python3 -m unittest $${t%.py} -v; echo; done

.PHONY: test
