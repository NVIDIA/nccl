# NCCL Tuner Plugin Unit Tests

This directory contains comprehensive unit tests for the NCCL tuner plugin. The tests verify all major functionality including configuration parsing, matching logic, and cost table updates.

## Test Structure

```
test/
├── test_plugin.c     # Main unit test file
├── Makefile          # Build system for tests
└── README.md         # This file
```

## Building and Running Tests

### Quick Start

```bash
# Build and run all tests
make test

# Or step by step
make           # Build test executable
./test_plugin  # Run tests
```

### Advanced Testing

```bash
# Run with memory leak detection (requires valgrind)
make test-memory

# Run with verbose logging
make test-verbose

# Generate code coverage report (requires gcov)
make coverage

# Create sample test configuration files
make test-configs
```

## Test Coverage

The unit tests cover the following functionality:

### 1. **Plugin Initialization (`test_plugin_init`)**
- Tests successful plugin initialization
- Verifies context allocation
- Tests cleanup on destroy

### 2. **Configuration Parsing (`test_config_parsing_valid`, `test_config_parsing_invalid`)**
- Valid CSV format parsing
- Comment and empty line handling
- Invalid format graceful handling
- Environment variable configuration

### 3. **Collective Type Matching (`test_collective_matching`)**
- Correct matching of allreduce, broadcast, etc.
- Algorithm/protocol selection
- Channel configuration

### 4. **Size Range Matching (`test_size_matching`)**
- Small, medium, large message size handling
- Proper range boundary checking
- Multiple size-based configurations

### 5. **Topology Matching (`test_topology_matching`)**
- Single-node vs multi-node configurations
- Exact nNodes/nRanks matching
- Wildcard matching (-1 values)

### 6. **Default Channels (`test_default_channels`)**
- Proper handling of -1 channel specification
- Preservation of NCCL default behavior

### 7. **Registered Buffer Matching (`test_regbuff_matching`)**
- Configurations based on regBuff parameter
- Registered vs non-registered buffer handling
- Backward compatibility with configs missing regBuff

### 8. **Pipeline Operations Matching (`test_pipeops_matching`)**
- Configurations based on numPipeOps parameter
- Single vs multiple pipeline operation handling
- Backward compatibility with configs missing numPipeOps

### 9. **Fallback Behavior (`test_no_match_fallback`)**
- Default behavior when no config matches
- Ring/Simple algorithm fallback

## Test Output

Successful test run:
```
Running NCCL Tuner Plugin Unit Tests
=====================================
PASS: test_plugin_init
PASS: test_config_parsing_valid
PASS: test_config_parsing_invalid
PASS: test_collective_matching
PASS: test_size_matching
PASS: test_topology_matching
PASS: test_default_channels
PASS: test_regbuff_matching
PASS: test_pipeops_matching
PASS: test_no_match_fallback

=====================================
Test Results: 9/9 tests passed
All tests PASSED!
```

Failed test example:
```
FAIL: test_collective_matching - Tree/Simple should have low cost
Test Results: 8/9 tests passed
Some tests FAILED!
```

## Mock NCCL Implementation

The tests use the actual NCCL header files from the `../nccl/` directory:

- `tuner.h` - Complete NCCL tuner interface and type definitions
- `common.h` - Common NCCL types and logging functions
- `err.h` - NCCL error codes

This allows testing with the real NCCL interface definitions while still being able to run tests without the full NCCL library installation.

## Integration with CI/CD

```bash
# Install tests for CI/CD pipeline
make install-test

# Run as part of automated testing
make test && echo "Tests passed" || echo "Tests failed"
```

## Memory Testing

The tests can be run with valgrind for memory leak detection:

```bash
make test-memory
```

This will detect:
- Memory leaks
- Invalid memory access
- Use of uninitialized memory

## Code Coverage

Generate code coverage reports to ensure comprehensive testing:

```bash
make coverage
# Creates test_plugin.c.gcov with line-by-line coverage
```

## Adding New Tests

To add a new test:

1. Create a new test function in `test_plugin.c`:
```c
int test_new_feature() {
  // Test setup
  TEST_ASSERT(condition, "description");
  // Test cleanup
  TEST_PASS();
}
```

2. Add the test to the main function:
```c
total++; passed += test_new_feature();
```

3. Rebuild and run:
```bash
make test
```

## Debugging Tests

For debugging failed tests:

```bash
# Compile with debug symbols
make CFLAGS="-g -O0 -DDEBUG"

# Run with gdb
gdb ./test_plugin
```

## Cleaning Up

```bash
# Remove all build artifacts and temporary files
make clean
```

This comprehensive test suite ensures the NCCL tuner plugin works correctly across all supported configurations and edge cases.
