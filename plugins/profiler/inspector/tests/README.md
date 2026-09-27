# Proxy regression tests

On Linux with Python 3 and a C++14 compiler, run from the Inspector directory:

```sh
python3 tests/test_proxy.py
TEST_CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  python3 tests/test_proxy.py
```

The fixture compiles the real callbacks, pools, ring and JSON writer. Only
platform adapters are supplied; unused CUDA entry points abort if reached.
No GPU or third-party Python package is required. Set `CXX` to choose a compiler.

Coverage includes 164 timing fixtures (missing/zero/reversed timestamps,
bandwidth and SN output); lazy growth and allocation-failure recovery; buffer
recycling; flat/Op-only modes; FIFO parent retention at ring capacities 1–20;
Op/Step finalization order; same-dump merging; cross-dump tails and marker counts.
The lifecycle suite runs with SN output both off and on.

These regressions complement, rather than replace, real-GPU AllReduce/SendRecv
validation of environment parsing, network callbacks and output-reader compatibility.
