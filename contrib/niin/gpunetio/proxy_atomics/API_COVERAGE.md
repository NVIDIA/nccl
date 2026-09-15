# NIIN proxy atomic API coverage contract

This document records the intended API contract for NIIN's experimental SRQ
proxy atomic plane. It deliberately lives beside the NIIN-owned proxy, rather
than in GIN or GPUNetIO: existing NIIN RMA, signaling, and synchronization
remain GIN operations.

The native GPUNetIO provider is the supported atomic path. This proxy remains
development-only until it has a safe CUDA ordering handoff for target heap
access and peer-visible failure recovery. In particular, a CPU GDRCopy BAR RMW
cannot safely service a self or LSA AMO issued by a still-running target GPU
kernel.

`include/niin/gpunetio/proxy_atomics/api.h` provides the corresponding
value-class and operation checks for the device dispatcher.

## Public operations and value classes

The prototype adapter aims to implement the integral rows below and the listed
floating base/extended forms. The proxy protocol carries raw 4- or 8-byte
integral values, so aliases such as `long`, `size_t`, and `ptrdiff_t` do not
need separate wire encodings.

| Public value class | Public API spelling(s) | Current NIIN surface | Required proxy operations |
| --- | --- | --- | --- |
| 32-bit integral | public aliases whose actual width is 32, such as `int`, `uint`, `int32`, `uint32` | yes | fetch-add/add, compare-swap, swap/set, fetch, inc/fetch-inc, fetch/non-fetch AND/OR/XOR |
| 64-bit integral | public aliases whose actual width is 64, such as `long`, `longlong`, `ulong`, `ulonglong`, `int64`, `uint64`, `size`, `ptrdiff` | yes | same integral operation set |
| `ptrdiff_t` | `nvshmem_ptrdiff_atomic_*` | yes | Encode as its actual 4- or 8-byte two's-complement representation; do not introduce a platform-width `ptrdiff` wire tag |
| `float` | base `nvshmem_float_atomic_{swap,fetch,set}` and extended `nvshmemx_float_atomic_{fetch_add,add}` | yes | raw-bit swap/fetch/set; target-side floating add/fetch-add |
| `double` | base `nvshmem_double_atomic_{swap,fetch,set}` and extended `nvshmemx_double_atomic_{fetch_add,add}` | yes | raw-bit swap/fetch/set; target-side floating add/fetch-add |
| `__half` | extended `nvshmemx_half_atomic_{fetch_add,add}` | yes | target-side binary16 add/fetch-add only |

The upstream NVSHMEM extended floating API is exactly the six
`nvshmemx_{half,float,double}_atomic_{fetch_add,add}` functions.  `__half`
is not a base `nvshmem_half_atomic_*` type.  In particular, this contract
does **not** invent floating compare-swap or bitwise AMOs that NVSHMEM does
not expose.

The floating adapter must not instantiate NIIN's existing generic
`niin_atomicAdd<T>` helper: that helper intentionally implements integral
modulo addition by passing raw bits through an unsigned CUDA atomic.  The
floating rows require their actual CUDA `atomicAdd(float*)`,
`atomicAdd(double*)`, and supported `atomicAdd(__half*)` overloads for
self/LSA access.  Likewise, a floating atomic fetch must use a bit-preserving
integer fetch (or the proxy) rather than `atomicAdd(value, 0)`, which can
canonicalize a NaN or change signed-zero representation.

The native GPUNetIO path continues to cover only the integral classes. The
prototype currently routes every AMO in the table through the proxy plane,
including self and LSA targets and otherwise-native integral ones. That is not
a safe public contract: a target CPU BAR RMW cannot be ordered with a target
GPU kernel that is already running. Mixing a target CPU read-modify-write
proxy with NIC AMOs against the same heap location is also not linearizable,
so a context binds one provider mode at a time.

## Wire-value rules

`niinGpunetioProxyAtomicType` identifies the *representation* rather than a
C typedef spelling:

- integral values are 32 or 64 bits;
- `float` is IEEE-754 binary32 and `double` is IEEE-754 binary64;
- `__half` is IEEE-754 binary16.

Operands, compare operands, and returned values are raw bits in 64-bit
protocol fields.  Only the low `sizeof(T)` bytes are meaningful.  Producers
and consumers must use `memcpy`/bit conversion rather than pointer casts so
that request-ring alignment does not become a language-aliasing requirement.
The request fields themselves need an explicitly specified byte order if the
protocol is ever used across heterogeneous hosts; the initial supported
deployment is homogeneous little-endian CUDA/IB hosts.

`ptrdiff_t` is a normal integral value on the wire.  The setup protocol must
collectively reject a PE set whose `sizeof(ptrdiff_t)` is not uniform instead
of assuming LP64 from a public symbol name.

## Required semantics

- Fetching operations return the value observed immediately before the
  target-side update.  Non-fetching operations still wait for the proxy
  completion before returning so program order is preserved.
- Compare-swap compares raw integral bits.  For the proxy's raw-bit
  float/double swap/fetch/set operations, NaN payloads and signed zero are
  copied unchanged.
- Integral add/inc use modulo-2^N arithmetic, matching the raw native AMO
  encoding.  Signedness does not change the wire opcode.
- Floating add is serialized at the target using the host's ordinary IEEE
  arithmetic and binary16 conversion for `__half`; it is not a source-GPU
  update or a NIC integer AMO. Normal finite operands are smoke-tested. Until
  dedicated edge tests exist, the proxy does not promise CUDA-bit-identical
  NaN, denormal, signed-zero, or half-rounding behavior for floating adds.
- A completion body (`resultBits`, `status`) must become visible before its
  ticket/valid word.  A two-write completion protocol with the ticket written
  last is the simplest safe design for a GPU polling the completion ring.

## ABI and routing boundaries

The proxy request needs an operation code, representation tag, target PE,
remote heap offset, operand bits, compare bits, slot, and monotonically
checked ticket.  Its completion needs result bits, status, and ticket.  Keep
both structures versioned and fixed-width; never place host pointers,
`size_t`, or CUDA/ibverbs structures on this wire.

The prototype device API currently selects an attached proxy-all context
before the self/LSA/native-direct branches. That preserves one CPU-RMW domain,
but is precisely why the current self/LSA behavior is development-only. A
supported design needs an explicit CUDA ordering handoff and a separate
failure-reporting path. The native-only mode retains the existing routing
exactly: self/LSA CUDA atomics and remote integral GPUNetIO WQEs.

Neither mode may borrow GIN state, QPs, MRs, or request formats.  The proxy
uses NIIN-owned queues/QPs/SRQ and is allowed to depend on public CUDA,
ibverbs, and GDRCopy APIs only.
