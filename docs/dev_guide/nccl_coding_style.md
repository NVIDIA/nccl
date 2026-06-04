# NCCL C/C++ Code Style Guide

## General Guidelines
- Use 2 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Follow K&R brace style for C/CUDA code
- Use clear, descriptive variable names

## Naming Conventions
- Functions and variables: `ncclCamelCase`
- Macros and constants: `UPPER_CASE_WITH_UNDERSCORES`
- Struct/type names: `ncclFooBar`
- Some names may contain a suffix separated by `_`, such as versioned names for struct. Those are special cases, and should not be mistaken as snake case naming.

## CUDA-Specific
- Minimize register usage in performance-critical kernels
- Avoid warp divergence where possible
- Document kernel launch configurations and occupancy considerations
- Prefer `cudaLaunchKernel` over `<<<>>>` syntax

## Return Codes and Argument Checks
- NCCL functions should return a `ncclResult_t`
- All function calls should be guarded by `NCCLCHECK` or similar macro
- All external function calls should be guarded with `CUDACHECK`, `SYSCHECK`, `PTHREADCHECK`, etc.
- Use public helpers like `CommCheck` and `PtrCheck` to check function arguments.

## Memory Management
- Always allocate memory through NCCL alloc functions, e.g. `ncclCalloc()`
- Use appropriate memory fences for synchronization
- Free resources in reverse order of allocation

## Comments and Documentation
- Write clear comments for complex algorithms
- Document assumptions and invariants
- Explain "why" not just "what" for non-obvious code

## Language Standard
- `nccl.h` need to be C99 compatible
- Other codes need to be C++14 compatible. For example, use `if NCCL_IF_CONSTEXPR ()` rather than `if constexpr ()`.

## Code Formatting
- Avoid trailing white-spaces
- Try to match the existing style in files you're modifying

## Detailed Formatting Rules

These are the formatting rules that will be enforced by a formatting tool, and will be checked against every commit.

---

### Line Length

120-character limit. Longer lines are wrapped (see [Line Wrapping](#line-wrapping)).

---

### Blank Lines

- At most **1 blank line** between statements.
- No blank lines after `{` or before `}`.

---

### Indentation

2 spaces. No tabs.

```cpp
ncclResult_t foo() {
  if (x) {
    bar();
  }
}
```

Continuation lines (wrapped expressions, parameters) also use 2-space indent relative to the
statement start — unless aligning after an open parenthesis produces a shorter line (see [Line Wrapping](#line-wrapping)).

---

### Braces

K&R style: opening brace on the same line as the control statement or declaration. No exceptions
for functions, structs, classes, or namespaces.

```cpp
struct Foo {
  int x;
};

ncclResult_t bar(int n) {
  for (int i = 0; i < n; ++i) {
    baz(i);
  }
  return ncclSuccess;
}
```

---

### Spacing

**Control statements**

Space between keyword and `(`, space after `)` before `{`, no space inside parens:

```cpp
if (cond) { // not: if(cond){
if (cond1 && cond2) {
for (int i = 0; i < n; ++i) {
while (running) {
switch (type) {
// Not allowed
if ( cond ) {
```

<!-- #### Operators -->
<!---->
<!-- Spacing around operators is **not modified** by the formatter (set to `ignore`) to preserve -->

**Comma and Semicolon**

Space after `,` and `;` (in `for`); no space before:

```cpp
foo(a, b, c);
for (int i = 0; i < n; ++i)
```

**Pointers and References**

The `*` and `&` attach to the **type**, not the variable name:

```cpp
int* ptr;
const char* name;
void foo(struct ncclComm* comm, int* out);
```
This applies to declaration, definition and cast. Multiple definition on the same line involving pointers or references should be broken up.
```cpp
vptr = (void*)ptr; // not: vptr = (void *)ptr;
vptr = static_cast<void*>(ptr); // not: vptr = static_cast<void *>(ptr);
int *ptr1, *ptr2; // not allowed, must be broken

```

**C-style Casts**

No space after the cast:

```cpp
(CUmemAllocationHandleType)requestedHandleTypes // not: (CUmemAllocationHandleType) x
(void*)devComm
```

---

### Short Constructs on a Single Line

**If / else** — any single-statement if or if-else may stay on one line:

```cpp
if (version == NULL) return ncclInvalidArgument;
if (x) do_this();
else do_that();
```

* If the statement cannot fit on a single-line, it must be enclosed in braces.
* If any of `if` or `else` branch has braces, they both need to have braces.
* Nesting of control statements should have braces except for the inner-most one.

```cpp
if (0 == p2p && i != cudaDev) {
  INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i);
}

if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
  if (flags & ConnFifoEnabled) {
    stmt_1;
    stmt_2;
  }
  if (cond) do_something; // OK to have one-liner here
}

// Not OK examples
// Should add braces if branch body does not fit in single line with if.
if (0 == p2p && i != cudaDev)
  INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i);

if (0 == p2p && i != cudaDev) INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d",
                                   cudaDev, i);

// Nesting ifs should have braces. Otherwise, adding else later could have unexpected result.
if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend))
  if (flags & ConnFifoEnabled)

// if and else branch need to be consistent with braces
if (x) do_this();
else {
  do_that();
  do_more();
}

```

**Loops** — single-statement loops may stay on one line:

```cpp
for (int i = 0; i < n; ++i) maxPolicy |= policies[i];
while (token) token = strtok(NULL, "|");
```

* If loop statement cannot fit on a single-line, it must be enclosed in braces.

**Inline functions** — inline function bodies that fit on one line stay on one line:

```cpp
inline int count() { return n_; }
```

---

### Line Wrapping

**Long Functions**

When a call or declaration exceeds column limit, arguments are **bin-packed** — as many fit on the
first line; the rest wrap and **align under the first argument** (after the opening parenthesis):

```cpp
// call
ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount,
                &chunkCount);

// definition
static ncclResult_t addProfilerProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                               struct ncclProxyOp* op) {
```

When there is no room to place even the first argument on the opening line, break immediately after
`(` and indent by 2:

```cpp
static ncclResult_t scheduleCollTasksToPlan(
  struct ncclComm* comm, struct ncclTaskColl* task, int nChannels, size_t nBytes,
  /*outputs*/ uint32_t* outChunkSize, struct ncclProxyOp* proxyOp);

// If force BinPacking, it would be
static ncclResult_t scheduleCollTasksToPlan(struct ncclComm* comm, struct ncclTaskColl* task,
                                            int nChannels, size_t nBytes,
                                            /*outputs*/ uint32_t* outChunkSize,
                                            struct ncclProxyOp* proxyOp);
```

**Breaking at Binary Operators**

Break **after** the operator, not before:

```cpp
// correct — operator stays on the line it belongs to
ok |= (batchBytes + workBytes <= budget->inArgsBytes);
ok |= (batchBytes <= budget->inArgsBytes) &&
      (workBytes <= budget->outArgsBytes);
```

Ternary operators (`? :`) are also not broken before `?`; they stay on the same line when they fit.

---

### switch / case

`case` labels are at the **same indent level** as `switch`. No extra indentation.

```cpp
switch (env[0]) {
case '0':
  ctaPolicyEnv = NCCL_CTA_POLICY_DEFAULT;
  break;
case '1':
  ctaPolicyEnv = NCCL_CTA_POLICY_EFFICIENCY;
  break;
default:
  INFO(NCCL_ENV, "Unknown policy.");
}
```

---

### Enums

Each enumerator on its own line. Short single-line enums are not allowed.

```cpp
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
} ncclResult_t;
```

---

### Bit-fields

No spaces around the colon:

```cpp
int32_t nMaxChannels:8;
int32_t algorithm:8, protocol:8;
uint32_t isCollnet:1, isNvls:1;
```

---

### Initializer Lists

No spaces inside `{}` for numeric/uniform initialization; spaces for string arrays:

```cpp
int arr[3] = {0, 1, 2};
void* args[] = {(void*)devComm};

// String arrays get spaces:
const char* ncclFuncStr[] = {"Broadcast", "Reduce", "AllGather", "ReduceScatter",
                             "AllReduce"};

static const float nvlinkBws[NCCL_NVLINK_BW_IDX_NUM] = {
  360.0f, // Hopper
  720.0f, // Blackwell
};
```

---

### Preprocessor Directives

`#if`, `#else`, `#endif`, `#define` are placed at **column 0**, regardless of surrounding
indentation. Code also do not indent relative to a preprocessor directive.

```cpp
__device__ __forceinline__ uint64_t ld_relaxed(uint64_t* ptr) {
  uint64_t ans;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.relaxed.sys.global.u64 ...");
#else
  asm volatile("ld.volatile.global.u64 ...");
#endif
  return ans;
}
```

Multiline `#define` continuation is indented 2 spaces from `#`:

```cpp
#define NCCL_FUNC(name, args) \
  ncclResult_t name(args)
```

---

### Preprocessor Macros

**Consecutive Macro Definitions** — values are not column-aligned

```cpp
#define STR2(v) #v
#define STR(v) STR2(v)

#define NCCL_SPLIT_NOCOLOR -1
#define NCCL_UNDEF_FLOAT -1.0f
```

This is to avoid change in on macro causing reformatting of entire macro definition block.

**Multi-line Macros** — backslash at one space after the line

```cpp
#define NOWARN(EXPR, FLAGS) \
  do { \
    int oldNoWarn = ncclDebugNoWarn; \
    ncclDebugNoWarn = FLAGS; \
    (EXPR); \
    ncclDebugNoWarn = oldNoWarn; \
  } while (0)
```

This also help avoid unnecessary reformatting when macro is changed.

---

### Include Order

New `#include` directive should be added following the existing groupping. Do not reorder or regroup existing include directives.

---

### Comments

Keep one space to separate comment content and comment token, unless it is embedded comment.
```cpp
// OK
/* OK */
//NOT OK
/*NOT OK*/
func(arg1, arg2, /*force=*/true); // OK
```

Block comments using C style `/* */` comment must use ` * ` continuation:
```cpp
/* Something
 * something
 */

// Not
/* Something
* something
*/
```

Trailing comments of adjacent statements do not align.
```cpp
a = b; // a simple assignment
c = dd; // comment do not align with previous line
```

Long trailing comments should be move to before the statement.
```cpp
// for initial rank <-> root information exchange
union ncclSocketAddress* rankAddressesRoot = NULL;

// Not this, trailing comment exceeds column limit
union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
```

<!-- C++ Related Styles -->

### Class, Struct and Similar Definitions

* No indentation for `public`, `private`, `protected`. They stay at the same column as their corresponding `class`.
* Add a blank line before access specifiers if they start a new logic block.
```cpp
class RingAlgorithm {
protected:
  int refCount;
  ...

public:
  virtual ~RingAlgorithm() {};
};
```

Spaces around `:` for inheritance or for underlying type of a enum class.

```cpp
struct IsFloatingPoint : std::false_type {};
class RingAlgorithm : public BaseAlgorithm {};
enum Flags : uint64_t {};
enum class Flags : uint64_t {};
```

---
### Templates

`template` is always on its own line. A space appears between `template` and `<`:

```cpp
template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work);
};

template <typename T>
T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n = 1);
```

When the template parameter list itself overflows column limit, it wraps with 2-space indent:

```cpp
template <typename T, typename RedOp, typename Proto,
  bool isNetOffload = false>
__device__ __forceinline__ void runRing(...) {
```

Forward declarations with the template on the same line are reformatted to break:

```cpp
// Before:
template <typename T> struct FuncPreMulSum;

// After:
template <typename T>
struct FuncPreMulSum;
```

---

### Constructor Initializer Lists

The list initializers of a constructor start on a new line, break before the `:`.

```cpp
ncclGin_C::ncclGin_C(ncclDevComm const& comm, unsigned backendMask, int contextIndex)
  : comm(comm), backendMask(backendMask) {
  ncclGinInitCommon(this, comm, contextIndex);
}
```

---

### Namespaces

Namespace bodies are **not** indented. The closing brace carries a `// namespace` comment
(auto-inserted by clang-format):

```cpp
namespace nccl {

void foo();

} // namespace nccl
```

Anonymous namespaces follow the same rule:

```cpp
namespace {

template <typename T>
__device__ void helper() { ... }

} // namespace
```

---

**Lambda**

* Lambdas follow the same brace, spacing and wrapping rules as regular functions.
* Lambda body should be aligned relative to the signature.
```cpp
someMethod([](SomeReallyLongLambdaSignatureArgument foo) {
             return;
           });

auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
  return dstPtr;
};
```
