This directory has been structured to make it easy for user to read the headers to learn the API. The files adjacent
to this README are meant for humans. They contain the essential declarations like which types exist and function prototypes and comments
indicating the contract/usage. Everything else goes into the "impl/" subdirectory. Most modules are stratified into three layers:

1) "foo.h" Public API declarations.
2) "impl/foo__types.h" struct definitions. Has #include of layer 1.
3) "impl/foo_funcs.h" inline functions. Has #include of layer 2.

The include dependencies should be acyclic for layers 1 and 2 since order matters for declarations and types. Layer 3 though
can freely have cycles amongst itself ("impl/foo__funcs.h" and "impl/bar__funcs.h" can mutually include each other) since
functions can be defined in any order once declared.

Translation units should just include "nccl_device.h" to ensure they get all the "impl/foo__funcs.h". But if a translation unit wants
to be more specific as to which module it pulls in it should include "impl/foo__funcs.h".

One of the nasty reasons this was required is because of C++ defaulted function parameters:

```
// +++ in foo.h +++
struct Foo; // defined in some __types.h

// +++ in "impl/foo__types.h" +++
struct Foo { int x; };

// +++ in "bar.h" +++
// Prototype function where default value is default construction of Foo. Since
// Foo would be incomplete if just including "foo.h" the compiler errors because
// it can't reason about the {}.
// I was able to solve this by including "impl/foo__types.h" instead.
#include "impl/foo__types.h"
void bar(Foo arg = {});
```
