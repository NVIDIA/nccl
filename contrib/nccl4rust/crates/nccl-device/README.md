# nccl-device

Rust-style `no_std` wrappers over `nccl-device-sys`. The crate provides borrowed
device communicator, window, and multimem-handle types; team queries; pointer
translation; LSA barriers; and the first typed LSA reduce/copy primitive.

See the workspace `README.md` for the CUDA-Oxide and nvJitLink flow. The current
pointer, barrier, and reduce/copy methods use documented `unsafe` contracts for
memory validity, mapping, convergence, and participation. Higher-level wrappers
can encode more of those requirements over time.
