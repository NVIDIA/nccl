# nccl-device-sys

Raw `no_std` Rust declarations for the C symbols emitted by
`../../shim/nccl4rust_device.cu`. Enable `cuda-oxide` when compiling device
code and link the resulting kernel PTX (or compatible kernel LTOIR) with the
matching per-architecture `libnccl4rust_device_sm_<arch>.ltoir`.

This crate is internal to the experimental `contrib/nccl4rust` workspace; its
ABI is versioned together with the shim and the NCCL public headers used to
compile it.
