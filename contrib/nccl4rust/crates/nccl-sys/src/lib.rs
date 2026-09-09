// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw FFI bindings to NCCL's public host-callable API.
//!
//! The bindings are generated at build time from the public [`nccl.h`] and
//! [`nccl_device.h`] headers. This crate intentionally adds no ownership or
//! synchronization policy; use the higher-level `nccl` crate for that.
//!
//! [`nccl.h`]: https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
//! [`nccl_device.h`]: https://github.com/NVIDIA/nccl/blob/master/src/include/nccl_device.h

#![no_std]
#![allow(
    clippy::missing_safety_doc,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Sentinel used by NCCL's public configuration initializer.
///
/// The C macro expands through `INT_MIN`, which libclang does not reliably
/// expose to bindgen as an allowlisted macro.
pub const NCCL_CONFIG_UNDEF_INT: core::ffi::c_int = core::ffi::c_int::MIN;

/// Null-pointer sentinel used by NCCL's public configuration initializer.
///
/// Cast this untyped C equivalent to the pointer type required by the field.
pub const NCCL_CONFIG_UNDEF_PTR: *const core::ffi::c_void = core::ptr::null();

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn unique_id_matches_public_abi() {
        assert_eq!(size_of::<ncclUniqueId>(), NCCL_UNIQUE_ID_BYTES as usize);
        assert_eq!(align_of::<ncclUniqueId>(), align_of::<core::ffi::c_char>());
    }

    #[test]
    fn opaque_handles_are_pointer_sized() {
        assert_eq!(size_of::<ncclComm_t>(), size_of::<*mut core::ffi::c_void>());
        assert_eq!(
            size_of::<ncclWindow_t>(),
            size_of::<*mut core::ffi::c_void>()
        );
    }

    #[test]
    fn result_constants_match_the_stable_abi() {
        assert_eq!(ncclSuccess, 0);
        assert_eq!(ncclUnhandledCudaError, 1);
        assert_eq!(ncclInvalidArgument, 4);
    }

    #[test]
    fn configuration_sentinels_match_the_c_macros() {
        assert_eq!(NCCL_CONFIG_UNDEF_INT, i32::MIN);
        assert!(NCCL_CONFIG_UNDEF_PTR.is_null());
    }

    #[test]
    fn public_device_host_layouts_are_concrete() {
        assert_eq!(size_of::<ncclTeam_t>(), 12);
        assert_eq!(align_of::<ncclTeam_t>(), 4);

        // The exact values below cover the NCCL 2.31.0 public ABI in this
        // source tree. These checks also catch bindgen regressions that
        // accidentally turn complete public C++ structures into one-byte
        // opaque types.
        if NCCL_VERSION_CODE == 23100 && cfg!(target_pointer_width = "64") {
            assert_eq!(size_of::<ncclConfig_t>(), 112);
            assert_eq!(size_of::<ncclDevCommRequirements_t>(), 120);
            assert_eq!(size_of::<ncclCommProperties_t>(), 144);
            assert_eq!(size_of::<ncclDevComm_t>(), 248);
        }
    }
}
