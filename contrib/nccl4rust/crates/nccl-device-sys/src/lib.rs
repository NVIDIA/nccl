// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw device-side ABI for nccl4rust.
//!
//! NCCL's public device API is a CUDA C++ template API. The C ABI declared in
//! this crate is implemented by `shim/nccl4rust_device.cu`, which includes only
//! public NCCL headers and is linked into a CUDA-Oxide kernel as LTOIR.

#![no_std]
#![allow(non_camel_case_types)]

use core::ffi::c_void;

#[cfg(feature = "cuda-oxide")]
use cuda_device::device;

/// Opaque public `ncclDevComm_t`.
///
/// Rust device code deliberately uses a pointer to a device-resident copy of
/// this object. That keeps the versioned C layout out of kernel argument ABIs.
#[repr(C)]
pub struct ncclDevComm_t {
    _private: [u8; 0],
}

/// Opaque public `ncclWindow_t` handle.
pub type ncclWindow_t = *mut c_void;

/// Public `ncclMultimemHandle_t` represented by its single pointer field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ncclMultimemHandle_t {
    pub mc_base_ptr: *mut c_void,
}

/// Public `ncclTeam_t` value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ncclTeam_t {
    pub n_ranks: i32,
    pub rank: i32,
    pub stride: i32,
}

#[cfg_attr(feature = "cuda-oxide", device)]
unsafe extern "C" {
    pub fn nccl4rust_dev_comm_rank(comm: *const ncclDevComm_t) -> i32;
    pub fn nccl4rust_dev_comm_n_ranks(comm: *const ncclDevComm_t) -> i32;
    pub fn nccl4rust_dev_comm_lsa_rank(comm: *const ncclDevComm_t) -> i32;
    pub fn nccl4rust_dev_comm_lsa_size(comm: *const ncclDevComm_t) -> i32;

    pub fn nccl4rust_team_rank_to_world(
        comm: *const ncclDevComm_t,
        team_n_ranks: i32,
        team_rank: i32,
        team_stride: i32,
        rank: i32,
    ) -> i32;
    pub fn nccl4rust_team_rank_to_lsa(
        comm: *const ncclDevComm_t,
        team_n_ranks: i32,
        team_rank: i32,
        team_stride: i32,
        rank: i32,
    ) -> i32;

    pub fn nccl4rust_get_local_pointer(window: ncclWindow_t, offset: usize) -> *mut c_void;
    pub fn nccl4rust_get_lsa_pointer(window: ncclWindow_t, offset: usize, peer: i32)
    -> *mut c_void;
    pub fn nccl4rust_get_peer_pointer(
        window: ncclWindow_t,
        offset: usize,
        peer: i32,
    ) -> *mut c_void;
    pub fn nccl4rust_get_peer_pointer_team(
        window: ncclWindow_t,
        offset: usize,
        team_n_ranks: i32,
        team_rank: i32,
        team_stride: i32,
        peer: i32,
    ) -> *mut c_void;
    pub fn nccl4rust_get_multimem_pointer(
        window: ncclWindow_t,
        offset: usize,
        mc_base_ptr: *mut c_void,
    ) -> *mut c_void;
    pub fn nccl4rust_get_lsa_multimem_pointer(
        window: ncclWindow_t,
        offset: usize,
        comm: *const ncclDevComm_t,
    ) -> *mut c_void;

    pub fn nccl4rust_lsa_barrier_thread(comm: *const ncclDevComm_t, index: u32, multimem: bool);
    pub fn nccl4rust_lsa_barrier_warp(comm: *const ncclDevComm_t, index: u32, multimem: bool);
    pub fn nccl4rust_lsa_barrier_cta(comm: *const ncclDevComm_t, index: u32, multimem: bool);

    /// LSA f32 reduce-sum-copy using all threads in the CTA.
    ///
    /// This primitive does not add entry/exit barriers. Callers must establish
    /// the ordering required by the public NCCL reduce/copy API.
    pub fn nccl4rust_lsa_reduce_sum_copy_f32_cta(
        comm: *const ncclDevComm_t,
        src_window: ncclWindow_t,
        src_offset: usize,
        dst_window: ncclWindow_t,
        dst_offset: usize,
        count: usize,
    );
}

const _: () = {
    assert!(core::mem::size_of::<ncclTeam_t>() == 12);
    assert!(core::mem::align_of::<ncclTeam_t>() == 4);
    assert!(core::mem::offset_of!(ncclTeam_t, n_ranks) == 0);
    assert!(core::mem::offset_of!(ncclTeam_t, rank) == 4);
    assert!(core::mem::offset_of!(ncclTeam_t, stride) == 8);
    assert!(core::mem::size_of::<ncclMultimemHandle_t>() == core::mem::size_of::<usize>());
    assert!(core::mem::align_of::<ncclMultimemHandle_t>() == core::mem::align_of::<usize>());
    assert!(core::mem::offset_of!(ncclMultimemHandle_t, mc_base_ptr) == 0);
    assert!(core::mem::size_of::<bool>() == 1);
    assert!(core::mem::align_of::<bool>() == 1);
};

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn team_layout_and_fields_match_the_public_abi() {
        let team = ncclTeam_t {
            n_ranks: 8,
            rank: 3,
            stride: 2,
        };

        assert_eq!(size_of::<ncclTeam_t>(), 3 * size_of::<i32>());
        assert_eq!(align_of::<ncclTeam_t>(), align_of::<i32>());
        assert_eq!(offset_of!(ncclTeam_t, n_ranks), 0);
        assert_eq!(offset_of!(ncclTeam_t, rank), size_of::<i32>());
        assert_eq!(offset_of!(ncclTeam_t, stride), 2 * size_of::<i32>());
        assert_eq!((team.n_ranks, team.rank, team.stride), (8, 3, 2));
    }

    #[test]
    fn multimem_handle_is_one_pointer_wide() {
        let pointer = core::ptr::without_provenance_mut::<c_void>(0x1234);
        let handle = ncclMultimemHandle_t {
            mc_base_ptr: pointer,
        };

        assert_eq!(size_of::<ncclMultimemHandle_t>(), size_of::<*mut c_void>());
        assert_eq!(
            align_of::<ncclMultimemHandle_t>(),
            align_of::<*mut c_void>()
        );
        assert_eq!(offset_of!(ncclMultimemHandle_t, mc_base_ptr), 0);
        assert_eq!(handle.mc_base_ptr, pointer);
    }

    #[test]
    fn opaque_handles_have_the_expected_representation() {
        assert_eq!(size_of::<ncclDevComm_t>(), 0);
        assert_eq!(align_of::<ncclDevComm_t>(), 1);
        assert_eq!(size_of::<ncclWindow_t>(), size_of::<*mut c_void>());
        assert_eq!(align_of::<ncclWindow_t>(), align_of::<*mut c_void>());
    }

    #[cfg(not(feature = "cuda-oxide"))]
    mod host_c_abi_signatures {
        use super::*;

        // These assignments intentionally have no runtime assertions:
        // compiling them detects accidental changes to the host C ABI
        // declarations. `#[device]` gives these declarations Rust ABI inside
        // a CUDA-Oxide build, so that configuration is compile-checked by the
        // device smoke package instead.
        const _: unsafe extern "C" fn(*const ncclDevComm_t) -> i32 = nccl4rust_dev_comm_rank;
        const _: unsafe extern "C" fn(*const ncclDevComm_t) -> i32 = nccl4rust_dev_comm_n_ranks;
        const _: unsafe extern "C" fn(*const ncclDevComm_t) -> i32 = nccl4rust_dev_comm_lsa_rank;
        const _: unsafe extern "C" fn(*const ncclDevComm_t) -> i32 = nccl4rust_dev_comm_lsa_size;
        const _: unsafe extern "C" fn(*const ncclDevComm_t, i32, i32, i32, i32) -> i32 =
            nccl4rust_team_rank_to_world;
        const _: unsafe extern "C" fn(*const ncclDevComm_t, i32, i32, i32, i32) -> i32 =
            nccl4rust_team_rank_to_lsa;
        const _: unsafe extern "C" fn(ncclWindow_t, usize) -> *mut c_void =
            nccl4rust_get_local_pointer;
        const _: unsafe extern "C" fn(ncclWindow_t, usize, i32) -> *mut c_void =
            nccl4rust_get_lsa_pointer;
        const _: unsafe extern "C" fn(ncclWindow_t, usize, i32) -> *mut c_void =
            nccl4rust_get_peer_pointer;
        const _: unsafe extern "C" fn(ncclWindow_t, usize, i32, i32, i32, i32) -> *mut c_void =
            nccl4rust_get_peer_pointer_team;
        const _: unsafe extern "C" fn(ncclWindow_t, usize, *mut c_void) -> *mut c_void =
            nccl4rust_get_multimem_pointer;
        const _: unsafe extern "C" fn(ncclWindow_t, usize, *const ncclDevComm_t) -> *mut c_void =
            nccl4rust_get_lsa_multimem_pointer;
        const _: unsafe extern "C" fn(*const ncclDevComm_t, u32, bool) =
            nccl4rust_lsa_barrier_thread;
        const _: unsafe extern "C" fn(*const ncclDevComm_t, u32, bool) = nccl4rust_lsa_barrier_warp;
        const _: unsafe extern "C" fn(*const ncclDevComm_t, u32, bool) = nccl4rust_lsa_barrier_cta;
        const _: unsafe extern "C" fn(
            *const ncclDevComm_t,
            ncclWindow_t,
            usize,
            ncclWindow_t,
            usize,
            usize,
        ) = nccl4rust_lsa_reduce_sum_copy_f32_cta;
    }
}
