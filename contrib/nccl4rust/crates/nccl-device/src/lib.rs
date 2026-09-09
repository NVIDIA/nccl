// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-style, `no_std` wrappers around [`nccl_device_sys`].
//!
//! The wrapper does not own NCCL host state or CUDA memory. A [`DevComm`] must
//! point at a device-resident copy of the public communicator produced by
//! `ncclDevCommCreate`, and a [`Window`] remains owned by its host communicator.
//! These functions are intended to be inlined into CUDA-Oxide device kernels;
//! they are not CPU-callable host wrappers.

#![no_std]

use core::ffi::c_void;
use core::marker::PhantomData;

pub use nccl_device_sys as sys;

/// Borrowed device pointer to a public NCCL device communicator.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DevComm<'a> {
    raw: *const sys::ncclDevComm_t,
    _lifetime: PhantomData<&'a sys::ncclDevComm_t>,
}

impl<'a> DevComm<'a> {
    /// Borrow a device-resident `ncclDevComm_t`.
    ///
    /// # Safety
    ///
    /// `raw` must point to a live, correctly aligned device copy produced by a
    /// compatible NCCL runtime, and that allocation must outlive `'a`.
    pub const unsafe fn from_raw(raw: *const sys::ncclDevComm_t) -> Self {
        Self {
            raw,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_raw(self) -> *const sys::ncclDevComm_t {
        self.raw
    }
}

/// A non-owning NCCL registered-window handle.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Window<'a> {
    raw: sys::ncclWindow_t,
    _lifetime: PhantomData<&'a c_void>,
}

impl<'a> Window<'a> {
    /// Borrow a window registered on the host.
    ///
    /// # Safety
    ///
    /// `raw` must remain registered for `'a`, and all requested byte ranges
    /// must be inside that registration.
    pub const unsafe fn from_raw(raw: sys::ncclWindow_t) -> Self {
        Self {
            raw,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_raw(self) -> sys::ncclWindow_t {
        self.raw
    }
}

/// A non-owning multimem mapping associated with an NCCL device communicator.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultimemHandle<'a> {
    raw: sys::ncclMultimemHandle_t,
    _lifetime: PhantomData<&'a sys::ncclDevComm_t>,
}

impl<'a> MultimemHandle<'a> {
    /// Borrow a multimem handle populated by `ncclDevCommCreate`.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid handle produced by a compatible NCCL runtime,
    /// and its parent device communicator and resources must outlive `'a`.
    pub const unsafe fn from_raw(raw: sys::ncclMultimemHandle_t) -> Self {
        Self {
            raw,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_raw(self) -> sys::ncclMultimemHandle_t {
        self.raw
    }
}

/// Value-type NCCL team descriptor.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Team(sys::ncclTeam_t);

impl Team {
    pub const fn size(self) -> i32 {
        self.0.n_ranks
    }

    pub const fn rank(self) -> i32 {
        self.0.rank
    }

    pub const fn stride(self) -> i32 {
        self.0.stride
    }

    pub const fn as_raw(self) -> sys::ncclTeam_t {
        self.0
    }
}

#[inline(always)]
pub fn rank(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_rank(comm.as_raw()) }
}

#[inline(always)]
pub fn size(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_n_ranks(comm.as_raw()) }
}

#[inline(always)]
pub fn lsa_rank(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_lsa_rank(comm.as_raw()) }
}

#[inline(always)]
pub fn lsa_size(comm: DevComm<'_>) -> i32 {
    unsafe { sys::nccl4rust_dev_comm_lsa_size(comm.as_raw()) }
}

#[inline(always)]
pub fn world(comm: DevComm<'_>) -> Team {
    Team(sys::ncclTeam_t {
        n_ranks: size(comm),
        rank: rank(comm),
        stride: 1,
    })
}

#[inline(always)]
pub fn lsa(comm: DevComm<'_>) -> Team {
    Team(sys::ncclTeam_t {
        n_ranks: lsa_size(comm),
        rank: lsa_rank(comm),
        stride: 1,
    })
}

/// Return the rail team containing the communicator's rank.
///
/// A live NCCL device communicator always has a positive LSA size, which is
/// required by the divisions used to derive the rail size and rank.
#[inline(always)]
pub fn rail(comm: DevComm<'_>) -> Team {
    let n_ranks = size(comm);
    let rank = rank(comm);
    let lsa_size = lsa_size(comm);
    Team(sys::ncclTeam_t {
        n_ranks: n_ranks / lsa_size,
        rank: rank / lsa_size,
        stride: lsa_size,
    })
}

/// Translate a rank in `team` to its world rank.
///
/// Returns `None` when `rank` is outside the team.
#[inline(always)]
pub fn rank_to_world(comm: DevComm<'_>, team: Team, rank: i32) -> Option<i32> {
    if rank < 0 || rank >= team.size() {
        None
    } else {
        Some(rank_to_world_device(
            comm,
            team.size(),
            team.rank(),
            team.stride(),
            rank,
        ))
    }
}

#[inline(always)]
fn rank_to_world_device(
    comm: DevComm<'_>,
    team_n_ranks: i32,
    team_rank: i32,
    team_stride: i32,
    rank: i32,
) -> i32 {
    unsafe {
        sys::nccl4rust_team_rank_to_world(comm.as_raw(), team_n_ranks, team_rank, team_stride, rank)
    }
}

/// Translate a rank in `team` to its LSA rank.
///
/// Returns `None` when `rank` is outside the team or the translated rank does
/// not belong to this communicator rank's local LSA team.
#[inline(always)]
pub fn rank_to_lsa(comm: DevComm<'_>, team: Team, rank: i32) -> Option<i32> {
    if rank < 0 || rank >= team.size() {
        None
    } else {
        let translated = rank_to_lsa_device(comm, team.size(), team.rank(), team.stride(), rank);
        if translated < 0 || translated >= lsa_size(comm) {
            None
        } else {
            Some(translated)
        }
    }
}

#[inline(always)]
fn rank_to_lsa_device(
    comm: DevComm<'_>,
    team_n_ranks: i32,
    team_rank: i32,
    team_stride: i32,
    rank: i32,
) -> i32 {
    unsafe {
        sys::nccl4rust_team_rank_to_lsa(comm.as_raw(), team_n_ranks, team_rank, team_stride, rank)
    }
}

#[inline(always)]
fn local_ptr_device(window: Window<'_>, byte_offset: usize) -> *mut c_void {
    unsafe { sys::nccl4rust_get_local_pointer(window.as_raw(), byte_offset) }
}

/// Translate a registered window offset to this rank's local pointer.
///
/// # Safety
///
/// `byte_offset` and the resulting `T` access must be in bounds and correctly
/// aligned for the registered window. The caller must also uphold all aliasing
/// and synchronization requirements for accesses through the returned pointer.
#[inline(always)]
pub unsafe fn local_ptr<T>(window: Window<'_>, byte_offset: usize) -> *mut T {
    local_ptr_device(window, byte_offset).cast()
}

#[inline(always)]
fn lsa_ptr_device(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut c_void {
    unsafe { sys::nccl4rust_get_lsa_pointer(window.as_raw(), byte_offset, peer) }
}

/// Translate a registered window offset for an LSA-local peer.
///
/// # Safety
///
/// `peer` must name a member of the window's LSA team. `byte_offset` and the
/// resulting `T` access must be in bounds and correctly aligned, and the
/// caller must uphold all aliasing and synchronization requirements.
#[inline(always)]
pub unsafe fn lsa_ptr<T>(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut T {
    lsa_ptr_device(window, byte_offset, peer).cast()
}

#[inline(always)]
fn peer_ptr_device(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut c_void {
    unsafe { sys::nccl4rust_get_peer_pointer(window.as_raw(), byte_offset, peer) }
}

/// Translate a registered window offset for a world peer.
///
/// # Safety
///
/// `peer` must be LSA-accessible from this rank. `byte_offset` and the
/// resulting `T` access must be in bounds and correctly aligned, and the
/// caller must uphold all aliasing and synchronization requirements.
#[inline(always)]
pub unsafe fn peer_ptr<T>(window: Window<'_>, byte_offset: usize, peer: i32) -> *mut T {
    peer_ptr_device(window, byte_offset, peer).cast()
}

#[inline(always)]
fn team_peer_ptr_device(
    window: Window<'_>,
    byte_offset: usize,
    team_n_ranks: i32,
    team_rank: i32,
    team_stride: i32,
    peer: i32,
) -> *mut c_void {
    unsafe {
        sys::nccl4rust_get_peer_pointer_team(
            window.as_raw(),
            byte_offset,
            team_n_ranks,
            team_rank,
            team_stride,
            peer,
        )
    }
}

/// Translate a registered window offset for a peer in `team`.
///
/// # Safety
///
/// `team` and `peer` must describe an LSA-accessible peer for this window.
/// `byte_offset` and the resulting `T` access must be in bounds and correctly
/// aligned, and the caller must uphold all aliasing and synchronization
/// requirements.
#[inline(always)]
pub unsafe fn team_peer_ptr<T>(
    window: Window<'_>,
    byte_offset: usize,
    team: Team,
    peer: i32,
) -> *mut T {
    team_peer_ptr_device(
        window,
        byte_offset,
        team.size(),
        team.rank(),
        team.stride(),
        peer,
    )
    .cast()
}

#[inline(always)]
fn multimem_ptr_device(
    window: Window<'_>,
    byte_offset: usize,
    handle: MultimemHandle<'_>,
) -> *mut c_void {
    unsafe {
        sys::nccl4rust_get_multimem_pointer(
            window.as_raw(),
            byte_offset,
            handle.as_raw().mc_base_ptr,
        )
    }
}

/// Translate a registered window offset through a multimem team mapping.
///
/// # Safety
///
/// `handle` must be compatible with `window`. `byte_offset` and the resulting
/// `T` access must be in bounds and correctly aligned, the selected system
/// must support multimem, and the caller must uphold all aliasing and
/// synchronization requirements.
#[inline(always)]
pub unsafe fn multimem_ptr<T>(
    window: Window<'_>,
    byte_offset: usize,
    handle: MultimemHandle<'_>,
) -> *mut T {
    multimem_ptr_device(window, byte_offset, handle).cast()
}

#[inline(always)]
fn lsa_multimem_ptr_device(
    window: Window<'_>,
    byte_offset: usize,
    comm: DevComm<'_>,
) -> *mut c_void {
    unsafe { sys::nccl4rust_get_lsa_multimem_pointer(window.as_raw(), byte_offset, comm.as_raw()) }
}

/// Translate a registered window offset through the communicator's LSA
/// multimem mapping.
///
/// # Safety
///
/// LSA multimem must have been enabled when `comm` was created, and `window`
/// must be registered with its parent host communicator. `byte_offset` and the
/// resulting `T` access must be in bounds and correctly aligned, and the
/// caller must uphold all aliasing and synchronization requirements.
#[inline(always)]
pub unsafe fn lsa_multimem_ptr<T>(
    window: Window<'_>,
    byte_offset: usize,
    comm: DevComm<'_>,
) -> *mut T {
    lsa_multimem_ptr_device(window, byte_offset, comm).cast()
}

/// Synchronize one participating thread per rank in the LSA team.
///
/// # Safety
///
/// Every rank in the LSA team must execute the same barrier index, and the
/// communicator must have reserved that index in `lsaBarrierCount`. A barrier
/// index must not be shared by concurrently active cooperative groups. When
/// `multimem` is true, the device communicator must have been created with
/// `DeviceCommRequirements::lsa_multimem(true)`, and the target must support
/// LSA multimem.
#[inline(always)]
pub unsafe fn lsa_barrier_thread(comm: DevComm<'_>, index: u32, multimem: bool) {
    lsa_barrier_thread_device(comm, index, multimem)
}

#[inline(always)]
fn lsa_barrier_thread_device(comm: DevComm<'_>, index: u32, multimem: bool) {
    unsafe { sys::nccl4rust_lsa_barrier_thread(comm.as_raw(), index, multimem) }
}

/// Synchronize a full warp on every rank in the LSA team.
///
/// # Safety
///
/// All lanes in each participating warp and every LSA rank must execute this
/// call convergently with the same reserved barrier index. A barrier index must
/// not be shared by concurrently active cooperative groups. When `multimem`
/// is true, the device communicator must have been created with
/// `DeviceCommRequirements::lsa_multimem(true)`, and the target must support
/// LSA multimem.
#[inline(always)]
pub unsafe fn lsa_barrier_warp(comm: DevComm<'_>, index: u32, multimem: bool) {
    lsa_barrier_warp_device(comm, index, multimem)
}

#[inline(always)]
fn lsa_barrier_warp_device(comm: DevComm<'_>, index: u32, multimem: bool) {
    unsafe { sys::nccl4rust_lsa_barrier_warp(comm.as_raw(), index, multimem) }
}

/// Synchronize a full CTA on every rank in the LSA team.
///
/// # Safety
///
/// All threads in each CTA and every LSA rank must execute this call
/// convergently with the same reserved barrier index. A barrier index must not
/// be shared by concurrently active cooperative groups. When `multimem` is
/// true, the device communicator must have been created with
/// `DeviceCommRequirements::lsa_multimem(true)`, and the target must support
/// LSA multimem.
#[inline(always)]
pub unsafe fn lsa_barrier_cta(comm: DevComm<'_>, index: u32, multimem: bool) {
    lsa_barrier_cta_device(comm, index, multimem)
}

#[inline(always)]
fn lsa_barrier_cta_device(comm: DevComm<'_>, index: u32, multimem: bool) {
    unsafe { sys::nccl4rust_lsa_barrier_cta(comm.as_raw(), index, multimem) }
}

/// Reduce f32 values from every LSA peer and copy the sum to every peer.
///
/// # Safety
///
/// `comm`, `src`, and `dst` must originate from the same parent communicator;
/// both windows must have matching collective registrations on every
/// participating rank. For any given source/destination region, only one rank
/// and cooperative group may issue this non-rank-collective operation;
/// different ranks may concurrently issue calls for disjoint regions. Every
/// CTA thread in the issuing group must participate convergently. Source and
/// destination ranges must be valid `count * size_of::<f32>()` ranges in their
/// windows and must not partially overlap. Callers must provide the entry/exit
/// ordering required by NCCL, normally with [`lsa_barrier_cta`] around this
/// primitive.
#[inline(always)]
pub unsafe fn lsa_reduce_sum_copy_f32_cta(
    comm: DevComm<'_>,
    src: Window<'_>,
    src_byte_offset: usize,
    dst: Window<'_>,
    dst_byte_offset: usize,
    count: usize,
) {
    lsa_reduce_sum_copy_f32_cta_device(comm, src, src_byte_offset, dst, dst_byte_offset, count)
}

#[inline(always)]
fn lsa_reduce_sum_copy_f32_cta_device(
    comm: DevComm<'_>,
    src: Window<'_>,
    src_byte_offset: usize,
    dst: Window<'_>,
    dst_byte_offset: usize,
    count: usize,
) {
    unsafe {
        sys::nccl4rust_lsa_reduce_sum_copy_f32_cta(
            comm.as_raw(),
            src.as_raw(),
            src_byte_offset,
            dst.as_raw(),
            dst_byte_offset,
            count,
        )
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, needs_drop, size_of};
    use std::sync::{Mutex, MutexGuard};
    use std::vec::Vec;

    #[derive(Debug, Eq, PartialEq)]
    enum Call {
        Rank(usize),
        Size(usize),
        LsaRank(usize),
        LsaSize(usize),
        RankToWorld {
            comm: usize,
            n_ranks: i32,
            team_rank: i32,
            stride: i32,
            rank: i32,
        },
        RankToLsa {
            comm: usize,
            n_ranks: i32,
            team_rank: i32,
            stride: i32,
            rank: i32,
        },
        LocalPtr {
            window: usize,
            offset: usize,
        },
        LsaPtr {
            window: usize,
            offset: usize,
            peer: i32,
        },
        PeerPtr {
            window: usize,
            offset: usize,
            peer: i32,
        },
        TeamPeerPtr {
            window: usize,
            offset: usize,
            n_ranks: i32,
            team_rank: i32,
            stride: i32,
            peer: i32,
        },
        MultimemPtr {
            window: usize,
            offset: usize,
            mc_base_ptr: usize,
        },
        LsaMultimemPtr {
            window: usize,
            offset: usize,
            comm: usize,
        },
        BarrierThread {
            comm: usize,
            index: u32,
            multimem: bool,
        },
        BarrierWarp {
            comm: usize,
            index: u32,
            multimem: bool,
        },
        BarrierCta {
            comm: usize,
            index: u32,
            multimem: bool,
        },
        ReduceSumCopy {
            comm: usize,
            src: usize,
            src_offset: usize,
            dst: usize,
            dst_offset: usize,
            count: usize,
        },
    }

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static CALLS: Mutex<Vec<Call>> = Mutex::new(Vec::new());

    fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn serial_test() -> MutexGuard<'static, ()> {
        lock_unpoisoned(&TEST_LOCK)
    }

    fn record(call: Call) {
        lock_unpoisoned(&CALLS).push(call);
    }

    fn take_calls() -> Vec<Call> {
        core::mem::take(&mut *lock_unpoisoned(&CALLS))
    }

    fn comm(address: usize) -> DevComm<'static> {
        // SAFETY: Tests only pass this opaque pointer through to C-ABI stubs;
        // no test dereferences it.
        unsafe { DevComm::from_raw(core::ptr::without_provenance(address)) }
    }

    fn window(address: usize) -> Window<'static> {
        // SAFETY: Tests only pass this opaque handle through to C-ABI stubs;
        // no test dereferences it.
        unsafe { Window::from_raw(core::ptr::without_provenance_mut(address)) }
    }

    fn multimem(address: usize) -> MultimemHandle<'static> {
        // SAFETY: Tests only pass this opaque handle through to C-ABI stubs;
        // no test dereferences it.
        unsafe {
            MultimemHandle::from_raw(sys::ncclMultimemHandle_t {
                mc_base_ptr: core::ptr::without_provenance_mut(address),
            })
        }
    }

    fn team(n_ranks: i32, rank: i32, stride: i32) -> Team {
        Team(sys::ncclTeam_t {
            n_ranks,
            rank,
            stride,
        })
    }

    #[test]
    fn wrappers_are_pointer_sized_non_owning_values() {
        let _serial = serial_test();
        let comm = comm(0x1000);
        let window = window(0x2000);
        let multimem = multimem(0x3000);
        let team = team(8, 3, 2);

        assert_eq!(size_of::<DevComm<'_>>(), size_of::<*const c_void>());
        assert_eq!(align_of::<DevComm<'_>>(), align_of::<*const c_void>());
        assert!(!needs_drop::<DevComm<'_>>());
        assert_eq!(comm.as_raw().addr(), 0x1000);

        assert_eq!(size_of::<Window<'_>>(), size_of::<*mut c_void>());
        assert_eq!(align_of::<Window<'_>>(), align_of::<*mut c_void>());
        assert!(!needs_drop::<Window<'_>>());
        assert_eq!(window.as_raw().addr(), 0x2000);

        assert_eq!(
            size_of::<MultimemHandle<'_>>(),
            size_of::<sys::ncclMultimemHandle_t>()
        );
        assert_eq!(
            align_of::<MultimemHandle<'_>>(),
            align_of::<sys::ncclMultimemHandle_t>()
        );
        assert!(!needs_drop::<MultimemHandle<'_>>());
        assert_eq!(multimem.as_raw().mc_base_ptr.addr(), 0x3000);

        assert_eq!(size_of::<Team>(), size_of::<sys::ncclTeam_t>());
        assert_eq!(align_of::<Team>(), align_of::<sys::ncclTeam_t>());
        assert!(!needs_drop::<Team>());
        assert_eq!((team.size(), team.rank(), team.stride()), (8, 3, 2));
        assert_eq!(
            team.as_raw(),
            sys::ncclTeam_t {
                n_ranks: 8,
                rank: 3,
                stride: 2,
            }
        );
        assert!(take_calls().is_empty());
    }

    #[test]
    fn communicator_and_team_queries_forward_the_raw_communicator() {
        let _serial = serial_test();
        take_calls();
        let comm = comm(0x1100);

        assert_eq!(rank(comm), 11);
        assert_eq!(size(comm), 16);
        assert_eq!(lsa_rank(comm), 3);
        assert_eq!(lsa_size(comm), 8);

        let world = world(comm);
        assert_eq!((world.size(), world.rank(), world.stride()), (16, 11, 1));
        let lsa = lsa(comm);
        assert_eq!((lsa.size(), lsa.rank(), lsa.stride()), (8, 3, 1));
        let rail = rail(comm);
        assert_eq!((rail.size(), rail.rank(), rail.stride()), (2, 1, 8));

        assert_eq!(
            take_calls(),
            std::vec![
                Call::Rank(0x1100),
                Call::Size(0x1100),
                Call::LsaRank(0x1100),
                Call::LsaSize(0x1100),
                Call::Size(0x1100),
                Call::Rank(0x1100),
                Call::LsaSize(0x1100),
                Call::LsaRank(0x1100),
                Call::Size(0x1100),
                Call::Rank(0x1100),
                Call::LsaSize(0x1100),
            ]
        );
    }

    #[test]
    fn rank_translation_forwards_team_fields_and_rejects_invalid_ranks() {
        let _serial = serial_test();
        take_calls();
        let comm = comm(0x1200);
        let team = team(6, 2, 3);

        assert_eq!(rank_to_world(comm, team, 4), Some(104));
        assert_eq!(rank_to_lsa(comm, team, 3), Some(6));
        assert_eq!(rank_to_lsa(comm, team, 5), None);
        assert_eq!(rank_to_world(comm, team, -1), None);
        assert_eq!(rank_to_world(comm, team, 6), None);
        assert_eq!(rank_to_lsa(comm, team, -1), None);
        assert_eq!(rank_to_lsa(comm, team, 6), None);

        assert_eq!(
            take_calls(),
            std::vec![
                Call::RankToWorld {
                    comm: 0x1200,
                    n_ranks: 6,
                    team_rank: 2,
                    stride: 3,
                    rank: 4,
                },
                Call::RankToLsa {
                    comm: 0x1200,
                    n_ranks: 6,
                    team_rank: 2,
                    stride: 3,
                    rank: 3,
                },
                Call::LsaSize(0x1200),
                Call::RankToLsa {
                    comm: 0x1200,
                    n_ranks: 6,
                    team_rank: 2,
                    stride: 3,
                    rank: 5,
                },
                Call::LsaSize(0x1200),
            ]
        );
    }

    #[test]
    fn pointer_wrappers_forward_handles_offsets_peers_and_team_fields() {
        let _serial = serial_test();
        take_calls();
        let comm = comm(0x1300);
        let window = window(0x2300);
        let multimem = multimem(0x3300);
        let team = team(4, 1, 2);

        // SAFETY: The test stubs only record arguments and return opaque test
        // addresses. No returned pointer is dereferenced.
        unsafe {
            assert_eq!(local_ptr::<u32>(window, 16).addr(), 0xa000);
            assert_eq!(lsa_ptr::<u64>(window, 24, 2).addr(), 0xb000);
            assert_eq!(peer_ptr::<u16>(window, 32, 7).addr(), 0xc000);
            assert_eq!(team_peer_ptr::<u8>(window, 40, team, 3).addr(), 0xd000);
            assert_eq!(multimem_ptr::<f32>(window, 48, multimem).addr(), 0xe000);
            assert_eq!(lsa_multimem_ptr::<f64>(window, 56, comm).addr(), 0xf000);
        }

        assert_eq!(
            take_calls(),
            std::vec![
                Call::LocalPtr {
                    window: 0x2300,
                    offset: 16,
                },
                Call::LsaPtr {
                    window: 0x2300,
                    offset: 24,
                    peer: 2,
                },
                Call::PeerPtr {
                    window: 0x2300,
                    offset: 32,
                    peer: 7,
                },
                Call::TeamPeerPtr {
                    window: 0x2300,
                    offset: 40,
                    n_ranks: 4,
                    team_rank: 1,
                    stride: 2,
                    peer: 3,
                },
                Call::MultimemPtr {
                    window: 0x2300,
                    offset: 48,
                    mc_base_ptr: 0x3300,
                },
                Call::LsaMultimemPtr {
                    window: 0x2300,
                    offset: 56,
                    comm: 0x1300,
                },
            ]
        );
    }

    #[test]
    fn synchronization_and_reduction_wrappers_forward_every_argument() {
        let _serial = serial_test();
        take_calls();
        let comm = comm(0x1400);
        let src = window(0x2400);
        let dst = window(0x3400);

        // SAFETY: Test C-ABI stubs only record these values and perform no
        // synchronization or memory access.
        unsafe {
            lsa_barrier_thread(comm, 1, false);
            lsa_barrier_warp(comm, 2, true);
            lsa_barrier_cta(comm, u32::MAX, false);
            lsa_reduce_sum_copy_f32_cta(comm, src, 64, dst, 128, 257);
        }

        assert_eq!(
            take_calls(),
            std::vec![
                Call::BarrierThread {
                    comm: 0x1400,
                    index: 1,
                    multimem: false,
                },
                Call::BarrierWarp {
                    comm: 0x1400,
                    index: 2,
                    multimem: true,
                },
                Call::BarrierCta {
                    comm: 0x1400,
                    index: u32::MAX,
                    multimem: false,
                },
                Call::ReduceSumCopy {
                    comm: 0x1400,
                    src: 0x2400,
                    src_offset: 64,
                    dst: 0x3400,
                    dst_offset: 128,
                    count: 257,
                },
            ]
        );
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_dev_comm_rank(comm: *const sys::ncclDevComm_t) -> i32 {
        record(Call::Rank(comm.addr()));
        11
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_dev_comm_n_ranks(comm: *const sys::ncclDevComm_t) -> i32 {
        record(Call::Size(comm.addr()));
        16
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_dev_comm_lsa_rank(comm: *const sys::ncclDevComm_t) -> i32 {
        record(Call::LsaRank(comm.addr()));
        3
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_dev_comm_lsa_size(comm: *const sys::ncclDevComm_t) -> i32 {
        record(Call::LsaSize(comm.addr()));
        8
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_team_rank_to_world(
        comm: *const sys::ncclDevComm_t,
        n_ranks: i32,
        team_rank: i32,
        stride: i32,
        rank: i32,
    ) -> i32 {
        record(Call::RankToWorld {
            comm: comm.addr(),
            n_ranks,
            team_rank,
            stride,
            rank,
        });
        100 + rank
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_team_rank_to_lsa(
        comm: *const sys::ncclDevComm_t,
        n_ranks: i32,
        team_rank: i32,
        stride: i32,
        rank: i32,
    ) -> i32 {
        record(Call::RankToLsa {
            comm: comm.addr(),
            n_ranks,
            team_rank,
            stride,
            rank,
        });
        3 + (rank - team_rank) * stride
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_get_local_pointer(
        window: sys::ncclWindow_t,
        offset: usize,
    ) -> *mut c_void {
        record(Call::LocalPtr {
            window: window.addr(),
            offset,
        });
        core::ptr::without_provenance_mut(0xa000)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_get_lsa_pointer(
        window: sys::ncclWindow_t,
        offset: usize,
        peer: i32,
    ) -> *mut c_void {
        record(Call::LsaPtr {
            window: window.addr(),
            offset,
            peer,
        });
        core::ptr::without_provenance_mut(0xb000)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_get_peer_pointer(
        window: sys::ncclWindow_t,
        offset: usize,
        peer: i32,
    ) -> *mut c_void {
        record(Call::PeerPtr {
            window: window.addr(),
            offset,
            peer,
        });
        core::ptr::without_provenance_mut(0xc000)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_get_peer_pointer_team(
        window: sys::ncclWindow_t,
        offset: usize,
        n_ranks: i32,
        team_rank: i32,
        stride: i32,
        peer: i32,
    ) -> *mut c_void {
        record(Call::TeamPeerPtr {
            window: window.addr(),
            offset,
            n_ranks,
            team_rank,
            stride,
            peer,
        });
        core::ptr::without_provenance_mut(0xd000)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_get_multimem_pointer(
        window: sys::ncclWindow_t,
        offset: usize,
        mc_base_ptr: *mut c_void,
    ) -> *mut c_void {
        record(Call::MultimemPtr {
            window: window.addr(),
            offset,
            mc_base_ptr: mc_base_ptr.addr(),
        });
        core::ptr::without_provenance_mut(0xe000)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_get_lsa_multimem_pointer(
        window: sys::ncclWindow_t,
        offset: usize,
        comm: *const sys::ncclDevComm_t,
    ) -> *mut c_void {
        record(Call::LsaMultimemPtr {
            window: window.addr(),
            offset,
            comm: comm.addr(),
        });
        core::ptr::without_provenance_mut(0xf000)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_lsa_barrier_thread(
        comm: *const sys::ncclDevComm_t,
        index: u32,
        multimem: bool,
    ) {
        record(Call::BarrierThread {
            comm: comm.addr(),
            index,
            multimem,
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_lsa_barrier_warp(
        comm: *const sys::ncclDevComm_t,
        index: u32,
        multimem: bool,
    ) {
        record(Call::BarrierWarp {
            comm: comm.addr(),
            index,
            multimem,
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_lsa_barrier_cta(
        comm: *const sys::ncclDevComm_t,
        index: u32,
        multimem: bool,
    ) {
        record(Call::BarrierCta {
            comm: comm.addr(),
            index,
            multimem,
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn nccl4rust_lsa_reduce_sum_copy_f32_cta(
        comm: *const sys::ncclDevComm_t,
        src: sys::ncclWindow_t,
        src_offset: usize,
        dst: sys::ncclWindow_t,
        dst_offset: usize,
        count: usize,
    ) {
        record(Call::ReduceSumCopy {
            comm: comm.addr(),
            src: src.addr(),
            src_offset,
            dst: dst.addr(),
            dst_offset,
            count,
        });
    }
}
