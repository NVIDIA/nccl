// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA-Oxide compile-and-link coverage for the Rust-facing device API.

use core::ffi::c_void;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use cuda_device::{kernel, thread};
use libnvvm_sys::{LibNvvm, Program};
use nccl_device::{self as nccl, DevComm, MultimemHandle, Window, sys};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};

const REQUIRED_DEVICE_SHIM_SYMBOLS: &[&str] = &[
    "nccl4rust_dev_comm_rank",
    "nccl4rust_dev_comm_n_ranks",
    "nccl4rust_dev_comm_lsa_rank",
    "nccl4rust_dev_comm_lsa_size",
    "nccl4rust_team_rank_to_world",
    "nccl4rust_team_rank_to_lsa",
    "nccl4rust_get_local_pointer",
    "nccl4rust_get_lsa_pointer",
    "nccl4rust_get_peer_pointer",
    "nccl4rust_get_peer_pointer_team",
    "nccl4rust_get_multimem_pointer",
    "nccl4rust_get_lsa_multimem_pointer",
    "nccl4rust_lsa_barrier_thread",
    "nccl4rust_lsa_barrier_warp",
    "nccl4rust_lsa_barrier_cta",
    "nccl4rust_lsa_reduce_sum_copy_f32_cta",
];

/// Query a device communicator into four consecutive `i32` outputs.
///
/// # Safety
///
/// `comm` must point to a live NCCL device communicator, and `output` must be
/// writable for at least four `i32` values for the duration of the kernel.
#[kernel]
pub unsafe fn nccl4rust_query_smoke(comm: *const sys::ncclDevComm_t, output: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        let comm = unsafe { DevComm::from_raw(comm) };
        unsafe {
            *output.add(0) = nccl::rank(comm);
            *output.add(1) = nccl::size(comm);
            *output.add(2) = nccl::lsa_rank(comm);
            *output.add(3) = nccl::lsa_size(comm);
        }
    }
}

/// Exercise team construction, field access, and rank translation wrappers.
///
/// The output layout is the `(size, rank, stride)` tuple for world, LSA, and
/// rail teams, followed by the candidate rank translated from the LSA team to
/// world and LSA ranks. An invalid candidate is written as `-1`.
///
/// # Safety
///
/// `comm` must point to a live NCCL device communicator, and `output` must be
/// writable for at least eleven `i32` values for the duration of the kernel.
#[kernel]
pub unsafe fn nccl4rust_team_smoke(
    comm: *const sys::ncclDevComm_t,
    candidate_rank: i32,
    output: *mut i32,
) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        let comm = unsafe { DevComm::from_raw(comm) };
        let world = nccl::world(comm);
        let lsa = nccl::lsa(comm);
        let rail = nccl::rail(comm);

        unsafe {
            *output.add(0) = world.size();
            *output.add(1) = world.rank();
            *output.add(2) = world.stride();
            *output.add(3) = lsa.size();
            *output.add(4) = lsa.rank();
            *output.add(5) = lsa.stride();
            *output.add(6) = rail.size();
            *output.add(7) = rail.rank();
            *output.add(8) = rail.stride();
            *output.add(9) = nccl::rank_to_world(comm, lsa, candidate_rank).unwrap_or(-1);
            *output.add(10) = nccl::rank_to_lsa(comm, lsa, candidate_rank).unwrap_or(-1);
        }
    }
}

/// Exercise all typed window-pointer mappings and raw pointer-like arguments.
///
/// The six outputs are local, LSA-peer, world-peer, team-peer, explicit
/// multimem, and communicator-LSA-multimem pointers, represented as `usize`.
/// The last two are zero when `enable_multimem` is zero.
/// The different pointee types intentionally ensure the generic typed API does
/// not accidentally constrain mappings to one Rust element type.
///
/// # Safety
///
/// The window, offset, LSA peer, world peer, and output storage must satisfy
/// the contracts of the corresponding NCCL device-pointer functions. When
/// `enable_multimem` is nonzero, both multimem mappings must also be enabled
/// and `multimem_base` must be valid. `output` must be writable for at least
/// six `usize` values.
#[allow(clippy::too_many_arguments)]
#[kernel]
pub unsafe fn nccl4rust_pointer_smoke(
    comm: *const sys::ncclDevComm_t,
    window: *mut c_void,
    multimem_base: *mut c_void,
    byte_offset: usize,
    lsa_peer: i32,
    world_peer: i32,
    enable_multimem: u8,
    output: *mut usize,
) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        let comm = unsafe { DevComm::from_raw(comm) };
        let window = unsafe { Window::from_raw(window) };
        let team = nccl::lsa(comm);

        unsafe {
            *output.add(0) = nccl::local_ptr::<u8>(window, byte_offset) as usize;
            *output.add(1) = nccl::lsa_ptr::<u16>(window, byte_offset, lsa_peer) as usize;
            *output.add(2) = nccl::peer_ptr::<u32>(window, byte_offset, world_peer) as usize;
            *output.add(3) =
                nccl::team_peer_ptr::<u64>(window, byte_offset, team, lsa_peer) as usize;

            if enable_multimem != 0 {
                let multimem = MultimemHandle::from_raw(sys::ncclMultimemHandle_t {
                    mc_base_ptr: multimem_base,
                });
                *output.add(4) = nccl::multimem_ptr::<f32>(window, byte_offset, multimem) as usize;
                *output.add(5) = nccl::lsa_multimem_ptr::<f64>(window, byte_offset, comm) as usize;
            } else {
                *output.add(4) = 0;
                *output.add(5) = 0;
            }
        }
    }
}

/// Exercise thread-, warp-, and CTA-cooperative barrier argument mappings.
///
/// The three calls deliberately use `false`, `true`, and a byte converted to
/// `bool`, respectively, so generated device calls cover every boolean form.
///
/// # Safety
///
/// The communicator must reserve three consecutive barrier indices beginning
/// at `first_index`. Every LSA rank must launch one matching CTA with at least
/// 32 threads. The target must support the warp barrier's multimem mode and,
/// when `multimem` is nonzero, the CTA barrier's multimem mode.
#[kernel]
pub unsafe fn nccl4rust_barrier_smoke(
    comm: *const sys::ncclDevComm_t,
    first_index: u32,
    multimem: u8,
) {
    let comm = unsafe { DevComm::from_raw(comm) };
    let thread_index = thread::threadIdx_x();

    if thread_index == 0 {
        unsafe { nccl::lsa_barrier_thread(comm, first_index, false) };
    }
    if thread_index < 32 {
        unsafe { nccl::lsa_barrier_warp(comm, first_index + 1, true) };
    }
    unsafe { nccl::lsa_barrier_cta(comm, first_index + 2, multimem != 0) };
}

/// One-CTA-per-rank LSA all-reduce building block.
///
/// The host must reserve two LSA barriers, register valid source/destination
/// windows, and launch identical grids on all LSA ranks.
///
/// # Safety
///
/// The communicator and windows must remain live for the launch; offsets and
/// `count` must describe valid `f32` regions; all participating LSA ranks must
/// execute this kernel with matching arguments and barrier reservations. Each
/// rank issues reduce-sum-copy for a disjoint partition of the region, as
/// required by NCCL's non-rank-collective ReduceCopy invocation model.
#[kernel]
pub unsafe fn nccl4rust_lsa_allreduce_f32_smoke(
    comm: *const sys::ncclDevComm_t,
    src_window: *mut c_void,
    src_offset: usize,
    dst_window: *mut c_void,
    dst_offset: usize,
    count: usize,
) {
    let comm = unsafe { DevComm::from_raw(comm) };
    let src = unsafe { Window::from_raw(src_window) };
    let dst = unsafe { Window::from_raw(dst_window) };
    let lsa_rank = nccl::lsa_rank(comm) as usize;
    let lsa_size = nccl::lsa_size(comm) as usize;
    let elements_per_rank = count / lsa_size;
    let remainder = count % lsa_size;
    let prefix_remainder = if lsa_rank < remainder {
        lsa_rank
    } else {
        remainder
    };
    let rank_start = lsa_rank * elements_per_rank + prefix_remainder;
    let rank_count = elements_per_rank + usize::from(lsa_rank < remainder);
    let rank_byte_offset = rank_start * core::mem::size_of::<f32>();

    unsafe {
        nccl::lsa_barrier_cta(comm, 0, false);
        if rank_count != 0 {
            nccl::lsa_reduce_sum_copy_f32_cta(
                comm,
                src,
                src_offset + rank_byte_offset,
                dst,
                dst_offset + rank_byte_offset,
                rank_count,
            );
        }
        nccl::lsa_barrier_cta(comm, 1, false);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let arch = std::env::var("NCCL4RUST_ARCH").unwrap_or_else(|_| "sm_90".to_owned());
    let compute_arch = arch.strip_prefix("sm_").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("NCCL4RUST_ARCH must have the form sm_XX, got {arch:?}"),
        )
    })?;
    let compute_capability = compute_arch.parse::<u32>().map_err(|source| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid NCCL4RUST_ARCH {arch:?}: {source}"),
        )
    })?;
    let link_mode = std::env::var("NCCL4RUST_LINK_MODE").unwrap_or_else(|_| "ptx".to_owned());
    let device_ltoir_path = env_path("NCCL4RUST_DEVICE_LTOIR").unwrap_or_else(|| {
        manifest_dir
            .join("../../build")
            .join(format!("libnccl4rust_device_{arch}.ltoir"))
    });
    let output_path = env_path("NCCL4RUST_CUBIN")
        .unwrap_or_else(|| manifest_dir.join(format!("nccl4rust_cuda_oxide_smoke_{arch}.cubin")));

    let device_ltoir = read_input(&device_ltoir_path, "NCCL device shim LTOIR")?;
    let (rust_input, rust_input_type, rust_input_name) = match link_mode.as_str() {
        "ptx" => {
            let path = env_path("NCCL4RUST_RUST_PTX")
                .unwrap_or_else(|| manifest_dir.join("nccl4rust_cuda_oxide_smoke.ptx"));
            let rust_ptx = read_input(&path, "Rust PTX")?;
            verify_device_api_references(&rust_ptx, "Rust PTX")?;
            (
                rust_ptx,
                InputType::Ptx,
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("nccl4rust_cuda_oxide_smoke.ptx")
                    .to_owned(),
            )
        }
        "nvvm-lto" => {
            if compute_capability < 100 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "this pinned CUDA-Oxide revision emits opaque-pointer NVVM 20 IR, which \
                     requires sm_100 or newer; use NCCL4RUST_LINK_MODE=ptx for pre-Blackwell",
                )
                .into());
            }
            let path = env_path("NCCL4RUST_RUST_NVVM_IR")
                .unwrap_or_else(|| manifest_dir.join("nccl4rust_cuda_oxide_smoke.ll"));
            let rust_ir = read_input(&path, "Rust NVVM IR")?;
            verify_device_api_references(&rust_ir, "Rust NVVM IR")?;
            (
                compile_nvvm_ir(&rust_ir, compute_arch)?,
                InputType::Ltoir,
                "nccl4rust_cuda_oxide_smoke.ltoir".to_owned(),
            )
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("NCCL4RUST_LINK_MODE must be `ptx` or `nvvm-lto`, got {other:?}"),
            )
            .into());
        }
    };

    let nvjitlink = LibNvJitLink::load()?;
    let link_arch = format!("-arch={arch}");
    let mut linker = Linker::new(&nvjitlink, &[&link_arch, "-lto"])?;
    linker.add(rust_input_type, &rust_input, &rust_input_name)?;
    linker.add(
        InputType::Ltoir,
        &device_ltoir,
        device_ltoir_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("libnccl4rust_device.ltoir"),
    )?;
    let cubin = linker.finish()?;
    fs::write(&output_path, &cubin).map_err(|source| {
        io::Error::new(
            source.kind(),
            format!("failed to write {}: {source}", output_path.display()),
        )
    })?;

    println!(
        "linked {} bytes of Rust {} and {} bytes of NCCL shim LTOIR into {} ({} bytes)",
        rust_input.len(),
        link_mode,
        device_ltoir.len(),
        output_path.display(),
        cubin.len()
    );
    Ok(())
}

fn compile_nvvm_ir(nvvm_ir: &[u8], compute_arch: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let libdevice_path = cuda_host::ltoir::find_libdevice()?;
    let libdevice = read_input(&libdevice_path, "CUDA libdevice bitcode")?;
    let nvvm = LibNvvm::load()?;
    let mut program = Program::new(&nvvm)?;
    program.add_module(&libdevice, "libdevice.10.bc")?;
    program.add_module(nvvm_ir, "nccl4rust_cuda_oxide_smoke.ll")?;
    let nvvm_arch = format!("-arch=compute_{compute_arch}");
    Ok(program.compile(&[&nvvm_arch, "-gen-lto"])?)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn read_input(path: &Path, description: &str) -> io::Result<Vec<u8>> {
    fs::read(path).map_err(|source| {
        io::Error::new(
            source.kind(),
            format!(
                "failed to read {description} at {}: {source}",
                path.display()
            ),
        )
    })
}

fn verify_device_api_references(input: &[u8], description: &str) -> io::Result<()> {
    let missing: Vec<_> = REQUIRED_DEVICE_SHIM_SYMBOLS
        .iter()
        .copied()
        .filter(|symbol| !contains_identifier(input, symbol.as_bytes()))
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{description} does not reference the complete NCCL device shim; missing: {}",
                missing.join(", ")
            ),
        ))
    }
}

fn contains_identifier(input: &[u8], identifier: &[u8]) -> bool {
    input
        .windows(identifier.len())
        .enumerate()
        .any(|(start, candidate)| {
            candidate == identifier
                && start
                    .checked_sub(1)
                    .is_none_or(|before| !is_identifier_byte(input[before]))
                && input
                    .get(start + identifier.len())
                    .is_none_or(|after| !is_identifier_byte(*after))
        })
}

const fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_match_does_not_accept_a_longer_symbol() {
        assert!(!contains_identifier(
            b"call nccl4rust_get_peer_pointer_team",
            b"nccl4rust_get_peer_pointer",
        ));
    }

    #[test]
    fn verifier_reports_an_absent_device_symbol() {
        let input = REQUIRED_DEVICE_SHIM_SYMBOLS
            .iter()
            .copied()
            .filter(|symbol| *symbol != "nccl4rust_team_rank_to_lsa")
            .collect::<Vec<_>>()
            .join("\n");

        let error = verify_device_api_references(input.as_bytes(), "test input").unwrap_err();
        assert!(error.to_string().contains("nccl4rust_team_rank_to_lsa"));
    }
}
