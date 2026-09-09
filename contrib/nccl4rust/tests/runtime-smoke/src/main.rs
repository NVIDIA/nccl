// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-process, two-GPU runtime validation for nccl4rust.

use nccl::{
    Communicator, CudaStream, DeviceCommRequirements, NcclMemory, ReductionOp, UniqueId, Version,
    WindowFlags,
};
use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::thread;

const WORLD_SIZE: i32 = 2;
const ELEMENT_COUNT: usize = 16;
const DEVICE_ELEMENT_COUNT: usize = 17;
const WINDOW_GUARD_ELEMENTS: usize = 1;
const WINDOW_ELEMENT_COUNT: usize = DEVICE_ELEMENT_COUNT + 2 * WINDOW_GUARD_ELEMENTS;
const WINDOW_BYTE_OFFSET: usize = WINDOW_GUARD_ELEMENTS * size_of::<f32>();
const SOURCE_GUARD: f32 = -101.0;
const DESTINATION_GUARD: f32 = -202.0;

type SmokeResult<T> = Result<T, SmokeError>;

#[derive(Debug)]
struct SmokeError(String);

impl SmokeError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for SmokeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for SmokeError {}

impl From<nccl::Error> for SmokeError {
    fn from(error: nccl::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Debug)]
struct RankReport {
    rank: i32,
    device: i32,
    all_reduce_value: f32,
    device_lsa_reduce_value: f32,
    device_query: [i32; 4],
    device_teams: [i32; 11],
}

fn main() -> SmokeResult<()> {
    check_driver(unsafe { ffi::cuInit(0) }, "cuInit")?;
    let device_count = cuda_device_count()?;
    if device_count < WORLD_SIZE {
        return Err(SmokeError::new(format!(
            "runtime smoke needs {WORLD_SIZE} visible GPUs, found {device_count}"
        )));
    }

    let cubin = cubin_path()?;
    let version = Version::current()?;
    let unique_id = UniqueId::generate()?;
    println!(
        "nccl4rust runtime smoke: NCCL {version}, {device_count} visible GPUs, cubin {}",
        cubin.display()
    );

    let mut workers = Vec::new();
    for rank in 0..WORLD_SIZE {
        let cubin = cubin.clone();
        workers.push(thread::spawn(move || run_rank(rank, unique_id, &cubin)));
    }

    let mut reports = Vec::new();
    for worker in workers {
        reports.push(
            worker
                .join()
                .map_err(|_| SmokeError::new("rank worker panicked"))??,
        );
    }
    reports.sort_by_key(|report| report.rank);
    for report in &reports {
        println!(
            "rank {} device {}: host-all-reduce={} device-lsa-reduce={} device-query={:?} device-teams={:?}",
            report.rank,
            report.device,
            report.all_reduce_value,
            report.device_lsa_reduce_value,
            report.device_query,
            report.device_teams
        );
    }
    println!("nccl4rust multi-GPU runtime smoke passed");
    Ok(())
}

fn run_rank(rank: i32, unique_id: UniqueId, cubin: &Path) -> SmokeResult<RankReport> {
    check_cuda(unsafe { ffi::cudaSetDevice(rank) }, "cudaSetDevice")?;
    let stream = Stream::new()?;
    let mut send = DeviceBuffer::<f32>::new(ELEMENT_COUNT)?;
    let receive = DeviceBuffer::<f32>::new(ELEMENT_COUNT)?;
    send.copy_from_slice(&[rank as f32 + 1.0; ELEMENT_COUNT])?;

    let mut communicator = Communicator::init_rank(WORLD_SIZE, &unique_id, rank)?;
    let reported_rank = communicator.rank()?;
    let reported_size = communicator.size()?;
    let reported_device = communicator.device()?;
    if (reported_rank, reported_size, reported_device) != (rank, WORLD_SIZE, rank) {
        return Err(SmokeError::new(format!(
            "rank {rank} communicator query mismatch: rank={reported_rank}, size={reported_size}, device={reported_device}"
        )));
    }

    unsafe {
        communicator.all_reduce(
            send.as_ptr(),
            receive.as_mut_ptr(),
            ELEMENT_COUNT,
            ReductionOp::Sum,
            stream.as_nccl(),
        )?;
    }
    stream.synchronize()?;
    let reduced = receive.copy_to_vec()?;
    let expected = (1..=WORLD_SIZE).map(|value| value as f32).sum::<f32>();
    if reduced.iter().any(|&value| value != expected) {
        return Err(SmokeError::new(format!(
            "rank {rank} all-reduce mismatch: expected {expected}, got {reduced:?}"
        )));
    }

    let mut window_source = NcclMemory::<f32>::new(WINDOW_ELEMENT_COUNT)?;
    let mut window_destination = NcclMemory::<f32>::new(WINDOW_ELEMENT_COUNT)?;
    let mut source_values = vec![SOURCE_GUARD; WINDOW_ELEMENT_COUNT];
    source_values[WINDOW_GUARD_ELEMENTS..WINDOW_GUARD_ELEMENTS + DEVICE_ELEMENT_COUNT]
        .fill(rank as f32 + 1.0);
    copy_to_device(window_source.as_mut_ptr(), &source_values)?;
    copy_to_device(
        window_destination.as_mut_ptr(),
        &[DESTINATION_GUARD; WINDOW_ELEMENT_COUNT],
    )?;
    let source_window =
        communicator.register_window(&mut window_source, WindowFlags::COLLECTIVE_SYMMETRIC)?;
    let mut destination_window =
        communicator.register_window(&mut window_destination, WindowFlags::COLLECTIVE_SYMMETRIC)?;

    let requirements = DeviceCommRequirements::default().lsa_barrier_count(2);
    let device_communicator = communicator.create_device_communicator(&requirements)?;
    let mut device_image = DeviceBuffer::<u8>::new(device_communicator.as_bytes().len())?;
    device_image.copy_from_slice(device_communicator.as_bytes())?;
    let query_output = DeviceBuffer::<i32>::new(4)?;
    let module = Module::load(cubin)?;
    module.launch_query(
        device_image.as_mut_ptr().cast::<c_void>(),
        query_output.as_mut_ptr(),
        stream.raw(),
    )?;
    stream.synchronize()?;
    let query = query_output.copy_to_vec()?;
    let device_query: [i32; 4] = query
        .try_into()
        .map_err(|_| SmokeError::new("device query returned the wrong output length"))?;
    if device_query[0] != rank || device_query[1] != WORLD_SIZE {
        return Err(SmokeError::new(format!(
            "rank {rank} device world query mismatch: {device_query:?}"
        )));
    }
    if device_query[2] < 0
        || device_query[3] <= 0
        || device_query[2] >= device_query[3]
        || device_query[3] > WORLD_SIZE
    {
        return Err(SmokeError::new(format!(
            "rank {rank} device LSA query is invalid: {device_query:?}"
        )));
    }
    if device_query[3] != WORLD_SIZE {
        return Err(SmokeError::new(format!(
            "rank {rank} runtime smoke requires both ranks in one LSA team, got {device_query:?}"
        )));
    }

    let team_output = DeviceBuffer::<i32>::new(11)?;
    module.launch_team(
        device_image.as_mut_ptr().cast::<c_void>(),
        device_query[2],
        team_output.as_mut_ptr(),
        stream.raw(),
    )?;
    stream.synchronize()?;
    let team_query: [i32; 11] = team_output
        .copy_to_vec()?
        .try_into()
        .map_err(|_| SmokeError::new("device team query returned the wrong output length"))?;
    let expected_team_query = [
        WORLD_SIZE,
        rank,
        1,
        device_query[3],
        device_query[2],
        1,
        WORLD_SIZE / device_query[3],
        rank / device_query[3],
        device_query[3],
        rank,
        device_query[2],
    ];
    if team_query != expected_team_query {
        return Err(SmokeError::new(format!(
            "rank {rank} device team query mismatch: expected {expected_team_query:?}, got {team_query:?}"
        )));
    }

    let pointer_output = DeviceBuffer::<usize>::new(6)?;
    module.launch_pointer(
        device_image.as_mut_ptr().cast::<c_void>(),
        source_window.as_raw().cast::<c_void>(),
        0,
        device_query[2],
        rank,
        pointer_output.as_mut_ptr(),
        stream.raw(),
    )?;
    stream.synchronize()?;
    let pointer_base_query: [usize; 6] = pointer_output
        .copy_to_vec()?
        .try_into()
        .map_err(|_| SmokeError::new("device pointer query returned the wrong output length"))?;
    module.launch_pointer(
        device_image.as_mut_ptr().cast::<c_void>(),
        source_window.as_raw().cast::<c_void>(),
        WINDOW_BYTE_OFFSET,
        device_query[2],
        rank,
        pointer_output.as_mut_ptr(),
        stream.raw(),
    )?;
    stream.synchronize()?;
    let pointer_query: [usize; 6] = pointer_output
        .copy_to_vec()?
        .try_into()
        .map_err(|_| SmokeError::new("device pointer query returned the wrong output length"))?;
    let pointer_base = pointer_base_query[0];
    let expected_offset_pointer = pointer_base + WINDOW_BYTE_OFFSET;
    if pointer_base == 0
        || pointer_base_query[..4]
            .iter()
            .any(|&pointer| pointer != pointer_base)
        || pointer_base_query[4..] != [0, 0]
        || pointer_query[..4]
            .iter()
            .any(|&pointer| pointer != expected_offset_pointer)
        || pointer_query[4..] != [0, 0]
    {
        return Err(SmokeError::new(format!(
            "rank {rank} device pointer query mismatch: base={pointer_base_query:#x?}, offset={pointer_query:#x?}"
        )));
    }

    let lsa_world_rank = rank - device_query[2];
    let expected_lsa_sum = (0..device_query[3])
        .map(|lsa_rank| (lsa_world_rank + lsa_rank + 1) as f32)
        .sum::<f32>();

    for iteration in 0..2 {
        if iteration != 0 {
            copy_to_device(
                destination_window.as_mut_ptr(),
                &[DESTINATION_GUARD; WINDOW_ELEMENT_COUNT],
            )?;
        }
        module.launch_lsa_allreduce(
            device_image.as_mut_ptr().cast::<c_void>(),
            source_window.as_raw().cast::<c_void>(),
            WINDOW_BYTE_OFFSET,
            destination_window.as_raw().cast::<c_void>(),
            WINDOW_BYTE_OFFSET,
            DEVICE_ELEMENT_COUNT,
            stream.raw(),
        )?;
        stream.synchronize()?;
        let device_reduced = copy_from_device(destination_window.as_ptr(), WINDOW_ELEMENT_COUNT)?;
        if device_reduced[..WINDOW_GUARD_ELEMENTS]
            .iter()
            .chain(device_reduced[WINDOW_GUARD_ELEMENTS + DEVICE_ELEMENT_COUNT..].iter())
            .any(|&value| value != DESTINATION_GUARD)
        {
            return Err(SmokeError::new(format!(
                "rank {rank} device LSA reduce iteration {iteration} overwrote guard elements: {device_reduced:?}"
            )));
        }
        if device_reduced[WINDOW_GUARD_ELEMENTS..WINDOW_GUARD_ELEMENTS + DEVICE_ELEMENT_COUNT]
            .iter()
            .any(|&value| value != expected_lsa_sum)
        {
            return Err(SmokeError::new(format!(
                "rank {rank} device LSA reduce iteration {iteration} mismatch: expected {expected_lsa_sum}, got {device_reduced:?}"
            )));
        }
    }

    drop(module);
    drop(pointer_output);
    drop(team_output);
    drop(query_output);
    drop(device_image);
    device_communicator.destroy()?;
    drop(destination_window);
    drop(source_window);
    communicator.finalize()?;
    drop(communicator);
    stream.synchronize()?;

    Ok(RankReport {
        rank,
        device: reported_device,
        all_reduce_value: reduced[0],
        device_lsa_reduce_value: expected_lsa_sum,
        device_query,
        device_teams: team_query,
    })
}

fn copy_to_device<T>(destination: *mut T, source: &[T]) -> SmokeResult<()> {
    check_cuda(
        unsafe {
            ffi::cudaMemcpy(
                destination.cast::<c_void>(),
                source.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(source),
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        },
        "cudaMemcpy host-to-device",
    )
}

fn copy_from_device<T: Copy + Default>(source: *const T, len: usize) -> SmokeResult<Vec<T>> {
    let mut destination = vec![T::default(); len];
    check_cuda(
        unsafe {
            ffi::cudaMemcpy(
                destination.as_mut_ptr().cast::<c_void>(),
                source.cast::<c_void>(),
                len * size_of::<T>(),
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        },
        "cudaMemcpy device-to-host",
    )?;
    Ok(destination)
}

fn cubin_path() -> SmokeResult<PathBuf> {
    let path = std::env::var_os("NCCL4RUST_CUBIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../cuda-oxide-smoke/nccl4rust_cuda_oxide_smoke_sm_90.cubin")
        });
    if path.is_file() {
        Ok(path)
    } else {
        Err(SmokeError::new(format!(
            "NCCL4RUST_CUBIN does not name a cubin file: {}",
            path.display()
        )))
    }
}

fn cuda_device_count() -> SmokeResult<i32> {
    let mut count = 0;
    check_cuda(
        unsafe { ffi::cudaGetDeviceCount(&mut count) },
        "cudaGetDeviceCount",
    )?;
    Ok(count)
}

struct Stream(NonNull<c_void>);

impl Stream {
    fn new() -> SmokeResult<Self> {
        let mut raw = ptr::null_mut();
        check_cuda(
            unsafe { ffi::cudaStreamCreate(&mut raw) },
            "cudaStreamCreate",
        )?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| SmokeError::new("cudaStreamCreate returned a null stream"))?;
        Ok(Self(raw))
    }

    fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    fn as_nccl(&self) -> CudaStream {
        unsafe { CudaStream::from_raw(self.raw()) }
    }

    fn synchronize(&self) -> SmokeResult<()> {
        check_cuda(
            unsafe { ffi::cudaStreamSynchronize(self.raw()) },
            "cudaStreamSynchronize",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            ffi::cudaStreamDestroy(self.raw());
        }
    }
}

struct DeviceBuffer<T> {
    pointer: NonNull<T>,
    len: usize,
}

impl<T> DeviceBuffer<T> {
    fn new(len: usize) -> SmokeResult<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| SmokeError::new("CUDA allocation size overflow"))?;
        if bytes == 0 {
            return Err(SmokeError::new(
                "runtime smoke does not support empty CUDA allocations",
            ));
        }
        let mut raw = ptr::null_mut();
        check_cuda(unsafe { ffi::cudaMalloc(&mut raw, bytes) }, "cudaMalloc")?;
        let pointer = NonNull::new(raw.cast::<T>())
            .ok_or_else(|| SmokeError::new("cudaMalloc returned a null pointer"))?;
        Ok(Self { pointer, len })
    }

    fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut T {
        self.pointer.as_ptr()
    }

    fn copy_from_slice(&mut self, source: &[T]) -> SmokeResult<()> {
        if source.len() != self.len {
            return Err(SmokeError::new("host-to-device copy length mismatch"));
        }
        check_cuda(
            unsafe {
                ffi::cudaMemcpy(
                    self.as_mut_ptr().cast::<c_void>(),
                    source.as_ptr().cast::<c_void>(),
                    self.len * size_of::<T>(),
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy host-to-device",
        )
    }
}

impl<T: Copy + Default> DeviceBuffer<T> {
    fn copy_to_vec(&self) -> SmokeResult<Vec<T>> {
        let mut destination = vec![T::default(); self.len];
        check_cuda(
            unsafe {
                ffi::cudaMemcpy(
                    destination.as_mut_ptr().cast::<c_void>(),
                    self.as_ptr().cast::<c_void>(),
                    self.len * size_of::<T>(),
                    ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy device-to-host",
        )?;
        Ok(destination)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            ffi::cudaFree(self.as_mut_ptr().cast::<c_void>());
        }
    }
}

struct Module {
    raw: NonNull<c_void>,
    query: NonNull<c_void>,
    team: NonNull<c_void>,
    pointer: NonNull<c_void>,
    lsa_allreduce: NonNull<c_void>,
}

impl Module {
    fn load(path: &Path) -> SmokeResult<Self> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| SmokeError::new("cubin path contains an interior NUL byte"))?;
        let mut raw = ptr::null_mut();
        check_driver(
            unsafe { ffi::cuModuleLoad(&mut raw, path.as_ptr()) },
            "cuModuleLoad",
        )?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| SmokeError::new("cuModuleLoad returned a null module"))?;
        let functions = (|| {
            Ok((
                load_function(raw, c"nccl4rust_query_smoke")?,
                load_function(raw, c"nccl4rust_team_smoke")?,
                load_function(raw, c"nccl4rust_pointer_smoke")?,
                load_function(raw, c"nccl4rust_lsa_allreduce_f32_smoke")?,
            ))
        })();
        let (query, team, pointer, lsa_allreduce) = match functions {
            Ok(functions) => functions,
            Err(error) => {
                unsafe {
                    ffi::cuModuleUnload(raw.as_ptr());
                }
                return Err(error);
            }
        };
        Ok(Self {
            raw,
            query,
            team,
            pointer,
            lsa_allreduce,
        })
    }

    fn launch_query(
        &self,
        communicator: *mut c_void,
        output: *mut i32,
        stream: *mut c_void,
    ) -> SmokeResult<()> {
        let mut communicator_argument = communicator;
        let mut output_argument = output;
        let mut arguments = [
            ptr::from_mut(&mut communicator_argument).cast::<c_void>(),
            ptr::from_mut(&mut output_argument).cast::<c_void>(),
        ];
        check_driver(
            unsafe {
                ffi::cuLaunchKernel(
                    self.query.as_ptr(),
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel(nccl4rust_query_smoke)",
        )
    }

    fn launch_team(
        &self,
        communicator: *mut c_void,
        candidate_rank: i32,
        output: *mut i32,
        stream: *mut c_void,
    ) -> SmokeResult<()> {
        let mut communicator_argument = communicator;
        let mut candidate_rank_argument = candidate_rank;
        let mut output_argument = output;
        let mut arguments = [
            ptr::from_mut(&mut communicator_argument).cast::<c_void>(),
            ptr::from_mut(&mut candidate_rank_argument).cast::<c_void>(),
            ptr::from_mut(&mut output_argument).cast::<c_void>(),
        ];
        check_driver(
            unsafe {
                ffi::cuLaunchKernel(
                    self.team.as_ptr(),
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel(nccl4rust_team_smoke)",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_pointer(
        &self,
        communicator: *mut c_void,
        window: *mut c_void,
        byte_offset: usize,
        lsa_peer: i32,
        world_peer: i32,
        output: *mut usize,
        stream: *mut c_void,
    ) -> SmokeResult<()> {
        let mut communicator_argument = communicator;
        let mut window_argument = window;
        let mut multimem_base_argument = ptr::null_mut::<c_void>();
        let mut byte_offset_argument = byte_offset;
        let mut lsa_peer_argument = lsa_peer;
        let mut world_peer_argument = world_peer;
        let mut enable_multimem_argument = 0_u8;
        let mut output_argument = output;
        let mut arguments = [
            ptr::from_mut(&mut communicator_argument).cast::<c_void>(),
            ptr::from_mut(&mut window_argument).cast::<c_void>(),
            ptr::from_mut(&mut multimem_base_argument).cast::<c_void>(),
            ptr::from_mut(&mut byte_offset_argument).cast::<c_void>(),
            ptr::from_mut(&mut lsa_peer_argument).cast::<c_void>(),
            ptr::from_mut(&mut world_peer_argument).cast::<c_void>(),
            ptr::from_mut(&mut enable_multimem_argument).cast::<c_void>(),
            ptr::from_mut(&mut output_argument).cast::<c_void>(),
        ];
        check_driver(
            unsafe {
                ffi::cuLaunchKernel(
                    self.pointer.as_ptr(),
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel(nccl4rust_pointer_smoke)",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_lsa_allreduce(
        &self,
        communicator: *mut c_void,
        source_window: *mut c_void,
        source_offset: usize,
        destination_window: *mut c_void,
        destination_offset: usize,
        count: usize,
        stream: *mut c_void,
    ) -> SmokeResult<()> {
        let mut communicator_argument = communicator;
        let mut source_window_argument = source_window;
        let mut source_offset_argument = source_offset;
        let mut destination_window_argument = destination_window;
        let mut destination_offset_argument = destination_offset;
        let mut count_argument = count;
        let mut arguments = [
            ptr::from_mut(&mut communicator_argument).cast::<c_void>(),
            ptr::from_mut(&mut source_window_argument).cast::<c_void>(),
            ptr::from_mut(&mut source_offset_argument).cast::<c_void>(),
            ptr::from_mut(&mut destination_window_argument).cast::<c_void>(),
            ptr::from_mut(&mut destination_offset_argument).cast::<c_void>(),
            ptr::from_mut(&mut count_argument).cast::<c_void>(),
        ];
        check_driver(
            unsafe {
                ffi::cuLaunchKernel(
                    self.lsa_allreduce.as_ptr(),
                    1,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel(nccl4rust_lsa_allreduce_f32_smoke)",
        )
    }
}

fn load_function(module: NonNull<c_void>, name: &CStr) -> SmokeResult<NonNull<c_void>> {
    let mut function = ptr::null_mut();
    check_driver(
        unsafe { ffi::cuModuleGetFunction(&mut function, module.as_ptr(), name.as_ptr()) },
        &format!("cuModuleGetFunction({})", name.to_string_lossy()),
    )?;
    NonNull::new(function).ok_or_else(|| {
        SmokeError::new(format!(
            "cuModuleGetFunction({}) returned a null function",
            name.to_string_lossy()
        ))
    })
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            ffi::cuModuleUnload(self.raw.as_ptr());
        }
    }
}

fn check_cuda(status: ffi::CudaError, operation: &str) -> SmokeResult<()> {
    if status == ffi::CUDA_SUCCESS {
        return Ok(());
    }
    let message = unsafe {
        let pointer = ffi::cudaGetErrorString(status);
        if pointer.is_null() {
            format!("CUDA runtime error {status}")
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(SmokeError::new(format!("{operation} failed: {message}")))
}

fn check_driver(status: ffi::CuResult, operation: &str) -> SmokeResult<()> {
    if status == ffi::CUDA_SUCCESS {
        return Ok(());
    }
    let mut name = ptr::null();
    let mut description = ptr::null();
    unsafe {
        ffi::cuGetErrorName(status, &mut name);
        ffi::cuGetErrorString(status, &mut description);
    }
    let name = c_string_or(name, "CUDA_ERROR_UNKNOWN");
    let description = c_string_or(description, "unknown CUDA driver error");
    Err(SmokeError::new(format!(
        "{operation} failed: {name} ({description})"
    )))
}

fn c_string_or(pointer: *const c_char, fallback: &str) -> String {
    if pointer.is_null() {
        fallback.to_owned()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[allow(non_snake_case)]
mod ffi {
    use super::{c_char, c_int, c_uint, c_void};

    pub type CudaError = c_int;
    pub type CuResult = c_int;
    pub const CUDA_SUCCESS: c_int = 0;
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

    unsafe extern "C" {
        pub fn cudaGetDeviceCount(count: *mut c_int) -> CudaError;
        pub fn cudaSetDevice(device: c_int) -> CudaError;
        pub fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> CudaError;
        pub fn cudaFree(pointer: *mut c_void) -> CudaError;
        pub fn cudaMemcpy(
            destination: *mut c_void,
            source: *const c_void,
            bytes: usize,
            kind: c_int,
        ) -> CudaError;
        pub fn cudaStreamCreate(stream: *mut *mut c_void) -> CudaError;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> CudaError;
        pub fn cudaStreamDestroy(stream: *mut c_void) -> CudaError;
        pub fn cudaGetErrorString(error: CudaError) -> *const c_char;

        pub fn cuInit(flags: c_uint) -> CuResult;
        pub fn cuModuleLoad(module: *mut *mut c_void, path: *const c_char) -> CuResult;
        pub fn cuModuleGetFunction(
            function: *mut *mut c_void,
            module: *mut c_void,
            name: *const c_char,
        ) -> CuResult;
        pub fn cuLaunchKernel(
            function: *mut c_void,
            grid_x: c_uint,
            grid_y: c_uint,
            grid_z: c_uint,
            block_x: c_uint,
            block_y: c_uint,
            block_z: c_uint,
            shared_memory_bytes: c_uint,
            stream: *mut c_void,
            kernel_parameters: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CuResult;
        pub fn cuModuleUnload(module: *mut c_void) -> CuResult;
        pub fn cuGetErrorName(error: CuResult, name: *mut *const c_char) -> CuResult;
        pub fn cuGetErrorString(error: CuResult, message: *mut *const c_char) -> CuResult;
    }
}
