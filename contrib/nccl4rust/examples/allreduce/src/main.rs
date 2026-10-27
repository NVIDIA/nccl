// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal NCCL-from-Rust example: every rank contributes a buffer of
//! rank + 1, and an all-reduce leaves the sum on every GPU.

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use nccl::{Communicator, CudaStream, ReductionOp, UniqueId};
use std::{env, error::Error, fs, thread, time::Duration};

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Who am I? The launcher (srun) tells every process its rank.
    let rank: i32 = env::var("SLURM_PROCID")?.parse()?;
    let world: i32 = env::var("SLURM_NTASKS")?.parse()?;
    let local: usize = env::var("SLURM_LOCALID").map_or(Ok(0), |v| v.parse())?;
    let host = env::var("SLURMD_NODENAME").unwrap_or_default();

    // 2. One GPU per process, and a CUDA stream to queue work on.
    let ctx = CudaContext::new(local)?;
    let stream = ctx.new_stream()?;
    // SAFETY: `stream` outlives every NCCL call that uses this handle.
    let nccl_stream = unsafe { CudaStream::from_raw(stream.cu_stream().cast()) };

    // 3. Rank 0 makes a unique ID, everyone joins the same communicator.
    let id = share_unique_id(rank)?;
    let comm = Communicator::init_rank(world, &id, rank)?;

    // 4. Each rank contributes a buffer full of (rank + 1).
    let n = 1 << 20;
    let send = stream.clone_htod(&vec![(rank + 1) as f32; n])?;
    let mut recv = stream.alloc_zeros::<f32>(n)?;

    // 5. Enqueue the all-reduce on the stream. This returns immediately;
    //    the GPU does the work, the CPU is free.
    {
        let (send_ptr, _send_sync) = send.device_ptr(&stream);
        let (recv_ptr, _recv_sync) = recv.device_ptr_mut(&stream);
        // SAFETY: both buffers hold `n` f32s on this GPU and stay alive until
        // the copy below, which runs after the all-reduce on the same stream.
        unsafe {
            comm.all_reduce(send_ptr as *const f32, recv_ptr as *mut f32, n,
                            ReductionOp::Sum, nccl_stream)?;
        }
    }

    // 6. Copy back (stream-ordered after the all-reduce) and check.
    let out = stream.clone_dtoh(&recv)?;
    let expected = (world * (world + 1) / 2) as f32;
    assert!(out.iter().all(|&x| x == expected));
    println!("[{host}] rank {rank}/{world}: sum = {} (expected {expected}) OK", out[0]);
    Ok(())
}

/// The NCCL unique ID is 128 opaque bytes. Pass it through a file in
/// NCCL_ID_DIR (default: the current directory), which all ranks must share.
fn share_unique_id(rank: i32) -> Result<UniqueId, Box<dyn Error>> {
    let dir = env::var("NCCL_ID_DIR").unwrap_or_else(|_| ".".to_string());
    let path = format!("{dir}/nccl-id.{}", env::var("SLURM_JOB_ID")?);
    if rank == 0 {
        let id = UniqueId::generate()?;
        fs::write(format!("{path}.tmp"), id.as_bytes())?;
        fs::rename(format!("{path}.tmp"), &path)?;
        return Ok(id);
    }
    let bytes = loop {
        match fs::read(&path) {
            Ok(b) if b.len() == UniqueId::BYTE_LEN => break b,
            _ => thread::sleep(Duration::from_millis(50)),
        }
    };
    Ok(UniqueId::from_bytes(bytes.try_into().unwrap()))
}
