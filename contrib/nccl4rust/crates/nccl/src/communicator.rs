// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::error::{Error, Result, Status, check};
use crate::group::{ensure_no_active_group, group_is_active, register_group_communicator};
use crate::sys;
use crate::types::{CudaStream, NcclDataType, ReductionOp, UniqueId};
use std::cell::Cell;
use std::env;
use std::io::Write;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::rc::Rc;

/// The lifecycle state of a live communicator handle.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CommunicatorState {
    Active,
    Finalized,
    Failed,
}

/// The state reported by `ncclCommGetAsyncError`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AsyncStatus {
    Ready,
    InProgress,
    Failed(Error),
}

/// Whether Rust-owned storage can be released after a management call.
pub(crate) enum ManagementCompletion {
    Complete,
    SynchronousFailure(Error),
    TerminalFailure(Error),
    Uncertain(Error),
}

/// An owned NCCL communicator.
///
/// The wrapper is deliberately neither `Send` nor `Sync`: moving a
/// communicator between host threads can silently change the current CUDA
/// context, while sharing one would permit unsynchronized state transitions.
///
/// This wrapper rejects an explicitly nonblocking config and forces its config
/// field to blocking before initialization. NCCL environment/config-file
/// overrides are outside Rust's control, so management APIs defensively retain
/// output storage and enqueue/group paths poll setup to completion if NCCL
/// nevertheless returns `ncclInProgress`.
///
/// # Drop behavior
///
/// Drop invokes destroy or abort once and cannot report cleanup failures. If
/// teardown does not complete synchronously, it retains the config strings
/// that NCCL may still access rather than polling a handle that background
/// teardown can free. Dropping a communicator inside an active
/// [`crate::Group`] similarly leaks the communicator and config instead of
/// risking deferred use-after-free.
pub struct Communicator {
    handle: Option<NonNull<sys::ncclComm>>,
    state: Rc<Cell<CommunicatorState>>,
    // NCCL stores config.commName without copying it.
    config: Option<Config>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for Communicator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Communicator")
            .field("handle", &self.handle.map(NonNull::as_ptr))
            .field("state", &self.state.get())
            .finish()
    }
}

impl Communicator {
    /// Initialize one rank on the thread's current CUDA device.
    pub fn init_rank(rank_count: i32, id: &UniqueId, rank: i32) -> Result<Self> {
        Self::init_rank_with_config(rank_count, id, rank, Config::default())
    }

    /// Initialize one rank using an owned NCCL config.
    ///
    /// Ownership is required because NCCL retains `commName` without copying
    /// it. An explicit `blocking(false)` request is rejected.
    pub fn init_rank_with_config(
        rank_count: i32,
        id: &UniqueId,
        rank: i32,
        config: Config,
    ) -> Result<Self> {
        ensure_no_active_group("communicator initialization")?;
        validate_rank(rank_count, rank)?;
        let mut config = prepare_blocking_config(config)?;
        let mut output = Box::new(ptr::null_mut::<sys::ncclComm>());
        let status = unsafe {
            sys::ncclCommInitRankConfig(
                output.as_mut(),
                rank_count,
                id.as_raw(),
                rank,
                &mut config.raw,
            )
        };
        Self::from_init_result(output, status, config)
    }

    fn from_init_result(
        output: Box<sys::ncclComm_t>,
        status: sys::ncclResult_t,
        config: Config,
    ) -> Result<Self> {
        let handle = NonNull::new(*output);
        match Status::from_raw(status) {
            Status::Success => {
                let handle = handle
                    .ok_or_else(|| Error::local("NCCL initialized a null communicator handle"))?;
                Ok(Self {
                    handle: Some(handle),
                    state: Rc::new(Cell::new(CommunicatorState::Active)),
                    config: Some(config),
                    _thread_bound: PhantomData,
                })
            }
            Status::InProgress => {
                let unsupported = Error::local(
                    "NCCL returned ncclInProgress from communicator initialization; safe initialization supports blocking completion only",
                );
                let Some(handle) = handle else {
                    Box::leak(output);
                    std::mem::forget(config);
                    return Err(unsupported
                        .with_context("NCCL did not publish a communicator handle for cleanup"));
                };
                let communicator = Self {
                    handle: Some(handle),
                    state: Rc::new(Cell::new(CommunicatorState::Failed)),
                    config: Some(config),
                    _thread_bound: PhantomData,
                };
                match communicator.abort() {
                    Ok(()) => Err(unsupported),
                    Err(cleanup) => {
                        // Cleanup did not establish that initialization has
                        // stopped targeting this output slot.
                        Box::leak(output);
                        Err(unsupported
                            .with_context(format!("communicator abort also failed ({cleanup})")))
                    }
                }
            }
            _ => {
                let initialization = Error::from_raw(status);
                if let Some(handle) = handle {
                    let communicator = Self {
                        handle: Some(handle),
                        state: Rc::new(Cell::new(CommunicatorState::Failed)),
                        config: Some(config),
                        _thread_bound: PhantomData,
                    };
                    return match communicator.abort() {
                        Ok(()) => Err(initialization),
                        Err(cleanup) => Err(initialization
                            .with_context(format!("communicator abort also failed ({cleanup})"))),
                    };
                }
                Err(initialization)
            }
        }
    }

    pub fn state(&self) -> CommunicatorState {
        self.state.get()
    }

    pub fn rank(&self) -> Result<i32> {
        ensure_no_active_group("communicator rank query")?;
        let mut rank = 0;
        check(unsafe { sys::ncclCommUserRank(self.queryable_raw()?, &mut rank) })?;
        Ok(rank)
    }

    pub fn size(&self) -> Result<i32> {
        ensure_no_active_group("communicator size query")?;
        let mut size = 0;
        check(unsafe { sys::ncclCommCount(self.queryable_raw()?, &mut size) })?;
        Ok(size)
    }

    pub fn device(&self) -> Result<i32> {
        ensure_no_active_group("communicator device query")?;
        let mut device = 0;
        check(unsafe { sys::ncclCommCuDevice(self.queryable_raw()?, &mut device) })?;
        Ok(device)
    }

    /// Query asynchronous collective completion or failure.
    pub fn async_error(&self) -> Result<AsyncStatus> {
        ensure_no_active_group("communicator asynchronous-error query")?;
        let mut asynchronous = sys::ncclSuccess;
        check(unsafe { sys::ncclCommGetAsyncError(self.raw(), &mut asynchronous) })?;
        Ok(match Status::from_raw(asynchronous) {
            Status::Success => AsyncStatus::Ready,
            Status::InProgress => AsyncStatus::InProgress,
            _ => {
                self.state.set(CommunicatorState::Failed);
                AsyncStatus::Failed(Error::from_raw(asynchronous))
            }
        })
    }

    /// Flush operations and synchronously reach finalized state.
    pub fn finalize(&mut self) -> Result<()> {
        ensure_no_active_group("communicator finalization")?;
        match self.state.get() {
            CommunicatorState::Finalized => return Ok(()),
            CommunicatorState::Active => {}
            CommunicatorState::Failed => {
                return Err(Error::local(
                    "cannot finalize a failed NCCL communicator; abort it instead",
                ));
            }
        }
        let status = unsafe { sys::ncclCommFinalize(self.raw()) };
        match self.complete_management_call(status, "communicator finalization") {
            ManagementCompletion::Complete => {
                self.state.set(CommunicatorState::Finalized);
                Ok(())
            }
            ManagementCompletion::SynchronousFailure(error)
            | ManagementCompletion::TerminalFailure(error)
            | ManagementCompletion::Uncertain(error) => {
                // Finalize can mutate communicator state before a later
                // management/group error is surfaced. Never permit new
                // collectives after an ambiguous finalize attempt.
                self.state.set(CommunicatorState::Failed);
                Err(error)
            }
        }
    }

    /// Abort outstanding work and consume the communicator.
    pub fn abort(mut self) -> Result<()> {
        ensure_no_active_group("communicator abort")?;
        // Never retry abort from Drop. Even an error return can follow partial
        // teardown, so a second call could dereference an invalid handle.
        let handle = self
            .handle
            .take()
            .expect("live Communicator must contain a raw handle");
        let status = unsafe { sys::ncclCommAbort(handle.as_ptr()) };
        match Status::from_raw(status) {
            Status::Success => Ok(()),
            Status::InProgress => {
                // Teardown owns the communicator and may free it at any time;
                // ncclCommGetAsyncError would race that free. Preserve strings
                // the background teardown can still access instead of polling.
                self.leak_config();
                Err(Error::local(
                    "NCCL communicator abort continued asynchronously; communicator config was retained for background teardown",
                ))
            }
            _ => {
                self.state.set(CommunicatorState::Failed);
                // NCCL may have begun teardown before reporting the failure.
                // Preserve any config strings the raw communicator can retain.
                self.leak_config();
                Err(Error::from_raw(status).with_context("communicator abort"))
            }
        }
    }

    pub(crate) fn raw(&self) -> sys::ncclComm_t {
        self.handle
            .expect("live Communicator must contain a raw handle")
            .as_ptr()
    }

    pub(crate) fn active_raw(&self) -> Result<sys::ncclComm_t> {
        if self.state.get() == CommunicatorState::Active {
            Ok(self.raw())
        } else {
            Err(Error::local(format!(
                "cannot enqueue an operation while the NCCL communicator is {:?}",
                self.state.get()
            )))
        }
    }

    fn active_enqueue_raw(&self) -> Result<sys::ncclComm_t> {
        let communicator = self.active_raw()?;
        register_group_communicator(communicator, Rc::clone(&self.state));
        Ok(communicator)
    }

    fn complete_enqueue_call(&self, status: sys::ncclResult_t, operation: &str) -> Result<()> {
        match self.complete_management_call(status, operation) {
            ManagementCompletion::Complete => Ok(()),
            ManagementCompletion::SynchronousFailure(error)
            | ManagementCompletion::TerminalFailure(error)
            | ManagementCompletion::Uncertain(error) => Err(error),
        }
    }

    fn queryable_raw(&self) -> Result<sys::ncclComm_t> {
        match self.state.get() {
            CommunicatorState::Active | CommunicatorState::Finalized => Ok(self.raw()),
            CommunicatorState::Failed => {
                Err(Error::local("NCCL communicator is in a failed state"))
            }
        }
    }

    pub(crate) fn complete_management_call(
        &self,
        status: sys::ncclResult_t,
        operation: &str,
    ) -> ManagementCompletion {
        match Status::from_raw(status) {
            Status::Success => ManagementCompletion::Complete,
            Status::InProgress => match self.poll_async_terminal() {
                Ok(Ok(())) => ManagementCompletion::Complete,
                Ok(Err(error)) => {
                    self.state.set(CommunicatorState::Failed);
                    ManagementCompletion::TerminalFailure(error.with_context(operation))
                }
                Err(error) => {
                    self.state.set(CommunicatorState::Failed);
                    ManagementCompletion::Uncertain(error.with_context(format!(
                        "could not establish terminal completion of {operation}"
                    )))
                }
            },
            _ => ManagementCompletion::SynchronousFailure(
                Error::from_raw(status).with_context(operation),
            ),
        }
    }

    /// Establish terminal completion for an API whose output pointer NCCL may
    /// retain in an internal group task.
    ///
    /// Current NCCL management entry points can consume an internal
    /// `ncclInProgress` from `ncclGroupEndInternal` and still return
    /// `ncclSuccess`. Poll after either accepted status; otherwise a fast
    /// background task can race Rust while it reads or frees the output slot.
    pub(crate) fn complete_deferred_output_call(
        &self,
        status: sys::ncclResult_t,
        operation: &str,
    ) -> ManagementCompletion {
        match Status::from_raw(status) {
            Status::Success | Status::InProgress => match self.poll_async_terminal() {
                Ok(Ok(())) => ManagementCompletion::Complete,
                Ok(Err(error)) => {
                    self.state.set(CommunicatorState::Failed);
                    ManagementCompletion::TerminalFailure(error.with_context(operation))
                }
                Err(error) => {
                    self.state.set(CommunicatorState::Failed);
                    ManagementCompletion::Uncertain(error.with_context(format!(
                        "could not establish terminal completion of {operation}"
                    )))
                }
            },
            _ => ManagementCompletion::SynchronousFailure(
                Error::from_raw(status).with_context(operation),
            ),
        }
    }

    fn poll_async_terminal(&self) -> Result<Result<()>> {
        loop {
            let mut asynchronous = sys::ncclInProgress;
            let query = unsafe { sys::ncclCommGetAsyncError(self.raw(), &mut asynchronous) };
            if query != sys::ncclSuccess {
                return Err(Error::from_raw(query));
            }
            match Status::from_raw(asynchronous) {
                Status::Success => return Ok(Ok(())),
                Status::InProgress => std::thread::yield_now(),
                _ => return Ok(Err(Error::from_raw(asynchronous))),
            }
        }
    }

    fn leak_config(&mut self) {
        if let Some(config) = self.config.take() {
            std::mem::forget(config);
        }
    }

    /// Enqueue a reduction to one root rank.
    ///
    /// # Safety
    ///
    /// The pointers must address CUDA-accessible storage for `count` values of
    /// `T`, obey NCCL's root/in-place rules, and remain valid with the required
    /// aliasing permissions until `stream` has completed the operation.
    pub unsafe fn reduce<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        count: usize,
        operation: ReductionOp,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclReduce(
                send.cast(),
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                operation.as_raw(),
                root,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "reduction enqueue")
    }

    /// Enqueue a broadcast from one root rank.
    ///
    /// # Safety
    ///
    /// The pointer, length, stream-lifetime, device-access, and aliasing
    /// requirements described by [`Self::reduce`] apply.
    pub unsafe fn broadcast<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclBroadcast(
                send.cast(),
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                root,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "broadcast enqueue")
    }

    /// Enqueue an all-reduce.
    ///
    /// # Safety
    ///
    /// Both pointers must address CUDA-accessible storage for `count` values of
    /// `T` and stay alive, with NCCL-compatible aliasing, until stream
    /// completion.
    pub unsafe fn all_reduce<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        count: usize,
        operation: ReductionOp,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclAllReduce(
                send.cast(),
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                operation.as_raw(),
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "all-reduce enqueue")
    }

    /// Enqueue a reduce-scatter.
    ///
    /// # Safety
    ///
    /// `send` must hold `receive_count * communicator_size` values and
    /// `receive` must hold `receive_count` values. Both must meet the same
    /// device, lifetime, and aliasing obligations as [`Self::all_reduce`].
    pub unsafe fn reduce_scatter<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        receive_count: usize,
        operation: ReductionOp,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclReduceScatter(
                send.cast(),
                receive.cast(),
                receive_count,
                T::NCCL_DATA_TYPE,
                operation.as_raw(),
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "reduce-scatter enqueue")
    }

    /// Enqueue an all-gather.
    ///
    /// # Safety
    ///
    /// `send` must hold `send_count` values and `receive` must hold
    /// `send_count * communicator_size` values. Both must remain valid until
    /// stream completion and follow NCCL's in-place rules.
    pub unsafe fn all_gather<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        send_count: usize,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclAllGather(
                send.cast(),
                receive.cast(),
                send_count,
                T::NCCL_DATA_TYPE,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "all-gather enqueue")
    }

    /// Enqueue an all-to-all.
    ///
    /// # Safety
    ///
    /// Each pointer must hold `count * communicator_size` values of `T` in
    /// CUDA-accessible memory and remain valid until stream completion.
    pub unsafe fn all_to_all<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        count: usize,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclAlltoAll(
                send.cast(),
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "all-to-all enqueue")
    }

    /// Enqueue a gather to one root rank.
    ///
    /// # Safety
    ///
    /// `send` must hold `count` values on every rank. At `root`, `receive` must
    /// hold `count * communicator_size` values. Pointers must stay valid until
    /// stream completion and follow NCCL's root and in-place rules.
    pub unsafe fn gather<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclGather(
                send.cast(),
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                root,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "gather enqueue")
    }

    /// Enqueue a scatter from one root rank.
    ///
    /// # Safety
    ///
    /// At `root`, `send` must hold `count * communicator_size` values.
    /// `receive` must hold `count` values on every rank. Pointers must remain
    /// valid until stream completion and follow NCCL's root/in-place rules.
    pub unsafe fn scatter<T: NcclDataType>(
        &self,
        send: *const T,
        receive: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclScatter(
                send.cast(),
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                root,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "scatter enqueue")
    }

    /// Enqueue a point-to-point send.
    ///
    /// # Safety
    ///
    /// `send` must address `count` CUDA-accessible values and remain readable
    /// until stream completion. The peer must post a matching receive; cycles
    /// that require concurrent progress must use [`crate::Group`].
    pub unsafe fn send<T: NcclDataType>(
        &self,
        send: *const T,
        count: usize,
        peer: i32,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclSend(
                send.cast(),
                count,
                T::NCCL_DATA_TYPE,
                peer,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "send enqueue")
    }

    /// Enqueue a point-to-point receive.
    ///
    /// # Safety
    ///
    /// `receive` must address `count` writable CUDA-accessible values and
    /// remain exclusively writable until stream completion. The peer must post
    /// a matching send.
    pub unsafe fn receive<T: NcclDataType>(
        &self,
        receive: *mut T,
        count: usize,
        peer: i32,
        stream: CudaStream,
    ) -> Result<()> {
        let comm = self.active_enqueue_raw()?;
        let status = unsafe {
            sys::ncclRecv(
                receive.cast(),
                count,
                T::NCCL_DATA_TYPE,
                peer,
                comm,
                stream.as_sys(),
            )
        };
        self.complete_enqueue_call(status, "receive enqueue")
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        if group_is_active() {
            warn_from_drop(format_args!(
                "warning: leaking an NCCL communicator dropped inside active nccl4rust Group scope; finish the Group before dropping communicator resources"
            ));
            self.handle.take();
            self.leak_config();
            return;
        }

        let (status, operation) = match self.state.get() {
            CommunicatorState::Active | CommunicatorState::Finalized => (
                unsafe { sys::ncclCommDestroy(self.raw()) },
                "communicator destroy",
            ),
            CommunicatorState::Failed => (
                unsafe { sys::ncclCommAbort(self.raw()) },
                "communicator abort",
            ),
        };
        self.handle.take();
        if status != sys::ncclSuccess {
            let error = if status == sys::ncclInProgress {
                "cleanup continued asynchronously".to_owned()
            } else {
                Error::from_raw(status).to_string()
            };
            warn_from_drop(format_args!(
                "warning: {operation} did not complete synchronously ({error}); retaining communicator config for possible background teardown"
            ));
            self.leak_config();
        }
    }
}

fn warn_from_drop(message: std::fmt::Arguments<'_>) {
    // Never panic from Drop merely because stderr is unavailable.
    let _ = writeln!(std::io::stderr().lock(), "{message}");
}

fn prepare_blocking_config(mut config: Config) -> Result<Config> {
    if config.raw.blocking == 0 {
        return Err(Error::local(
            "Config::blocking(false) is unsupported by nccl4rust safe communicator initialization",
        ));
    }
    if env::var_os("NCCL_COMM_BLOCKING").is_some_and(|value| value.to_string_lossy().trim() == "0")
    {
        return Err(Error::local(
            "NCCL_COMM_BLOCKING=0 is unsupported by nccl4rust safe communicator initialization",
        ));
    }
    config.raw.blocking = 1;
    Ok(config)
}

fn validate_rank(rank_count: i32, rank: i32) -> Result<()> {
    if rank_count <= 0 {
        return Err(Error::local(
            "NCCL communicator rank count must be positive",
        ));
    }
    if !(0..rank_count).contains(&rank) {
        return Err(Error::local(format!(
            "NCCL rank {rank} is outside 0..{rank_count}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn rejects_invalid_ranks_without_touching_a_gpu() {
        assert!(validate_rank(0, 0).is_err());
        assert!(validate_rank(2, -1).is_err());
        assert!(validate_rank(2, 2).is_err());
        assert!(validate_rank(2, 1).is_ok());
    }

    #[test]
    fn safe_initialization_rejects_explicit_nonblocking_config() {
        let error = prepare_blocking_config(Config::default().blocking(false)).unwrap_err();
        assert!(error.message().contains("blocking(false)"));
    }

    #[test]
    fn prepared_config_forces_blocking_and_keeps_owned_name_storage() {
        let config = Config::default().try_comm_name("kept-alive").unwrap();
        let pointer = config.raw.commName;
        let prepared = prepare_blocking_config(config).unwrap();

        assert_eq!(prepared.raw.blocking, 1);
        assert_eq!(prepared.raw.commName, pointer);
        assert_eq!(
            unsafe { CStr::from_ptr(prepared.raw.commName) }.to_bytes(),
            b"kept-alive"
        );
    }
}
