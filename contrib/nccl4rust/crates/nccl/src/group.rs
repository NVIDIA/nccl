// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::communicator::CommunicatorState;
use crate::error::{Error, Result, Status, check};
use crate::sys;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    /// Depth of groups started through this safe wrapper on the current host
    /// thread. Raw `nccl-sys` group calls cannot be observed here.
    static GROUP_DEPTH: Cell<usize> = const { Cell::new(0) };

    /// Communicators touched through the Rust wrapper in the current outer
    /// group. NCCL can make group completion asynchronous if an external
    /// config file overrides the wrapper's requested blocking mode.
    static GROUP_COMMUNICATORS: RefCell<Vec<GroupCommunicator>> = const { RefCell::new(Vec::new()) };
}

struct GroupCommunicator {
    raw: sys::ncclComm_t,
    state: Rc<Cell<CommunicatorState>>,
}

pub(crate) fn ensure_no_active_group(operation: &str) -> Result<()> {
    if group_is_active() {
        Err(crate::Error::local(format!(
            "{operation} is not allowed inside nccl4rust Group scope because NCCL may defer writes to Rust-owned output storage"
        )))
    } else {
        Ok(())
    }
}

pub(crate) fn group_is_active() -> bool {
    GROUP_DEPTH.with(|depth| depth.get() != 0)
}

fn next_group_depth() -> Result<usize> {
    GROUP_DEPTH.with(|depth| {
        depth
            .get()
            .checked_add(1)
            .ok_or_else(|| crate::Error::local("nccl4rust Group nesting depth overflow"))
    })
}

fn set_group_depth(depth: usize) {
    GROUP_DEPTH.with(|current| current.set(depth));
}

fn leave_group() -> usize {
    GROUP_DEPTH.with(|depth| {
        let current = depth.get();
        debug_assert!(current != 0, "unbalanced nccl4rust Group tracking");
        let remaining = current.saturating_sub(1);
        depth.set(remaining);
        remaining
    })
}

fn clear_group_communicators() {
    GROUP_COMMUNICATORS.with(|communicators| communicators.borrow_mut().clear());
}

fn take_group_communicators() -> Vec<GroupCommunicator> {
    GROUP_COMMUNICATORS.with(|communicators| std::mem::take(&mut *communicators.borrow_mut()))
}

pub(crate) fn register_group_communicator(
    communicator: sys::ncclComm_t,
    state: Rc<Cell<CommunicatorState>>,
) {
    if !group_is_active() {
        return;
    }
    GROUP_COMMUNICATORS.with(|communicators| {
        let mut communicators = communicators.borrow_mut();
        if !communicators.iter().any(|entry| entry.raw == communicator) {
            communicators.push(GroupCommunicator {
                raw: communicator,
                state,
            });
        }
    });
}

fn complete_group_call(
    status: sys::ncclResult_t,
    communicators: &[GroupCommunicator],
) -> Result<()> {
    match Status::from_raw(status) {
        Status::Success => {
            if communicators.is_empty() {
                return Ok(());
            }
        }
        Status::InProgress => {
            if communicators.is_empty() {
                return Err(Error::local(
                    "ncclGroupEnd continued asynchronously, but no communicator was registered through the Rust wrapper for completion polling",
                ));
            }
        }
        _ => {
            for communicator in communicators {
                communicator.state.set(CommunicatorState::Failed);
            }
            return Err(Error::from_raw(status));
        }
    }

    let mut first_error = None;
    for communicator in communicators {
        loop {
            let mut asynchronous = sys::ncclInProgress;
            let query = unsafe { sys::ncclCommGetAsyncError(communicator.raw, &mut asynchronous) };
            if query != sys::ncclSuccess {
                communicator.state.set(CommunicatorState::Failed);
                if first_error.is_none() {
                    first_error = Some(
                        Error::from_raw(query)
                            .with_context("could not query asynchronous ncclGroupEnd completion"),
                    );
                }
                break;
            }
            match Status::from_raw(asynchronous) {
                Status::Success => break,
                Status::InProgress => std::thread::yield_now(),
                _ => {
                    communicator.state.set(CommunicatorState::Failed);
                    if first_error.is_none() {
                        first_error = Some(
                            Error::from_raw(asynchronous)
                                .with_context("asynchronous ncclGroupEnd failed"),
                        );
                    }
                    break;
                }
            }
        }
    }

    first_error.map_or(Ok(()), Err)
}

/// An active NCCL group on the current host thread.
///
/// Dropping a live guard always calls `ncclGroupEnd`, including while unwinding.
/// Prefer [`Group::scope`] when an end error must be reported.
///
/// While this guard is active, safe nccl4rust APIs reject initialization,
/// resource creation, and other calls whose Rust-owned output pointers NCCL
/// could defer until group end. Communicators used through the Rust wrapper are
/// tracked so an externally forced nonblocking group can be polled through
/// setup completion. Do not mix raw `nccl-sys` group calls with this guard: raw
/// calls bypass the thread-local tracking that enforces these guarantees.
pub struct Group {
    active: bool,
    // Group state is thread-local in NCCL. Prevent moving a live guard to a
    // different thread even if raw pointer auto-traits change in the future.
    _thread_bound: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for Group {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Group")
            .field("active", &self.active)
            .finish()
    }
}

impl Group {
    /// Start collecting NCCL calls into a group.
    pub fn start() -> Result<Self> {
        let depth = next_group_depth()?;
        check(unsafe { sys::ncclGroupStart() })?;
        if depth == 1 {
            clear_group_communicators();
        }
        set_group_depth(depth);
        Ok(Self {
            active: true,
            _thread_bound: PhantomData,
        })
    }

    /// End the group and enqueue its fused operation.
    pub fn end(mut self) -> Result<()> {
        self.finish()
    }

    /// Run a closure inside NCCL group semantics and always close the group.
    ///
    /// If both the closure and `ncclGroupEnd` fail, the closure's error is
    /// returned with the group-end failure attached as context.
    pub fn scope<T>(body: impl FnOnce(&mut Group) -> Result<T>) -> Result<T> {
        let mut group = Self::start()?;
        let body_result = body(&mut group);
        let end_result = group.finish();

        match (body_result, end_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(end_error)) => Err(end_error.with_context("ncclGroupEnd failed")),
            (Err(error), Err(end_error)) => {
                Err(error.with_context(format!("ncclGroupEnd also failed ({end_error})")))
            }
        }
    }

    fn finish(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        // Clear before entering NCCL so a failure is not retried by Drop.
        self.active = false;
        let status = unsafe { sys::ncclGroupEnd() };
        if leave_group() == 0 {
            complete_group_call(status, &take_group_communicators())
        } else {
            check(status)
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            let status = unsafe { sys::ncclGroupEnd() };
            if leave_group() == 0 {
                let _ = complete_group_call(status, &take_group_communicators());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::NonNull;

    #[test]
    fn active_group_guard_tracks_nesting_on_this_thread() {
        assert!(!group_is_active());
        let first = next_group_depth().unwrap();
        set_group_depth(first);
        assert!(group_is_active());
        assert!(ensure_no_active_group("test output call").is_err());
        assert!(crate::Version::current().is_err());
        assert!(crate::NcclMemory::<u8>::new(1).is_err());
        let id = crate::UniqueId::from_bytes([0; crate::UniqueId::BYTE_LEN]);
        assert!(crate::Communicator::init_rank(1, &id, 0).is_err());

        let second = next_group_depth().unwrap();
        set_group_depth(second);
        leave_group();
        assert!(group_is_active());
        leave_group();
        assert!(!group_is_active());
        assert!(ensure_no_active_group("test output call").is_ok());
    }

    #[test]
    fn start_end_and_drop_update_tracking_without_a_gpu() {
        let group = Group::start().unwrap();
        assert!(group_is_active());
        group.end().unwrap();
        assert!(!group_is_active());

        let group = Group::start().unwrap();
        assert!(group_is_active());
        drop(group);
        assert!(!group_is_active());
    }

    #[test]
    fn group_end_error_marks_registered_communicator_failed() {
        clear_group_communicators();
        set_group_depth(1);
        let state = Rc::new(Cell::new(CommunicatorState::Active));
        let raw = NonNull::<sys::ncclComm>::dangling().as_ptr();
        register_group_communicator(raw, Rc::clone(&state));
        register_group_communicator(raw, Rc::clone(&state));

        let communicators = take_group_communicators();
        assert_eq!(communicators.len(), 1);
        assert!(complete_group_call(sys::ncclInvalidUsage, &communicators).is_err());
        assert_eq!(state.get(), CommunicatorState::Failed);
        leave_group();
    }
}
