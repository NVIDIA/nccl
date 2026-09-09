// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::communicator::{Communicator, ManagementCompletion};
use crate::error::{Error, Result, check};
use crate::group::ensure_no_active_group;
use crate::sys;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::{BitOr, BitOrAssign};
use std::ptr::{self, NonNull};

/// An allocation made by `ncclMemAlloc`.
///
/// The allocation is CUDA device memory; this type intentionally does not
/// implement `Deref` or create host slices.
pub struct NcclMemory<T> {
    pointer: NonNull<T>,
    len: usize,
    allocated: bool,
    _element: PhantomData<T>,
}

impl<T> std::fmt::Debug for NcclMemory<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NcclMemory")
            .field("pointer", &self.pointer)
            .field("len", &self.len)
            .finish()
    }
}

impl<T> NcclMemory<T> {
    /// Allocate space for `len` values of `T` using NCCL's allocator.
    pub fn new(len: usize) -> Result<Self> {
        ensure_no_active_group("NCCL memory allocation")?;
        if size_of::<T>() == 0 {
            return Err(Error::local(
                "NcclMemory does not support zero-sized element types",
            ));
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::local("NCCL allocation size overflow"))?;

        if bytes == 0 {
            return Ok(Self {
                pointer: NonNull::dangling(),
                len,
                allocated: false,
                _element: PhantomData,
            });
        }

        let mut pointer: *mut c_void = ptr::null_mut();
        check(unsafe { sys::ncclMemAlloc(&mut pointer, bytes) })?;
        let pointer = NonNull::new(pointer.cast::<T>()).ok_or_else(|| {
            Error::local(format!(
                "ncclMemAlloc reported success but returned null for {bytes} bytes"
            ))
        })?;
        Ok(Self {
            pointer,
            len,
            allocated: true,
            _element: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        // Checked by new and invariant while self is alive.
        self.len * size_of::<T>()
    }

    pub const fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }

    /// Return the device pointer for APIs that write this allocation.
    ///
    /// The mutable borrow only prevents overlapping access through this Rust
    /// owner. Current raw-pointer enqueue APIs do not track CUDA stream ordering
    /// or outstanding device aliases; callers coordinate them according to each
    /// method's documented contract.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr()
    }

    fn leak_allocation(&mut self) {
        self.allocated = false;
    }

    fn transfer_allocation_to_window(&mut self) -> bool {
        let restore_on_deregister = self.allocated;
        self.allocated = false;
        restore_on_deregister
    }

    fn restore_allocation_after_deregister(&mut self, restore: bool) {
        if restore {
            self.allocated = true;
        }
    }
}

impl<T> Drop for NcclMemory<T> {
    fn drop(&mut self) {
        if self.allocated {
            self.allocated = false;
            unsafe {
                sys::ncclMemFree(self.pointer.as_ptr().cast::<c_void>());
            }
        }
    }
}

/// Flags accepted by `ncclCommWindowRegister`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct WindowFlags(i32);

impl WindowFlags {
    pub const DEFAULT: Self = Self(sys::NCCL_WIN_DEFAULT as i32);
    pub const COLLECTIVE_SYMMETRIC: Self = Self(sys::NCCL_WIN_COLL_SYMMETRIC as i32);
    pub const STRICT_ORDERING: Self = Self(sys::NCCL_WIN_STRICT_ORDERING as i32);

    pub const fn bits(self) -> i32 {
        self.0
    }
}

impl BitOr for WindowFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for WindowFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A registered NCCL memory window.
///
/// The two borrows ensure that the communicator and its backing allocation
/// both outlive deregistration.
pub struct Window<'comm, 'memory, T> {
    handle: Option<NonNull<sys::ncclWindow_vidmem>>,
    communicator: &'comm Communicator,
    memory: &'memory mut NcclMemory<T>,
    restore_allocation_on_drop: bool,
}

impl<T> std::fmt::Debug for Window<'_, '_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Window")
            .field("handle", &self.handle.map(NonNull::as_ptr))
            .field("pointer", &self.memory.pointer)
            .field("len", &self.memory.len)
            .finish()
    }
}

impl Communicator {
    /// Register an NCCL allocation as a memory window.
    ///
    /// This operation is collective. Every rank must call it in matching order
    /// with memory satisfying the selected symmetry flags.
    pub fn register_window<'comm, 'memory, T>(
        &'comm self,
        memory: &'memory mut NcclMemory<T>,
        flags: WindowFlags,
    ) -> Result<Window<'comm, 'memory, T>> {
        ensure_no_active_group("NCCL memory-window registration")?;
        let communicator = self.active_raw()?;
        if memory.is_empty() {
            return Err(Error::local("cannot register an empty NCCL memory window"));
        }

        let mut output = Box::new(ptr::null_mut::<sys::ncclWindow_vidmem>());
        let status = unsafe {
            sys::ncclCommWindowRegister(
                communicator,
                memory.as_mut_ptr().cast::<c_void>(),
                memory.byte_len(),
                output.as_mut(),
                flags.bits(),
            )
        };
        match self.complete_deferred_output_call(status, "NCCL memory-window registration") {
            ManagementCompletion::Complete => {}
            ManagementCompletion::SynchronousFailure(error)
            | ManagementCompletion::TerminalFailure(error) => {
                if let Some(handle) = NonNull::new(*output) {
                    let cleanup =
                        unsafe { sys::ncclCommWindowDeregister(communicator, handle.as_ptr()) };
                    if cleanup != sys::ncclSuccess {
                        memory.leak_allocation();
                        return Err(error.with_context(format!(
                            "registration published a window and cleanup failed ({}); retained the NCCL allocation",
                            Error::from_raw(cleanup)
                        )));
                    }
                }
                return Err(error);
            }
            ManagementCompletion::Uncertain(error) => {
                // NCCL may still write the output handle. It is safer to leak
                // both the output target and backing device allocation than
                // let either disappear while registration may still use it.
                Box::leak(output);
                memory.leak_allocation();
                return Err(error.with_context(
                    "retained the NCCL allocation and output slot because registration completion is uncertain",
                ));
            }
        }
        let handle = NonNull::new(*output)
            .ok_or_else(|| Error::local("NCCL registered a null memory-window handle"))?;
        // Transfer allocation-release responsibility to the Window before it
        // becomes possible to forget the guard. If Window is forgotten, the
        // backing allocation intentionally leaks instead of being freed while
        // NCCL still retains the registration.
        let restore_allocation_on_drop = memory.transfer_allocation_to_window();
        Ok(Window {
            handle: Some(handle),
            communicator: self,
            memory,
            restore_allocation_on_drop,
        })
    }
}

impl<T> Window<'_, '_, T> {
    pub const fn len(&self) -> usize {
        self.memory.len
    }

    pub const fn is_empty(&self) -> bool {
        self.memory.len == 0
    }

    pub const fn as_ptr(&self) -> *const T {
        self.memory.pointer.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.pointer.as_ptr()
    }

    /// Query the local user pointer associated with this window.
    pub fn user_ptr(&self) -> Result<*mut T> {
        ensure_no_active_group("NCCL memory-window user-pointer query")?;
        let mut pointer: *mut c_void = ptr::null_mut();
        check(unsafe {
            sys::ncclWinGetUserPtr(self.communicator.raw(), self.as_raw(), &mut pointer)
        })?;
        Ok(pointer.cast::<T>())
    }

    /// Return the borrowed public NCCL window handle.
    ///
    /// This handle is the value passed to device kernels using the NCCL
    /// window API. It remains valid only while this `Window` is alive; kernel
    /// launches using it must complete before the window is dropped.
    pub fn as_raw(&self) -> sys::ncclWindow_t {
        self.handle
            .expect("live Window must contain a raw handle")
            .as_ptr()
    }
}

impl<T> Drop for Window<'_, '_, T> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let status =
                unsafe { sys::ncclCommWindowDeregister(self.communicator.raw(), handle.as_ptr()) };
            if status != sys::ncclSuccess {
                // NCCL may still retain the registration and its user pointer.
                // Keep the device allocation alive rather than let the now-
                // released Rust borrow enable a use-after-free.
                self.memory.leak_allocation();
            } else {
                self.memory
                    .restore_allocation_after_deregister(self.restore_allocation_on_drop);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_flags_compose_without_touching_cuda() {
        let flags = WindowFlags::COLLECTIVE_SYMMETRIC | WindowFlags::STRICT_ORDERING;
        assert_eq!(
            flags.bits(),
            (sys::NCCL_WIN_COLL_SYMMETRIC | sys::NCCL_WIN_STRICT_ORDERING) as i32
        );
    }

    #[test]
    fn allocation_size_checks_happen_before_nccl() {
        let error = NcclMemory::<[u8; 2]>::new(usize::MAX).unwrap_err();
        assert!(error.message().contains("overflow"));
        let zst = NcclMemory::<()>::new(1).unwrap_err();
        assert!(zst.message().contains("zero-sized"));
    }
}
