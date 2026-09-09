// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::sys;
use std::error;
use std::ffi::CStr;
use std::fmt;

/// A decoded NCCL status value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Status {
    Success,
    UnhandledCudaError,
    SystemError,
    InternalError,
    InvalidArgument,
    InvalidUsage,
    RemoteError,
    InProgress,
    Timeout,
    Unknown(sys::ncclResult_t),
}

impl Status {
    /// Decode a raw `ncclResult_t`, preserving values introduced by newer NCCL
    /// libraries.
    pub const fn from_raw(raw: sys::ncclResult_t) -> Self {
        match raw {
            sys::ncclSuccess => Self::Success,
            sys::ncclUnhandledCudaError => Self::UnhandledCudaError,
            sys::ncclSystemError => Self::SystemError,
            sys::ncclInternalError => Self::InternalError,
            sys::ncclInvalidArgument => Self::InvalidArgument,
            sys::ncclInvalidUsage => Self::InvalidUsage,
            sys::ncclRemoteError => Self::RemoteError,
            sys::ncclInProgress => Self::InProgress,
            sys::ncclTimeout => Self::Timeout,
            other => Self::Unknown(other),
        }
    }

    /// Return the raw NCCL status value.
    pub const fn as_raw(self) -> sys::ncclResult_t {
        match self {
            Self::Success => sys::ncclSuccess,
            Self::UnhandledCudaError => sys::ncclUnhandledCudaError,
            Self::SystemError => sys::ncclSystemError,
            Self::InternalError => sys::ncclInternalError,
            Self::InvalidArgument => sys::ncclInvalidArgument,
            Self::InvalidUsage => sys::ncclInvalidUsage,
            Self::RemoteError => sys::ncclRemoteError,
            Self::InProgress => sys::ncclInProgress,
            Self::Timeout => sys::ncclTimeout,
            Self::Unknown(raw) => raw,
        }
    }
}

/// An error reported by NCCL or detected before entering NCCL.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    status: Option<Status>,
    message: String,
}

impl Error {
    pub(crate) fn from_raw(raw: sys::ncclResult_t) -> Self {
        let status = Status::from_raw(raw);
        let pointer = unsafe { sys::ncclGetErrorString(raw) };
        let message = if pointer.is_null() {
            format!("NCCL status {raw}")
        } else {
            // NCCL documents this as a static, null-terminated error string.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        Self {
            status: Some(status),
            message,
        }
    }

    pub(crate) fn local(message: impl Into<String>) -> Self {
        Self {
            status: None,
            message: message.into(),
        }
    }

    pub(crate) fn with_context(mut self, context: impl AsRef<str>) -> Self {
        self.message = format!("{}: {}", context.as_ref(), self.message);
        self
    }

    /// The NCCL status, or `None` for an error detected by this wrapper.
    pub const fn status(&self) -> Option<Status> {
        self.status
    }

    /// The human-readable NCCL error string or wrapper diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(status) = self.status {
            write!(formatter, "{status:?}: {}", self.message)
        } else {
            formatter.write_str(&self.message)
        }
    }
}

impl error::Error for Error {}

/// Result type used by the host wrapper.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn check(raw: sys::ncclResult_t) -> Result<()> {
    if raw == sys::ncclSuccess {
        Ok(())
    } else {
        Err(Error::from_raw(raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_and_future_status_values_are_preserved() {
        assert_eq!(
            Status::from_raw(sys::ncclInvalidUsage),
            Status::InvalidUsage
        );
        let future = sys::ncclNumResults + 41;
        assert_eq!(Status::from_raw(future), Status::Unknown(future));
        assert_eq!(Status::Unknown(future).as_raw(), future);
    }

    #[test]
    fn local_error_has_no_nccl_status() {
        let error = Error::local("bad local input");
        assert_eq!(error.status(), None);
        assert_eq!(error.to_string(), "bad local input");
    }
}
