// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error, Result, check};
use crate::group::ensure_no_active_group;
use crate::sys;
use std::ffi::c_void;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;

const UNIQUE_ID_BYTES: usize = sys::NCCL_UNIQUE_ID_BYTES as usize;

/// The version of the loaded NCCL library.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Version {
    code: u32,
    major: u32,
    minor: u32,
    patch: u32,
}

impl Version {
    /// Query the version of the dynamically linked NCCL library.
    pub fn current() -> Result<Self> {
        ensure_no_active_group("NCCL version query")?;
        let mut code = 0;
        check(unsafe { sys::ncclGetVersion(&mut code) })?;
        let code = u32::try_from(code)
            .map_err(|_| Error::local(format!("NCCL returned a negative version code: {code}")))?;
        Ok(Self::from_code(code))
    }

    /// Decode an NCCL version code.
    pub const fn from_code(code: u32) -> Self {
        // NCCL_VERSION used M*1000+m*100+p through 2.8 and has used
        // M*10000+m*100+p since 2.9.
        let (major, minor, patch) = if code < 10_000 {
            (code / 1_000, (code % 1_000) / 100, code % 100)
        } else {
            (code / 10_000, (code % 10_000) / 100, code % 100)
        };
        Self {
            code,
            major,
            minor,
            patch,
        }
    }

    pub const fn code(self) -> u32 {
        self.code
    }

    pub const fn major(self) -> u32 {
        self.major
    }

    pub const fn minor(self) -> u32 {
        self.minor
    }

    pub const fn patch(self) -> u32 {
        self.patch
    }
}

impl fmt::Display for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The 128-byte token used to initialize a communicator clique.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct UniqueId(sys::ncclUniqueId);

impl UniqueId {
    pub const BYTE_LEN: usize = UNIQUE_ID_BYTES;

    /// Ask NCCL to generate a unique communicator identifier.
    pub fn generate() -> Result<Self> {
        ensure_no_active_group("NCCL unique-ID generation")?;
        let mut raw = MaybeUninit::<sys::ncclUniqueId>::uninit();
        check(unsafe { sys::ncclGetUniqueId(raw.as_mut_ptr()) })?;
        // ncclGetUniqueId initializes every byte on success.
        Ok(Self(unsafe { raw.assume_init() }))
    }

    /// Reconstruct an identifier distributed by another rank.
    pub fn from_bytes(bytes: [u8; UNIQUE_ID_BYTES]) -> Self {
        let mut raw = MaybeUninit::<sys::ncclUniqueId>::uninit();
        // ncclUniqueId is exactly one byte array; every bit pattern is valid.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                raw.as_mut_ptr().cast::<u8>(),
                UNIQUE_ID_BYTES,
            );
            Self(raw.assume_init())
        }
    }

    /// Bytes suitable for transport to the other ranks.
    pub fn as_bytes(&self) -> &[u8; UNIQUE_ID_BYTES] {
        // The public NCCL ABI defines ncclUniqueId as exactly this byte array.
        unsafe { &*(ptr::from_ref(&self.0).cast::<[u8; UNIQUE_ID_BYTES]>()) }
    }

    pub fn into_bytes(self) -> [u8; UNIQUE_ID_BYTES] {
        *self.as_bytes()
    }

    pub(crate) const fn as_raw(self) -> sys::ncclUniqueId {
        self.0
    }
}

impl PartialEq for UniqueId {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for UniqueId {}

impl fmt::Debug for UniqueId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UniqueId")
            .field("bytes", &format_args!("[{} bytes]", UNIQUE_ID_BYTES))
            .finish()
    }
}

/// An opaque CUDA stream handle.
///
/// This avoids coupling the host wrapper to a particular CUDA Rust crate.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CudaStream(sys::cudaStream_t);

impl CudaStream {
    /// CUDA's legacy default stream.
    pub const DEFAULT: Self = Self(ptr::null_mut());

    /// Borrow a raw CUDA runtime stream handle.
    ///
    /// # Safety
    ///
    /// `raw` must be null (the default stream) or a live `cudaStream_t` that is
    /// valid for every NCCL enqueue using the returned value. The owner must
    /// keep it alive until all such work is complete.
    pub const unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw.cast())
    }

    /// Return the untyped CUDA runtime stream handle without transferring
    /// ownership.
    pub const fn as_raw(self) -> *mut c_void {
        self.0.cast()
    }

    pub(crate) const fn as_sys(self) -> sys::cudaStream_t {
        self.0
    }
}

/// A built-in NCCL reduction operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReductionOp {
    Sum,
    Product,
    Maximum,
    Minimum,
    Average,
}

impl ReductionOp {
    pub(crate) const fn as_raw(self) -> sys::ncclRedOp_t {
        match self {
            Self::Sum => sys::ncclSum,
            Self::Product => sys::ncclProd,
            Self::Maximum => sys::ncclMax,
            Self::Minimum => sys::ncclMin,
            Self::Average => sys::ncclAvg,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust scalar with an ABI-compatible NCCL datatype.
///
/// This trait is sealed because an incorrect datatype mapping can make a typed
/// pointer operation access the wrong number of bytes.
pub trait NcclDataType: sealed::Sealed {
    const NCCL_DATA_TYPE: sys::ncclDataType_t;
}

macro_rules! impl_data_type {
    ($rust:ty, $nccl:ident) => {
        impl sealed::Sealed for $rust {}
        impl NcclDataType for $rust {
            const NCCL_DATA_TYPE: sys::ncclDataType_t = sys::$nccl;
        }
    };
}

impl_data_type!(i8, ncclInt8);
impl_data_type!(u8, ncclUint8);
impl_data_type!(i32, ncclInt32);
impl_data_type!(u32, ncclUint32);
impl_data_type!(i64, ncclInt64);
impl_data_type!(u64, ncclUint64);
impl_data_type!(f32, ncclFloat32);
impl_data_type!(f64, ncclFloat64);

/// Bit-level storage for an IEEE binary16 value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Float16(pub u16);

/// Bit-level storage for a bfloat16 value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BFloat16(pub u16);

/// Bit-level storage for an FP8 e4m3 value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Float8E4M3(pub u8);

/// Bit-level storage for an FP8 e5m2 value.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Float8E5M2(pub u8);

impl_data_type!(Float16, ncclFloat16);
impl_data_type!(BFloat16, ncclBfloat16);
impl_data_type!(Float8E4M3, ncclFloat8e4m3);
impl_data_type!(Float8E5M2, ncclFloat8e5m2);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn decodes_old_and_current_version_schemes() {
        let old = Version::from_code(2_708);
        assert_eq!((old.major(), old.minor(), old.patch()), (2, 7, 8));
        assert_eq!(old.to_string(), "2.7.8");

        let current = Version::from_code(23_100);
        assert_eq!(
            (current.major(), current.minor(), current.patch()),
            (2, 31, 0)
        );
    }

    #[test]
    fn loaded_library_matches_generated_headers() {
        assert_eq!(Version::current().unwrap().code(), sys::NCCL_VERSION_CODE);
    }

    #[test]
    fn unique_id_bytes_round_trip() {
        let mut bytes = [0_u8; UNIQUE_ID_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let id = UniqueId::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
        assert_eq!(id.into_bytes(), bytes);
        assert_eq!(size_of::<UniqueId>(), UNIQUE_ID_BYTES);
    }

    #[test]
    fn scalar_mappings_have_the_expected_width() {
        assert_eq!(size_of::<Float16>(), 2);
        assert_eq!(align_of::<Float16>(), align_of::<u16>());
        assert_eq!(size_of::<BFloat16>(), 2);
        assert_eq!(size_of::<Float8E4M3>(), 1);
        assert_eq!(size_of::<Float8E5M2>(), 1);
        assert_eq!(<f32 as NcclDataType>::NCCL_DATA_TYPE, sys::ncclFloat32);
    }
}
