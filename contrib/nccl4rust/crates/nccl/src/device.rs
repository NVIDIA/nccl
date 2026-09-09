// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::communicator::{Communicator, ManagementCompletion};
use crate::error::{Error, Result, check};
use crate::group::ensure_no_active_group;
use crate::sys;
use std::alloc::Layout;
use std::mem::{MaybeUninit, align_of, size_of};
use std::ptr;

/// The GIN connections requested for a device communicator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GinConnectionType {
    None,
    Full,
    Rail,
    /// Connect ranks using the supplied positive world-rank stride.
    CustomStride(i32),
}

impl GinConnectionType {
    const fn as_raw(self) -> sys::ncclGinConnectionType_t {
        match self {
            Self::None => sys::NCCL_GIN_CONNECTION_NONE,
            Self::Full => sys::NCCL_GIN_CONNECTION_FULL,
            Self::Rail => sys::NCCL_GIN_CONNECTION_RAIL,
            Self::CustomStride(_) => sys::NCCL_GIN_CONNECTION_CUSTOM_STRIDE,
        }
    }
}

/// GIN backend requested for a device communicator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GinType {
    /// Accept any backend supported by every communicator rank.
    Any,
    Proxy,
    Gdaki,
    Gpi,
    EfaGda,
}

impl GinType {
    const fn as_raw(self) -> sys::ncclGinType_t {
        match self {
            Self::Any => sys::NCCL_GIN_TYPE_NONE,
            Self::Proxy => sys::NCCL_GIN_TYPE_PROXY,
            Self::Gdaki => sys::NCCL_GIN_TYPE_GDAKI,
            Self::Gpi => sys::NCCL_GIN_TYPE_GPI,
            Self::EfaGda => sys::NCCL_GIN_TYPE_EFA_GDA,
        }
    }
}

/// Initialized requirements for `ncclDevCommCreate`.
///
/// `Default` exactly mirrors `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER` from the
/// selected NCCL headers. Resource/team linked-list builders are intentionally
/// not exposed yet because their output-handle pointers need a separate pinned
/// ownership API. Runtime-version device communicators are also left disabled:
/// supporting them requires querying and allocating the runtime-selected image
/// size instead of this wrapper's compile-time `ncclDevComm_t` layout.
#[derive(Clone)]
pub struct DeviceCommRequirements {
    raw: sys::ncclDevCommRequirements_t,
}

impl Default for DeviceCommRequirements {
    fn default() -> Self {
        Self {
            raw: sys::ncclDevCommRequirements_t {
                size: size_of::<sys::ncclDevCommRequirements_t>(),
                magic: sys::NCCL_API_MAGIC,
                version: sys::NCCL_VERSION_CODE,
                resourceRequirementsList: ptr::null_mut(),
                teamRequirementsList: ptr::null_mut(),
                lsaMultimem: false,
                barrierCount: 0,
                lsaBarrierCount: 0,
                railGinBarrierCount: 0,
                lsaLLA2ABlockCount: 0,
                lsaLLA2ASlotCount: 0,
                ginForceEnable: false,
                ginContextCount: 4,
                ginSignalCount: 0,
                ginCounterCount: 0,
                ginConnectionType: sys::NCCL_GIN_CONNECTION_NONE,
                ginExclusiveContexts: false,
                ginQueueDepth: 0,
                ginTrafficClass: sys::NCCL_CONFIG_UNDEF_INT,
                worldGinBarrierCount: 0,
                ginStrongSignalsRequired: true,
                ginVaSignalsRequired: true,
                ginCustomStride: 1,
                ginType: sys::NCCL_GIN_TYPE_NONE,
                useRuntimeVersion: false,
                cftCaps: sys::NCCL_CFT_NONE as i32,
                cftBarrierCount: 0,
            },
        }
    }
}

impl std::fmt::Debug for DeviceCommRequirements {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceCommRequirements")
            .field("lsa_multimem", &self.raw.lsaMultimem)
            .field("barrier_count", &self.raw.barrierCount)
            .field("lsa_barrier_count", &self.raw.lsaBarrierCount)
            .field("rail_gin_barrier_count", &self.raw.railGinBarrierCount)
            .field("lsa_lla2a_block_count", &self.raw.lsaLLA2ABlockCount)
            .field("lsa_lla2a_slot_count", &self.raw.lsaLLA2ASlotCount)
            .field("gin_force_enable", &self.raw.ginForceEnable)
            .field("gin_context_count", &self.raw.ginContextCount)
            .field("gin_signal_count", &self.raw.ginSignalCount)
            .field("gin_counter_count", &self.raw.ginCounterCount)
            .field("gin_connection_type", &self.raw.ginConnectionType)
            .field("gin_exclusive_contexts", &self.raw.ginExclusiveContexts)
            .field("gin_queue_depth", &self.raw.ginQueueDepth)
            .field("gin_traffic_class", &self.raw.ginTrafficClass)
            .field("world_gin_barrier_count", &self.raw.worldGinBarrierCount)
            .field(
                "gin_strong_signals_required",
                &self.raw.ginStrongSignalsRequired,
            )
            .field("gin_va_signals_required", &self.raw.ginVaSignalsRequired)
            .field("gin_custom_stride", &self.raw.ginCustomStride)
            .field("gin_type", &self.raw.ginType)
            .field("use_runtime_version", &self.raw.useRuntimeVersion)
            .field("cft_caps", &self.raw.cftCaps)
            .field("cft_barrier_count", &self.raw.cftBarrierCount)
            .finish()
    }
}

impl DeviceCommRequirements {
    pub fn lsa_multimem(mut self, enabled: bool) -> Self {
        self.raw.lsaMultimem = enabled;
        self
    }

    pub fn barrier_count(mut self, count: i32) -> Self {
        self.raw.barrierCount = count;
        self
    }

    pub fn lsa_barrier_count(mut self, count: i32) -> Self {
        self.raw.lsaBarrierCount = count;
        self
    }

    pub fn rail_gin_barrier_count(mut self, count: i32) -> Self {
        self.raw.railGinBarrierCount = count;
        self
    }

    pub fn lsa_lla2a(mut self, block_count: i32, slot_count: i32) -> Self {
        self.raw.lsaLLA2ABlockCount = block_count;
        self.raw.lsaLLA2ASlotCount = slot_count;
        self
    }

    pub fn gin_force_enable(mut self, enabled: bool) -> Self {
        self.raw.ginForceEnable = enabled;
        self
    }

    pub fn gin_context_count(mut self, count: i32) -> Self {
        self.raw.ginContextCount = count;
        self
    }

    pub fn gin_signal_count(mut self, count: i32) -> Self {
        self.raw.ginSignalCount = count;
        self
    }

    pub fn gin_counter_count(mut self, count: i32) -> Self {
        self.raw.ginCounterCount = count;
        self
    }

    pub fn gin_connection_type(mut self, connection_type: GinConnectionType) -> Self {
        self.raw.ginConnectionType = connection_type.as_raw();
        self.raw.ginCustomStride = match connection_type {
            GinConnectionType::CustomStride(stride) => stride,
            _ => 1,
        };
        self
    }

    pub fn gin_type(mut self, gin_type: GinType) -> Self {
        self.raw.ginType = gin_type.as_raw();
        self
    }

    pub fn gin_exclusive_contexts(mut self, enabled: bool) -> Self {
        self.raw.ginExclusiveContexts = enabled;
        self
    }

    pub fn gin_queue_depth(mut self, depth: i32) -> Self {
        self.raw.ginQueueDepth = depth;
        self
    }

    pub fn gin_traffic_class(mut self, traffic_class: i32) -> Self {
        self.raw.ginTrafficClass = traffic_class;
        self
    }

    pub fn world_gin_barrier_count(mut self, count: i32) -> Self {
        self.raw.worldGinBarrierCount = count;
        self
    }

    pub fn gin_strong_signals_required(mut self, required: bool) -> Self {
        self.raw.ginStrongSignalsRequired = required;
        self
    }

    pub fn gin_va_signals_required(mut self, required: bool) -> Self {
        self.raw.ginVaSignalsRequired = required;
        self
    }

    fn validate_for_communicator(&self, communicator: sys::ncclComm_t) -> Result<()> {
        // These public host helpers return value types and are valid for every
        // active communicator. The team sizes are needed to mirror NCCL's
        // device-resource sizing expressions before entering C++.
        let world = unsafe { sys::ncclTeamWorld(communicator) };
        let lsa = unsafe { sys::ncclTeamLsa(communicator) };
        let rail = unsafe { sys::ncclTeamRail(communicator) };
        self.validate_team_sizes(world.nRanks, lsa.nRanks, rail.nRanks)
    }

    fn validate_team_sizes(&self, world_size: i32, lsa_size: i32, rail_size: i32) -> Result<()> {
        for (name, value) in [
            ("world team size", world_size),
            ("LSA team size", lsa_size),
            ("rail team size", rail_size),
        ] {
            if value <= 0 {
                return Err(Error::local(format!(
                    "NCCL returned an invalid {name}: {value}"
                )));
            }
        }

        for (name, value) in [
            ("barrier count", self.raw.barrierCount),
            ("LSA barrier count", self.raw.lsaBarrierCount),
            ("rail GIN barrier count", self.raw.railGinBarrierCount),
            ("LSA LLA2A block count", self.raw.lsaLLA2ABlockCount),
            ("LSA LLA2A slot count", self.raw.lsaLLA2ASlotCount),
            ("GIN context count", self.raw.ginContextCount),
            ("GIN signal count", self.raw.ginSignalCount),
            ("GIN counter count", self.raw.ginCounterCount),
            ("GIN queue depth", self.raw.ginQueueDepth),
            ("world GIN barrier count", self.raw.worldGinBarrierCount),
            ("CFT barrier count", self.raw.cftBarrierCount),
        ] {
            require_nonnegative(name, value)?;
        }

        if self.raw.ginConnectionType == sys::NCCL_GIN_CONNECTION_CUSTOM_STRIDE
            && self.raw.ginCustomStride <= 0
        {
            return Err(Error::local(format!(
                "GIN custom stride must be positive, got {}",
                self.raw.ginCustomStride
            )));
        }
        if self.raw.ginConnectionType == sys::NCCL_GIN_CONNECTION_CUSTOM_STRIDE
            && self.raw.ginCustomStride > lsa_size
        {
            return Err(Error::local(format!(
                "GIN custom stride {} exceeds the LSA size {lsa_size}",
                self.raw.ginCustomStride
            )));
        }

        // ncclLsaBarrierCreateRequirement computes
        //   (3*nBarriers + nBarriers*team.nRanks)
        // as signed int before converting to size_t.
        checked_lsa_barrier_resources(
            "hybrid LSA barrier resources",
            self.raw.barrierCount,
            lsa_size,
        )?;
        checked_lsa_barrier_resources("LSA barrier resources", self.raw.lsaBarrierCount, lsa_size)?;

        // The GIN helpers multiply barrier counts by team sizes as signed int,
        // then ncclDevrCommCreateInternal accumulates every signal count in an
        // int and multiplies it by the context count.
        let hybrid_rail_signals = checked_nonnegative_product(
            "hybrid rail GIN barrier resources",
            &[self.raw.barrierCount, rail_size],
        )?;
        let hybrid_world_signals = checked_nonnegative_product(
            "hybrid world GIN barrier resources",
            &[self.raw.barrierCount, world_size],
        )?;
        let rail_signals = checked_nonnegative_product(
            "rail GIN barrier resources",
            &[self.raw.railGinBarrierCount, rail_size],
        )?;
        let world_signals = checked_nonnegative_product(
            "world GIN barrier resources",
            &[self.raw.worldGinBarrierCount, world_size],
        )?;
        let signal_total = checked_nonnegative_sum(
            "GIN signal resources",
            &[
                self.raw.ginSignalCount,
                hybrid_rail_signals,
                hybrid_world_signals,
                rail_signals,
                world_signals,
            ],
        )?;
        checked_nonnegative_product(
            "GIN context signal shadows",
            &[self.raw.ginContextCount, signal_total],
        )?;

        // GIN enablement is sticky on a communicator: a previous device
        // communicator may have activated it even when this request selects
        // `None`. A backend may expose up to four GIN connections. ROUNDUP
        // first adds connection_count-1 in signed arithmetic, then the GDAKI
        // backend sizes QPs and signal/counter tables with signed products.
        // Validate those expressions for every request because the public host
        // API does not expose the communicator's current sticky state.
        let rounded_context_bound = checked_add("GIN context count", self.raw.ginContextCount, 3)?;
        let world_with_local = checked_add("world team size", world_size, 1)?;
        checked_nonnegative_product(
            "GIN communication QPs",
            &[rounded_context_bound, world_size],
        )?;
        checked_nonnegative_product("GIN companion QPs", &[rounded_context_bound, world_size, 2])?;
        checked_nonnegative_product(
            "GIN local and communication QPs",
            &[rounded_context_bound, world_with_local],
        )?;
        checked_nonnegative_product("GIN signal table", &[rounded_context_bound, signal_total])?;
        checked_nonnegative_product(
            "GIN counter table",
            &[rounded_context_bound, self.raw.ginCounterCount],
        )?;

        // ncclLLA2ACreateRequirement uses nBlocks*(1 + 2*nSlots)*16 in
        // signed arithmetic. This release does not consume these fields yet,
        // but validating the public contract keeps the wrapper safe when it
        // does and matches the helper backing that API.
        let slots_twice = checked_nonnegative_product(
            "LSA LLA2A slot resources",
            &[self.raw.lsaLLA2ASlotCount, 2],
        )?;
        let slots_with_control = checked_add("LSA LLA2A slot resources", slots_twice, 1)?;
        checked_nonnegative_product(
            "LSA LLA2A resources",
            &[self.raw.lsaLLA2ABlockCount, slots_with_control, 16],
        )?;

        Ok(())
    }
}

fn require_nonnegative(name: &str, value: i32) -> Result<()> {
    if value < 0 {
        Err(Error::local(format!(
            "{name} must be nonnegative, got {value}"
        )))
    } else {
        Ok(())
    }
}

fn checked_add(name: &str, left: i32, right: i32) -> Result<i32> {
    left.checked_add(right)
        .ok_or_else(|| Error::local(format!("{name} exceeds NCCL's signed resource limit")))
}

fn checked_nonnegative_product(name: &str, factors: &[i32]) -> Result<i32> {
    factors.iter().try_fold(1_i32, |product, &factor| {
        require_nonnegative(name, factor)?;
        product
            .checked_mul(factor)
            .ok_or_else(|| Error::local(format!("{name} exceeds NCCL's signed resource limit")))
    })
}

fn checked_lsa_barrier_resources(name: &str, count: i32, team_size: i32) -> Result<i32> {
    let control_words = checked_nonnegative_product(name, &[3, count])?;
    let rank_words = checked_nonnegative_product(name, &[count, team_size])?;
    checked_add(name, control_words, rank_words)
}

fn checked_nonnegative_sum(name: &str, terms: &[i32]) -> Result<i32> {
    terms.iter().try_fold(0_i32, |sum, &term| {
        require_nonnegative(name, term)?;
        sum.checked_add(term)
            .ok_or_else(|| Error::local(format!("{name} exceeds NCCL's signed resource limit")))
    })
}

/// An owned host image of `ncclDevComm_t` and its NCCL-side resources.
///
/// The device wrapper expects a pointer to a device-resident copy of this
/// image. [`Self::as_bytes`] exposes the exact initialized host representation
/// so a CUDA integration can allocate device storage and copy it there without
/// this crate depending on a particular CUDA runtime wrapper.
pub struct DeviceCommunicator<'comm> {
    communicator: &'comm Communicator,
    raw: Option<Box<sys::ncclDevComm_t>>,
}

impl std::fmt::Debug for DeviceCommunicator<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceCommunicator")
            .field("byte_len", &Self::BYTE_LEN)
            .field("copy_alignment", &Self::COPY_ALIGNMENT)
            .finish_non_exhaustive()
    }
}

impl Communicator {
    /// Create the device-side communicator image and associated NCCL
    /// resources.
    pub fn create_device_communicator<'comm>(
        &'comm self,
        requirements: &DeviceCommRequirements,
    ) -> Result<DeviceCommunicator<'comm>> {
        ensure_no_active_group("NCCL device-communicator creation")?;
        let communicator = self.active_raw()?;
        requirements.validate_for_communicator(communicator)?;
        // NCCL's implementation initializes the entire public image (including
        // padding) before returning. Starting zeroed also keeps failure cleanup
        // and byte-copy instrumentation deterministic.
        let mut raw = Box::new(MaybeUninit::<sys::ncclDevComm_t>::zeroed());
        let status =
            unsafe { sys::ncclDevCommCreate(communicator, &requirements.raw, raw.as_mut_ptr()) };
        match self.complete_deferred_output_call(status, "NCCL device-communicator creation") {
            ManagementCompletion::Complete => {}
            ManagementCompletion::SynchronousFailure(error)
            | ManagementCompletion::TerminalFailure(error) => return Err(error),
            ManagementCompletion::Uncertain(error) => {
                // A deferred NCCL task may still target this host image.
                Box::leak(raw);
                return Err(error);
            }
        }
        // Validate the sentinel fields after terminal polling before treating
        // the zero-initialized output image as this call's result.
        let initialized = unsafe { raw.assume_init_ref() };
        if initialized.magic != sys::NCCL_API_MAGIC || initialized.version != sys::NCCL_VERSION_CODE
        {
            return Err(Error::local(format!(
                "NCCL did not initialize the device communicator image (magic={:#x}, version={})",
                initialized.magic, initialized.version
            )));
        }
        let raw_pointer = Box::into_raw(raw).cast::<sys::ncclDevComm_t>();
        let raw = unsafe { Box::from_raw(raw_pointer) };
        Ok(DeviceCommunicator {
            communicator: self,
            raw: Some(raw),
        })
    }
}

impl DeviceCommunicator<'_> {
    /// Number of bytes that must be copied for device use.
    pub const BYTE_LEN: usize = size_of::<sys::ncclDevComm_t>();

    /// Required alignment for a typed device-side `ncclDevComm_t` image.
    pub const COPY_ALIGNMENT: usize = align_of::<sys::ncclDevComm_t>();

    pub const fn copy_layout() -> Layout {
        // Both values come from one concrete Rust type and therefore always
        // form a valid Layout.
        match Layout::from_size_align(Self::BYTE_LEN, Self::COPY_ALIGNMENT) {
            Ok(layout) => layout,
            Err(_) => unreachable!(),
        }
    }

    /// Return the complete host image for a host-to-device copy.
    ///
    /// Copy all bytes, unchanged, to storage satisfying [`Self::copy_layout`].
    /// Do not interpret or patch pointer fields. Every copied image becomes
    /// invalid when this owner is dropped; all kernels using a copy must finish
    /// first. The current `as_bytes` API does not track the copied image through
    /// CUDA or kernel completion, so keep this owner and every copied image alive
    /// until all dependent kernels finish.
    pub fn as_bytes(&self) -> &[u8] {
        let raw = self
            .raw
            .as_ref()
            .expect("live DeviceCommunicator must contain its host image");
        unsafe {
            std::slice::from_raw_parts(ptr::from_ref(raw.as_ref()).cast::<u8>(), Self::BYTE_LEN)
        }
    }

    /// Destroy the NCCL-side device-communicator resources and report errors.
    ///
    /// All kernels using a device-resident copy of [`Self::as_bytes`] must have
    /// completed before this call. The host image is removed before entering
    /// NCCL, so `Drop` cannot issue a second destruction even when NCCL reports
    /// an error. Owners that do not call this method retain the best-effort
    /// `Drop` fallback.
    pub fn destroy(mut self) -> Result<()> {
        self.destroy_once()
    }

    fn destroy_once(&mut self) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        check(unsafe { sys::ncclDevCommDestroy(self.communicator.raw(), raw.as_ref()) })
    }
}

impl Drop for DeviceCommunicator<'_> {
    fn drop(&mut self) {
        let _ = self.destroy_once();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_device_requirements_initializer() {
        let requirements = DeviceCommRequirements::default();
        let raw = &requirements.raw;

        assert_eq!(raw.size, size_of::<sys::ncclDevCommRequirements_t>());
        assert_eq!(raw.magic, sys::NCCL_API_MAGIC);
        assert_eq!(raw.version, sys::NCCL_VERSION_CODE);
        assert!(raw.resourceRequirementsList.is_null());
        assert!(raw.teamRequirementsList.is_null());
        assert!(!raw.lsaMultimem);
        assert_eq!(raw.barrierCount, 0);
        assert_eq!(raw.lsaBarrierCount, 0);
        assert_eq!(raw.railGinBarrierCount, 0);
        assert_eq!(raw.lsaLLA2ABlockCount, 0);
        assert_eq!(raw.lsaLLA2ASlotCount, 0);
        assert!(!raw.ginForceEnable);
        assert_eq!(raw.ginContextCount, 4);
        assert_eq!(raw.ginSignalCount, 0);
        assert_eq!(raw.ginCounterCount, 0);
        assert_eq!(raw.ginConnectionType, sys::NCCL_GIN_CONNECTION_NONE);
        assert!(!raw.ginExclusiveContexts);
        assert_eq!(raw.ginQueueDepth, 0);
        assert_eq!(raw.ginTrafficClass, sys::NCCL_CONFIG_UNDEF_INT);
        assert_eq!(raw.worldGinBarrierCount, 0);
        assert!(raw.ginStrongSignalsRequired);
        assert!(raw.ginVaSignalsRequired);
        assert_eq!(raw.ginCustomStride, 1);
        assert_eq!(raw.ginType, sys::NCCL_GIN_TYPE_NONE);
        assert!(!raw.useRuntimeVersion);
        assert_eq!(raw.cftCaps, sys::NCCL_CFT_NONE as i32);
        assert_eq!(raw.cftBarrierCount, 0);
    }

    #[test]
    fn device_copy_contract_uses_the_public_abi_layout() {
        assert_eq!(
            DeviceCommunicator::BYTE_LEN,
            size_of::<sys::ncclDevComm_t>()
        );
        assert_eq!(
            DeviceCommunicator::COPY_ALIGNMENT,
            align_of::<sys::ncclDevComm_t>()
        );
        assert_eq!(
            DeviceCommunicator::copy_layout().size(),
            DeviceCommunicator::BYTE_LEN
        );
    }

    #[test]
    fn requirements_builders_update_only_requested_fields() {
        let requirements = DeviceCommRequirements::default()
            .lsa_multimem(true)
            .barrier_count(3)
            .gin_context_count(8)
            .gin_connection_type(GinConnectionType::CustomStride(2))
            .gin_type(GinType::Gdaki)
            .gin_strong_signals_required(false);
        assert!(requirements.raw.lsaMultimem);
        assert_eq!(requirements.raw.barrierCount, 3);
        assert_eq!(requirements.raw.ginContextCount, 8);
        assert_eq!(
            requirements.raw.ginConnectionType,
            sys::NCCL_GIN_CONNECTION_CUSTOM_STRIDE
        );
        assert_eq!(requirements.raw.ginCustomStride, 2);
        assert_eq!(requirements.raw.ginType, sys::NCCL_GIN_TYPE_GDAKI);
        assert!(!requirements.raw.ginStrongSignalsRequired);
        assert!(requirements.raw.ginVaSignalsRequired);
        assert!(!requirements.raw.useRuntimeVersion);
    }

    #[test]
    fn device_resource_counts_must_be_nonnegative() {
        macro_rules! assert_rejected {
            ($field:ident) => {{
                let mut requirements = DeviceCommRequirements::default();
                requirements.raw.$field = -1;
                let error = requirements.validate_team_sizes(8, 4, 2).unwrap_err();
                assert!(
                    error.message().contains("nonnegative"),
                    "{} produced unexpected error: {error}",
                    stringify!($field)
                );
            }};
        }

        assert_rejected!(barrierCount);
        assert_rejected!(lsaBarrierCount);
        assert_rejected!(railGinBarrierCount);
        assert_rejected!(lsaLLA2ABlockCount);
        assert_rejected!(lsaLLA2ASlotCount);
        assert_rejected!(ginContextCount);
        assert_rejected!(ginSignalCount);
        assert_rejected!(ginCounterCount);
        assert_rejected!(ginQueueDepth);
        assert_rejected!(worldGinBarrierCount);
        assert_rejected!(cftBarrierCount);
    }

    #[test]
    fn device_resource_arithmetic_is_checked_before_nccl() {
        let lsa_barriers = DeviceCommRequirements::default().lsa_barrier_count(i32::MAX);
        assert!(
            lsa_barriers
                .validate_team_sizes(8, 4, 2)
                .unwrap_err()
                .message()
                .contains("signed resource limit")
        );

        let mut gin_shadows = DeviceCommRequirements::default().gin_context_count(2);
        gin_shadows.raw.ginSignalCount = i32::MAX;
        assert!(
            gin_shadows
                .validate_team_sizes(8, 4, 2)
                .unwrap_err()
                .message()
                .contains("signed resource limit")
        );

        let lla2a = DeviceCommRequirements::default().lsa_lla2a(i32::MAX, 1);
        assert!(
            lla2a
                .validate_team_sizes(8, 4, 2)
                .unwrap_err()
                .message()
                .contains("signed resource limit")
        );

        // GIN can already be active because of an earlier device communicator,
        // so this must be checked even when this request selects `None`.
        let contexts = DeviceCommRequirements::default().gin_context_count(i32::MAX);
        assert!(
            contexts
                .validate_team_sizes(8, 4, 2)
                .unwrap_err()
                .message()
                .contains("signed resource limit")
        );

        let mut context_counters = DeviceCommRequirements::default()
            .gin_connection_type(GinConnectionType::Full)
            .gin_context_count(i32::MAX / 4);
        context_counters.raw.ginCounterCount = 8;
        assert!(
            context_counters
                .validate_team_sizes(1, 1, 1)
                .unwrap_err()
                .message()
                .contains("signed resource limit")
        );
    }

    #[test]
    fn custom_gin_stride_must_be_positive() {
        for stride in [i32::MIN, -1, 0] {
            let requirements = DeviceCommRequirements::default()
                .gin_connection_type(GinConnectionType::CustomStride(stride));
            let error = requirements.validate_team_sizes(8, 4, 2).unwrap_err();
            assert!(
                error.message().contains("custom stride must be positive"),
                "stride {stride} produced unexpected error: {error}"
            );
        }

        DeviceCommRequirements::default()
            .gin_connection_type(GinConnectionType::CustomStride(2))
            .validate_team_sizes(8, 4, 2)
            .unwrap();

        let too_large = DeviceCommRequirements::default()
            .gin_connection_type(GinConnectionType::CustomStride(5))
            .validate_team_sizes(8, 4, 2)
            .unwrap_err();
        assert!(too_large.message().contains("exceeds the LSA size"));
    }

    #[test]
    fn runtime_smoke_device_requirements_pass_resource_validation() {
        DeviceCommRequirements::default()
            .lsa_barrier_count(2)
            .validate_team_sizes(2, 2, 1)
            .unwrap();
    }

    #[test]
    fn explicit_destroy_is_consuming_and_fallible() {
        let _destroy: fn(DeviceCommunicator<'static>) -> Result<()> = DeviceCommunicator::destroy;
    }
}
