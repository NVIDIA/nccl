// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error, Result};
use crate::sys;
use std::ffi::CString;
use std::mem::size_of;
use std::ptr;

/// NCCL's communicator CTA scheduling policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(i32)]
pub enum CtaPolicy {
    Default = sys::NCCL_CTA_POLICY_DEFAULT as i32,
    Efficiency = sys::NCCL_CTA_POLICY_EFFICIENCY as i32,
    Zero = sys::NCCL_CTA_POLICY_ZERO as i32,
}

/// NCCL's CUDA graph usage policy for a communicator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(i32)]
pub enum GraphUsageMode {
    /// Do not use CUDA graphs with the communicator.
    NoGraphs = 0,
    /// Use the communicator with one CUDA graph.
    SingleGraph = 1,
    /// Permit multiple graphs or a mix of graph and non-graph launches.
    MultipleOrMixed = 2,
}

/// An initialized `ncclConfig_t` with owned storage for string fields.
///
/// `Default` exactly mirrors `NCCL_CONFIG_INITIALIZER` from the selected NCCL
/// headers. Builder methods only change the documented user fields.
pub struct Config {
    pub(crate) raw: sys::ncclConfig_t,
    net_name: Option<CString>,
    comm_name: Option<CString>,
}

impl Default for Config {
    fn default() -> Self {
        let undefined = sys::NCCL_CONFIG_UNDEF_INT;
        Self {
            raw: sys::ncclConfig_t {
                size: size_of::<sys::ncclConfig_t>(),
                magic: sys::NCCL_API_MAGIC,
                version: sys::NCCL_VERSION_CODE,
                blocking: undefined,
                cgaClusterSize: undefined,
                minCTAs: undefined,
                maxCTAs: undefined,
                netName: ptr::null(),
                splitShare: undefined,
                trafficClass: undefined,
                commName: ptr::null(),
                collnetEnable: undefined,
                CTAPolicy: undefined,
                shrinkShare: undefined,
                nvlsCTAs: undefined,
                nChannelsPerNetPeer: undefined,
                nvlinkCentricSched: undefined,
                graphUsageMode: undefined,
                numRmaCtx: undefined,
                maxP2pPeers: undefined,
                graphStreamOrdering: undefined,
                launchOrderImplicit: undefined,
                numRmaSig: undefined,
                rmaEagerInit: undefined,
            },
            net_name: None,
            comm_name: None,
        }
    }
}

impl Clone for Config {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            raw: self.raw,
            net_name: self.net_name.clone(),
            comm_name: self.comm_name.clone(),
        };
        cloned.refresh_string_pointers();
        cloned
    }
}

impl std::fmt::Debug for Config {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Config")
            .field("blocking", &self.raw.blocking)
            .field("cga_cluster_size", &self.raw.cgaClusterSize)
            .field("min_ctas", &self.raw.minCTAs)
            .field("max_ctas", &self.raw.maxCTAs)
            .field("net_name", &self.net_name)
            .field("split_share", &self.raw.splitShare)
            .field("traffic_class", &self.raw.trafficClass)
            .field("comm_name", &self.comm_name)
            .field("collnet_enable", &self.raw.collnetEnable)
            .field("cta_policy", &self.raw.CTAPolicy)
            .field("shrink_share", &self.raw.shrinkShare)
            .field("nvls_ctas", &self.raw.nvlsCTAs)
            .field("channels_per_net_peer", &self.raw.nChannelsPerNetPeer)
            .field("nvlink_centric_sched", &self.raw.nvlinkCentricSched)
            .field("graph_usage_mode", &self.raw.graphUsageMode)
            .field("rma_contexts", &self.raw.numRmaCtx)
            .field("max_p2p_peers", &self.raw.maxP2pPeers)
            .field("graph_stream_ordering", &self.raw.graphStreamOrdering)
            .field("launch_order_implicit", &self.raw.launchOrderImplicit)
            .field("rma_signals", &self.raw.numRmaSig)
            .field("rma_eager_init", &self.raw.rmaEagerInit)
            .finish()
    }
}

impl Config {
    fn refresh_string_pointers(&mut self) {
        self.raw.netName = self
            .net_name
            .as_ref()
            .map_or(ptr::null(), |name| name.as_ptr());
        self.raw.commName = self
            .comm_name
            .as_ref()
            .map_or(ptr::null(), |name| name.as_ptr());
    }

    pub fn blocking(mut self, enabled: bool) -> Self {
        self.raw.blocking = i32::from(enabled);
        self
    }

    pub fn cga_cluster_size(mut self, size: i32) -> Self {
        self.raw.cgaClusterSize = size;
        self
    }

    pub fn min_ctas(mut self, count: i32) -> Self {
        self.raw.minCTAs = count;
        self
    }

    pub fn max_ctas(mut self, count: i32) -> Self {
        self.raw.maxCTAs = count;
        self
    }

    /// Set the network plugin name from an already validated C string.
    pub fn net_name(mut self, name: CString) -> Self {
        self.net_name = Some(name);
        self.refresh_string_pointers();
        self
    }

    /// Validate and set the network plugin name.
    pub fn try_net_name(self, name: impl Into<Vec<u8>>) -> Result<Self> {
        let name = CString::new(name)
            .map_err(|_| Error::local("NCCL network name contains an interior NUL byte"))?;
        Ok(self.net_name(name))
    }

    pub fn split_share(mut self, enabled: bool) -> Self {
        self.raw.splitShare = i32::from(enabled);
        self
    }

    pub fn traffic_class(mut self, traffic_class: i32) -> Self {
        self.raw.trafficClass = traffic_class;
        self
    }

    /// Set the communicator name from an already validated C string.
    pub fn comm_name(mut self, name: CString) -> Self {
        self.comm_name = Some(name);
        self.refresh_string_pointers();
        self
    }

    /// Validate and set the communicator name.
    pub fn try_comm_name(self, name: impl Into<Vec<u8>>) -> Result<Self> {
        let name = CString::new(name)
            .map_err(|_| Error::local("NCCL communicator name contains an interior NUL byte"))?;
        Ok(self.comm_name(name))
    }

    pub fn collnet_enable(mut self, enabled: bool) -> Self {
        self.raw.collnetEnable = i32::from(enabled);
        self
    }

    pub fn cta_policy(mut self, policy: CtaPolicy) -> Self {
        self.raw.CTAPolicy = policy as i32;
        self
    }

    pub fn shrink_share(mut self, enabled: bool) -> Self {
        self.raw.shrinkShare = i32::from(enabled);
        self
    }

    pub fn nvls_ctas(mut self, count: i32) -> Self {
        self.raw.nvlsCTAs = count;
        self
    }

    pub fn channels_per_net_peer(mut self, count: i32) -> Self {
        self.raw.nChannelsPerNetPeer = count;
        self
    }

    pub fn nvlink_centric_sched(mut self, enabled: bool) -> Self {
        self.raw.nvlinkCentricSched = i32::from(enabled);
        self
    }

    pub fn graph_usage_mode(mut self, mode: GraphUsageMode) -> Self {
        self.raw.graphUsageMode = mode as i32;
        self
    }

    pub fn rma_contexts(mut self, count: i32) -> Self {
        self.raw.numRmaCtx = count;
        self
    }

    pub fn max_p2p_peers(mut self, count: i32) -> Self {
        self.raw.maxP2pPeers = count;
        self
    }

    pub fn graph_stream_ordering(mut self, enabled: bool) -> Self {
        self.raw.graphStreamOrdering = i32::from(enabled);
        self
    }

    pub fn launch_order_implicit(mut self, enabled: bool) -> Self {
        self.raw.launchOrderImplicit = i32::from(enabled);
        self
    }

    pub fn rma_signals(mut self, count: i32) -> Self {
        self.raw.numRmaSig = count;
        self
    }

    pub fn rma_eager_init(mut self, enabled: bool) -> Self {
        self.raw.rmaEagerInit = i32::from(enabled);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_nccl_config_initializer() {
        let config = Config::default();
        let raw = &config.raw;
        let undefined = sys::NCCL_CONFIG_UNDEF_INT;

        assert_eq!(raw.size, size_of::<sys::ncclConfig_t>());
        assert_eq!(raw.magic, sys::NCCL_API_MAGIC);
        assert_eq!(raw.version, sys::NCCL_VERSION_CODE);
        assert_eq!(raw.blocking, undefined);
        assert_eq!(raw.cgaClusterSize, undefined);
        assert_eq!(raw.minCTAs, undefined);
        assert_eq!(raw.maxCTAs, undefined);
        assert!(raw.netName.is_null());
        assert_eq!(raw.splitShare, undefined);
        assert_eq!(raw.trafficClass, undefined);
        assert!(raw.commName.is_null());
        assert_eq!(raw.collnetEnable, undefined);
        assert_eq!(raw.CTAPolicy, undefined);
        assert_eq!(raw.shrinkShare, undefined);
        assert_eq!(raw.nvlsCTAs, undefined);
        assert_eq!(raw.nChannelsPerNetPeer, undefined);
        assert_eq!(raw.nvlinkCentricSched, undefined);
        assert_eq!(raw.graphUsageMode, undefined);
        assert_eq!(raw.numRmaCtx, undefined);
        assert_eq!(raw.maxP2pPeers, undefined);
        assert_eq!(raw.graphStreamOrdering, undefined);
        assert_eq!(raw.launchOrderImplicit, undefined);
        assert_eq!(raw.numRmaSig, undefined);
        assert_eq!(raw.rmaEagerInit, undefined);
    }

    #[test]
    fn builder_and_clone_keep_owned_string_pointers_valid() {
        let config = Config::default()
            .blocking(false)
            .min_ctas(2)
            .max_ctas(8)
            .rma_signals(3)
            .rma_eager_init(true)
            .try_net_name("IB")
            .unwrap()
            .try_comm_name("pipeline")
            .unwrap();
        let clone = config.clone();

        assert_eq!(config.raw.blocking, 0);
        assert_eq!(config.raw.minCTAs, 2);
        assert_eq!(config.raw.maxCTAs, 8);
        assert_eq!(config.raw.numRmaSig, 3);
        assert_eq!(config.raw.rmaEagerInit, 1);
        assert!(!config.raw.netName.is_null());
        assert!(!clone.raw.netName.is_null());
        assert_ne!(config.raw.netName, clone.raw.netName);
        assert_ne!(config.raw.commName, clone.raw.commName);
    }

    #[test]
    fn string_builders_reject_interior_nul() {
        let error = Config::default().try_comm_name(b"bad\0name".to_vec());
        assert!(error.is_err());
    }

    #[test]
    fn graph_usage_builder_preserves_all_supported_modes() {
        for (mode, expected) in [
            (GraphUsageMode::NoGraphs, 0),
            (GraphUsageMode::SingleGraph, 1),
            (GraphUsageMode::MultipleOrMixed, 2),
        ] {
            assert_eq!(
                Config::default().graph_usage_mode(mode).raw.graphUsageMode,
                expected
            );
        }
    }
}
