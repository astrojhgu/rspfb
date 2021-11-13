//! This module contains a set of cfg structs, from which actual working structs are constructed
//! It is convient to generate working structs from cfg files on disks

use serde::{Deserialize, Serialize};

/// cfg to generate pfb
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct PfbCfg {
    /// number of total number of channels
    pub nch: usize,
    /// tap per channel
    pub tap_per_ch: usize,

    // pass band width factor 1.1 should be a common value
    pub k: f64,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct TwoStageCfg {
    pub coarse_cfg: PfbCfg,
    pub fine_cfg: PfbCfg,
    pub selected_coarse_ch: Vec<usize>,
}

/// cfg to generate FracDelayer
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub struct DelayerCfg {
    /// max value of possible delay
    pub max_delay: usize,
    /// 2*half_tap+1 is the filter tap
    pub half_tap: usize,
}
