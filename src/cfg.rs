use serde::{Serialize, Deserialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct PfbCfg{
    pub nch:usize, 
    pub tap_per_ch: usize,
    pub k: f64, 
}


#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct TwoStageCfg{
    pub coarse_cfg: PfbCfg,
    pub fine_cfg: PfbCfg,
    pub selected_coarse_ch: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub struct DelayerCfg{
    pub max_delay: usize,
    pub tap: usize, 
}
