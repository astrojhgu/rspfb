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
    pub freq_range: [f64;2],//in PI
    pub nfreq: usize,
    pub signal_len: usize,
    pub niter:usize,
    pub selected_coarse_ch: Vec<usize>,
}
