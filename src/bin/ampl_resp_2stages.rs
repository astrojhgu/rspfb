extern crate rspfb;

use std::{
    fs::{
        File
    }
};

use rspfb::{
    cfg::{
        PfbCfg
        , TwoStageCfg
    }
    , ospfb
    , cspfb
    , csp_pfb::{
        CspPfb
    }
    , windowed_fir::{
        coeff
    }
    , oscillator::{
        COscillator
    }
};

use num_complex::{
    Complex
};

use num_traits::{
    FloatConst
};

use serde_yaml::{
    from_reader
};


use clap::{
    App,
    Arg
};

use ndarray::{
    Array1
    , Array2
    ,ArrayView1
    , Axis
    , parallel::prelude::*
};

use itertools_num::{
    linspace
};

use ndarray_npy::{
    NpzWriter
};

pub fn main(){
    let matches=App::new("ampl_resp_2stages")
    .arg(Arg::new("chcfg")
        .short('c')
        .long("cfg")
        .takes_value(true)
        .value_name("config file")
        .required(true)
    )
    .arg(
        Arg::new("outfile")
        .short('o')
        .long("out")
        .takes_value(true)
        .value_name("output name")
        .required(true)
    ).get_matches();

    let mut cfg_file=File::open(matches.value_of("chcfg").unwrap()).unwrap();
    let TwoStageCfg{
        coarse_cfg:PfbCfg{
            nch: nch_coarse,
            k: k_coarse,
            tap_per_ch: tap_coarse
        }
        , fine_cfg:PfbCfg{
            nch: nch_fine, 
            k: k_fine, 
            tap_per_ch: tap_fine
        }
        , freq_range
        , nfreq
        , signal_len
        , niter
        , selected_coarse_ch
    }=from_reader(&mut cfg_file).unwrap();

    let coeff_coarse = coeff::<f64>(nch_coarse / 2, tap_coarse, k_coarse);
    let coeff_fine = coeff::<f64>(nch_fine*2, tap_fine, k_fine);
    let bandwidth=(freq_range[1]-freq_range[0])*f64::PI();
    let df=bandwidth/(nfreq+1) as f64;
    let freqs=Array1::from(linspace(f64::PI()*freq_range[0], f64::PI()*freq_range[1]-df, nfreq).collect::<Vec<_>>());
    let mut coarse_spec=Array2::<f64>::zeros((nfreq, selected_coarse_ch.len()));
    let mut fine_spec=Array2::<f64>::zeros((nfreq, selected_coarse_ch.len()*nch_fine));
    println!("{:?}", freqs);


    fine_spec.axis_iter_mut(Axis(0)).into_par_iter().zip_eq(coarse_spec.axis_iter_mut(Axis(0)).into_par_iter())
    .zip_eq(freqs.axis_iter(Axis(0)).into_par_iter())
    .for_each(|((mut fine_resp, mut coarse_resp), freq)|{
        let freq=freq[()];
        let mut coarse_pfb = ospfb::Analyzer::<Complex<f64>, f64>::new(nch_coarse, ArrayView1::from(&coeff_coarse));
        let fine_pfb=cspfb::Analyzer::<Complex<f64>, f64>::new(nch_fine*2, ArrayView1::from(&coeff_fine));
        
        let mut csp = CspPfb::new(&selected_coarse_ch, &fine_pfb);
        let mut osc = COscillator::new(0.0, freq);
        for _i in 0..niter-1{
            let mut signal = vec![Complex::<f64>::default(); signal_len];
            signal.iter_mut().for_each(|x| *x = osc.get());
            let mut coarse_data = coarse_pfb.analyze_par(&signal);
            let _ = csp.analyze_par(coarse_data.view());
        }

        let mut signal = vec![Complex::<f64>::default(); signal_len];
        signal.iter_mut().for_each(|x| *x = osc.get());
        let mut coarse_data = coarse_pfb.analyze_par(&signal);
        let coarse_spec=coarse_data.map(|x| x.norm_sqr()).sum_axis(Axis(1));

        for (i, &c) in selected_coarse_ch.iter().enumerate(){
            coarse_resp[i]=coarse_spec[c];
        }

        let fine_data = csp.analyze_par(coarse_data.view());

        let fine_spec=fine_data.map(|x| x.norm_sqr()).sum_axis(Axis(1));
        fine_resp.assign(&fine_spec.view());
    });

    let outfile=std::fs::File::create(matches.value_of("outfile").unwrap()).unwrap();
    let mut npz=NpzWriter::new(outfile);
    let _=npz.add_array("freq", &freqs).unwrap();
    let _=npz.add_array("coarse", &coarse_spec).unwrap();
    let _=npz.add_array("fine", &fine_spec).unwrap();
    let _=npz.add_array("coarse_ch", &ArrayView1::from(&selected_coarse_ch).map(|&x| x as i32));
}
