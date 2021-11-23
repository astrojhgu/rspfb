extern crate rspfb;

use std::fs::File;

use rspfb::{
    cfg::{PfbCfg, TwoStageCfg},
    csp_pfb::CspPfb,
    cspfb,
    oscillator::COscillator,
    ospfb,
    windowed_fir::coeff,
};

use num_complex::Complex;

use num_traits::FloatConst;

use serde_yaml::from_reader;

use clap::{App, Arg};

use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView1, Axis};

use itertools_num::linspace;

use ndarray_npy::NpzWriter;

pub fn main() {
    let matches = App::new("ampl_resp_2stages")
        .arg(
            Arg::new("chcfg")
                .short('c')
                .long("cfg")
                .takes_value(true)
                .value_name("config file")
                .required(true),
        )
        .arg(
            Arg::new("fmin")
                .short('f')
                .long("fmin")
                .allow_hyphen_values(true)
                .takes_value(true)
                .value_name("freq")
                .default_value("-1")
                .required(false),
        )
        .arg(
            Arg::new("fmax")
                .short('F')
                .long("fmax")
                .allow_hyphen_values(true)
                .takes_value(true)
                .value_name("freq")
                .default_value("1")
                .required(false),
        )
        .arg(
            Arg::new("nfreq")
                .short('n')
                .long("nfreq")
                .takes_value(true)
                .value_name("nfreq")
                .default_value("1024")
                .required(false),
        )
        .arg(
            Arg::new("siglen")
                .short('l')
                .long("len")
                .takes_value(true)
                .value_name("signal length")
                .default_value("8192")
                .required(false),
        )
        .arg(
            Arg::new("niter")
                .short('t')
                .long("niter")
                .takes_value(true)
                .value_name("niter")
                .default_value("2")
                .required(false),
        )
        .arg(
            Arg::new("outfile")
                .short('o')
                .long("out")
                .takes_value(true)
                .value_name("output name")
                .required(true),
        )
        .get_matches();

    let mut cfg_file = File::open(matches.value_of("chcfg").unwrap()).unwrap();
    let TwoStageCfg {
        coarse_cfg:
            PfbCfg {
                nch: nch_coarse,
                k: k_coarse,
                tap_per_ch: tap_coarse,
            },
        fine_cfg:
            PfbCfg {
                nch: nch_fine,
                k: k_fine,
                tap_per_ch: tap_fine,
            },
        selected_coarse_ch,
    } = from_reader(&mut cfg_file).unwrap();

    let fmin = matches.value_of("fmin").unwrap().parse::<f64>().unwrap();
    let fmax = matches.value_of("fmax").unwrap().parse::<f64>().unwrap();
    let nfreq = matches.value_of("nfreq").unwrap().parse::<usize>().unwrap();
    let signal_len = matches
        .value_of("siglen")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let niter = matches.value_of("niter").unwrap().parse::<usize>().unwrap();

    let coeff_coarse = coeff::<f64>(nch_coarse / 2, tap_coarse, k_coarse);
    let coeff_fine = coeff::<f64>(nch_fine * 2, tap_fine, k_fine);
    let bandwidth = (fmax - fmin) * f64::PI();
    let df = bandwidth / (nfreq + 1) as f64;
    let freqs =
        Array1::from(linspace(f64::PI() * fmin, f64::PI() * fmax - df, nfreq).collect::<Vec<_>>());
    let mut coarse_spec = Array2::<f64>::zeros((nfreq, selected_coarse_ch.len()));
    let mut fine_spec = Array2::<f64>::zeros((nfreq, selected_coarse_ch.len() * nch_fine));
    println!("{:?}", freqs);

    fine_spec
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip_eq(coarse_spec.axis_iter_mut(Axis(0)).into_par_iter())
        .zip_eq(freqs.axis_iter(Axis(0)).into_par_iter())
        .for_each(|((mut fine_resp, mut coarse_resp), freq)| {
            let freq = freq[()];
            let mut coarse_pfb = ospfb::Analyzer::<Complex<f64>, f64>::new(
                nch_coarse,
                ArrayView1::from(&coeff_coarse),
            );
            let fine_pfb = cspfb::Analyzer::<Complex<f64>, f64>::new(
                nch_fine * 2,
                ArrayView1::from(&coeff_fine),
            );

            let mut csp = CspPfb::new(&selected_coarse_ch, &fine_pfb);
            let mut osc = COscillator::new(0.0, freq);
            for _i in 0..niter - 1 {
                let mut signal = vec![Complex::<f64>::default(); signal_len];
                signal.iter_mut().for_each(|x| *x = osc.get());
                let coarse_data = coarse_pfb.analyze_par(&signal);
                let _ = csp.analyze_par(coarse_data.view());
            }

            let mut signal = vec![Complex::<f64>::default(); signal_len];
            signal.iter_mut().for_each(|x| *x = osc.get());
            let mut coarse_data = coarse_pfb.analyze_par(&signal);
            let coarse_spec = coarse_data.map(|x| x.norm_sqr()).sum_axis(Axis(1));

            for (i, &c) in selected_coarse_ch.iter().enumerate() {
                coarse_resp[i] = coarse_spec[c];
            }

            let fine_data = csp.analyze_par(coarse_data.view());

            let fine_spec = fine_data.map(|x| x.norm_sqr()).sum_axis(Axis(1));
            fine_resp.assign(&fine_spec.view());
        });

    let outfile = std::fs::File::create(matches.value_of("outfile").unwrap()).unwrap();
    let mut npz = NpzWriter::new(outfile);
    let _ = npz.add_array("freq", &freqs).unwrap();
    let _ = npz.add_array("coarse", &coarse_spec).unwrap();
    let _ = npz.add_array("fine", &fine_spec).unwrap();
    let _ = npz.add_array(
        "coarse_ch",
        &ArrayView1::from(&selected_coarse_ch).map(|&x| x as i32),
    );
}
