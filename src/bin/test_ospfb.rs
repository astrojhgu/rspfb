extern crate rspfb;
use ndarray::{concatenate, ArrayView1, Axis};
use num_complex::Complex;
use num_traits::FloatConst;
use rspfb::{oscillator::COscillator, ospfb::Analyzer, windowed_fir::coeff};

use ndarray_npy::NpzWriter;

fn main() {
    let nch = 32;
    let c = coeff::<f64>(nch / 2, 16, 0.51);

    let phi = f64::PI() / nch as f64 * 0.0 + 0.0001;
    let mut osc = COscillator::new(0.0, phi);

    let mut ana = Analyzer::<Complex<f64>, f64>::new(nch, ArrayView1::from(&c));
    let mut signal = vec![Complex::<f64>::default(); 65536];

    signal.iter_mut().for_each(|x| *x = osc.get());
    let _x1 = ana.analyze(&signal);

    signal.iter_mut().for_each(|x| *x = osc.get());
    let x2 = ana.analyze(&signal);

    signal.iter_mut().for_each(|x| *x = osc.get());
    let x3 = ana.analyze_par(&signal);
    let x = concatenate!(Axis(1), x2, x3);

    let mut npz = NpzWriter::new(std::fs::File::create("out.npz").unwrap());
    npz.add_array("x", &x).unwrap();
}
