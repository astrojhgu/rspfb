extern crate rspfb;
use ndarray::{Array1, Array2, ArrayView1, Axis, concatenate};
use num_complex::Complex;
use num_traits::FloatConst;
use rspfb::{csp_pfb::CspPfb, cspfb, oscillator::COscillator, ospfb, windowed_fir::coeff};

use ndarray_npy::NpzWriter;

use itertools_num::linspace;

fn main() {
    let nch = 32;
    let c = coeff::<f64>(nch / 2, 16, 0.51);
    
    let phi=f64::PI()/nch as f64*10.0+0.0001;
    let mut osc = COscillator::new(0.0, phi);

    let mut ana = ospfb::Analyzer::<Complex<f64>, f64>::new(nch, ArrayView1::from(&c));
    let mut signal = vec![Complex::<f64>::default(); 65536];
    
    signal.iter_mut().for_each(|x| *x = osc.get());
    let x1=ana.analyze(&signal);

    signal.iter_mut().for_each(|x| *x = osc.get());
    let x2=ana.analyze(&signal);

    signal.iter_mut().for_each(|x| *x = osc.get());
    let x3=ana.analyze(&signal);
    let x=concatenate!(Axis(1), x2, x3);

    let mut npz = NpzWriter::new(std::fs::File::create("out.npz").unwrap());
    npz.add_array("x", &x).unwrap();
}
