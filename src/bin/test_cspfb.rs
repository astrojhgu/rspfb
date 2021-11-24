extern crate rspfb;
use ndarray::{
    Array2,
    ArrayView1,
    Axis, //    , s
};
use num_complex::Complex;
use num_traits::FloatConst;
use rspfb::{
    cspfb::Analyzer,
    oscillator::COscillator,
    windowed_fir::coeff, //, window_funcs::{
                         //blackman_window
                         //}
};

use ndarray_npy::NpzWriter;

use itertools_num::linspace;

//use gnuplot::Figure;

fn main() -> std::io::Result<()> {
    let nch = 32;
    let c = coeff::<f64>(nch, 32, 0.4);
    let nphi = 8192;

    let mut specs = Array2::zeros((nch, nphi));

    for (phi, mut s1) in linspace(-f64::PI(), f64::PI(), nphi).zip(specs.axis_iter_mut(Axis(1))) {
        let mut ana = Analyzer::<Complex<f64>, f64>::new(nch, ArrayView1::from(&c));
        let mut osc = COscillator::new(0.0, phi);

        let mut signal = vec![Complex::<f64>::default(); 1024];
        signal.iter_mut().for_each(|x| *x = osc.get());

        let _ = ana.analyze(&signal);
        signal.iter_mut().for_each(|x| *x = osc.get());
        let channelized = ana.analyze(&signal);
        let spec = channelized.map(|x| x.norm_sqr()).sum_axis(Axis(1));
        s1.assign(&spec);
    }

    let mut npz = NpzWriter::new(std::fs::File::create("out.npz").unwrap());
    let _ = npz.add_array("channelized", &specs).unwrap();
    Ok(())
}
