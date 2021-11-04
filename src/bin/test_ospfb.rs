extern crate rspfb;
use ndarray::{Array2, ArrayView1, Axis};
use num_complex::Complex;
use num_traits::FloatConst;
use rspfb::{csp_pfb::CspPfb, cspfb, oscillator::COscillator, ospfb, windowed_fir::coeff};

use ndarray_npy::NpzWriter;

use itertools_num::linspace;

fn main() {
    let nch = 512;
    let c = coeff::<f64>(nch / 2, 12, 0.49);
    
    let nphi = 2048;
    let nfine = 16;
    let cfine = coeff::<f64>(nfine*2, 12, 0.5);
    let selected_ch: Vec<_> = (0..32).collect();

    let mut specs = Array2::<f64>::zeros((selected_ch.len() * nfine, nphi));

    for (cnt, (phi, mut s1)) in linspace(0.0, f64::PI()/512.0*32.0, nphi).zip(specs.axis_iter_mut(Axis(1))).enumerate() {
        let mut ana = ospfb::Analyzer::<Complex<f64>, f64>::new(nch, ArrayView1::from(&c));
        let ana_fine =
            cspfb::Analyzer::<Complex<f64>, f64>::new(nfine * 2, ArrayView1::from(&cfine));

        let mut csp = CspPfb::new(&selected_ch, &ana_fine);

        let mut osc = COscillator::new(0.0, phi);

        let mut signal = vec![Complex::<f64>::default(); 65536];
        signal.iter_mut().for_each(|x| *x = osc.get());
        let mut channelized = ana.analyze_par(&signal);
        let x = csp.analyze_par(channelized.view_mut());

        signal.iter_mut().for_each(|x| *x = osc.get());
        let mut channelized = ana.analyze_par(&signal);
        let x = csp.analyze_par(channelized.view_mut());

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                s1[i] += x[(i, j)].norm_sqr();
            }
        }
    }

    let mut npz = NpzWriter::new(std::fs::File::create("out.npz").unwrap());
    npz.add_array("channelized", &specs).unwrap();
}
