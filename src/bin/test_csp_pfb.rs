extern crate rspfb;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use num_complex::Complex;
use num_traits::FloatConst;
use rspfb::{csp_pfb::CspPfb, cspfb, oscillator::COscillator, ospfb, windowed_fir::coeff};

use ndarray_npy::NpzWriter;

use itertools_num::linspace;

fn main() {
    let nch = 512;
    //let c = coeff::<f64>(nch / 2, 16, 1.1);//best
    let c = coeff::<f64>(nch / 2, 12, 1.3);
    let mut npz = NpzWriter::new(std::fs::File::create("out.npz").unwrap());
    npz.add_array("coeff", &c).unwrap();

    let nphi = 8192;
    let nfine = 16;
    let cfine = coeff::<f64>(nfine*2, 4, 2.2);
    let selected_ch: Vec<_> = (0..nch/2).collect();
    //let selected_ch=vec![0,1,2];

    let mut spec_fines = Array2::<f64>::zeros((selected_ch.len() * nfine, nphi));
    let mut spec_coarse = Array2::<f64>::zeros((selected_ch.len(), nphi));

    let freq:Vec<_>=linspace(0.0, f64::PI()-f64::PI()/nphi as f64, nphi).collect();

    for (cnt, (&phi, (mut s1, mut s2))) in freq.iter().zip(spec_fines.axis_iter_mut(Axis(1)).zip(spec_coarse.axis_iter_mut(Axis(1)))).enumerate() {
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
        for i in 0..selected_ch.len(){
            for j in 0..channelized.ncols(){
                s2[i]+=channelized[(selected_ch[i], j)].norm_sqr();
            }            
        }


        let x = csp.analyze_par(channelized.view_mut());
        if cnt%100==0{
            eprintln!("{}", cnt);
        }

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                s1[i] += x[(i, j)].norm_sqr();
            }
        }
    }

    //let mut npz = NpzWriter::new(std::fs::File::create("out.npz").unwrap());
    npz.add_array("fine", &spec_fines).unwrap();
    npz.add_array("coarse", &spec_coarse).unwrap();
    npz.add_array("freq", &Array1::from_vec(freq)).unwrap();
}
