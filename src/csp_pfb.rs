use crate::{cspfb, oscillator::HalfChShifter, utils::fftshift2};

use rustfft::FftNum;

//type FFTa=Fft1Dn;
//type FFTs=Fft1D;

use ndarray::{parallel::prelude::*, s, Array2, ArrayViewMut2, Axis, ScalarOperand};
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

pub struct CspPfb<T> {
    pfb: Vec<cspfb::Analyzer<Complex<T>, T>>,
    coarse_ch_selected: Vec<usize>,
    shifter: HalfChShifter<T>,
}

impl<T> CspPfb<T>
where
    T: Copy
        + Float
        + FloatConst
        + std::ops::MulAssign<T>
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sync
        + Send
        + FftNum,
    Complex<T>: Copy
        + Add<Complex<T>, Output = Complex<T>>
        + Mul<T, Output = Complex<T>>
        + Mul<Complex<T>, Output = Complex<T>>
        + Sum
        + Default
        + ScalarOperand
        + Sync,
{
    pub fn new(coarse_ch_selected: &[usize], pfb1: &cspfb::Analyzer<Complex<T>, T>) -> CspPfb<T> {
        let coarse_ch_selected = Vec::from(coarse_ch_selected);
        let pfb: Vec<_> = coarse_ch_selected.iter().map(|_| pfb1.clone()).collect();
        let shifter = HalfChShifter::<T>::new(pfb1.nch(), false);
        CspPfb {
            pfb,
            coarse_ch_selected,
            shifter,
        }
    }

    pub fn analyze(&mut self, mut x: ArrayViewMut2<Complex<T>>) -> Array2<Complex<T>> {
        x.axis_iter_mut(Axis(1))
            .enumerate()
            .for_each(|(_i, mut c)| {
                let f = self.shifter.get();
                c.iter_mut().for_each(|c1| {
                    *c1 *= f;
                })
            });
        let nch_fine = self.pfb[0].nch();
        let data_len = x.ncols();
        let nch_output = self.coarse_ch_selected.len() * nch_fine / 2;
        let mut result = unsafe { Array2::uninit((nch_output, data_len / nch_fine)).assume_init() };
        for (i, (&c, pfb)) in self
            .coarse_ch_selected
            .iter()
            .zip(self.pfb.iter_mut())
            .enumerate()
        {
            let y = pfb.analyze(x.slice(s![c, ..]).as_slice().unwrap());
            let y = fftshift2(y.view());
            result
                .slice_mut(s![i * nch_fine / 2..(i + 1) * nch_fine / 2, ..])
                .assign(&y.slice(s![nch_fine / 4..nch_fine / 4 * 3, ..]));
        }
        result
    }

    pub fn analyze_par(&mut self, mut x: ArrayViewMut2<Complex<T>>) -> Array2<Complex<T>> {
        x.axis_iter_mut(Axis(1))
            .enumerate()
            .for_each(|(_i, mut c)| {
                let f = self.shifter.get();
                c.iter_mut().for_each(|c1| {
                    *c1 *= f;
                })
            });
        let nch_fine = self.pfb[0].nch();
        let data_len = x.ncols();
        let nch_output = self.coarse_ch_selected.len() * nch_fine / 2;
        let mut result = unsafe { Array2::uninit((nch_output, data_len / nch_fine)).assume_init() };

        //self.coarse_ch_selected.iter().into_par_iter();
        let _ = result
            .axis_chunks_iter_mut(Axis(0), nch_fine / 2)
            .into_par_iter()
            .zip(self.pfb.par_iter_mut())
            .zip(x.axis_iter(Axis(0)))
            .for_each(|((mut r, pfb), x1)| {
                let y = pfb.analyze(x1.as_slice().unwrap());
                let y = fftshift2(y.view());
                r.assign(&y.slice(s![nch_fine / 4..nch_fine / 4 * 3, ..]));
            });
        result
    }
}
