//! Central Signal Processor PFB
//! Implemented with critical sampling pfb

use crate::{cspfb, oscillator::HalfChShifter, utils::fftshift2};
use ndarray::{parallel::prelude::*, s, Array2, ArrayView2, ArrayViewMut2, Axis, ScalarOperand};
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use rustfft::FftNum;
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

/// Struct for central processor pfb
pub struct CspPfb<T> {
    /// an array of cspfbs, each one for one coarse channel that is selected
    pfb: Vec<cspfb::Analyzer<Complex<T>, T>>,
    /// coarse channels selected
    coarse_ch_selected: Vec<usize>,
    /// shift frequency in each coarse channel by half of the width of a fine channel
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
    /// constructor of CsPfb
    /// coarse_ch_selected
    /// * `coarse_ch_selected` - slected coarse channels
    /// * `pfb1` - a template, which is cloned for each selected coarse channel
    /// * return value - a constructed CsPfb
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

    /// Further channelize the input coarse channels to finer channels
    /// * `x` - input coarse channels, 2D array view, with `number of coarse channels` rows.
    /// * return value - a 2D array with fine channel data, with `number of fine channels` rows.
    pub fn analyze(&mut self, x: ArrayView2<Complex<T>>) -> Array2<Complex<T>> {
        let mut x = x.select(Axis(0), &self.coarse_ch_selected);
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
            .zip(self.pfb.iter_mut())
            .zip(x.axis_iter(Axis(0)))
            .for_each(|((mut r, pfb), x1)| {
                let y = pfb.analyze(x1.as_slice().unwrap());
                let y = fftshift2(y.view());
                r.assign(&y.slice(s![nch_fine / 4..nch_fine / 4 * 3, ..]));
            });
        //println!("{:?} {}", result.shape(), nch_fine);
        result
    }

    /// Parallel version of [`Self::analyze`].
    pub fn analyze_par(&mut self, x: ArrayView2<Complex<T>>) -> Array2<Complex<T>> {
        let mut x = x.select(Axis(0), &self.coarse_ch_selected);
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
            .zip_eq(self.pfb.par_iter_mut())
            .zip_eq(x.axis_iter(Axis(0)).into_par_iter())
            .for_each(|((mut r, pfb), x1)| {
                let y = pfb.analyze_par(x1.as_slice().unwrap());
                let y = fftshift2(y.view());
                r.assign(&y.slice(s![nch_fine / 4..nch_fine / 4 * 3, ..]));
            });
        //println!("{:?} {}", result.shape(), nch_fine);
        result
    }
}
