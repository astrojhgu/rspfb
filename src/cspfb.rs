//! Containing critical sampling poly phase filter bank

#![allow(clippy::uninit_vec)]

use crate::{batch_fir::BatchFilter, utils::transpose_par_map};
use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, Axis, ScalarOperand};
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use rustfft::{FftNum, FftPlanner};
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

/// Analyze channelizer
#[derive(Clone)]
pub struct Analyzer<R, T> {
    /// A vec of filters, one for each branch
    batch_filter:BatchFilter<T>,
    /// a buffer, ensurning that the input signal length need not to be nch*tap. The remaining elements will be stored and be concated with the input next time.
    buffer: Vec<R>,
}

impl<R, T> Analyzer<R, T>
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
    R: Copy
        + Add<R, Output = R>
        + Mul<R, Output = R>
        + std::ops::MulAssign<R>
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sync
        + Send,
    Complex<T>: Copy
        + Add<Complex<T>, Output = Complex<T>>
        + Mul<T, Output = Complex<T>>
        + Mul<R, Output = Complex<T>>
        + Mul<Complex<T>, Output = Complex<T>>
        + std::convert::From<R>
        + Sum
        + Default
        + ScalarOperand
        + Sync,
{
    /// constructor
    /// * `nch` - number of channels including both pos and neg ones
    /// * `coeff` - prototype low-pass filter, the tap of which should be nch times of the tap of each branch.
    /// * return value - `Analyzer`
    pub fn new(nch: usize, coeff: ArrayView1<T>) -> Analyzer<R, T> {
        let tap = coeff.len() / nch;
        assert!(nch * tap == coeff.len());
        let coeff=coeff.into_shape((tap, nch)).unwrap().t().as_standard_layout().to_owned();
        let coeff = coeff.slice(s![..;-1,..]);

        let batch_filter=BatchFilter::new(coeff);

        Analyzer {
            batch_filter,
            buffer: Vec::<R>::new(),
        }
    }

    /// return the number of channels
    /// return value - the number of channels
    pub fn nch(&self) -> usize {
        self.batch_filter.filters.len()
    }

    /// Channelize input signal
    /// * `input_signal` - a 1-d slice containing time domain input signal
    /// * return value - channelized data, with `nch` rows
    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.batch_filter.filters.len();
        let batch = (self.buffer.len() + input_signal.len()) / nch;

        let signal = Array1::from_iter(
            self.buffer
                .iter()
                .chain(input_signal)
                .take(nch * batch)
                .cloned(),
        );
        self.buffer
            .reserve(input_signal.len() - nch * batch + self.buffer.len());
        unsafe {
            self.buffer
                .set_len(input_signal.len() - nch * batch + self.buffer.len())
        };
        let l = self.buffer.len();
        self.buffer
            .iter_mut()
            .zip(&input_signal[input_signal.len() - l..])
            .for_each(|(a, &b)| *a = b);


        let mut x1=self.batch_filter.filter(signal.view());
        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch, batch)).assume_init() };
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch);

        result.axis_iter_mut(Axis(1))
            .zip(x1.axis_iter_mut(Axis(0)))
            .for_each(|(mut r_col, mut x1_row)|{
                fft.process(x1_row.as_slice_mut().unwrap());
                r_col.assign(&x1_row.view());
            })

        ;
        /*
        result
            .axis_iter_mut(Axis(1))
            .zip(x1.axis_iter(Axis(1)))
            .for_each(|(mut r_col, x1_col)| {
                let mut fx1 = x1_col.to_owned();
                fft.process(fx1.as_slice_mut().unwrap());
                r_col.slice_mut(s![..]).assign(&ArrayView1::from(&fx1));
            });
        */
        result
    }

    /// A parallel version of [`Self::analyze`]
    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        self.analyze(input_signal)
    }
}
