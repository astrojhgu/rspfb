//! Containing critical sampling poly phase filter bank

#![allow(clippy::uninit_vec)]

use crate::{batch_filter::Filter, utils::transpose_par_map};
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
    //filters: Vec<Filter<T, Complex<T>>>,
    filter: Filter<T, Complex<T>>,
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
        let coeff = coeff.into_shape((tap, nch)).unwrap();
        let coeff = coeff.slice(s![..,..;-1]);

        let filter=Filter::new(coeff);
        //println!("{:?}", filter.coeff_rev);

        Analyzer {
            filter,
            buffer: Vec::<R>::new(),
        }
    }

    /// return the number of channels
    /// return value - the number of channels
    pub fn nch(&self) -> usize {
        self.filter.coeff_rev.ncols()
    }

    /// Channelize input signal
    /// * `input_signal` - a 1-d slice containing time domain input signal
    /// * return value - channelized data, with `nch` rows
    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.nch();
        let batch = (self.buffer.len() + input_signal.len()) / nch;

        let signal = Array1::from_iter(
            self.buffer
                .iter()
                .chain(input_signal)
                .take(nch * batch)
                .cloned(),
        );

        //self.buffer = ArrayView1::from(&input_signal[nch * batch - self.buffer.len()..]).to_vec();
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

        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.map(|&x| Complex::<T>::from(x));


        //let mut x1=transpose_par_map(signal.into_shape((m,n)).unwrap(), |x|Complex::<T>::from(*x) );
        let _im = Complex::new(T::zero(), T::one());

        let mut x1=self.filter.filter(x1.view());
        //let mut x1=x2;

        //println!("{:?}", x1.t());

        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch);
        x1.axis_iter_mut(Axis(0)).for_each(|mut x|{
            fft.process(x.as_slice_mut().unwrap());
        });

        x1.t().as_standard_layout().to_owned()
    }


    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch = self.nch();
        let batch = (self.buffer.len() + input_signal.len()) / nch;

        let signal = Array1::from_iter(
            self.buffer
                .iter()
                .chain(input_signal)
                .take(nch * batch)
                .cloned(),
        );

        //self.buffer = ArrayView1::from(&input_signal[nch * batch - self.buffer.len()..]).to_vec();
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

        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.map(|&x| Complex::<T>::from(x));


        //let mut x1=transpose_par_map(signal.into_shape((m,n)).unwrap(), |x|Complex::<T>::from(*x) );
        let _im = Complex::new(T::zero(), T::one());

        let mut x1=self.filter.filter(x1.view());
        //let mut x1=x2;

        //println!("{:?}", x1.t());

        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch);
        x1.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x|{
            fft.process(x.as_slice_mut().unwrap());
        });

        x1.t().as_standard_layout().to_owned()
    }
}
