use crate::{filter::Filter, oscillator::HalfChShifter, utils::transpose_par_map};

use rustfft::{FftNum, FftPlanner};

//type FFTa=Fft1Dn;
//type FFTs=Fft1D;

use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, Axis, ScalarOperand};
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

pub struct Analyzer<R, T> {
    filters_even: Vec<Filter<T, Complex<T>>>,
    filters_odd: Vec<Filter<T, Complex<T>>>,
    buffer: Vec<R>,
    shifter: HalfChShifter<T>,
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
    pub fn new(nch_total: usize, coeff: ArrayView1<T>) -> Analyzer<R, T> {
        let nch_each = nch_total / 2;
        let tap = coeff.len() / nch_each;
        assert!(nch_each * tap == coeff.len());
        let coeff = coeff.into_shape((tap, nch_each)).unwrap();
        let coeff = coeff.t();
        let coeff = coeff.as_standard_layout();
        let coeff = coeff.slice(s![..;-1,..]);

        let filters_even: Vec<_> = (0..nch_each)
            .map(|i| Filter::<T, Complex<T>>::new(coeff.slice(s![i, ..]).to_vec()))
            .collect();

        let filters_odd: Vec<_> = (0..nch_each)
            .map(|i| Filter::<T, Complex<T>>::new(coeff.slice(s![i, ..]).to_vec()))
            .collect();

        let shifter = HalfChShifter::<T>::new(nch_each, false);

        Analyzer {
            filters_even,
            filters_odd,
            buffer: Vec::<R>::new(),
            shifter,
        }
    }

    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch_each = self.filters_even.len();
        let nch_total = nch_each * 2;

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;
        let mut signal = unsafe { Array1::<R>::uninit(batch * nch_each).assume_init() };
        signal
            .slice_mut(s![..self.buffer.len()])
            .assign(&ArrayView1::from(&self.buffer[..]));
        signal
            .slice_mut(s![self.buffer.len()..])
            .assign(&ArrayView1::from(
                &input_signal[..(nch_each * batch - self.buffer.len())],
            ));
        self.buffer =
            ArrayView1::from(&input_signal[nch_each * batch - self.buffer.len()..]).to_vec();

        let x1 = signal.into_shape((batch, nch_each)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        let mut x1 = x1.map(|&x| Complex::<T>::from(x));
        let mut x2 = x1.clone();
        for j in 0..x2.ncols() {
            for i in 0..x2.nrows() {
                x2[(i, j)] *= self.shifter.get();
            }
        }

        self.filters_even
            .iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });

        self.filters_odd
            .iter_mut()
            .zip(x2.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(_i, (ft, mut x2_row))| {
                let x = Array1::from(ft.filter(x2_row.as_slice().unwrap()));
                x2_row.assign(&x);
            });

        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch_total, batch)).assume_init() };
        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch_each);
        result
            .axis_iter_mut(Axis(1))
            .zip(x1.axis_iter(Axis(1)))
            .for_each(|(mut r_col, x1_col)| {
                let mut fx1 = x1_col.to_owned();
                fft.process(fx1.as_slice_mut().unwrap());
                r_col.slice_mut(s![0..;2]).assign(&ArrayView1::from(&fx1));
            });

        result
            .axis_iter_mut(Axis(1))
            .zip(x2.axis_iter(Axis(1)))
            .for_each(|(mut r_col, x2_col)| {
                let mut fx2 = x2_col.to_owned();
                fft.process(fx2.as_slice_mut().unwrap());
                r_col.slice_mut(s![1..;2]).assign(&ArrayView1::from(&fx2));
            });

        result
    }

    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch_each = self.filters_even.len();
        let nch_total = nch_each * 2;

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;
        let mut signal = unsafe { Array1::<R>::uninit(batch * nch_each).assume_init() };
        signal
            .slice_mut(s![..self.buffer.len()])
            .assign(&ArrayView1::from(&self.buffer[..]));
        signal
            .slice_mut(s![self.buffer.len()..])
            .assign(&ArrayView1::from(
                &input_signal[..(nch_each * batch - self.buffer.len())],
            ));
        self.buffer =
            ArrayView1::from(&input_signal[nch_each * batch - self.buffer.len()..]).to_vec();

        let mut x1 =
            transpose_par_map(signal.into_shape((batch, nch_each)).unwrap().view(), |&x| {
                Complex::<T>::from(x)
            });

        let mut x2 = x1.clone();
        for j in 0..x2.ncols() {
            for i in 0..x2.nrows() {
                x2[(i, j)] *= self.shifter.get();
            }
        }

        self.filters_even
            .par_iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });

        self.filters_odd
            .par_iter_mut()
            .zip(x2.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(_i, (ft, mut x2_row))| {
                let x = Array1::from(ft.filter(x2_row.as_slice().unwrap()));
                x2_row.assign(&x);
            });

        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch_total, batch)).assume_init() };
        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch_each);
        result
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(x1.axis_iter(Axis(1)).into_par_iter())
            .for_each(|(mut r_col, x1_col)| {
                let mut fx1 = x1_col.to_owned();
                fft.process(fx1.as_slice_mut().unwrap());
                r_col.slice_mut(s![0..;2]).assign(&ArrayView1::from(&fx1));
            });

        result
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(x2.axis_iter(Axis(1)).into_par_iter())
            .for_each(|(mut r_col, x2_col)| {
                let mut fx2 = x2_col.to_owned();
                fft.process(fx2.as_slice_mut().unwrap());
                r_col.slice_mut(s![1..;2]).assign(&ArrayView1::from(&fx2));
            });

        result
    }
}
