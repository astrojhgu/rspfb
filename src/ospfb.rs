//! oversampling poly phase filter bank

#![allow(clippy::uninit_vec)]
use crate::{batch_filter::Filter, oscillator::HalfChShifter, utils::transpose_par_map};
use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, Axis, ScalarOperand};
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign};
use rustfft::{FftNum, FftPlanner};
use std::{
    iter::Sum,
    ops::{Add, Mul},
};

/// Pfb for channelizing
pub struct Analyzer<R, T> {
    /// filters for even channels
    filter_even: Filter<T, Complex<T>>,

    /// filters for odd channels
    filter_odd: Filter<T, Complex<T>>,

    /// a buffer, ensurning that the input signal length need not to be nch*tap. The remaining elements will be stored and be concated with the input next time.
    buffer: Vec<R>,

    /// shifting input signal by half of the channel spacing
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
    /// constructor
    /// * `nch_total` - total number of channels, including even and odd, pos and neg channels
    /// * `coeff` - property low pass filter coefficients, the length of which should be equal to `nch_total`/2*`tap_per_ch`
    /// ```
    /// extern crate rspfb;
    /// use num_complex::Complex;
    /// use rspfb::{
    ///     windowed_fir
    ///     , ospfb::Analyzer
    /// };
    ///
    /// let nch=32;
    /// let tap_per_ch=16;
    /// let k=1.1;
    /// let coeff=windowed_fir::coeff::<f64>(nch/2, tap_per_ch, k);
    /// let mut pfb=Analyzer::<Complex<f64>, f64>::new(nch, coeff.view());
    /// ```
    pub fn new(nch_total: usize, coeff: ArrayView1<T>) -> Analyzer<R, T> {
        let nch_each = nch_total / 2;
        let tap = coeff.len() / nch_each;
        assert!(nch_each * tap == coeff.len());
        let coeff = coeff.into_shape((tap, nch_each)).unwrap();
        let coeff = coeff.slice(s![..,..;-1]);

        let filter_even=Filter::new(coeff);
        let filter_odd=Filter::new(coeff);



        let shifter = HalfChShifter::<T>::new(nch_each, false);

        Analyzer {
            filter_even,
            filter_odd,
            buffer: Vec::<R>::new(),
            shifter,
        }
    }

    /// performing the channelizing
    /// * `input_signal` - input 1-d time series of the input signal
    /// * return value - channelized signal, with `nch_total` rows
    /// ```
    /// extern crate rspfb;
    /// use num_complex::Complex;
    /// use rspfb::{
    ///     windowed_fir
    ///     , ospfb::Analyzer
    ///     , oscillator::COscillator
    /// };
    /// use num_traits::{FloatConst};
    ///
    /// let nch=32;
    /// let tap_per_ch=16;
    /// let k=1.1;
    /// let coeff=windowed_fir::coeff::<f64>(nch/2, tap_per_ch, k);
    /// let mut pfb=Analyzer::<Complex<f64>, f64>::new(nch, coeff.view());
    /// let mut osc=COscillator::<f64>::new(0.0, f64::PI()/(nch/2) as f64*4.0);//some certain frequency
    /// let input_signal:Vec<_>=(0..256).map(|_| osc.get()).collect();
    /// let channelized_signal=pfb.analyze(&input_signal);
    /// assert_eq!(channelized_signal.nrows(), nch);
    /// ```
    pub fn analyze(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch_each = self.filter_even.coeff_rev.ncols();
        let nch_total = nch_each * 2;

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;
        /*
        let mut signal = unsafe { Array1::<R>::uninit(batch * nch_each).assume_init() };
        signal
            .slice_mut(s![..self.buffer.len()])
            .assign(&ArrayView1::from(&self.buffer[..]));
        signal
            .slice_mut(s![self.buffer.len()..])
            .assign(&ArrayView1::from(
                &input_signal[..(nch_each * batch - self.buffer.len())],
            ));
        */

        let signal = Array1::from_iter(
            self.buffer
                .iter()
                .chain(input_signal)
                .take(nch_each * batch)
                .cloned(),
        );
        //self.buffer =
        //    ArrayView1::from(&input_signal[nch_each * batch - self.buffer.len()..]).to_vec();
        self.buffer
            .reserve(input_signal.len() - nch_each * batch + self.buffer.len());
        unsafe {
            self.buffer
                .set_len(input_signal.len() - nch_each * batch + self.buffer.len())
        };
        let l = self.buffer.len();
        self.buffer
            .iter_mut()
            .zip(&input_signal[input_signal.len() - l..])
            .for_each(|(a, &b)| *a = b);

        let x1 = signal.into_shape((batch, nch_each)).unwrap();
        let x1 = x1.map(|&x| Complex::<T>::from(x));
        let mut x2 = x1.clone();


        for i in 0..x2.nrows() {
            for j in 0..x2.ncols() {
                x2[(i, j)] *= self.shifter.get();
            }
        }

        let mut x1=self.filter_even.filter(x1.view());
        let mut x2=self.filter_odd.filter(x2.view());


        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch_total, batch)).assume_init() };
        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch_each);

        x1.axis_iter_mut(Axis(0)).for_each(|mut x|{
            fft.process(x.as_slice_mut().unwrap());
        });

        x2.axis_iter_mut(Axis(0)).for_each(|mut x|{
            fft.process(x.as_slice_mut().unwrap());
        });

        result.axis_iter_mut(Axis(1)).zip(x1.axis_iter(Axis(0)).zip(x2.axis_iter(Axis(0)))).for_each(|(mut r, (x1, x2))|{
            r.slice_mut(s![..;2]).assign(&x1);
            r.slice_mut(s![1..;2]).assign(&x2);
        });

        result
    }

    pub fn analyze_par(&mut self, input_signal: &[R]) -> Array2<Complex<T>> {
        let nch_each = self.filter_even.coeff_rev.ncols();
        let nch_total = nch_each * 2;

        let batch = (self.buffer.len() + input_signal.len()) / nch_each;
        /*
        let mut signal = unsafe { Array1::<R>::uninit(batch * nch_each).assume_init() };
        signal
            .slice_mut(s![..self.buffer.len()])
            .assign(&ArrayView1::from(&self.buffer[..]));
        signal
            .slice_mut(s![self.buffer.len()..])
            .assign(&ArrayView1::from(
                &input_signal[..(nch_each * batch - self.buffer.len())],
            ));
        */

        let signal = Array1::from_iter(
            self.buffer
                .iter()
                .chain(input_signal)
                .take(nch_each * batch)
                .cloned(),
        );
        //self.buffer =
        //    ArrayView1::from(&input_signal[nch_each * batch - self.buffer.len()..]).to_vec();
        self.buffer
            .reserve(input_signal.len() - nch_each * batch + self.buffer.len());
        unsafe {
            self.buffer
                .set_len(input_signal.len() - nch_each * batch + self.buffer.len())
        };
        let l = self.buffer.len();
        self.buffer
            .iter_mut()
            .zip(&input_signal[input_signal.len() - l..])
            .for_each(|(a, &b)| *a = b);

        let x1 = signal.into_shape((batch, nch_each)).unwrap();
        let x1 = x1.map(|&x| Complex::<T>::from(x));
        let mut x2 = x1.clone();


        for i in 0..x2.nrows() {
            for j in 0..x2.ncols() {
                x2[(i, j)] *= self.shifter.get();
            }
        }

        let mut x1=self.filter_even.filter(x1.view());
        let mut x2=self.filter_odd.filter(x2.view());


        let mut result = unsafe { Array2::<Complex<T>>::uninit((nch_total, batch)).assume_init() };
        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nch_each);

        x1.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x|{
            fft.process(x.as_slice_mut().unwrap());
        });

        x2.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut x|{
            fft.process(x.as_slice_mut().unwrap());
        });

        result.axis_iter_mut(Axis(1)).into_par_iter().zip(x1.axis_iter(Axis(0)).into_par_iter().zip(x2.axis_iter(Axis(0)).into_par_iter())).for_each(|(mut r, (x1, x2))|{
            r.slice_mut(s![..;2]).assign(&x1);
            r.slice_mut(s![1..;2]).assign(&x2);
        });

        result
    }

}
