use crate::utils::{ifft0, transpose_par_map};

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

pub struct Analyzer<R> {
    nch: usize,
    buffer: Vec<R>,
}

impl<R> Analyzer<R>
where
    R: Copy
        + Add<R, Output = R>
        + Mul<R, Output = R>
        + std::ops::MulAssign<R>
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sync
        + Send,
{
    pub fn new(nch: usize) -> Analyzer<R> {
        Analyzer {
            nch,
            buffer: Vec::<R>::new(),
        }
    }

    pub fn analyze<T>(&mut self, input_signal: &[R]) -> Array2<Complex<T>>
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
            + Mul<R, Output = Complex<T>>
            + Mul<Complex<T>, Output = Complex<T>>
            + std::convert::From<R>
            + Sum
            + Default
            + ScalarOperand
            + Sync,
    {
        let n = self.nch;
        let m = (self.buffer.len() + input_signal.len()) / n;
        let mut signal = unsafe { Array1::<R>::uninit(m * n).assume_init() };
        signal
            .slice_mut(s![..self.buffer.len()])
            .assign(&ArrayView1::from(&self.buffer[..]));
        signal
            .slice_mut(s![self.buffer.len()..])
            .assign(&ArrayView1::from(
                &input_signal[..(n * m - self.buffer.len())],
            ));
        self.buffer = ArrayView1::from(&input_signal[n * m - self.buffer.len()..]).to_vec();

        let mut x1 = signal
            .into_shape((m, n))
            .unwrap()
            .t()
            .as_standard_layout()
            .map(|&x| Complex::<T>::from(x).conj());
        //let mut x1=transpose_par_map(signal.into_shape((m,n)).unwrap(), |x|Complex::<T>::from(*x) );
        let im = Complex::new(T::zero(), T::one());

        x1.axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut x1_row)| {
                let k: Complex<_> =
                    (im * T::PI() * T::from(i).unwrap() / T::from(n).unwrap()).exp();
                x1_row.iter_mut().enumerate().for_each(|(j, x)| {
                    *x = *x * k * if j % 2 == 0 { T::one() } else { -T::one() };
                });
                //let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                //x1_row.assign(&x);
            });

        //let x1=x1.t().as_standard_layout().to_owned();
        //let x2=x2.t().as_standard_layout().to_owned();
        let mut result = unsafe { Array2::<Complex<T>>::uninit((n, m)).assume_init() };
        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(n);
        result
            .axis_iter_mut(Axis(1))
            .zip(x1.axis_iter(Axis(1)))
            .for_each(|(mut r_col, x1_col)| {
                //let fx1 = ifft0(x1_col.to_owned().as_slice().unwrap());
                let mut fx1 = x1_col.to_owned();
                fft.process(fx1.as_slice_mut().unwrap());
                //let fx2 = ifft0(x2_col.to_owned().as_slice().unwrap());

                //let fx1 = fftwifft::<T>(x1_col.to_owned().as_slice_mut().unwrap());
                r_col.slice_mut(s![..]).assign(&ArrayView1::from(&fx1));
            });

        result.map(|x| x.conj())
    }

    pub fn analyze_par<T>(&mut self, input_signal: &[R]) -> Array2<Complex<T>>
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
            + Mul<R, Output = Complex<T>>
            + Mul<Complex<T>, Output = Complex<T>>
            + std::convert::From<R>
            + Sum
            + Default
            + ScalarOperand
            + Sync,
    {
        let n = self.nch;
        let m = (self.buffer.len() + input_signal.len()) / n;
        let mut signal = unsafe { Array1::<R>::uninit(m * n).assume_init() };
        signal
            .slice_mut(s![..self.buffer.len()])
            .assign(&ArrayView1::from(&self.buffer[..]));
        signal
            .slice_mut(s![self.buffer.len()..])
            .assign(&ArrayView1::from(
                &input_signal[..(n * m - self.buffer.len())],
            ));
        self.buffer = ArrayView1::from(&input_signal[n * m - self.buffer.len()..]).to_vec();

        //let mut x1=signal.into_shape((m, n)).unwrap().t().as_standard_layout().map(|&x| Complex::<T>::from(x));
        let mut x1 = transpose_par_map(signal.into_shape((m, n)).unwrap().view(), |x| {
            Complex::<T>::from(*x).conj()
        });
        let im = Complex::new(T::zero(), T::one());

        x1.axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut x1_row)| {
                let k: Complex<_> =
                    (im * T::PI() * T::from(i).unwrap() / T::from(n).unwrap()).exp();
                x1_row.iter_mut().enumerate().for_each(|(j, x)| {
                    *x = *x * k * if j % 2 == 0 { T::one() } else { -T::one() };
                });
            });

        //let x1=x1.t().as_standard_layout().to_owned();
        //let x2=x2.t().as_standard_layout().to_owned();
        let mut result = unsafe { Array2::<Complex<T>>::uninit((n, m)).assume_init() };
        //let mut fft_plan=CFFT::<T>::with_len(x1.shape()[0]);
        result
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(x1.axis_iter(Axis(1)).into_par_iter())
            .for_each(|(mut r_col, x1_col)| {
                let fx1 = ifft0(x1_col.to_owned().as_slice().unwrap());
                //let fx1 = fftwifft(x1_col.to_owned().as_slice_mut().unwrap());
                r_col.slice_mut(s![..]).assign(&ArrayView1::from(&fx1));
            });
        result.map(|x| x.conj())
    }
}
