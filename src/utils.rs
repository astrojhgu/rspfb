#![allow(non_snake_case)]
use std::ops::Index;

use fftw::{
    plan::{C2CPlan, C2CPlan32, C2CPlan64},
    types::{Flag, Sign},
};
use rustfft::{FftNum, FftPlanner};

use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use num_complex::Complex;
use num_traits::{Float, FloatConst, Num, NumAssign};

pub fn fft<T>(in_data: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    //let mut fft = CFft1D::<T>::with_len(in_data.len());
    let mut output = Vec::from(in_data);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(in_data.len());
    fft.process(&mut output);
    output
}

pub fn ifft<T>(in_data: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    let mut output = Vec::from(in_data);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(in_data.len());
    fft.process(&mut output);
    let norm = T::from(in_data.len()).unwrap();
    output.iter_mut().for_each(|x| *x /= norm);
    output
}

pub fn ifft0<T>(in_data: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    let mut output = Vec::from(in_data);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(in_data.len());
    fft.process(&mut output);
    output
}

pub trait FftwImplemented<T> {
    fn fft(input: &mut [Complex<T>], sign: Sign) -> Vec<Complex<T>>;
}

impl FftwImplemented<f64> for f64 {
    fn fft(input: &mut [Complex<f64>], sign: Sign) -> Vec<Complex<f64>> {
        let mut output = Vec::with_capacity(input.len());
        let mut plan =
            C2CPlan64::new(&[input.len()], input, &mut output, sign, Flag::ESTIMATE).unwrap();
        plan.c2c(input, &mut output).unwrap();
        unsafe { output.set_len(input.len()) };
        output
    }
}

impl FftwImplemented<f32> for f32 {
    fn fft(input: &mut [Complex<f32>], sign: Sign) -> Vec<Complex<f32>> {
        let mut output = Vec::with_capacity(input.len());
        let mut plan =
            C2CPlan32::new(&[input.len()], input, &mut output, sign, Flag::ESTIMATE).unwrap();
        plan.c2c(input, &mut output).unwrap();
        unsafe { output.set_len(input.len()) };
        output
    }
}

pub fn fftwfft<T>(input: &mut [Complex<T>]) -> Vec<Complex<T>>
where
    T: FftwImplemented<T>,
{
    T::fft(input, Sign::Forward)
}

pub fn fftwifft<T>(input: &mut [Complex<T>]) -> Vec<Complex<T>>
where
    T: FftwImplemented<T>,
{
    T::fft(input, Sign::Backward)
}

pub fn fftshift<T>(in_data: &[T]) -> Vec<T>
where
    T: Copy,
{
    let mut result = Vec::with_capacity(in_data.len());
    let n = (in_data.len() + 1) as usize / 2;
    //for i in n..in_data.len() {
    for item in in_data.iter().skip(n).cloned() {
        //result[i-n-1]=in_data[i];
        result.push(item);
    }

    for item in in_data.iter().take(n).cloned() {
        //result[i+n]=in_data[i];
        result.push(item);
    }
    result
}

pub fn fftshift2<T>(in_data: ArrayView2<T>) -> Array2<T>
where
    T: Copy,
{
    assert!(in_data.shape()[0] % 2 == 0);
    let n2 = in_data.shape()[0] / 2;
    let mut result =
        unsafe { Array2::uninit((in_data.shape()[0], in_data.shape()[1])).assume_init() };

    for i in 0..n2 {
        result
            .slice_mut(s![i, ..])
            .assign(&in_data.slice(s![i + n2, ..]));
        result
            .slice_mut(s![i + n2, ..])
            .assign(&in_data.slice(s![i, ..]));
    }
    result
}

pub fn sort_ch_and_cut<T>(in_data: ArrayView2<T>) -> Array2<T>
where
    T: Copy,
{
    assert!(in_data.shape()[0] % 4 == 0);
    let n = in_data.shape()[0];
    let n2 = n / 2;
    let mut result = unsafe { Array2::uninit((n2, in_data.shape()[1])).assume_init() };
    result
        .slice_mut(s![n2 / 2.., ..])
        .assign(&in_data.slice(s![..n2 / 2, ..]));
    result
        .slice_mut(s![..n2 / 2, ..])
        .assign(&in_data.slice(s![n - n2 / 2.., ..]));
    result
}

pub fn fftfreq<T>(n: usize) -> Vec<T>
where
    T: Float,
{
    let n = n as isize;
    let result = (0..=((n - 1) / 2))
        .chain(-n / 2..=-1)
        .map(|x| T::from(x).unwrap() / T::from(n).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(result.len(), n as usize);
    result
}

pub fn transpose_par<T>(x: ArrayView2<T>) -> Array2<T>
where
    T: Copy + Send + Sync,
{
    let mut result = unsafe { Array2::<T>::uninit((x.shape()[1], x.shape()[0])).assume_init() };
    x.axis_iter(Axis(1))
        .into_par_iter()
        .zip(result.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|(x_c, mut y_r)| {
            y_r.assign(&x_c);
        });
    result
}

pub fn transpose_par_map<T, U, F>(x: ArrayView2<T>, f: F) -> Array2<U>
where
    T: Copy + Send + Sync,
    U: Copy + Send + Sync,
    F: Fn(&T) -> U + Sync + Send,
{
    let mut result = unsafe { Array2::<U>::uninit((x.shape()[1], x.shape()[0])).assume_init() };
    x.axis_iter(Axis(1))
        .into_par_iter()
        .zip(result.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|(x_c, mut y_r)| {
            y_r.assign(&ArrayView1::from(&x_c.iter().map(&f).collect::<Vec<_>>()));
        });
    result
}

pub struct ConcatedSlice<'a, 'b, T> {
    pub old: &'a [T],
    pub appended: &'b [T],
}

impl<'a, 'b, T> ConcatedSlice<'a, 'b, T> {
    pub fn new(a: &'a [T], b: &'b [T]) -> ConcatedSlice<'a, 'b, T> {
        ConcatedSlice {
            old: a,
            appended: b,
        }
    }

    pub fn len(&self) -> usize {
        self.old.len() + self.appended.len()
    }

    pub fn is_empty(&self) -> bool {
        self.old.is_empty() && self.appended.is_empty()
    }
}

impl<'a, 'b, T> Index<usize> for ConcatedSlice<'a, 'b, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        let b = self.old.len();
        if idx < b {
            &self.old[idx]
        } else {
            &self.appended[idx - b]
        }
    }
}

pub fn add_neg_freq_no_dc<T>(input: Array2<Complex<T>>) -> Array2<Complex<T>>
where
    T: Num + Copy + std::ops::Neg<Output = T>,
{
    let mut result = Array2::<Complex<T>>::zeros((input.shape()[0] * 2, input.shape()[1]));

    result
        .slice_mut(s![1..input.shape()[0], ..])
        .assign(&input.slice(s![..input.shape()[0] - 1, ..]));
    result.slice_mut(s![input.shape()[0] + 1.., ..]).assign(
        &input
            .slice(s![..input.shape()[0]-1;-1,..])
            .map(|x| x.conj()),
    );
    result
}

pub fn apply_delay<T>(x: &mut Array2<Complex<T>>, d: T)
where
    T: Float + Copy + FloatConst + std::fmt::Debug,
{
    let two = T::one() + T::one();
    let freqs = fftfreq::<T>(x.shape()[0]);

    for (r, k) in x.rows_mut().into_iter().zip(
        freqs
            .into_iter()
            .map(|f| Complex::<T>::new(T::zero(), two * T::PI() * f * d).exp()),
    ) {
        for x1 in r {
            *x1 = *x1 * k;
        }
    }
}

pub fn corr<T>(x: &[T], y: &[T], fold_len: usize) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumAssign + FftNum,
{
    let result: Array1<Complex<T>> = x
        .chunks(fold_len)
        .zip(y.chunks(fold_len))
        .map(|(x1, y1)| {
            let x1: Vec<_> = x1
                .iter()
                .map(|&x11| Complex::<T>::new(x11, T::zero()))
                .collect();
            let y1: Vec<_> = y1
                .iter()
                .map(|&y11| Complex::<T>::new(y11, T::zero()))
                .collect();
            let X1 = fft(&x1[..]);
            let Y1 = fft(&y1[..]);
            Array1::from(
                X1.iter()
                    .zip(Y1.iter())
                    .map(|(&x1, &y1)| x1 * y1.conj())
                    .collect::<Vec<_>>(),
            )
        })
        .fold(Array1::<Complex<T>>::zeros(fold_len), |x, y| {
            if y.len() == fold_len {
                x + y
            } else {
                x
            }
        });
    //fftshift(result.as_slice().unwrap())
    result.to_vec()
}
