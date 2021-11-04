#![allow(clippy::many_single_char_names)]
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, Zero};
use special::Error;

use crate::utils::{fftshift, ifft};
use rustfft::FftNum;

fn rrerf<T>(f: T, k: T, m: T) -> T
where
    T: Float + Error,
{
    let two = T::one() + T::one();
    let half = T::one() / two;
    let x = k * (two * m * f - half);
    (x.compl_error() / two).sqrt()
}
pub fn rrerf_coeff<T>(l: usize, m: usize, k: T) -> Vec<T>
where
    T: Float + Error,
{
    (0..l * m)
        .map(|x| {
            rrerf(
                T::from(x).unwrap() / T::from(l * m).unwrap(),
                k,
                T::from(m).unwrap(),
            )
        })
        .collect::<Vec<_>>()
}

pub fn hann_coeff<T>(l: usize, m: usize, _k: T) -> Vec<T>
where
    T: Float + Error + FloatConst,
{
    let beta = T::from(0.65).unwrap();
    let period = T::from(1.0).unwrap() / T::from(l as f64 * 1.08).unwrap();
    let two = T::one() + T::one();
    (0..l * m)
        .map(|x| {
            if T::from(x).unwrap() < (T::one() - beta) / (two * period) {
                T::one()
            } else if T::from(x).unwrap() < (T::one() + beta) / (two * period) {
                T::one() / two
                    * (T::one()
                        + T::cos(
                            T::PI() * period / beta
                                * (T::from(x).unwrap().abs() - (T::one() - beta) / (two * period)),
                        ))
            } else {
                T::zero()
            }
        })
        .collect::<Vec<_>>()
}

pub fn lp_coeff<T>(l: usize, m: usize) -> Vec<T>
where
    T: Float + FloatConst,
{
    (0..l * m)
        .map(|i| {
            if i <= l / 4 || i >= l * m - l / 4 {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect()
}

pub fn symmetrize<T: Copy + Zero>(workpiece: &mut [T]) {
    let n1 = workpiece.len();

    for n in 0..=(n1 / 2 - 2) {
        workpiece[n1 - n - 1] = workpiece[n + 1];
    }
    //workpiece[n1 / 2] = T::zero();
}

pub fn to_time_domain<T>(input: &[T]) -> Vec<T>
where
    T: Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug + FftNum,
{
    let a: Vec<_> = input.iter().map(Complex::<T>::from).collect();
    let b: Vec<_> = ifft(&a[..]).iter().map(|x| x.re).collect();
    let b = fftshift(&b);
    let s = b.iter().cloned().sum();
    b.iter().map(|&x| x / s).collect()
}

pub fn coeff<T>(n: usize, l: usize, k: Option<T>) -> Array2<T>
where
    T: Error + Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug + FftNum,
{
    let k: T = if let Some(x) = k {
        x
    } else {
        T::from(8).unwrap()
    };

    let m = n / 2;
    let mut a = rrerf_coeff(l, m, k);
    //let mut a=hann_coeff(l, m, k);
    //let mut a=lp_coeff(l,m);
    symmetrize(&mut a);
    let b = to_time_domain(&a);
    //Array1::from(b).into_shape((l, m)).unwrap().t().to_owned()
    Array1::from(b)
        .into_shape((l, m))
        .unwrap()
        .t()
        .as_standard_layout()
        .to_owned()
}

pub fn coeff_cs<T>(n: usize, l: usize, k: Option<T>) -> Array2<T>
where
    T: Error + Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug + FftNum,
{
    let k: T = if let Some(x) = k {
        x
    } else {
        T::from(8).unwrap()
    };

    let m = n;

    let mut a = hann_coeff(l, m, k);
    symmetrize(&mut a);
    let b = to_time_domain(&a);
    //Array1::from(b).into_shape((l, m)).unwrap().t().to_owned()
    Array1::from(b)
        .into_shape((l, m))
        .unwrap()
        .t()
        .as_standard_layout()
        .to_owned()
}
