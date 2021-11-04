#![allow(clippy::many_single_char_names)]
use num_traits::{Float, FloatConst, NumAssign};
use special::Error;

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

pub fn hann_window<T>(i: isize, n: usize) -> T
where
    T: Float + FloatConst,
{
    let m = T::from(n as isize / 2).unwrap();
    let i = T::from(i).unwrap();
    let one = T::one();
    let two = one + one;
    let half = one / two;
    let pi = T::PI();
    half * (one + (pi * i / m).cos())
}

pub fn hamming_window<T>(i: isize, n: usize) -> T
where
    T: Float + FloatConst,
{
    let m = T::from(n as isize / 2).unwrap();
    let i = T::from(i).unwrap();
    let one = T::one();
    let two = one + one;
    let _half = one / two;
    let pi = T::PI();
    T::from(0.54).unwrap() + T::from(0.46).unwrap() * (pi * i / m).cos()
}

pub fn apply_hann_window<T>(workpiece: &mut [T])
where
    T: Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug,
{
    let n = workpiece.len();
    workpiece.iter_mut().enumerate().for_each(|(i, x)| {
        let j = i as isize - n as isize / 2;
        *x = hann_window::<T>(j, n) * (*x);
    });
}

pub fn apply_hamming_window<T>(workpiece: &mut [T])
where
    T: Float + FloatConst + NumAssign + std::iter::Sum<T> + std::fmt::Debug,
{
    let n = workpiece.len();
    workpiece.iter_mut().enumerate().for_each(|(i, x)| {
        let j = i as isize - n as isize / 2;
        *x = hamming_window::<T>(j, n) * (*x);
    });
}

pub fn blackman_window<T>(i: usize, n: usize) -> T
where
    T: Float + FloatConst,
{
    let a0 = T::from(0.3635819).unwrap();
    let a1 = T::from(0.4891775).unwrap();
    let a2 = T::from(0.1365995).unwrap();
    let a3 = T::from(0.0106411).unwrap();
    let x = T::from(i).unwrap() / T::from(n).unwrap() * T::PI();
    let two = T::one() + T::one();
    let four = two + two;
    let six = two + four;
    a0 - a1 * (two * x).cos() + a2 * (four * x).cos() - a3 * (six * x).cos()
}

pub fn apply_blackman_window<T>(workpiece: &mut [T])
where
    T: Float + FloatConst,
{
    let n = workpiece.len();
    workpiece.iter_mut().enumerate().for_each(|(i, x)| {
        *x = *x * blackman_window(i, n);
    });
}
