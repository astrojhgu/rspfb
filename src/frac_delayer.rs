//! fractional delayer

use ndarray::ScalarOperand;
use num_traits::{Float, FloatConst, NumAssign, NumCast};

use std::{
    iter::Sum,
    ops::{Add, Mul, MulAssign},
};

use crate::{cfg::DelayerCfg, utils::ConcatedSlice, window_funcs::apply_blackman_window};

/// Delay value
#[derive(Clone, Copy)]
pub struct DelayValue<T>
where
    T: Float,
{
    /// integral delay part
    pub i: isize,
    /// fractional delay part
    pub f: T,
}

/// types that can be casted to [`DelayValue`]
pub trait ToDelayValue<T>: Copy
where
    T: Float,
{
    /// convert to [`DelayValue`]
    fn to_delay_value(&self) -> DelayValue<T>;
}

impl<T> ToDelayValue<T> for T
where
    T: Float,
{
    fn to_delay_value(&self) -> DelayValue<T> {
        let i = <isize as NumCast>::from(self.signum()).unwrap()
            * (<isize as NumCast>::from(self.abs()).unwrap());
        let f = *self - T::from(i).unwrap();
        DelayValue { i, f }
    }
}

impl<T> ToDelayValue<T> for (isize, T)
where
    T: Float,
{
    fn to_delay_value(&self) -> DelayValue<T> {
        DelayValue {
            i: self.0,
            f: self.1,
        }
    }
}

/// help function for constructing fractional delayer
fn sinc_pi<T>(x: T) -> T
where
    T: Float + FloatConst,
{
    if x == T::zero() {
        T::one()
    } else {
        let y = x * T::PI();
        y.sin() / y
    }
}

/// calculating reversed coefficients for the delayer
pub fn delayer_coeff_rev<T>(dt: T, half_tap: usize) -> Vec<T>
where
    T: Float + FloatConst + NumAssign + std::fmt::Debug + std::iter::Sum<T>,
{
    //let mut result=vec![T::zero(); half_tap*2+1];
    let mut result: Vec<_> = (0..2 * half_tap + 1)
        .map(|i| {
            let x = T::from(i as isize - half_tap as isize).unwrap() + dt;
            sinc_pi(x)
        })
        .collect();
    apply_blackman_window(&mut result);
    //apply_hamming_window(&mut result);
    result
}

/// Fractional delayer
#[derive(Clone)]
pub struct FracDelayer<T, R = T> {
    pub coeff_rev: Vec<T>,
    pub buffer: Vec<R>,
    pub max_delay: usize,
}

impl<T, R> FracDelayer<T, R>
where
    T: Copy
        + Float
        + FloatConst
        + std::ops::MulAssign<T>
        + ScalarOperand
        + NumAssign
        + std::iter::Sum
        + std::fmt::Debug
        + Sync
        + Send,
    R: Copy
        + Add<R, Output = R>
        + Mul<R, Output = R>
        + Mul<T, Output = R>
        + MulAssign<R>
        + ScalarOperand
        + NumAssign
        + Sum
        + std::fmt::Debug
        + Sync
        + Send,
{
    /// construct a FracDelayer
    pub fn new(max_delay: usize, half_tap: usize) -> FracDelayer<T, R> {
        FracDelayer {
            coeff_rev: delayer_coeff_rev(T::zero(), half_tap),
            buffer: vec![R::zero(); 2 * max_delay + half_tap * 2 + 1],
            max_delay,
        }
    }

    /// delay the input signal
    /// Note that there is an unchangable intrinsic delay related to the filter tap
    /// * `signal` - input signal
    /// * `dv` - delay value
    pub fn delay<U>(&mut self, signal: &[R], dv: U) -> Vec<R>
    where
        U: ToDelayValue<T> + std::fmt::Debug,
    {
        let DelayValue {
            i: delay_i,
            f: delay_f,
        } = dv.to_delay_value();
        //println!("{:?} {:?}", delay_i, delay_f);
        //println!("{:?} {:?}", delay_i, delay_f);

        self.coeff_rev = delayer_coeff_rev(delay_f, (self.coeff_rev.len() - 1) / 2);
        //let extended_signal:Vec<T>=self.buffer.iter().cloned().chain(signal.iter().cloned()).collect();
        let concated = ConcatedSlice::new(&self.buffer, signal);
        let first_idx = self.max_delay;
        let end_idx = concated.len() - self.coeff_rev.len() - self.max_delay;
        let result: Vec<_> = (first_idx..end_idx)
            .map(|i| {
                (0..self.coeff_rev.len())
                    .map(|j| {
                        concated[(i as isize + j as isize - delay_i) as usize] * self.coeff_rev[j]
                    })
                    .sum()
            })
            .collect();
        let l = concated.len();
        let l1 = self.buffer.len();

        self.buffer = (0..l1).map(|i| concated[l - l1 + i]).collect();
        result
    }
}

/// construct a delayer from [`crate::cfg::DelayerCfg`]
pub fn cfg2delayer(cfg: &DelayerCfg) -> FracDelayer<f64, f64> {
    FracDelayer::new(cfg.max_delay, cfg.half_tap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    use rayon::prelude::*;

    fn validate_frac_delayer(dt: f64, signal_omega: f64, signal_len: usize) -> (f64, f64) {
        let mut delayer1 = FracDelayer::<f64, Complex<f64>>::new(500, 100);
        let mut delayer2 = FracDelayer::<f64, Complex<f64>>::new(500, 100);
        let dt_idx = (dt.ceil() as isize).abs() as usize;
        let signal: Vec<_> = (0..signal_len)
            .map(|i| ((i as f64 * signal_omega) * Complex::new(0.0, 1.0)).exp())
            .collect();
        let delayed_signal1 = delayer1.delay(&signal, 0.0);
        let delayed_signal2 = delayer2.delay(&signal, dt);
        //println!("{}", delayed_signal1.len());
        let corr = &delayed_signal1[dt_idx..signal.len() - dt_idx]
            .iter()
            .zip(&delayed_signal2[dt_idx..signal.len() - dt_idx])
            .map(|(&a, &b)| a * b.conj())
            .sum::<Complex<f64>>();
        let answer = (signal_omega * dt).to_degrees();
        let result = corr.arg().to_degrees();
        (answer, result)
    }

    #[test]
    fn frac_delayer_test() {
        let dt_min = -2.0;
        let dt_max = 2.0;
        let nsteps = 100;
        let dt: Vec<_> = (0..=nsteps)
            .map(|i| (dt_max - dt_min) / nsteps as f64 * i as f64 + dt_min)
            .collect();
        println!("{:?}", dt);
        dt.into_par_iter().for_each(|dt| {
            let (a, b) = validate_frac_delayer(dt, f64::PI() / 64.0, 65536);
            assert!((a - b).abs() < 0.0025);
        });
    }
}
