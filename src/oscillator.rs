//! digital oscillator for single frequency complex signal

use num_complex::Complex;
use num_traits::{Float, FloatConst};

/// Complex oscillator
pub struct COscillator<T> {
    /// current phase
    phi: T,
    /// phase difference between points
    dphi_dpt: T,
}

impl<T> COscillator<T>
where
    T: Float,
{
    /// constructor
    pub fn new(phi: T, dphi_dpt: T) -> COscillator<T> {
        COscillator { phi, dphi_dpt }
    }

    /// get the next value
    pub fn get(&mut self) -> Complex<T> {
        let y = (Complex::<T>::new(T::zero(), T::one()) * self.phi).exp();
        self.phi = self.phi + self.dphi_dpt;
        y
    }
}

/// Shifting signal by half of the channel spacing
pub struct HalfChShifter<T> {
    /// number of channels
    nch: usize,
    /// buffered factor, so that they need not to be computed repeatly
    factor: Vec<Complex<T>>,
    idx: usize,
}

impl<T> HalfChShifter<T>
where
    T: Float + FloatConst + std::fmt::Debug,
{
    /// constructor from number of channels and the direction
    /// * `nch` -  number of channels
    /// * `upshift` - shifting the frequency upward (`true`) or downward (`false`)
    pub fn new(nch: usize, upshift: bool) -> HalfChShifter<T> {
        let mut osc = COscillator::<T>::new(
            T::zero(),
            if upshift {
                T::PI() / T::from(nch).unwrap()
            } else {
                -T::PI() / T::from(nch).unwrap()
            },
        );
        let mut factor = Vec::new();
        for _ in 0..nch * 2 {
            factor.push(osc.get());
        }
        HalfChShifter {
            nch,
            factor,
            idx: 0,
        }
    }

    /// get the next factor
    pub fn get(&mut self) -> Complex<T> {
        let x = self.factor[self.idx];
        self.idx = (self.idx + 1) % (2 * self.nch);
        x
    }
}
