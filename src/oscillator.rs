use num_complex::Complex;

use num_traits::{Float, FloatConst};

pub struct COscillator<T> {
    phi: T,
    dphi_dpt: T,
}

impl<T> COscillator<T>
where
    T: Float,
{
    pub fn new(phi: T, dphi_dpt: T) -> COscillator<T> {
        COscillator { phi, dphi_dpt }
    }

    pub fn get(&mut self) -> Complex<T> {
        let y = (Complex::<T>::new(T::zero(), T::one()) * self.phi).exp();
        self.phi = self.phi + self.dphi_dpt;
        y
    }
}

pub struct HalfChShifter<T> {
    nch: usize,
    factor: Vec<Complex<T>>,
    idx: usize,
}

impl<T> HalfChShifter<T>
where
    T: Float + FloatConst + std::fmt::Debug,
{
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

    pub fn get(&mut self) -> Complex<T> {
        let x = self.factor[self.idx];
        self.idx = (self.idx + 1) % (2 * self.nch);
        x
    }
}
