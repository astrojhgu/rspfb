//! A module containing FIR filter

use std::{
    iter::Sum,
    ops::{Add, Mul},
    marker::PhantomData,
};

use num_complex::Complex;

use ndarray::{parallel::prelude::*, s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::filter;

/// FIR filter
#[derive(Clone)]
pub struct BatchFilter<U, T> {
    /// reversed coefficients, i.e., impulse respone
    pub filters: Vec<filter::Filter<U, T>>,
    pub u: PhantomData<U>
}

impl<U, T> BatchFilter<U, T>
where
    T: Copy + Sync+Send,
    U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default+Sync+Send,
    Complex<T>:std::convert::From<U>
{
    /// construct a FIR with its coefficients    
    pub fn new(coeff: ArrayView2<T>) -> Self {
        //coeff.reverse();
        //let tap = coeff.shape()[0];
        let nch = coeff.shape()[0];
        let filters: Vec<_> = (0..nch)
            .map(|i| filter::Filter::<U, T>::new(coeff.slice(s![i, ..]).to_vec()))
            .collect();

        BatchFilter { filters, u: PhantomData{} }
    }

    /// filter a time series signal
    /// return the filtered signal

    pub fn filter(&mut self, signal: ArrayView1<U>) -> Array2<Complex<T>>
    {
        let nch = self.filters.len();
        let batch = signal.len() / nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let mut x1:Array2<U> = signal
            .into_shape((batch, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        self.filters
            .iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });
        x1.t().as_standard_layout().map(|&x|Complex::<T>::from(x))
        //x1
    }

    pub fn filter_par(&mut self, signal: ArrayView1<U>) -> Array2<Complex<T>>
    {
        let nch = self.filters.len();
        let batch = signal.len() / nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let mut x1:Array2<U> = signal
            .into_shape((batch, nch))
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();
        self.filters
            .par_iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });
        x1.t().as_standard_layout().map(|&x|Complex::<T>::from(x))
    }
}
