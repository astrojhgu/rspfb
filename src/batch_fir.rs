//! A module containing FIR filter

use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use num_traits::{
    Zero
};

use num_complex::{
    Complex
};

use ndarray::{
    Array1
    , Array2
    , ArrayView2
    , ArrayView1
    , s
    , Axis
    , parallel::prelude::*
};

use crate::{
    filter
};

/// FIR filter
#[derive(Clone)]
pub struct BatchFilter<T> {
    /// reversed coefficients, i.e., impulse respone
    pub filters: Vec<filter::Filter<T, Complex<T>>>,
}

impl<T> BatchFilter<T>
where
    T: Copy + Sync + Send,
    Complex<T>: Copy + Add<Complex<T>, Output = Complex<T>> + Mul<T, Output = Complex<T>> + Sum + Default + Send + Sync,
{
    /// construct a FIR with its coefficients    
    pub fn new(coeff: ArrayView2<T>) -> Self {
        //coeff.reverse();
        //let tap = coeff.shape()[0];
        let nch=coeff.shape()[0];
        let filters: Vec<_> = (0..nch)
            .map(|i| filter::Filter::<T, Complex<T>>::new(coeff.slice(s![i, ..]).to_vec()))
            .collect();


        BatchFilter {
            filters
        }
    }

    /// filter a time series signal
    /// return the filtered signal
    
    pub fn filter<R>(&mut self, signal: ArrayView1<R>) -> Array2<Complex<T>> 
    where R: Copy
        + Add<R, Output = R>
        + Mul<R, Output = R>
        + std::ops::MulAssign<R>
        + std::fmt::Debug
        + Sync
        + Send,
        Complex<T>:
        Mul<R, Output = Complex<T>>
        + Mul<Complex<T>, Output = Complex<T>>
        + std::convert::From<R>
        + Sum
        + Default
        + Sync
        + Send,
    {
        let nch=self.filters.len();
        let batch=signal.len()/nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let mut x1 = signal.into_shape((batch, nch)).unwrap().t().as_standard_layout().map(|&x| Complex::<T>::from(x));
        self.filters
            .iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });
        x1.t().as_standard_layout().to_owned()
        //x1
    }

    pub fn filter_par<R>(&mut self, signal: ArrayView1<R>) -> Array2<Complex<T>> 
    where R: Copy
        + Add<R, Output = R>
        + Mul<R, Output = R>
        + std::ops::MulAssign<R>
        + std::fmt::Debug
        + Sync
        + Send,
        Complex<T>:
        Mul<R, Output = Complex<T>>
        + Mul<Complex<T>, Output = Complex<T>>
        + std::convert::From<R>
        + Sum
        + Default
        + Sync
        + Send,
    {
        let nch=self.filters.len();
        let batch=signal.len()/nch;
        /*
        let x1 = signal.into_shape((batch, nch)).unwrap();
        let x1 = x1.t();
        let x1 = x1.as_standard_layout();
        */
        let mut x1 = signal.into_shape((batch, nch)).unwrap().t().as_standard_layout().map(|&x| Complex::<T>::from(x));
        self.filters
            .par_iter_mut()
            .zip(x1.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .for_each(|(_i, (ft, mut x1_row))| {
                let x = Array1::from(ft.filter(x1_row.as_slice().unwrap()));
                x1_row.assign(&x);
            });
        x1.t().as_standard_layout().to_owned()
        //x1
    }
}
