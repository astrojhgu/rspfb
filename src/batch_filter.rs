//! A module containing FIR filter

use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use num_traits::{
    Zero
};

use ndarray::{
    Array2
    , ArrayView2
    , s
    , Axis
    , parallel::prelude::*
};

/// FIR filter
#[derive(Clone)]
pub struct Filter<T, U> {
    /// reversed coefficients, i.e., impulse respone
    pub coeff_rev: Array2<T>,
    /// filter state
    pub initial_state: Array2<U>,
}

impl<T, U> Filter<T, U>
where
    T: Copy+Sync+Send,
    U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default+Zero+Sync+Send,
{
    /// construct a FIR with its coefficients    
    pub fn new(coeff: ArrayView2<T>) -> Self {
        //coeff.reverse();
        let tap = coeff.shape()[0];
        let nch=coeff.shape()[1];
        Filter {
            coeff_rev: coeff.slice(s![..;-1, ..]).to_owned(),
            initial_state: Array2::default((tap-1, nch)),
        }
    }

    /// set the initial state
    pub fn with_initial_state(mut self, initial_state: ArrayView2<U>) -> Self {
        assert!(initial_state.shape()[0] == self.coeff_rev.shape()[0] - 1);
        assert!(initial_state.shape()[1] == self.coeff_rev.shape()[1]);
        self.initial_state = initial_state.to_owned();
        self
    }

    /// filter a time series signal
    /// return the filtered signal
    pub fn filter(&mut self, signal: ArrayView2<U>) -> Array2<U> {
        let tap = self.coeff_rev.shape()[0];
        let signal_len=signal.nrows();
        //let output_length = signal.len();

        //self.initial_state.reserve(tap - 1 + signal.len());
        //self.initial_state.extend_from_slice(signal);
        self.initial_state.append(Axis(0), signal).unwrap();

        //self.initial_state.axis_iter(Axis(0)).into_par_iter();
        let mut result=Array2::<U>::default((signal.nrows(), signal.ncols()));
        /*
        self.initial_state.windows([nch, tap]).into_iter().zip(result.axis_iter_mut(Axis(0)))
        .for_each(|(input, mut output)|{
            output.assign(&(&input*&self.coeff_rev).sum_axis(Axis(0)));
        });*/

        result.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut output)|{
            output.assign(&(&self.initial_state.slice(s![i..i+tap, ..])*&self.coeff_rev).sum_axis(Axis(0)));
        });

        let remained=self.initial_state.slice(s![signal_len..,..]).to_owned();
        assert!(remained.nrows()==tap-1);
        self.initial_state=remained;
        result
    }
}
