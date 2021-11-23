#![allow(unused_imports)]

extern crate rspfb;

use num_complex::Complex;

use num_traits::FloatConst;

//use rspfb::frac_delayer::{DelayValue, FracDelayer};
use rspfb::{
    batch_filter::Filter
};

use ndarray::{
    Array2,
    array
};

fn main() {
    let coeff=array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];
    let mut filter=Filter::<f64,f64>::new(coeff.view());
    let signal=array![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let filtered=filter.filter(signal.view());
    println!("{:?}", filtered);
}
