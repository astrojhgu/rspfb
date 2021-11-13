#![allow(unused_imports)]

extern crate rspfb;

use num_complex::{
    Complex
};

use num_traits::{
    FloatConst
};

use rspfb::{
    frac_delayer::{
        FracDelayer
        , DelayValue
    }
};


fn validate_frac_delayer(dt:f64, signal_omega:f64, signal_len: usize)->(f64,f64){
    let mut delayer1=FracDelayer::<f64, Complex<f64>>::new(500, 100);
    let mut delayer2=FracDelayer::<f64, Complex<f64>>::new(500, 100);
    let dt_idx=(dt.ceil() as isize).abs() as usize;
    let signal:Vec<_>=(0..signal_len).map(|i|{
        ((i as f64*signal_omega)*Complex::new(0.0, 1.0)).exp()
    }).collect();
    let delayed_signal1=delayer1.delay(&signal, 0.0);
    let delayed_signal2=delayer2.delay(&signal, dt);
    //println!("{}", delayed_signal1.len());
    let corr=&delayed_signal1[dt_idx..signal.len()-dt_idx].iter().zip(&delayed_signal2[dt_idx..signal.len()-dt_idx]).map(|(&a,&b)|{
        a*b.conj()
    }).sum::<Complex<f64>>();
    let answer=(signal_omega*dt).to_degrees();
    let result=corr.arg().to_degrees();
    (answer, result)    
}

fn main() {
        
}
