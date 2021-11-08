use std::{
    iter::Sum,
    ops::{Add, Mul},
};

#[derive(Clone)]
pub struct Filter<T, U> {
    pub coeff_rev: Vec<T>,
    pub initial_state: Vec<U>,
}

impl<T, U> Filter<T, U>
where
    T: Copy,
    U: Copy + Add<U, Output = U> + Mul<T, Output = U> + Sum + Default,
{
    pub fn new(mut coeff: Vec<T>) -> Self {
        coeff.reverse();
        let tap = coeff.len();
        Filter {
            coeff_rev: coeff,
            initial_state: vec![<U as Default>::default(); tap - 1],
        }
    }

    pub fn with_initial_state(mut self, initial_state: Vec<U>) -> Self {
        assert!(initial_state.len() == self.coeff_rev.len() - 1);
        self.initial_state = initial_state;
        self
    }

    pub fn filter(&mut self, signal: &[U]) -> Vec<U> {
        let tap = self.coeff_rev.len();
        let output_length = signal.len();
        let mut result=Vec::with_capacity(output_length);

        let mut iter=self.initial_state.iter().chain(signal);
        for i in 0..output_length{
            let mut iter1=iter.clone();
            result.push(self.coeff_rev.iter().zip(iter1).map(|(&a,&b)| b*a).sum::<U>());
            iter.next();
        }
        let remained:Vec<_>=iter.cloned().collect();
        self.initial_state=remained;
        result
    }
}
