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
        
        self.initial_state.reserve(tap-1+signal.len());
        self.initial_state.extend_from_slice(signal);
        

        let result = self
            .initial_state
            .windows(tap)
            .map(|x| x.iter().zip(&self.coeff_rev).map(|(&a, &b)| a * b).sum())
            .collect::<Vec<_>>();

        //self.initial_state=self.initial_state[output_length..].to_vec();
        for (i,j) in (0..tap-1).zip(self.initial_state.len()+1-tap..self.initial_state.len()){
            self.initial_state[i]=self.initial_state[j];
        }
        unsafe{self.initial_state.set_len(tap-1)};
        assert_eq!(result.len(), output_length);
        assert_eq!(self.initial_state.len(), tap-1);

        result
    }
}
