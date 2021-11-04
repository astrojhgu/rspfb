use num_complex::Complex;

pub trait ToReal<Output> {
    fn to_real(&self) -> Output;
}

impl<T> ToReal<T> for Complex<T>
where
    T: Copy + std::fmt::Debug,
{
    fn to_real(&self) -> T {
        //print!("({:?} {:?}) ", self.re, self.im);
        self.re
    }
}

impl<T> ToReal<T> for T
where
    T: Copy,
{
    fn to_real(&self) -> T {
        *self
    }
}
