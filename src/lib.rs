//! A fast, easy-to-use library for implementing neural networks.
#![warn(missing_docs)]

extern crate nalgebra;
extern crate num_traits;
extern crate rand;

pub mod distribution;
pub mod feedforward;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
