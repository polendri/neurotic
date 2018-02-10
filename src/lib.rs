//! A fast, easy-to-use library for implementing neural networks.
#![warn(missing_docs)]

extern crate nalgebra;
extern crate num_traits;
extern crate rand;

pub mod activation;
pub mod cost;
pub mod distribution;
pub mod feedforward;
pub mod initializer;
pub mod optimizer;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
