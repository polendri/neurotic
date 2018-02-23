//! A fast, easy-to-use library for implementing neural networks.
//#![warn(missing_docs)]

extern crate nalgebra;
extern crate rand;
extern crate typenum;

pub mod activation;
pub mod cost;
pub mod distribution;
pub mod initializer;
mod network;
pub mod optimizer;

pub use network::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
