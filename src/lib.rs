//! The compile-time feedforward neural network library that no one asked for.
#![warn(missing_docs)]

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
