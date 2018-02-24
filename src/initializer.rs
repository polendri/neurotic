//! Weight and bias initialization strategies

use nalgebra::{DefaultAllocator, DimAdd, DimName, DimSum, MatrixMN, U1};
use nalgebra::allocator::Allocator;

use distribution::{self, Distribution};

/// Initializes the weights and biases of a neural network.
pub trait Initializer<X, H, Y>
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, H, DimSum<X, U1>> + Allocator<f64, Y, DimSum<H, U1>>
{
    /// Generate a set of initial weight matrices.
    fn weights() -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>);
}

/// Initializer using a Standard Normal distribution.
pub struct StandardNormal;

impl<X, H, Y> Initializer<X, H, Y> for StandardNormal
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, H, DimSum<X, U1>> + Allocator<f64, Y, DimSum<H, U1>>
{
    fn weights() -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>) {
        let mut sampler = distribution::StandardNormal::new();
        (
            MatrixMN::<f64, H, DimSum<X, U1>>::from_fn(|_, _| sampler.sample()),
            MatrixMN::<f64, Y, DimSum<H, U1>>::from_fn(|_, _| sampler.sample()),
        )
    }
}

/// Normal distribution normalized by input connection count, i.e. with mean 0 and with standard
/// deviation `1/sqrt(num_inputs)`.
pub struct InputNormalizedNormal;

impl<X, H, Y> Initializer<X, H, Y> for InputNormalizedNormal
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, H, DimSum<X, U1>> + Allocator<f64, Y, DimSum<H, U1>>
{
    fn weights() -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>) {
        let mut sampler = distribution::StandardNormal::new();
        (
            MatrixMN::<f64, H, DimSum<X, U1>>::from_fn(|r, _| {
                if r == 0 {
                    // Biases should not be input-normalized, only connection weights
                    sampler.sample()
                } else {
                    sampler.sample() / (X::try_to_usize().unwrap() as f64).sqrt()
                }
            }),
            MatrixMN::<f64, Y, DimSum<H, U1>>::from_fn(|r, _| {
                if r == 0 {
                    sampler.sample()
                } else {
                    sampler.sample() / (H::try_to_usize().unwrap() as f64).sqrt()
                }
            }),
        )
    }
}
