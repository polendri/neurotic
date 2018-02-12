//! Weight and bias initialization strategies

use std::ops::Mul;

use generic_array::ArrayLength;
use nalgebra::{DimAdd, DimName, DimSum, MatrixMN, Scalar, U1, VectorN};
use typenum::{NonZero, Prod};

use distribution::{self, Distribution};

/// Trait for types which can initialize the weights and biases of a neural network.
pub trait Initializer<X, H, Y>
where
    X: NonZero + DimAdd<U1>,
    H: NonZero + DimName + DimAdd<U1>,
    Y: NonZero + DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    H::Value: Mul<<<X as DimAdd<U1>>::Output as DimName>::Value>,
    Y::Value: Mul<<<H as DimAdd<U1>>::Output as DimName>::Value>,
    Prod<H::Value, <<X as DimAdd<U1>>::Output as DimName>::Value>: ArrayLength<f64>,
    Prod<Y::Value, <<H as DimAdd<U1>>::Output as DimName>::Value>: ArrayLength<f64>,
{
    /// Generate a set of initial weight matrices.
    fn weights(&mut self) -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>);
}

/// Initializer using a Standard Normal distribution.
pub struct StandardNormal {
    sampler: distribution::StandardNormal,
}

impl StandardNormal {
    /// Constructs a new `StandardNormal` initializer.
    pub fn new() -> Self {
        StandardNormal {
            sampler: distribution::StandardNormal::new(),
        }
    }
}

impl<X, H, Y> Initializer<X, H, Y> for StandardNormal
where
    X: NonZero + DimAdd<U1>,
    H: NonZero + DimName + DimAdd<U1>,
    Y: NonZero + DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    H::Value: Mul<<<X as DimAdd<U1>>::Output as DimName>::Value>,
    Y::Value: Mul<<<H as DimAdd<U1>>::Output as DimName>::Value>,
    Prod<H::Value, <<X as DimAdd<U1>>::Output as DimName>::Value>: ArrayLength<f64>,
    Prod<Y::Value, <<H as DimAdd<U1>>::Output as DimName>::Value>: ArrayLength<f64>,
{
    fn weights(&mut self) -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>) {
        (
            MatrixMN::<f64, H, DimSum<X, U1>>::from_fn(|_, _| self.sampler.sample()),
            MatrixMN::<f64, Y, DimSum<H, U1>>::from_fn(|_, _| self.sampler.sample()),
        )
    }
}

/// Normal distribution normalized by input connection count, i.e. with mean 0 and with standard
/// deviation `1/sqrt(num_inputs)`.
pub struct InputNormalizedNormal {
    sampler: distribution::StandardNormal,
}

impl InputNormalizedNormal {
    /// Constructs a new `InputNormalizedNormal` initializer.
    pub fn new() -> Self {
        InputNormalizedNormal {
            sampler: distribution::StandardNormal::new(),
        }
    }
}

impl<X, H, Y> Initializer<X, H, Y> for InputNormalizedNormal
where
    X: NonZero + DimAdd<U1>,
    H: NonZero + DimName + DimAdd<U1>,
    Y: NonZero + DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    H::Value: Mul<<<X as DimAdd<U1>>::Output as DimName>::Value>,
    Y::Value: Mul<<<H as DimAdd<U1>>::Output as DimName>::Value>,
    Prod<H::Value, <<X as DimAdd<U1>>::Output as DimName>::Value>: ArrayLength<f64>,
    Prod<Y::Value, <<H as DimAdd<U1>>::Output as DimName>::Value>: ArrayLength<f64>,
{
    fn weights(&mut self) -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>) {
        (
            MatrixMN::<f64, H, DimSum<X, U1>>::from_fn(|r, _| {
                if r == 0 {
                    // Biases should not be input-normalized, only connection weights
                    self.sampler.sample()
                } else {
                    self.sampler.sample() / (X::try_to_usize().unwrap() as f64).sqrt()
                }
            }),
            MatrixMN::<f64, Y, DimSum<H, U1>>::from_fn(|r, _| {
                if r == 0 {
                    self.sampler.sample()
                } else {
                    self.sampler.sample() / (H::try_to_usize().unwrap() as f64).sqrt()
                }
            }),
        )
    }
}
