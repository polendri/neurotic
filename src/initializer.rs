//! Weight and bias initialization strategies

use nalgebra::{DMatrix, DVector};

use distribution::{self, Distribution};

/// Trait for types which can initialize the weights and biases of a neural network.
pub trait Initializer {
    /// Generate initial biases for the specified configuration of layers.
    fn biases(&mut self, layer_sizes: &[usize; 3]) -> [DVector<f64>; 2];

    /// Generate initial weights for the specified configuration of layers.
    fn weights(&mut self, layer_sizes: &[usize; 3]) -> [DMatrix<f64>; 2];
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

impl Initializer for StandardNormal {
    fn biases(&mut self, layer_sizes: &[usize; 3]) -> [DVector<f64>; 2] {
        [
            DVector::from_fn(layer_sizes[1], |_, _| self.sampler.sample()),
            DVector::from_fn(layer_sizes[2], |_, _| self.sampler.sample()),
        ]
    }

    fn weights(&mut self, layer_sizes: &[usize; 3]) -> [DMatrix<f64>; 2] {
        [
            DMatrix::from_fn(layer_sizes[1], layer_sizes[0], |_, _| self.sampler.sample()),
            DMatrix::from_fn(layer_sizes[2], layer_sizes[1], |_, _| self.sampler.sample()),
        ]
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

impl Initializer for InputNormalizedNormal {
    fn biases(&mut self, layer_sizes: &[usize; 3]) -> [DVector<f64>; 2] {
        [
            DVector::from_fn(layer_sizes[1], |_, _| self.sampler.sample()),
            DVector::from_fn(layer_sizes[2], |_, _| self.sampler.sample()),
        ]
    }

    fn weights(&mut self, layer_sizes: &[usize; 3]) -> [DMatrix<f64>; 2] {
        [
            DMatrix::from_fn(layer_sizes[1], layer_sizes[0], |_, _| self.sampler.sample() / (layer_sizes[0] as f64).sqrt()),
            DMatrix::from_fn(layer_sizes[2], layer_sizes[1], |_, _| self.sampler.sample() / (layer_sizes[1] as f64).sqrt()),
        ]
    }
}
