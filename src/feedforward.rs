//! Feedforward neural networks.

use nalgebra::{DMatrix, DVector};

use distribution::Distribution;

/// Defines the parameters for creating a `NeuralNetwork`.
pub struct NeuralNetworkParameters<'l, 'd, D>
    where D: 'd + Distribution
{
    layer_sizes: &'l [usize],
    initialization_distribution: &'d mut D
}

impl<'l, 'd, D> NeuralNetworkParameters<'l, 'd, D>
    where D: 'd + Distribution
{
    /// Constructs a new `NeuralNetworkParameters` object, returning `None` if any of the
    /// specified parameters are invalid.
    ///
    /// `layer_sizes` describes the number of neurons in each layer of the network, including the
    /// input and output layers; consequently, it must have length at least 2 and every value must
    /// be positive.
    pub fn new(layer_sizes: &'l [usize], initialization_distribution: &'d mut D) -> Option<Self> {
        if layer_sizes.len() < 2 || layer_sizes.iter().any(|s| *s == 0) {
            None
        } else {
            Some(NeuralNetworkParameters {
                layer_sizes,
                initialization_distribution,
            })
        }
    }
}

/// A feedforward neural network.
pub struct NeuralNetwork<'l> {
    layer_sizes: &'l [usize],
    biases: Vec<DVector<f64>>,
    weights: Vec<DMatrix<f64>>,
}

impl<'l> NeuralNetwork<'l> {
    /// Constructs a new `NeuralNetwork` with the provided parameters.
    pub fn new<'d, D>(parameters: &mut NeuralNetworkParameters<'l, 'd, D>) -> NeuralNetwork<'l>
        where D: 'd + Distribution
    {
        let mut biases = Vec::with_capacity(parameters.layer_sizes.len() - 1);
        let mut weights = Vec::with_capacity(parameters.layer_sizes.len() - 1);

        for i in 1..parameters.layer_sizes.len() {
            biases[i - 1] = DVector::from_fn(
                parameters.layer_sizes[i],
                |_, _| parameters.initialization_distribution.sample()
            );
            weights[i - 1] = DMatrix::from_fn(
                parameters.layer_sizes[i],
                parameters.layer_sizes[i - 1],
                |_, _| parameters.initialization_distribution.sample()
            );
        }

        NeuralNetwork {
            layer_sizes: parameters.layer_sizes,
            biases,
            weights,
        }
    }

    fn feedforward(&self, x: DVector<f64>) -> DVector<f64> {
        let mut a = x;

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a = w * a + b;
        }

        a
    }
}
