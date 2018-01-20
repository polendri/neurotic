//! Feedforward neural networks.

/// A feedforward neural network.
pub struct NeuralNetwork<'l> {
    layer_sizes: &'l [usize],
}

impl<'l> NeuralNetwork<'l> {
    /// Constructs a new `NeuralNetwork`, where `layer_sizes` is the number of neurons in each layer
    /// of the network (including the input and output layers). Returns `None` if the provided
    /// sizes are invalid (i.e. there are fewer than two layers, or any layer has size zero).
    pub fn new(layer_sizes: &'l [usize]) -> Option<NeuralNetwork<'l>> {
        if layer_sizes.len() < 2 || layer_sizes.iter().any(|s| *s == 0) {
            None
        } else {
            Some(NeuralNetwork { layer_sizes })
        }
    }
}
