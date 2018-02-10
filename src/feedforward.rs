//! Feedforward neural networks.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector};

use activation::ActivationFunction;
use cost::CostFunction;
use initializer::Initializer;
use optimizer::Optimizer;

/// Defines the parameters for creating a `NeuralNetwork`.
pub struct NeuralNetworkParameters<'i, A, C, I, O>
where
    A: ActivationFunction,
    C: CostFunction,
    I: 'i + Initializer,
    O: Optimizer<A, C>,
{
    layer_sizes: [usize; 3],
    initializer: &'i mut I,
    activation_function: PhantomData<A>,
    cost_function: PhantomData<C>,
    optimizer: PhantomData<O>,
}

impl<'i, A, C, I, O> NeuralNetworkParameters<'i, A, C, I, O>
where
    A: ActivationFunction,
    C: CostFunction,
    I: 'i + Initializer,
    O: Optimizer<A, C>,
{
    /// Constructs a new `NeuralNetworkParameters` object, returning `None` if any of the
    /// specified parameters are invalid.
    ///
    /// `layer_sizes` describes the number of neurons in each layer of the network (input, hidden
    /// and output); each value in it must be positive.
    pub fn new(layer_sizes: [usize; 3], initializer: &'i mut I) -> Option<Self> {
        if layer_sizes.iter().any(|s| *s == 0) {
            None
        } else {
            Some(NeuralNetworkParameters {
                layer_sizes,
                initializer,
                activation_function: PhantomData,
                cost_function: PhantomData,
                optimizer: PhantomData,
            })
        }
    }
}

/// A feedforward neural network.
pub struct NeuralNetwork<A, C, O>
where
    A: ActivationFunction,
    C: CostFunction,
    O: Optimizer<A, C>,
{
    /// Neuron biases for the hidden and output layers.
    biases: [DVector<f64>; 2],

    /// Connection weights between each layer. `weights[i]` is the connections from the `i-1`th
    /// layer to the `ith` layer.
    weights: [DMatrix<f64>; 2],

    /// The activation function to apply to the output values of neurons.
    activation_function: PhantomData<A>,

    /// The cost function for evaluating the cost of an output compared to the expected value.
    cost_function: PhantomData<C>,

    /// TODO
    optimizer: PhantomData<O>,
}

impl<A, C, O> NeuralNetwork<A, C, O>
where
    A: ActivationFunction,
    C: CostFunction,
    O: Optimizer<A, C>,
{
    /// Constructs a new `NeuralNetwork` with the provided parameters.
    pub fn new<'i, I>(parameters: &mut NeuralNetworkParameters<'i, A, C, I, O>) -> NeuralNetwork<A, C, O>
    where
        I: 'i + Initializer,
    {
        let biases = parameters.initializer.biases(&parameters.layer_sizes);
        let weights = parameters.initializer.weights(&parameters.layer_sizes);

        for i in 0..biases.len() {
            debug_assert_eq!(biases[i].nrows(), parameters.layer_sizes[i + 1]);
        }
        for i in 0..weights.len() {
            debug_assert_eq!(weights[i].nrows(), parameters.layer_sizes[i + 1]);
            debug_assert_eq!(weights[i].ncols(), parameters.layer_sizes[i]);
        }

        NeuralNetwork {
            biases,
            weights,
            activation_function: PhantomData,
            cost_function: PhantomData,
            optimizer: PhantomData,
        }
    }

    fn feedforward(&self, input: DVector<f64>) -> DVector<f64> {
        let mut a = input;

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a = w * a + b;
            a.apply(A::eval);
        }

        a
    }

    /// TODO
    pub fn compute_grad(&self, input: &DVector<f64>, target: &DVector<f64>) -> ([DVector<f64>; 2], [DMatrix<f64>; 2]) {
        let mut activations: [DVector<f64>; 3] = [DVector::<f64>::zeros(0), DVector::<f64>::zeros(0), DVector::<f64>::zeros(0)];
        let mut output_values: [DVector<f64>; 2] = [DVector::<f64>::zeros(0), DVector::<f64>::zeros(0)];

        // Set activation for the input layer
        activations[0] = input.clone();

        // Feedforward to compute output values and activations for subsequent layers
        for i in 0..1 {
            let z = &self.weights[i] * &activations[i + 1] + &self.biases[i];
            activations[i + 1] = z.map(A::eval);
            output_values[0] = z;
        }

        let mut output_errors: [DVector<f64>; 2] = [DVector::<f64>::zeros(0), DVector::<f64>::zeros(0)];

        // Compute the output layer's output error
        output_errors[1] = {
            let ref zl = output_values[1];
            let ref al = activations[2];
            let mut result = C::eval_grad(al, &target);
            result.component_mul_assign(&zl.map(A::eval_grad));
            result
        };

        // Backpropagate the output error
        output_errors[0] = {
            let ref prev_w = self.weights[1].transpose();
            let ref prev_error = output_errors[1];
            let ref z = output_values[0];

            let mut result = prev_w * prev_error;
            result.component_mul_assign(&z.map(A::eval_grad));
            result
        };

        // Compute the gradient of the cost function by each bias and weight

        let mut weight_grads: [DMatrix<f64>; 2] = [DMatrix::<f64>::zeros(0, 0), DMatrix::<f64>::zeros(0, 0)];

        for i in 0..1 {
            weight_grads[i] = activations[i + 1].clone() * output_errors[i].transpose();
        }

        let bias_grads: [DVector<f64>; 2] = output_errors;

        (bias_grads, weight_grads)
    }

    /// TODO
    pub fn optimize(&mut self, delta_b: &[DVector<f64>; 2], delta_w: &[DMatrix<f64>; 2]) {
        for i in 0..2 {
            self.biases[i] += &delta_b[i];
            self.weights[i] += &delta_w[i];
        }
    }

    //fn train<T>(&mut self, training_data: T)
}
