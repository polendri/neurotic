//! Feedforward neural networks.

use std::marker::PhantomData;
use std::ops::Mul;

use generic_array::ArrayLength;
use nalgebra::{Dim, DimAdd, DimMul, DimName, DimSum, MatrixMN, Scalar, U1, VectorN};
use typenum::{NonZero, Prod};

use activation::ActivationFunction;
use cost::CostFunction;
use initializer::Initializer;

/// Defines the parameters for creating a `NeuralNetwork`.
pub struct NeuralNetworkParameters<'i, X, H, Y, A, C, I>
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
    A: ActivationFunction,
    C: CostFunction,
    I: 'i + Initializer<X, H, Y>,
{
    initializer: &'i mut I,
    x: PhantomData<X>,
    h: PhantomData<H>,
    y: PhantomData<Y>,
    activation_function: PhantomData<A>,
    cost_function: PhantomData<C>,
}

impl<'i, X, H, Y, A, C, I> NeuralNetworkParameters<'i, X, H, Y, A, C, I>
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
    A: ActivationFunction,
    C: CostFunction,
    I: 'i + Initializer<X, H, Y>,
{
    /// Constructs a new `NeuralNetworkParameters` object, returning `None` if any of the
    /// specified parameters are invalid.
    pub fn new(initializer: &'i mut I) -> Self {
        NeuralNetworkParameters {
            initializer,
            x: PhantomData,
            h: PhantomData,
            y: PhantomData,
            activation_function: PhantomData,
            cost_function: PhantomData,
        }
    }

    /// Constructs a new `NeuralNetwork` with the current parameters.
    pub fn build(&mut self) -> NeuralNetwork< X, H, Y, A, C>
    {
        unimplemented!();
        /*
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
        */
    }

}

/// A feedforward neural network.
pub struct NeuralNetwork<X, H, Y, A, C>
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
    A: ActivationFunction,
    C: CostFunction,
{
    /// The weights and biases of the neural network. The first component of the tuple is an
    /// `H*(X+1)` matrix whose first row represents the biases for each neuron in the hidden layer,
    /// and each subsequent row `r` represents the connection weights between each input neuron and
    /// the `r`th neuron in the hidden layer. The second component of the tuple is similarly a
    /// `Y*(H+1)` matrix representing the biases of the output layer and the weights connecting the
    /// hidden and output layers.
    weights: (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>),

    /// The activation function applied to the output values of neurons.
    activation_function: PhantomData<A>,

    /// The cost function for evaluating output values.
    cost_function: PhantomData<C>,
}

/*
impl<X, H, Y, A, C> NeuralNetwork<X, H, Y, A, C>
where
    X: DimName,
    H: DimName,
    H::Value: Mul<X::Value>,
    Prod<<H as DimName>::Value, <X as DimName>::Value>: ArrayLength<f64>,
    Y: DimName,
    Y::Value: Mul<H::Value>,
    //Prod<Y::Value, H::Value>: ArrayLength<f64>,
    A: ActivationFunction,
    C: CostFunction,
{
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
}
*/
