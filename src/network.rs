//! Feedforward neural networks.

use std::marker::PhantomData;

use nalgebra::{DefaultAllocator, DimAdd, DimName, DimSum, MatrixMN, U1, VectorN};
use nalgebra::allocator::{Allocator, Reallocator};

use activation::ActivationFunction;
use cost::CostFunction;
use initializer::Initializer;

/// A feedforward neural network.
pub struct NeuralNetwork<X, H, Y, A, C>
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    A: ActivationFunction,
    C: CostFunction<Y>,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, X, U1> +
                      Allocator<f64, H, U1> +
                      Allocator<f64, Y, U1> +
                      Allocator<f64, U1, DimSum<X, U1>> +
                      Allocator<f64, H, DimSum<X, U1>> +
                      Allocator<f64, Y, DimSum<H, U1>> +
                      Allocator<f64, DimSum<H, U1>, Y> +
                      Reallocator<f64, X, U1, DimSum<X, U1>, U1> +
                      Reallocator<f64, H, U1, DimSum<H, U1>, U1> +
                      Reallocator<f64, U1, H, U1, DimSum<H, U1>>
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

impl<X, H, Y, A, C> NeuralNetwork<X, H, Y, A, C>
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    A: ActivationFunction,
    C: CostFunction<Y>,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, X, U1> +
                      Allocator<f64, H, U1> +
                      Allocator<f64, Y, U1> +
                      Allocator<f64, U1, DimSum<X, U1>> +
                      Allocator<f64, H, DimSum<X, U1>> +
                      Allocator<f64, Y, DimSum<H, U1>> +
                      Allocator<f64, DimSum<H, U1>, Y> +
                      Reallocator<f64, X, U1, DimSum<X, U1>, U1> +
                      Reallocator<f64, H, U1, DimSum<H, U1>, U1> +
                      Reallocator<f64, U1, H, U1, DimSum<H, U1>>
{
    pub fn new<I: Initializer<X, H, Y>>() -> Self {
        NeuralNetwork {
            weights: I::weights(),
            activation_function: PhantomData,
            cost_function: PhantomData,
        }
    }

    pub fn feedforward(&self, input: &VectorN<f64, X>) -> VectorN<f64, Y> {
        let mut a1: VectorN<f64, H> = &self.weights.0 * input.clone().insert_row(0, 1.0);
        a1.apply(A::eval);
        let mut a2: VectorN<f64, Y> = &self.weights.1 * a1.insert_row(0, 1.0);
        a2.apply(A::eval);

        a2
    }

    pub fn compute_grad(&self, input: &VectorN<f64, X>, target: &VectorN<f64, Y>) -> (MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>) {
        let x = input.clone().insert_row(0, 1.0);

        // Feedforward to compute output values and activations
        let z1: VectorN<f64, H> = &self.weights.0 * &x;
        let a1: VectorN<f64, DimSum<H, U1>> = z1.map(A::eval).insert_row(0, 1.0);
        let z2: VectorN<f64, Y> = &self.weights.1 * &a1;
        let a2: VectorN<f64, Y> = z2.map(A::eval);

        // Compute the output layer's output error
        let err2: VectorN<f64, Y> = {
            let mut result = C::eval_grad(&a2, target);
            result.component_mul_assign(&z2.map(A::eval_grad));
            result
        };

        // Backpropagate the output error
        let err1: VectorN<f64, H> = {
            let mut result: VectorN<f64, H> = &self.weights.1.transpose().fixed_rows::<H>(1) * &err2;
            result.component_mul_assign(&z1.map(A::eval_grad));
            result
        };

        // Compute the gradient of the cost function by each bias and weight
        let w_grad1: MatrixMN<f64, H, DimSum<X, U1>> = &err1 * &x.transpose();
        let w_grad2: MatrixMN<f64, Y, DimSum<H, U1>> = &err2 * &a1.transpose();

        (w_grad1, w_grad2)
    }

    pub fn apply_grad(&mut self, grads: &(MatrixMN<f64, H, DimSum<X, U1>>, MatrixMN<f64, Y, DimSum<H, U1>>)) {
        self.weights.0 += &grads.0;
        self.weights.1 += &grads.1;
    }
}
