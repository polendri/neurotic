//! Optimization algorithms

use rand::{thread_rng, Rng};
use std::cmp;

use nalgebra::{DefaultAllocator, DimAdd, DimName, DimSum, U1, VectorN};
use nalgebra::allocator::{Allocator, Reallocator};

use activation::ActivationFunction;
use cost::CostFunction;
use network::NeuralNetwork;

pub trait Optimizer<X, H, Y, A, C>
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
    fn optimize(&self, model: &mut NeuralNetwork<X, H, Y, A, C>, data: &[VectorN<f64, X>], labels: &[VectorN<f64, Y>]);
}

/// Gradient descent optimizer
#[derive(Clone, Copy, Debug)]
pub struct GradientDescent {
    pub learning_rate: f64,
}

impl<X, H, Y, A, C> Optimizer<X, H, Y, A, C> for GradientDescent
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
    fn optimize(&self, model: &mut NeuralNetwork<X, H, Y, A, C>, data: &[VectorN<f64, X>], labels: &[VectorN<f64, Y>]) {
        let mut avg_grads = model.compute_grad(&data[0], &labels[0]);

        for i in 1..data.len() {
            let grads = model.compute_grad(&data[i], &labels[i]);
            avg_grads.0 += &grads.0;
            avg_grads.1 += &grads.1;
        }

        avg_grads.0.apply(|w| -self.learning_rate * w / (data.len() as f64));
        avg_grads.1.apply(|w| -self.learning_rate * w / (data.len() as f64));

        model.apply_grad(&avg_grads);
    }
}

/// Stochastic gradient descent optimizer
#[derive(Clone, Copy, Debug)]
pub struct StochasticGradientDescent {
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl<X, H, Y, A, C> Optimizer<X, H, Y, A, C> for StochasticGradientDescent
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
    fn optimize(&self, model: &mut NeuralNetwork<X, H, Y, A, C>, data: &[VectorN<f64, X>], labels: &[VectorN<f64, Y>]) {
        let mut rng = thread_rng();
        let mut shuffled_indices: Vec<usize> = (0..data.len()).collect();

        for batch in 0..(data.len() / self.batch_size) {
            rng.shuffle(&mut shuffled_indices);
            let batch_start = batch * self.batch_size;
            let batch_end = cmp::min(batch_start + self.batch_size, data.len());
            let mut avg_grads = model.compute_grad(
                &data[shuffled_indices[batch_start]],
                &labels[shuffled_indices[batch_start]]
            );

            for i in (batch_start + 1)..batch_end {
                let grads = model.compute_grad(&data[i], &labels[i]);

                avg_grads.0 += &grads.0;
                avg_grads.1 += &grads.1;
            }

            avg_grads.0.apply(|w| -self.learning_rate * w / (self.batch_size as f64));
            avg_grads.1.apply(|w| -self.learning_rate * w / (self.batch_size as f64));

            model.apply_grad(&avg_grads);
        }
    }
}
