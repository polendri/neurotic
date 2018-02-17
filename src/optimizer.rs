//! Optimization algorithms

use rand::{thread_rng, Rng};
use std::cmp;

use nalgebra::{DefaultAllocator, DimAdd, DimName, DimSum, MatrixMN, U1, VectorN};
use nalgebra::allocator::{Allocator, Reallocator};

use activation::ActivationFunction;
use cost::CostFunction;
use network::NeuralNetwork;

pub trait Optimizer<X, H, Y, A, C, N>
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    A: ActivationFunction,
    C: CostFunction<Y>,
    N: DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, H, DimSum<X, U1>> +
                      Allocator<f64, Y, DimSum<H, U1>> +
                      Allocator<f64, Y, U1> +
                      Allocator<f64, N, X> +
                      Allocator<f64, N, Y>
{
    fn optimize(&self, model: &mut NeuralNetwork<X, H, Y, A, C>, data: MatrixMN<f64, N, X>, labels: MatrixMN<f64, N, Y>);
}

/// Gradient descent optimizer
#[derive(Clone, Copy, Debug)]
pub struct GradientDescent {
    learning_rate: f64,
}

/*
impl<X, H, Y, A, C, N> Optimizer<X, H, Y, A, C, N> for GradientDescent
where
    X: DimName + DimAdd<U1>,
    H: DimName + DimAdd<U1>,
    Y: DimName,
    A: ActivationFunction,
    C: CostFunction<Y>,
    N: DimName,
    <X as DimAdd<U1>>::Output: DimName,
    <H as DimAdd<U1>>::Output: DimName,
    DefaultAllocator: Allocator<f64, H, DimSum<X, U1>> +
                      Allocator<f64, Y, DimSum<H, U1>> +
                      Allocator<f64, Y, U1> +
                      Allocator<f64, N, X> +
                      Allocator<f64, N, Y>
{
    fn optimize(&self, model: &mut NeuralNetwork<X, H, Y, A, C>, data: MatrixMN<f64, N, X>, labels: MatrixMN<f64, N, Y>);
        let mut grads = model.compute_grad(&data[0].0, &data[0].1);

        for i in 1..data.len() {
            let (b_grads, w_grads) = model.compute_grad(&data[i].0, &data[i].1);

            for j in 0..grads.0.len() {
                grads.0[j] += b_grads[j].clone();
                grads.1[j] += w_grads[j].clone();
            }
        }

        for i in 0..grads.0.len() {
            grads.0[i].apply(|b| -self.learning_rate * b / (data.len() as f64));
            grads.1[i].apply(|w| -self.learning_rate * w / (data.len() as f64));
        }

        model.optimize(&grads.0, &grads.1)
    }
}

/// Stochastic descent optimizer
#[derive(Clone, Copy, Debug)]
pub struct StochasticGradientDescent {
    batch_size: usize,
    learning_rate: f64,
}

impl<A, C> Optimizer<A, C> for StochasticGradientDescent
where
    A: ActivationFunction,
    C: CostFunction,
{
    fn optimize(&self, model: &mut NeuralNetwork<A, C, Self>, data: &[(DVector<f64>, DVector<f64>)]) {
        let mut rng = thread_rng();
        let mut shuffled_indices: Vec<usize> = (0..data.len()).collect();

        for batch in 0..(data.len() / self.batch_size) {
            rng.shuffle(&mut shuffled_indices);
            let batch_start = batch * self.batch_size;
            let batch_end = cmp::min(batch_start + self.batch_size, data.len());
            let (mut avg_b_grads, mut avg_w_grads) = model.compute_grad(
                &data[shuffled_indices[batch_start]].0,
                &data[shuffled_indices[batch_start]].1
            );

            for i in (batch_start + 1)..batch_end {
                let data_point = &data[shuffled_indices[batch * self.batch_size + i]];
                let (b_grads, w_grads) = model.compute_grad(&data_point.0, &data_point.1);

                for j in 0..2 {
                    avg_b_grads[j] += &b_grads[j];
                    avg_w_grads[j] += &w_grads[j];
                }
            }

            for i in 0..2 {
                avg_b_grads[i].apply(|b| -self.learning_rate * b / (data.len() as f64));
                avg_w_grads[i].apply(|w| -self.learning_rate * w / (data.len() as f64));
            }

            model.optimize(&avg_b_grads, &avg_w_grads)
        }
    }
}
*/
