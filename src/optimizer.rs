//! Optimization algorithms

use rand::{thread_rng, Rng};
use std::cmp;

use nalgebra::DVector;

use activation::ActivationFunction;
use cost::CostFunction;
use feedforward::NeuralNetwork;

/// Optimizer function
pub trait Optimizer<A, C> : Sized
where
    A: ActivationFunction,
    C: CostFunction,
{
    /// TODO
    fn optimize(&self, model: &mut NeuralNetwork<A, C, Self>, data: &[(DVector<f64>, DVector<f64>)]);
}

/// Gradient descent optimizer
#[derive(Clone, Copy, Debug)]
pub struct GradientDescent {
    learning_rate: f64,
}

impl<A, C> Optimizer<A, C> for GradientDescent
where
    A: ActivationFunction,
    C: CostFunction,
{
    fn optimize(&self, model: &mut NeuralNetwork<A, C, Self>, data: &[(DVector<f64>, DVector<f64>)]) {
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
