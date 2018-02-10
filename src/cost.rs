//! Cost functions

use nalgebra::DVector;

/// Cost function
pub trait CostFunction {
    /// Evaluates the cost function
    fn eval(input: &DVector<f64>, target: &DVector<f64>) -> f64;

    /// Evaluates the gradient of the cost function
    fn eval_grad(input: &DVector<f64>, target: &DVector<f64>) -> DVector<f64>;
}

/// Quadratic cost function
#[derive(Clone, Copy, Debug)]
pub struct MeanSquared;

impl CostFunction for MeanSquared {
    fn eval(input: &DVector<f64>, target: &DVector<f64>) -> f64 {
        let mut diff = input - target;
        diff.apply(|e| e.powf(2.));
        let sum: f64 = diff.iter().sum();
        sum / 2.
    }

    fn eval_grad(input: &DVector<f64>, target: &DVector<f64>) -> DVector<f64> {
        input - target
    }
}
