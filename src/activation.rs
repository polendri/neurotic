//! Activation functions

/// An activation function, which converts an neuron input to its activation value (output).
pub trait ActivationFunction {
    /// Evaluate the activation function on the provided input.
    fn eval(x: f64) -> f64;

    /// Evaluate the gradient of the activation function on the provided input.
    fn eval_grad(x: f64) -> f64;
}

/// Exponential activation function.
///
/// `f(x) = e^x`
pub struct Exponential;

impl ActivationFunction for Exponential {
    #[inline]
    fn eval(x: f64) -> f64 {
        x.exp()
    }

    #[inline]
    fn eval_grad(x: f64) -> f64 {
        Self::eval(x)
    }
}

/// Linear activation function.
///
/// `f(x) = x`
pub struct Linear;

impl ActivationFunction for Linear {
    #[inline]
    fn eval(x: f64) -> f64 {
        x
    }

    #[inline]
    #[allow(unused_variables)]
    fn eval_grad(x: f64) -> f64 {
        1.
    }
}

/// Sigmoid activation function.
///
/// `f(x) = 1 / (1 + e^(-x))`
#[derive(Clone, Copy, Debug)]
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    #[inline]
    fn eval(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    #[inline]
    fn eval_grad(x: f64) -> f64 {
        Self::eval(x) * (1. - Self::eval(x))
    }
}
