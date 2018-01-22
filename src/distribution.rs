//! Defines several random distributions for initializing weights and biases.

use rand::{XorShiftRng, weak_rng};
use rand::distributions::{Normal, Sample};

/// A type that can sample values according to a random distribution.
pub trait Distribution: Sized {
    /// Generate a random `f64` according to the distribution.
    fn sample(&mut self) -> f64;
}

/// The Normal distribution.
pub struct NormalDistribution {
    rng: XorShiftRng,
    normal: Normal,
}

impl NormalDistribution {
    /// Constructs a new `NormalDistribution` with the specified mean and standard deviation.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        NormalDistribution {
            rng: weak_rng(),
            normal: Normal::new(mean, std_dev),
        }
    }
}

impl Distribution for NormalDistribution {
    fn sample(&mut self) -> f64 {
        self.normal.sample(&mut self.rng)
    }
}

/// The Gaussian (or Standard Normal) distribution, a Normal distribution with mean of 0 and a
/// standard deviation of 1.
pub struct GaussianDistribution(NormalDistribution);

impl GaussianDistribution {
    /// Constructs a new `GaussianDistribution`.
    pub fn new() -> Self {
        GaussianDistribution(NormalDistribution::new(0., 1.))
    }
}
