//! Random distributions for initializing weights.

use rand::{weak_rng, XorShiftRng};
use rand::distributions::{self, Sample};

/// Sample values according to a random distribution.
pub trait Distribution {
    /// Generate a random `f64` according to the distribution.
    fn sample(&mut self) -> f64;
}

/// The Normal (a.k.a. Gaussian) distribution.
pub struct Normal {
    rng: XorShiftRng,
    normal: distributions::Normal,
}

impl Normal {
    /// Constructs a new `NormalDistribution` with the specified mean and standard deviation.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Normal {
            rng: weak_rng(),
            normal: distributions::Normal::new(mean, std_dev),
        }
    }
}

impl Distribution for Normal {
    fn sample(&mut self) -> f64 {
        self.normal.sample(&mut self.rng)
    }
}

/// The Standard Normal distribution, a Normal distribution with mean of 0 and a
/// standard deviation of 1.
pub struct StandardNormal(Normal);

impl StandardNormal {
    /// Constructs a new `StandardNormal`.
    pub fn new() -> Self {
        StandardNormal(Normal::new(0., 1.))
    }
}

impl Distribution for StandardNormal {
    fn sample(&mut self) -> f64 {
        let StandardNormal(ref mut normal) = *self;
        normal.sample()
    }
}
