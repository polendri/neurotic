extern crate nalgebra;
extern crate neurotic;
extern crate typenum;

use typenum::{U150, U151, U152};

use neurotic::NeuralNetwork;
use neurotic::activation::Sigmoid;
use neurotic::cost::MeanSquared;
use neurotic::initializer::InputNormalizedNormal;

#[test]
fn instantiation() {
    let _ = NeuralNetwork::<U150, U151, U152, Sigmoid, MeanSquared>::new::<InputNormalizedNormal>();
}
