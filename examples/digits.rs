extern crate byteorder;
extern crate flate2;
extern crate itertools;
extern crate nalgebra;
extern crate neurotic;
extern crate typenum;

use std::io::Read;

use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use itertools::Itertools;
use nalgebra::{DefaultAllocator, DimName, U1, U10, U30, VectorN};
use nalgebra::allocator::Allocator;
use typenum::U784;

use neurotic::NeuralNetwork;
use neurotic::activation::Sigmoid;
use neurotic::cost::MeanSquared;
use neurotic::initializer::InputNormalizedNormal;
use neurotic::optimizer::{StochasticGradientDescent, Optimizer};

fn read_images<M>(data: &[u8]) -> Vec<VectorN<f64, M>>
where
    M: DimName,
    DefaultAllocator: Allocator<f64, M, U1>,
{
    let mut reader = GzDecoder::new(&data[..]);

    let magic: u32 = reader.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(2051u32, magic);

    let count: u32 = reader.read_u32::<BigEndian>().unwrap();

    let num_rows: u32 = reader.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(28, num_rows);

    let num_cols: u32 = reader.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(28, num_cols);

    reader.bytes()
        .map(|b| (b.unwrap() as f64) / 255.0)
        .batching(|it| Some(VectorN::<f64, M>::from_iterator(it.take(M::dim()))))
        .take(count as usize)
        .collect()
}

fn read_labels<M: DimName>(data: &[u8]) -> Vec<VectorN<f64, M>>
where
    M: DimName,
    DefaultAllocator: Allocator<f64, M, U1>,
{
    let mut reader = GzDecoder::new(&data[..]);

    let magic: u32 = reader.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(2049u32, magic);

    let count: u32 = reader.read_u32::<BigEndian>().unwrap();

    reader.bytes()
        .map(|b| {
            let mut val: VectorN<f64, M> = VectorN::zeros();
            val[b.unwrap() as usize] = 1.0;
            val
        }).take(count as usize)
        .collect()
}

fn main() {
    const ITERATION_COUNT: usize = 30;

    let training_image_bytes = include_bytes!("data/train-images-idx3-ubyte.gz");
    let training_label_bytes = include_bytes!("data/train-labels-idx1-ubyte.gz");
    let test_image_bytes = include_bytes!("data/t10k-images-idx3-ubyte.gz");
    let test_label_bytes = include_bytes!("data/t10k-labels-idx1-ubyte.gz");

    println!("Loading data...");
    let training_images = read_images::<U784>(training_image_bytes);
    let training_labels = read_labels::<U10>(training_label_bytes);
    let test_images = read_images::<U784>(test_image_bytes);
    let test_labels = read_labels::<U10>(test_label_bytes);
    println!("...Done");

    println!("Initializing neural network...");
    let mut network: NeuralNetwork<U784, U30, U10, Sigmoid, MeanSquared> = NeuralNetwork::new::<InputNormalizedNormal>();
    let optimizer: StochasticGradientDescent = StochasticGradientDescent { batch_size: 30, learning_rate: 3.0 };
    println!("...Done");

    for i in 0..ITERATION_COUNT {
        println!("Iteration {}:", i + 1);
        optimizer.optimize(&mut network, &training_images[..], &training_labels[..]);
    }
}
