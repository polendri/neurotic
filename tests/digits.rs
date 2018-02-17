extern crate byteorder;
extern crate nalgebra;
extern crate neurotic;
extern crate typenum;

use std::fs::File;
use std::io::{Cursor, Read};

use byteorder::{BigEndian, ReadBytesExt};
use nalgebra::{DefaultAllocator, DimName, MatrixMN, U10, U30};
use nalgebra::allocator::Allocator;
use typenum::{B0, B1, U784, U10000, UInt, UTerm};

use neurotic::NeuralNetwork;
use neurotic::activation::Sigmoid;
use neurotic::cost::MeanSquared;
use neurotic::initializer::InputNormalizedNormal;

type U60000 = UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, B1>, B1>, B0>, B1>, B0>, B1>, B0>, B0>, B1>, B1>, B0>, B0>, B0>, B0>, B0>;

fn read_images<M, N>(data: &[u8]) -> MatrixMN<f64, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<f64, M, N>
{
    let mut cursor = Cursor::new(data);

    let magic: u32 = cursor.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(2051u32, magic);

    let count: u32 = cursor.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(M::dim() as u32, count);

    let num_rows: u32 = cursor.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(28, num_rows);

    let num_cols: u32 = cursor.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(28, num_cols);

    MatrixMN::<f64, M, N>::from_iterator(
        cursor.bytes().map(|b| (b.unwrap() as f64) / 255.0)
    )
}

fn read_labels<M, N>(data: &[u8]) -> MatrixMN<f64, M, N>
where
    M: DimName,
    N: DimName,
    DefaultAllocator: Allocator<f64, M, N>
{
    let mut cursor = Cursor::new(data);

    let magic: u32 = cursor.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(2049u32, magic);

    let count: u32 = cursor.read_u32::<BigEndian>().unwrap();
    debug_assert_eq!(M::dim() as u32, count);

    let mut result: MatrixMN<f64, M, N> = MatrixMN::<f64, M, N>::zeros();
    for i in 0..M::dim() {
        result[(i, cursor.read_u8().unwrap() as usize)] = 1.0;
    }

    result
}

#[test]
fn digits() {
    let training_image_bytes = include_bytes!("data/train-images-idx3-ubyte.gz");
    let training_label_bytes = include_bytes!("data/train-labels-idx1-ubyte.gz");
    let test_image_bytes = include_bytes!("data/t10k-images-idx3-ubyte.gz");
    let test_label_bytes = include_bytes!("data/t10k-labels-idx1-ubyte.gz");

    let training_images: MatrixMN<f64, U60000, U784> = read_images::<U60000, U784>(training_image_bytes);
    let training_labels: MatrixMN<f64, U60000, U10> = read_labels::<U60000, U10>(training_image_bytes);
}
