# Neurotic

The compile-time feedforward neural network library that no one asked for, written in Rust.

## What is this?

It's a [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network)
implementation (currently with only a single hidden layer), roughly following the methods described
in Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).
What's special about it is that the whole model, including the size of each layer, is defined at
compile time; this is possible in Rust using the compile-time numbers from the
[typenum](https://github.com/paholg/typenum) crate.

If you want to define a `neurotic` neural network with 784 input neurons, 30 hidden neurons,
and 10 output neurons, initialized using a standard normal distribution, using a sigmoid activation
function and a mean-squared cost function, and optimized via stochastic gradient descent, it would
look like this:

```rust
let mut network: NeuralNetwork<U784, U30, U10, Sigmoid, MeanSquared> =
    NeuralNetwork::new::<InputNormalizedNormal>();
let optimizer = StochasticGradientDescent::new(3., 30);
```

Then, to train it with some data:

```rust
for i in 0..10 {
    optimizer.optimize(&mut network, &training_images[..], &training_labels[..]);
}
```

And finally, to get the output of the trained model for an input:

```rust
let y = network.feedforward(&test_images[0]);
```

## Why?

Since the sizes of all matrices involved in the algorithm are known at compile
time, they can be stored in contiguous memory. In theory, this ought to be noticeably
more efficient than any other CPU-based neural network implementation, since it makes much better
use of CPU caches.

The only downside is a hideous and inflexible type-parameter-filled API!

## So it's fast?

It has the potential to be, maybe? I profiled it just enough to determine that the the matrix
multiplications in the backpropagation and feedforward steps were the vast majority of the CPU
load, but nothing beyond that. It's not even multitheaded; I tried, but `nalgebra::Matrix` currently doesn't
implement `Send`, which is necessary for parallel threads to pass data around.

Furthermore, this is a CPU-based implementation, so it's not going to compete with actual ML
frameworks which support GPU acceleration. This was a hobby project to explore neural networks and
to see if Rust's type was up to the task of building everything up at compile time.

## Documentation

There is none, sorry; this isn't intended to be a proper library one might, you know, use. The
code itself is quite well-documented though, so you could easily generate some prettier
documentation from the source if you'd prefer that.

## Example

For a full working example, look at `examples/digits.rs`, which uses the MNIST handwritten digits
dataset to learn digit recognition. Run it with `cargo run --release --example digits`; after
1 iteration it will reach ~93% accuracy and by the end of the 10 iterations it ought to converge
 to 95% or so. On my i5-4670K CPU this takes about 60 seconds.

## Features

This library has a limited number of implemented parameters, but it could easily be extended.
Here's what's currently supported:

* Activation functions:
  * Exponential
  * Linear
  * Sigmoid
* Cost functions:
  * Mean Squared
* Weight Initializers:
  * Standard Normal Distribution
  * Input-normalized Normal Distribution
* Optimizers:
  * Gradient Descent
  * Stochastic Gradient Descent

One thing I would like to have implemented is support for deep networks; by using [generic-array](https://github.com/fizyk20/generic-array),
it would be totally doable to have a compile-time sized array of `H*H` weight matrices (where
`H` is the number of neurons in each hidden layer). I never got around to it.

## License

This project is licensed under the [WTFPL](http://www.wtfpl.net/); you may use it for whatever
purposes you'd like.
