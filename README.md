# TensorSlow

**[WIP]** Machine learning library implementing neural networks from scratch for a school project.


## How to build ?

You can build the library with the following `make` commands :

Command | Effect | Output folder
--- | --- | --
`make` / `make all` | Build library, tests, benchmarks and examples | `lib` && `bin`
`make lib` | Build library | `lib`
`make tests` | Build tests | `bin`
`make perf` | Build benchmarks | `bin`
`make examples` | Build examples | `bin`

The library's `.o` and `.so` files are built in the `lib` directory.
Other executables are built in the `bin` directory with an adequate suffix.

**Note** : Executables in the bin directory must be run from the root of the
project to be able to find `libtensorslow.so`, or sometimes the examples
data sets. As a general rule, always run binaries and scripts with the following
command :

`./bin/SOME_FILE`


In order to clean the build directories, you can use the following phonies :


Command | Effect | Target folder
--- | --- | ---
`make clean` | Clean everything | `lib` && `bin`
`make clean_lib` | Clean everything in `lib` | `lib`
`make clean_o` | Clean `.o` files in `lib` | `lib`
`make clean_so` |  Clean `.so` files in `lib` | `lib`
`make clean_bin` |  Clean everything in `bin` | `bin`
`make clean_tests` | Clean tests in `bin` | `bin`
`make clean_perf` | Clean benchmarks in `bin` | `bin`
`make clean_examples` | Clean examples in `bin` | `bin`


## Examples

Before running the MNIST example, you must download the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/). You can do this by running
the provided script (once again, from the root of this repository) :

`bash examples/get-mnist.sh`

This will create an `examples/mnist` folder containing the uncompressed dataset
for both training and testing phases.


## Requirements

- [Eigen](http://eigen.tuxfamily.org) : your compiler should have access
  to the `Eigen` directory
- [googletest](https://github.com/google/googletest) : required to build the tests
