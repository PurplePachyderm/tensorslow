# TensorSlow

**[WIP]** Machine learning library implementing neural networks from scratch for a school project.


## How to build ?

You can build the library with the following `make` commands :

Command | Effect | Output folder
--- | --- | --
`make` / `make all` | Build library, tests, benchmarks and examples | `lib` && `bin`
`make lib` | Build library | `lib`
`make test` | Build tests | `bin`
`make perf` | Build benchmarks | `bin`
`make examples` | Build examples | `bin`

The library's `.o` and `.so` files are built in the `lib` directory.
Other executables are built in the `bin` directory with an adequate suffix.

**Note** : Executables in the bin directory must be run from the root of the
project to be able to find `libtensorslow.so`, so you have to use the
following command :

`./bin/example_test`


In order to clean the build directories, you can use the following phonies :


Command | Effect | Target folder
--- | --- | ---
`make clean` | Clean everything | `lib` && `bin`
`make clean_lib` | Clean everything in `lib` | `lib`
`make clean_o` | Clean `.o` files in `lib` | `lib`
`make clean_so` |  Clean `.so` files in `lib` | `lib`
`make clean_bin` |  Clean everything in `bin` | `bin`
`make clean_test` | Clean tests in `bin` | `bin`
`make clean_perf` | Clean benchmarks in `bin` | `bin`
`make clean_examples` | Clean examples in `bin` | `bin`


## Requirements

- [googletest](https://github.com/google/googletest) is required to build the tests.
