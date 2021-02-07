# Architecture

This file describes the high level architecture of TensorSlow. The core library
code is in the `src` (for the `.cpp` files) and `include` directories (for the
`.hpp` files).


## Autodiff

The autodiff engine is at the core of the library. It defines the data
structure (`ts::Tensor`) that will be used for all computations, as well as
operators and their derivatives. The `autodiff` files are the base of the
autodiff engine. Additional operators/functions are defined in
`autodiff-operations` and `convolution`.


## Models

`ts::Model` is the class used to define optimizable mathematical models
(neural networks, etc...). The `model` files define the base of this class,
as well as three example children classes : polynoms, as well as MLP and CNN
networks. These models use `ts::Tensor` for computations, thus it is easy to get
their gradients for optimization.


## Optimization

Finally, we implemented optimizers, which are the final step to make the library
functional. The `optimizer` files lay the basis of these algorithms, and defines
two examples : the SGD and Adam optimizers. These optimizers take training data
and a model as a parameter for their `run` method, and will execute the model
on all the training data, adjusting the model's parameters in the process
(by obtaining the gradient with the autodiff system).


## Other files

These files aren't part of one of the three "modules" described aboved, and
thus, are not as important to understand the library architecture.

- `datatypes` : instantiation of template classes described aboves for different
floating data types.
- `serializer` : utility functions to parse/serialize models from/to external
files.
- `utils`: other utility functions.
