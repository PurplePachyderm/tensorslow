#!/bin/bash

# NOTE Run this script from the root of the repository :
# $ bash examples/get-cifar.sh

cd examples
mkdir cifar
cd cifar

wget https://www.cs.toronto.edu/%7Ekriz/cifar-10-binary.tar.gz

tar -xzvf cifar-10-binary.tar.gz

mv cifar-10-batches-bin/* .
rm cifar-10-binary.tar.gz
rm -rf cifar-10-batches-bin
