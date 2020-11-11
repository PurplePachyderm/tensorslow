/*
* Virtual model class that can be optimized using the ts::Optimizer class.
* Some basic neural networks models will be provided, but new ones can be
* user-defined.
*/

#pragma once

#include "autodiff.hpp"

namespace ts {
	template <typename T> class Model;
	template <typename T> class MultiLayerPerceptron;

	// Friends forward declaration
	template <typename T> class GradientAccumulator;
}



	// ts::Model

template <typename T>
class ts::Model {
private:

public:
	ts::WengertList<T> wList;

	// General method for computing the model forward pass
	virtual ts::Tensor<T> compute(ts::Tensor<T> input) = 0;

	friend ts::GradientAccumulator<T>;
};



	// ts::MultiLayerPerceptron

template <typename T>
class ts::MultiLayerPerceptron : public ts::Model<T> {
private:
	ts::Tensor<T> (*activationFunction)(const ts::Tensor<T>&) = &(ts::sigmoid);

public:
	MultiLayerPerceptron(unsigned iInputSize, std::vector<unsigned> layers);

	std::vector<ts::Tensor<T>> weights = {};
	std::vector<ts::Tensor<T>> biases = {};

	// Method accepting 1 input tensor and returning the model's output tensor
	ts::Tensor<T> compute(ts::Tensor<T> input);
};
