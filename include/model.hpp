/*
* Virtual model class that can be optimized using the ts::Optimizer class.
* Some basic neural networks models will be provided, but new ones can be
* user-defined.
*/

#pragma once

#include "autodiff.hpp"

namespace ts {
	template <typename T> class Model;

	template <typename T> class Polynom;
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

	// Call the WengertList toggleOptimize method
	void toggleOptimize(ts::Tensor<T> * tensor, bool enable);

	// Helper function to optimize the whole model
	void toggleGlobalOptimize(bool enable);

	// General method for computing the model forward pass
	virtual ts::Tensor<T> compute(ts::Tensor<T> input) = 0;

	friend ts::GradientAccumulator<T>;
};



	// ts::Polynom
	// (element-wise polynom for nxn tensors)

template <typename T>
class ts::Polynom : public ts::Model<T> {
private:
	long nRows = 0;
	long nCols = 0;

public:
	Polynom(unsigned order, std::vector<long> size);

	std::vector<ts::Tensor<T>> coefficients = {};

	void toggleGlobalOptimize(bool enable);

	ts::Tensor<T> compute(ts::Tensor<T> input);

	long rows();
	long cols();
};



	// ts::MultiLayerPerceptron

template <typename T>
class ts::MultiLayerPerceptron : public ts::Model<T> {
private:

public:
	MultiLayerPerceptron(unsigned inputSize, std::vector<unsigned> layers);

	ts::Tensor<T> (*activationFunction)(const ts::Tensor<T>&) = &(ts::sigmoid);

	std::vector<ts::Tensor<T>> weights = {};
	std::vector<ts::Tensor<T>> biases = {};

	void toggleGlobalOptimize(bool enable);

	ts::Tensor<T> compute(ts::Tensor<T> input);
};
