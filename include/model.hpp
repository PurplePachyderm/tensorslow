/*
* Virtual model class that can be optimized using the ts::Optimizer class.
* Some basic neural networks models will be provided, but new ones can be
* user-defined.
*/

#pragma once

#include "autodiff.hpp"
#include "serializer.hpp"

#include <string>
#include <fstream>
#include <iostream>

namespace ts {
	template <typename T> class Model;

	template <typename T> class Polynom;
	template <typename T> class MultiLayerPerceptron;
	template <typename T> class ConvolutionalNetwork;

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
	virtual void toggleGlobalOptimize(bool enable) = 0;

	// General method for computing the model forward pass
	virtual ts::Tensor<T> compute(ts::Tensor<T> input) = 0;

	// Serializes / parses model into / from a file
	virtual void save(std::string filePath) = 0;
	virtual void load(std::string filePath) = 0;

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

	void save(std::string filePath);
	void load(std::string filePath);

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

	void save(std::string filePath);
	void load(std::string filePath);
};



	// ts::ConvolutionalNetwork

template <typename T>
class ts::ConvolutionalNetwork : public ts::Model<T> {
private:

public:
	ConvolutionalNetwork(
		std::vector<unsigned> inputSize,
		std::vector<std::vector<unsigned>> convLayers,
		std::vector<std::vector<unsigned>> poolingLayers,
		std::vector<unsigned> denseLayers
	);

	ts::Tensor<T> (*activationFunction)(const ts::Tensor<T>&) = &(ts::sigmoid);


	std::vector<unsigned> expectedInput;
	std::vector<ts::Tensor<T>> convKernels = {};
	std::vector<std::vector<unsigned>> pooling;
	std::vector<ts::Tensor<T>> weights = {};
	std::vector<ts::Tensor<T>> biases = {};

	void toggleGlobalOptimize(bool enable);

	ts::Tensor<T> compute(ts::Tensor<T> input);

	void save(std::string filePath);
	void load(std::string filePath);
};
