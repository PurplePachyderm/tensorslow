/*
* Optimizer class to adjust parameters of a ts::Model.
*/

#pragma once

#include <Eigen/Dense>

#include "autodiff.hpp"
#include "model.hpp"

#include <vector>
#include <iostream>

namespace ts {
	template <typename T> class TrainingData;

	template <typename T> class GaElement;
	template <typename T> class GradientAccumulator;

	template <typename T> class Optimizer;
	template <typename T> class GradientDescentOptimizer;

};



	// ts::TrainingData
	// (helper class containing both input data and its expected result)

template <typename T>
class ts::TrainingData {
private:

public:
	TrainingData(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newInput,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newExpected
	);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> input;
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> expected;
};



	// ts::GaElement
	// (Gradient accumulator element that keeps summed derivatives and index of
	// an optimized node/tensor)

template <typename T>
class ts::GaElement {
private:
	GaElement(ts::Tensor<T> * inputTensor);

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> gradSum;
	unsigned index;	// in the ts::WengertList / ts::Gradient (see WARNING below)

	void reset();

public:

	friend ts::GradientAccumulator<T>;
	friend ts::GradientDescentOptimizer<T>;
};



	// ts::GradientAccumulator
	// A collection of accumulated gradient elements for all optimizable tensors
	// of a model.
	// WARNING The indices in this array are completely independent of those in
	// the Wengert List, since some nodes may not be input or optimizable.

template <typename T>
class ts::GradientAccumulator {
private:
	GradientAccumulator();
	GradientAccumulator(ts::Model<T> &model);

	std::vector<ts::GaElement<T>> elements = {};

	void reset();
	void increment(ts::Gradient<T> &gradient);
	void updateTensor(
		ts::Model<T> &model, unsigned i,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> value
	);
	void clear();


public:

	friend ts::Optimizer<T>;
	friend ts::GradientDescentOptimizer<T>;
};



	// ts::Optimizer

template <typename T>
class ts::Optimizer {
private:

protected:
	ts::GradientAccumulator<T> gradAccumulator;

	void resetGradAccumulator();	// Set values to 0
	void setupGradAccumulator(ts::Model<T> &model);	// Generate 0-filled elements
	void incrementGradAccumulator(ts::Gradient<T> &gradient);

	// Dependent of optimizer type. Applies and the accumulated gradient.
	virtual void updateModel(ts::Model<T> &model, unsigned batchSize) = 0;

public:
	Optimizer();

	ts::Tensor<T> (*normFunction)(const ts::Tensor<T>&) = &(ts::squaredNorm);

	unsigned epochs = 1;

	// Optimizes the model by running its compute() method on the batches data
	virtual std::vector<std::vector<std::vector< T >>> run(
		ts::Model<T> &model, std::vector<std::vector< ts::TrainingData<T> >> &batches
	) = 0;

};



	// ts::GradientDescentOptimizer

template <typename T>
class ts::GradientDescentOptimizer : public ts::Optimizer<T> {
private:
	void updateModel(ts::Model<T> &model, unsigned batchSize);

public:
	GradientDescentOptimizer(T newLearningRate);

	T learningRate = 0.1;

	std::vector<std::vector<std::vector< T >>> run(
		ts::Model<T> &model, std::vector<std::vector< ts::TrainingData<T> >> &batches
	);
};
