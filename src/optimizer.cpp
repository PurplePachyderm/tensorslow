/*
* Optimizer class to adjust parameters of a ts::Model.
*/


#include "../include/optimizer.hpp"



	// ts::TrainingData

template <typename T>
ts::TrainingData<T>::TrainingData(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newInput,
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> newExpected
) {
	input = newInput;
	expected = newExpected;
}



	// ts::GaElement

template <typename T>
ts::GaElement<T>::GaElement(ts::Tensor<T> * inputTensor) {

	unsigned rows = inputTensor->value.rows();
	unsigned cols = inputTensor->value.cols();

	gradSum = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);

	index = inputTensor->index;

}



template <typename T>
void ts::GaElement<T>::reset()  {
	// Reset the value of accumulated gradient
	gradSum.setZero();
}



	// ts::GradientAccumulator

template <typename T>
ts::GradientAccumulator<T>::GradientAccumulator() {
}



template <typename T>
ts::GradientAccumulator<T>::GradientAccumulator(ts::Model<T> &model) {

	// Reset wengertList in case it has been used before
	model.wList.reset();

	for(unsigned i=0; i<model.wList.nodes.size(); i++) {

		std::shared_ptr<ts::InputNode<T>> inputPtr =
		std::static_pointer_cast<ts::InputNode<T>>(model.wList.nodes[i]);

		// Check if it is associated with a tensor (== optimizable)
		if(inputPtr->optimizedTensor != NULL) {

			// Then append it to the gradient accumulator
			elements.push_back(ts::GaElement<T>(inputPtr->optimizedTensor));
		}
	}
}



template <typename T>
void ts::GradientAccumulator<T>::reset() {
	// #pragma omp parallel for
	for(unsigned i=0; i<elements.size(); i++) {
		elements[i].reset();
	}
}



template <typename T>
void ts::GradientAccumulator<T>::increment(ts::Gradient<T> &gradient) {
	// Increment all elements of gradAccumulator according to gradient

	// #pragma omp parallel for
	for(unsigned i=0; i<elements.size(); i++) {
		// We use two different indices systems here
		// (one for the wList/grad and one for the gradient accumulator)
		elements[i].gradSum += gradient.derivatives[elements[i].index];
	}
}



template <typename T>
void ts::GradientAccumulator<T>::updateTensor(
	ts::Model<T> &model, unsigned i,
	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> value
) {
	// Update a tensor via the gradient accumulator
	std::shared_ptr<ts::InputNode<T>> inputPtr =
	std::static_pointer_cast<ts::InputNode<T>>(model.wList.nodes[i]);

	inputPtr->optimizedTensor->value -= value;
}



template <typename T>
void ts::GradientAccumulator<T>::clear() {
	// Empty elements
	elements.clear();
}



	// ts::Optimizer

template <typename T>
ts::Optimizer<T>::Optimizer() {
}



	// ts::GradientDescentOptimizer

template <typename T>
ts::GradientDescentOptimizer<T>::GradientDescentOptimizer(T newLearningRate) {
	learningRate = newLearningRate;
}



template <typename T>
void ts::GradientDescentOptimizer<T>::updateModel(
	ts::Model<T> &model, unsigned batchSize
) {
	// #pragma omp parallel for
	for(unsigned i=0; i<this->gradAccumulator.elements.size(); i++) {
		this->gradAccumulator.updateTensor(
			model, i,
			learningRate * this->gradAccumulator.elements[i].gradSum / batchSize
		);
	}
}



template <typename T>
std::vector<std::vector<std::vector< T >>> ts::GradientDescentOptimizer<T>::run(
	ts::Model<T> &model, std::vector<std::vector< ts::TrainingData<T> >> &batches
) {

		// Set up gradient accumulator (this also resets wList)

	this->gradAccumulator = ts::GradientAccumulator<T>(model);


		// Start running and training the model

		std::vector<std::vector<std::vector< T >>> losses(this->epochs, (std::vector<std::vector<T>>) {});

	// Epochs
	for(unsigned i=0; i<this->epochs; i++) {

		losses[i] = std::vector<std::vector<T>>(batches.size(), (std::vector<T>) {});

		// Batches
		for(unsigned j=0; j<batches.size(); j++) {

			losses[i][j] = std::vector<T>(batches[j].size(), 0);

			// Data instance
			for(unsigned k=0; k<batches[j].size(); k++) {

				ts::Tensor<T> input = ts::Tensor<T>(
					batches[j][k].input, &(model.wList)
				);
				ts::Tensor<T> expected = ts::Tensor<T>(
					batches[j][k].expected, &(model.wList)
				);

				// Compute model and norm
				ts::Tensor<T> output = model.compute(input);
				ts::Tensor<T> norm = (*this->normFunction)(output - expected);

				// Get gradient and increment gradient accumulator
				ts::Gradient<T> gradient = norm.grad();
				this->gradAccumulator.increment(gradient);

				model.wList.reset();

				losses[i][j][k] = norm.getValue()(0, 0);
			}

			updateModel(model, batches[j].size());
			this->gradAccumulator.reset();

			// TODO Progress bar ?
			std::cout << "Epoch " << i << ", Batch " << j << std::endl;
		}
	}


		// Clean

	this->gradAccumulator.clear();
	model.wList.reset();

	return losses;
}



	// ts::AdamOptimizer

template <typename T>
ts::AdamOptimizer<T>::AdamOptimizer(
	T newAlpha,
	T newBeta1, T newBeta2,
	T newEpsilon
) {
	alpha = newAlpha;
	beta1 = newBeta1;
	beta2 = newBeta2;
	epsilon = newEpsilon;
}



template <typename T>
void ts::AdamOptimizer<T>::updateModel(
	ts::Model<T> &model, unsigned batchSize
) {
	// #pragma omp parallel for
	for(unsigned i=0; i<this->gradAccumulator.elements.size(); i++) {
		this->gradAccumulator.updateTensor(
			model, i,
			alpha * this->gradAccumulator.elements[i].gradSum / batchSize
		);
	}
}



template <typename T>
void ts::AdamOptimizer<T>::initMomentEstimates(
	std::vector< std::shared_ptr<ts::Node<T>> > nodes
) {
	// Initialize shape of moment estimates
	// (same size as reset wList, zero filled)

	Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> tmp;
	for(unsigned i = 0; i<nodes.size(); i++) {

		tmp.setZero(nodes[i]->rows, nodes[i]->cols);
		m.push_back(tmp);
		v.push_back(tmp);

		mHat.push_back(Eigen::Array<T, 0, 0>());
		vHat.push_back(Eigen::Array<T, 0, 0>());
	}
}



template <typename T>
void ts::AdamOptimizer<T>::computeIncrement(
	std::vector< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >& derivatives,
	std::vector<ts::GaElement<T>>& elements
) {
	// gradAccumulator will only be used to convert the indices of optimizable
	// tensors from the gradient / wList indices system (iGrad) to the
	// gradAccumulator indices system (iAcc)

	unsigned iGrad;
	for(unsigned iAcc=0; iAcc<elements.size(); iAcc++) {

		// Get index in the gradAccumulator system
		iGrad = elements[iAcc].index;

		// Compute biased moment estimates
		m[iAcc] = beta1 * m[iAcc] + (1-beta1) * derivatives[iGrad];
		v[iAcc] = beta2 * v[iAcc] + (1-beta2) * derivatives[iGrad] * derivatives[iGrad];

		// Compute bias-corrected moment estimates
		mHat[iAcc] = m[iAcc] / (1 - decayedBeta1);
		vHat[iAcc] = v[iAcc] / (1 - decayedBeta2);

		// Replace gradient with its corrected value
		// (since gradient is used in the gradAccumulator increment method)
		derivatives[iGrad] = mHat[iAcc] / (vHat[iAcc].sqrt() + epsilon);
	}
}



template <typename T>
std::vector<std::vector<std::vector< T >>> ts::AdamOptimizer<T>::run(
	ts::Model<T> &model, std::vector<std::vector< ts::TrainingData<T> >> &batches
) {

		// Set up gradient accumulator (this also resets wList)
	this->gradAccumulator = ts::GradientAccumulator<T>(model);


		//Init parameters

	initMomentEstimates(model.wList.nodes);
	decayedBeta1 = beta1;
	decayedBeta2 = beta2;


		// Start running and training the model

		std::vector<std::vector<std::vector< T >>> losses(this->epochs, (std::vector<std::vector<T>>) {});

	// Epochs
	for(unsigned i=0; i<this->epochs; i++) {

		losses[i] = std::vector<std::vector<T>>(batches.size(), (std::vector<T>) {});

		// Batches
		for(unsigned j=0; j<batches.size(); j++) {

			losses[i][j] = std::vector<T>(batches[j].size(), 0);

			// Data instance
			for(unsigned k=0; k<batches[j].size(); k++) {

				ts::Tensor<T> input = ts::Tensor<T>(
					batches[j][k].input, &(model.wList)
				);
				ts::Tensor<T> expected = ts::Tensor<T>(
					batches[j][k].expected, &(model.wList)
				);

				// Compute model and norm
				ts::Tensor<T> output = model.compute(input);
				ts::Tensor<T> norm = (*this->normFunction)(output - expected);

				// Get & correct gradient, then increment gradient accumulator
				ts::Gradient<T> gradient = norm.grad();
				computeIncrement(
					gradient.derivatives,
					this->gradAccumulator.elements
				);
				this->gradAccumulator.increment(gradient);

				model.wList.reset();

				losses[i][j][k] = norm.getValue()(0, 0);
			}

			updateModel(model, batches[j].size());
			this->gradAccumulator.reset();

			// Decay betas
			decayedBeta1 = decayedBeta1 * beta1;
			decayedBeta2 = decayedBeta2 * beta2;

			// TODO Progress bar ?
			std::cout << "Epoch " << i << ", Batch " << j << std::endl;
		}
	}


		// Clean

	this->gradAccumulator.clear();
	model.wList.reset();

	m = {};
	v = {};
	mHat = {};
	vHat = {};


	return losses;
}
