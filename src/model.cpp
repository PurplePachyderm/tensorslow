/*
* Virtual model class that can be optimized using the ts::Optimizer class.
* Some basic neural networks models will be provided, but new ones can be
* user-defined.
*/

#include "../include/model.hpp"



	// ts::MultiLayerPerceptron

template <typename T>
ts::MultiLayerPerceptron<T>::MultiLayerPerceptron(
	unsigned inputSize, std::vector<unsigned> layers
) {

	// Each element of the layers vector is a new layer, its value represents
	// the layer size. Values are randomly initialized between 0 and 1.

	layers.insert(layers.begin(), inputSize);

	for(unsigned i=1; i<layers.size(); i++) {
		// Add layer i-1 -> layer i weights (and make it optimizable)
		weights.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(layers[i], layers[i-1]),
			&(this->wList), true)
		);

		// Add layer i biases (and make it optimizable)
		biases.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(layers[i], 1),
			&(this->wList), true)
		);
	}
}



template <typename T>
ts::Tensor<T> ts::MultiLayerPerceptron<T>::compute(ts::Tensor<T> input) {

	// Assert expected size
	if(input.getValue().rows() != weights[0].getValue().cols() ||
	input.getValue().cols() != 1) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// Assert weights and biases vectors have the same size
	if(weights.size() != biases.size()) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
	}

	// Begin computation loop
	for(unsigned i=0; i<weights.size(); i++) {
		input = (*activationFunction)(matProd(weights[i], input) + biases[i]);
	}

	return input;

}
