/*
* Virtual model class that can be optimized using the ts::Optimizer class.
* Some basic neural networks models will be provided, but new ones can be
* user-defined.
*/

#include "../include/model.hpp"



	// ts::Model

template <typename T>
void ts::Model<T>::toggleOptimize(ts::Tensor<T> * tensor, bool enable) {
	wList.toggleOptimize(tensor, enable);
}



	// ts::Polynom

template <typename T>
ts::Polynom<T>::Polynom(unsigned order, std::vector<long> size) {

	// +1 for deg 0 coefficient
	for(unsigned i=0; i<order+1; i++) {
		coefficients.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(size[0], size[1]),
			&(this->wList))
		);
	}

	// Size of tensors
	nRows = size[0];
	nCols = size[1];

}



template <typename T>
void ts::Polynom<T>::toggleGlobalOptimize(bool enable) {
	for(unsigned i=0; i<coefficients.size(); i++) {
		this->wList.toggleOptimize(&(coefficients[i]), enable);
	}
}



template <typename T>
ts::Tensor<T> ts::Polynom<T>::compute(ts::Tensor<T> input) {

	// Assert input and coefficients have the same size
	for(unsigned i=0; i<coefficients.size(); i++) {
		if(
			input.getValue().cols() != coefficients[i].getValue().cols() ||
			input.getValue().rows() != coefficients[i].getValue().rows()
		) {
			return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), NULL);
		}
	}


	ts::Tensor<T> result = ts::Tensor<T>(
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(
			coefficients[0].getValue().cols(),
			coefficients[0].getValue().rows()
		),
		&(this->wList)
	);

	ts::Tensor<T> element;


	// Begin computation loop
	for(unsigned i=0; i<coefficients.size(); i++) {
		// Reset element
		element = ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(
				coefficients[i].getValue().cols(),
				coefficients[i].getValue().rows()
			),
			&(this->wList)
		);

		element = element + coefficients[i];

		// Compute element
		for(unsigned j=0; j<i; j++) {
			element = element * input;
		}

		// Increment result
		result = result + element;
	}

	return result;
}



template <typename T>
void ts::Polynom<T>::save(std::string filePath) {
	std::ofstream out(filePath);
	out << ts::serializeTensorsVector(coefficients);
	out.close();
}



template <typename T>
void ts::Polynom<T>::load(std::string filePath) {
	// Delete current tensors and reset wList
	coefficients = {};
	this->wList.reset();

	// Load new tensors
	std::ifstream in(filePath);
	coefficients = ts::parseTensorsVector(in, &(this->wList));
	in.close();

	// Set number of wors and cols
	if(coefficients.size() > 0) {
		nRows = coefficients[0].getValue().rows();
		nCols = coefficients[0].getValue().cols();
	} else {
		nRows = 0;
		nCols = 0;
	}
}



template <typename T>
long ts::Polynom<T>::rows() {
	return nRows;
}



template <typename T>
long ts::Polynom<T>::cols() {
	return nCols;
}



	// ts::MultiLayerPerceptron

template <typename T>
ts::MultiLayerPerceptron<T>::MultiLayerPerceptron(
	unsigned inputSize, std::vector<unsigned> layers
) {
	// Each element of the layers vector is a new layer, its value represents
	// the layer size. Values are randomly initialized between 0 and 1.

	// Make sure layers/input are not of size 0
	if(inputSize == 0) {
		return;
	}

	for(unsigned i=1; i<layers.size(); i++) {
		if(layers[i] == 0) {
			return;
		}
	}


	layers.insert(layers.begin(), inputSize);

	for(unsigned i=1; i<layers.size(); i++) {
		// Add layer i-1 -> layer i weights
		weights.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(layers[i], layers[i-1]),
			&(this->wList))
		);

		// Add layer i biases
		biases.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(layers[i], 1),
			&(this->wList))
		);
	}

}



template <typename T>
void ts::MultiLayerPerceptron<T>::toggleGlobalOptimize(bool enable) {
	for(unsigned i=0; i<weights.size(); i++) {
		this->toggleOptimize(&(weights[i]), enable);
		this->toggleOptimize(&(biases[i]), enable);
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



template <typename T>
void ts::MultiLayerPerceptron<T>::save(std::string filePath) {
	std::ofstream out(filePath);

	out << ts::serializeTensorsVector(weights);
	out << ts::serializeTensorsVector(biases);

	out.close();
}



template <typename T>
void ts::MultiLayerPerceptron<T>::load(std::string filePath) {
	// Delete current tensors and reset wList
	weights = {};
	biases = {};
	this->wList.reset();

	// Load new tensors
	std::ifstream in(filePath);

	weights = ts::parseTensorsVector(in, &(this->wList));
	biases = ts::parseTensorsVector(in, &(this->wList));

	in.close();
}



	// ts::ConvolutionalNetwork

template <typename T>
ts::ConvolutionalNetwork<T>::ConvolutionalNetwork(
	std::vector<unsigned> inputSize,
	std::vector<std::vector<unsigned>> convLayers,
	std::vector<std::vector<unsigned>> poolingLayers,
	std::vector<unsigned> denseLayers
) {
	// inputSize : std::vector of size 3 for dimensions of 2D image / matrix
	//	+ number of channels (number of conv kernels for each layer)
	// convLayers : sizes of convolution kernels (std::vector of dimension 2)
	// fullLayers: sizes of fully connected layers


		// Validate dimensions of network

	if(inputSize.size() != 2) {
		std::cout << "ERROR: Input is not of dimension 2" << std::endl;
		return;
	}
	if(inputSize[0] == 0 || inputSize[1] == 0) {
		std::cout << "ERROR: Input is of size 0" << std::endl;
		return;
	}

	// Do we have an equal number of convolution and pooling layers ?
	if(convLayers.size() != poolingLayers.size()) {
		std::cout << "ERROR: Different numbers for convolution and pooling layers"
		<< std::endl;
		return;
	}


	std::vector<int> intermediarySize = {(int) inputSize[0], (int) inputSize[1]};

	// Make sure convolutions / poolings are possible
	for(unsigned i=0; i<convLayers.size(); i++) {
		// Is size of kernel correctly described
		if(convLayers[i].size() != 3) {
			std::cout << "ERROR: Convolution layer " << i <<
			" is not of dimension 3" << std::endl;
			return;
		}
		// Are the different numbers of channels > 0 ?
		// (the numbers must be in growing order, and multipliers between each others)
		if(i != 0) {
			if(
				convLayers[i][2] == 0
			) {
				std::cout << "ERROR: Number of channels for " << i <<
				" is 0" << std::endl;
				return;
			}
		}

		// Is size of pooling correctly described
		if(poolingLayers[i].size() != 2) {
			std::cout << "ERROR: Pooling layer " << i <<
			" is not of dimension 2" << std::endl;
			return;
		}


		// Compute size of matrix after convolution
		intermediarySize[0] = intermediarySize[0] - convLayers[i][0] + 1;
		intermediarySize[1] = intermediarySize[1] - convLayers[i][1] + 1;


		if(intermediarySize[0] <= 0 || intermediarySize[1] <= 0) {
			std::cout << "ERROR: Convolution layer " << i <<
			" is impossible" << std::endl;
			return;
		}

		// Compute size of matrix after pooling
		if(poolingLayers[i][0] != 0 && poolingLayers[i][1] != 0) {
			if(
				intermediarySize[0] % poolingLayers[i][0] != 0 ||
				intermediarySize[1] % poolingLayers[i][1] != 0
			) {
				std::cout << "ERROR: Pooling layer " << i <<
				" is impossible" << std::endl;
				return;
			}

			intermediarySize[0] = intermediarySize[0] / poolingLayers[i][0];
			intermediarySize[1] = intermediarySize[1] / poolingLayers[i][1];
		}

	}


		// Randomly init kernels, weights and biases

	// Convolution layers

	// This is the input of the network, of dimension 1
	convLayers.insert(convLayers.begin(), {0, 0, 1});

	convKernels = {};

	// For each layer ...
	for(unsigned i=1; i<convLayers.size(); i++) {

		convKernels.push_back({});

		// For each output ...
		for(unsigned j=0; j<convLayers[i][2]; j++) {

			convKernels[i-1].push_back({});

			// For each input ...
			for(unsigned k=0; k<convLayers[i-1][2]; k++) {

				// ... add a conv kernel
				convKernels[i-1][j].push_back(ts::Tensor<T>(
					Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
					.setRandom(convLayers[i][0], convLayers[i][1]),
					&(this->wList))
				);

			}
		}
	}

	// Fully connected layers
	denseLayers.insert(
		// First dense layer input will be the flattened convolution output
		denseLayers.begin(),
		intermediarySize[0] * intermediarySize[1] * convLayers[convLayers.size() - 1][2]
	);

	for(unsigned i=1; i<denseLayers.size(); i++) {
		// Add layer i-1 -> layer i weights
		weights.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(denseLayers[i], denseLayers[i-1]),
			&(this->wList))
		);

		// Add layer i biases
		biases.push_back(ts::Tensor<T>(
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(denseLayers[i], 1),
			&(this->wList))
		);
	}

	// Set up data fields
	expectedInput = inputSize;
	pooling = poolingLayers;
}



template <typename T>
void ts::ConvolutionalNetwork<T>::toggleGlobalOptimize(bool enable) {
	if(weights.size() != biases.size()) {
		return;
	}

	for(unsigned i=0; i<convKernels.size(); i++) {
		for(unsigned j=0; j<convKernels[i].size(); j++) {
			for(unsigned k=0; k<convKernels[i][j].size(); k++) {
				this->toggleOptimize(&(convKernels[i][j][k]), enable);
			}
		}
	}

	for(unsigned i=0; i<weights.size(); i++) {
		this->toggleOptimize(&(weights[i]), enable);
		this->toggleOptimize(&(biases[i]), enable);
	}
}



template <typename T>
ts::Tensor<T> ts::ConvolutionalNetwork<T>::compute(ts::Tensor<T> input) {

	// NOTE It might be a good idea to add an entire function to make sure that
	// all parameters are compatible (in terms of size), and that output is
	// computable

	// Make sure input is large enough for first convolution
	if(
		input.getValue().rows() != expectedInput[0] ||
		input.getValue().cols() != expectedInput[1]
	) {
		return ts::Tensor<T>(Eigen::Array<T, 0, 0>(), &(this->wList));
	}


	// Convert input to a size 1 vector (for number of channels)
	std::vector<ts::Tensor<T>> inputVec = {input};


	// 1) Convolution / pooling computation loop
	for(unsigned i=0; i<convKernels.size(); i++) {

		// We will need to backup the values of the previous layer to compute
		// the new one
		std::vector<ts::Tensor<T>> tmp = inputVec;

		inputVec = {};

		// For each output channel ...
		for(unsigned j=0; j<convKernels[i].size(); j++) {
			inputVec.push_back( ts::convolution(tmp[0], convKernels[i][j][0]) );

			// For each input channel ...
			// (add other elements)
			for(unsigned k=1; k<convKernels[i][j].size(); k++) {
				inputVec[j] = inputVec[j] + ts::convolution(tmp[k], convKernels[i][j][k]);
			}


			// A pooling layer of size 0 means we want to skip it
			if(pooling[i][0] != 0 || pooling[i][1] != 0) {
				inputVec[j] = ts::maxPooling(inputVec[j], pooling[i]);
			}

			inputVec[j] = (*activationFunction)(inputVec[j]);

		}

	}



	// 2) Gather all channels back to input tensor,
	// and flatten convolution outputs
	input = flattening(vertCat(inputVec));


	// 3) Dense layers computation loop
	for(unsigned i=0; i<weights.size(); i++) {
		input = (*activationFunction)(matProd(weights[i], input) + biases[i]);
	}

	return input;
}



template <typename T>
void ts::ConvolutionalNetwork<T>::save(std::string filePath) {
	std::ofstream out(filePath);


	// TODO Recursive function to save tensor of dimension n as a vector of
	// a tensor of dimension n-1
	out << convKernels.size() << std::endl;
	for(unsigned i=0; i<convKernels.size(); i++) {
		out << convKernels[i].size() << std::endl;
		for(unsigned j=0; j<convKernels[i].size(); j++) {
			out << ts::serializeTensorsVector(convKernels[i][j]);
		}
	}

	out << ts::serializeTensorsVector(weights);
	out << ts::serializeTensorsVector(biases);

	out.close();
}



template <typename T>
void ts::ConvolutionalNetwork<T>::load(std::string filePath) {
	// Delete current tensors and reset wList
	convKernels = {};
	weights = {};
	biases = {};
	this->wList.reset();

	// Load new tensors
	std::ifstream in(filePath);


	// TODO Same recursive function for parsing ?

	std::string line;
	std::getline(in, line);
	unsigned size = std::stoi(line);
	unsigned subsize;

	convKernels = {};

	for(unsigned i=0; i<size; i++) {
		std::getline(in, line);
		subsize = std::stoi(line);

		convKernels.push_back({});

		for(unsigned j=0; j<subsize; j++) {
			convKernels[i].push_back(ts::parseTensorsVector(in, &(this->wList)));
		}

	}


	weights = ts::parseTensorsVector(in, &(this->wList));
	biases = ts::parseTensorsVector(in, &(this->wList));


	in.close();
}
