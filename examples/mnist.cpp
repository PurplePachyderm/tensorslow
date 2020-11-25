/*
* Example usage of the library on the MNIST database.
* It is solved using a multilayer perceptron.
* The structure of the network will be the one described in :
*
* https://leonardoaraujosantos.gitbook.io/artificial-inteligence/appendix/tensorflow/multi_layer_perceptron_mnist
*
* NOTE: Run the examples/get-mnist.sh script before this example to download the
* MNIST dataset. Both the script and this example must be run from the root of
* the repository.
*/

#include <iostream>
#include <fstream>

#include "../include/tensorslow.h"

#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define EXPECTED_IMAGE_SIZE 784



// Helper function to read infos of MNIST files
// (since numbers are encoded with MSB first)
int bitReversal(int i) {
	// Assuming size of an int is 4 bytes

	unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



int main(void) {

		// Define size of dataset, network, ...
		// (everything you might want to change should be here)

	unsigned batchSize = 50;
	unsigned nBatches = 100;
	// unsigned nEpochs = 3;
	// unsigned learningRate = 0.001;

	// unsigned nTests = 10;

	std::vector<unsigned> hiddenLayers = {256, 256};



		// Open data files

	// Train images
	std::ifstream trainImagesBinary(
		"examples/mnist/train-images-idx3-ubyte", std::ios::binary
	);

	// Train labels
	std::ifstream trainLabelsBinary(
		"examples/mnist/train-labels-idx1-ubyte", std::ios::binary
	);

	// Test images
	std::ifstream testImagesBinary(
		"examples/mnist/t10k-images-idx3-ubyte", std::ios::binary
	);

	// Test labels
	std::ifstream testLabelsBinary(
		"examples/mnist/t10k-labels-idx1-ubyte", std::ios::binary
	);

	if(
		!trainImagesBinary.is_open() || !trainLabelsBinary.is_open() ||
		!testImagesBinary.is_open() || !testLabelsBinary.is_open()
	) {
		std::cout << "Error: MNIST dataset not found" << std::endl;
		std::cout << "Make sure you ran the examples/get-mnist.sh script" << std::endl;
		return -1;
	}



		// Read and generate training / testing data
		// (this part will probavly be moved in a function in the newt commit)

	std::vector<std::vector<ts::TrainingData<float>>> trainingData = {};
	std::vector<ts::TrainingData<float>> testingData = {};

	// Infos about the file format of MNIST files can be found at :
	// http://yann.lecun.com/exdb/mnist/
	// (see last two sections of the page)


	// Read trainImagesBinary
	int magicNumber = 0;
	int nImagesTraining = 0;
	int nRowsTraining = 0;
	int nColsTraining = 0;
	int imageSize = 0;

	trainImagesBinary.read((char *)&magicNumber, sizeof(magicNumber));
	magicNumber = bitReversal(magicNumber);
	if(magicNumber != IMAGE_MAGIC_NUMBER) {
		std::cout << "ERROR: image file seems invalid" << std::endl;
		return -1;
	}

	trainImagesBinary.read((char *)&nImagesTraining, sizeof(nImagesTraining));
	nImagesTraining = bitReversal(nImagesTraining);

	trainImagesBinary.read((char *)&nRowsTraining, sizeof(nRowsTraining));
	nRowsTraining = bitReversal(nRowsTraining);

	trainImagesBinary.read((char *)&nColsTraining, sizeof(nColsTraining));
	nColsTraining = bitReversal(nColsTraining);

	imageSize = nRowsTraining * nColsTraining;
	if(imageSize != EXPECTED_IMAGE_SIZE) {
		std::cout << "ERROR: image size is different than expected" << std::endl;
		return -1;
	}

	// Arranging raw dataset by grouping images as a single vector
	u_char ** rawTrainingImages = new u_char*[nImagesTraining];
	for(int i=0; i<nImagesTraining; i++) {
		rawTrainingImages[i] = new u_char[imageSize];
		trainImagesBinary.read((char *)rawTrainingImages[i], imageSize);
	}



	// Read trainLabelsBinary
	magicNumber = 0;
	int nLabelsTraining = 0;

	trainLabelsBinary.read((char *)&magicNumber, sizeof(magicNumber));
	magicNumber = bitReversal(magicNumber);
	if(magicNumber != LABEL_MAGIC_NUMBER) {
		std::cout << "ERROR: label file seems invalid" << std::endl;
		return -1;
	}

	trainLabelsBinary.read((char *)&nLabelsTraining, sizeof(nLabelsTraining));
	nLabelsTraining = bitReversal(nLabelsTraining);

	u_char* rawTrainingLabels = new u_char[nLabelsTraining];
	for(int i = 0; i < nLabelsTraining; i++) {
		trainLabelsBinary.read((char*)&rawTrainingLabels[i], 1);
	}



	if(nImagesTraining != nLabelsTraining) {
		std::cout << "ERROR: numbers of images and labels are different" << std::endl;
		return -1;
	}



	// Generate the ts::TrainingData
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> image;
	image.resize(imageSize, 1);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> label;
	label.resize(1, 1);

	for(unsigned i=0; i<nBatches; i++) {
		trainingData.push_back({});
		for(unsigned j=0; j<batchSize; j++) {
			// Initialize Eigen Array for image
			// This loop seems inefficient but it seems to be the only way ?
			for(int k=0; k<imageSize; k++) {
				image(k, 0) = (float) rawTrainingImages[i * batchSize + j][k];
			}

			// Initialize Eigen Array for label
			label(0, 0) = (float) rawTrainingLabels[i * batchSize + j];

			// Push a new ts::TrainingData
			trainingData[i].push_back(
				ts::TrainingData<float>(image, label)
			);
		}
	}

		// Create and optimize the MultiLayerPerceptron (training phase)

	// TODO



		// Run tests (prediction phase)

	// TODO


	return 0;
}
