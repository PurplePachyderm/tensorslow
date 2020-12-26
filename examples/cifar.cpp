/*
* Example usage of the library on the CIFAR dataset.
* It is solved using a convolutional neural network.
*
* NOTE: Run the examples/get-cifar.sh script before this example to download the
* CIFAR dataset. Both the script and this example must be run from the root of
* the repository.
*/

#include <iostream>
#include <fstream>
#include <iomanip>

#include "../include/tensorslow.h"

#define FILE_SIZE 10000
#define IMAGE_SIZE 3072
#define IMAGE_WIDTH 32
// Images are stored in row-major order with each color component separated
#define IMAGE_HEIGHT 96



// Function to generate a 2D vector of ts::TrainingData from CIFAR file
// descriptor

std::vector<std::vector< ts::TrainingData<float> >> readCifar(
	std::ifstream &file, unsigned nBatches, unsigned batchSize
) {

	// Infos about the file format of CIFAR-10 files can be found at :
	// https://www.cs.toronto.edu/%7Ekriz/cifar.html
	// (see the "Binary version" section of the page)

	std::vector<std::vector<ts::TrainingData<float>>> data = {};

	if(nBatches * batchSize > FILE_SIZE) {
		std::cout << "ERROR: too few images for training data" << std::endl;
		exit(-1);
	}


	// Arranging raw dataset by grouping images & labels in separate vectors
	u_char * rawLabels = new u_char[nBatches * batchSize];
	u_char ** rawImages = new u_char*[nBatches * batchSize];

	for(unsigned i=0; i<nBatches * batchSize; i++) {
		// Read label
		file.read((char*)&rawLabels[i], 1);

		// Read raw image
		rawImages[i] = new u_char[IMAGE_SIZE];
		file.read((char *)rawImages[i], IMAGE_SIZE);
	}


	// Generate the ts::TrainingData
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> image;
	image.resize(IMAGE_HEIGHT, IMAGE_WIDTH);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> label;
	label.resize(10, 1);

	for(unsigned i=0; i<nBatches; i++) {
		data.push_back({});
		for(unsigned j=0; j<batchSize; j++) {

			// Initialize Eigen Array for image
			// This loop seems inefficient but it seems to be the only way ?
			for(int k=0; k<IMAGE_HEIGHT; k++) {
				for(int l=0; l<IMAGE_WIDTH; l++) {
					image(k, l) = (float)
					rawImages[i * batchSize + j][k * IMAGE_WIDTH + l] / 255.0f;
				}
			}

			// Initialize Eigen Array for label
			label.setZero(10, 1);
			label(rawLabels[i * batchSize + j], 0) = 1.0f;

			// Push a new ts::TrainingData
			data[i].push_back(
				ts::TrainingData<float>(image, label)
			);
		}
	}


	// Free u_char arrays
	for(unsigned i = 0; i < nBatches * batchSize; i++) {
		free(rawImages[i]);
	}
	free(rawImages);
	free(rawLabels);

	return data;
}



int main(void) {

	std::cout << std::setprecision(3);
	srand(time(NULL));


	unsigned batchSize = 5;
	unsigned nBatches = 1000;


	// Open data files
	std::ifstream batch1(
		"examples/cifar/data_batch_1.bin", std::ios::binary
	);

	std::vector<std::vector<ts::TrainingData<float>>> trainingData =
	readCifar(batch1, nBatches, batchSize);

	batch1.close();

	return 0;
}
