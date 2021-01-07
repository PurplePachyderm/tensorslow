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
	unsigned nBatches = 100;
	unsigned nEpochs = 3;

	unsigned nTests = 100;


		// Open data files

	std::ifstream batch1(
		"examples/cifar/data_batch_1.bin", std::ios::binary
	);

	std::ifstream batch2(
		"examples/cifar/data_batch_2.bin", std::ios::binary
	);

	std::ifstream batch3(
		"examples/cifar/data_batch_3.bin", std::ios::binary
	);

	std::ifstream batch4(
		"examples/cifar/data_batch_4.bin", std::ios::binary
	);

	std::ifstream batch5(
		"examples/cifar/data_batch_5.bin", std::ios::binary
	);

	std::ifstream batch6(
		"examples/cifar/data_batch_6.bin", std::ios::binary
	);


		// You can change batches for training / prediction phases

	std::vector<std::vector<ts::TrainingData<float>>> trainingData =
	readCifar(batch1, nBatches, batchSize);

	std::vector<ts::TrainingData<float>> testingData =
	readCifar(batch2, 1, nTests)[0];


		// Create and optimize the MultiLayerPerceptron (training phase)

	std::cout << "Creating model..." << std::endl;

	ts::ConvolutionalNetwork<float> model(
		// Input
		{IMAGE_HEIGHT, IMAGE_WIDTH},

		// Convolution / pooling
		{{9, 3, 32}, {17, 4, 64}, {9, 3, 64}},
		{{2,2}, {2,2}, {0, 0}},

		// Dense layers (with output vector & not including first layer)
		{64, 10}
	);

	model.toggleGlobalOptimize(true);


	// Adam optimizer is now the default one
	ts::AdamOptimizer<float> optimizer;

	optimizer.epochs = nEpochs;

	std::cout << "Training model..." << std::endl;
	std::vector<std::vector<std::vector< float >>> losses =
	optimizer.run(model, trainingData);


	std::cout << "Training phase complete !" << std::endl << std::endl;

	// TESTING
	ts::Tensor<float> input = ts::Tensor<float>(trainingData[0][0].input, &model.wList);
	ts::Tensor<float> output = model.compute(input);
	std::cout << "ACTUAL" << std::endl;
	std::cout << output.getValue() << std::endl;
	std::cout << "EXPECTED" << std::endl;
	std::cout << trainingData[0][0].expected << std::endl;


	// Run tests (prediction phase)

	unsigned nSuccesses = 0;
	unsigned nErrors = 0;

	for(unsigned i=0; i<nTests; i++) {
		ts::Tensor<float> input = ts::Tensor<float>(
			testingData[i].input, &(model.wList)
		);

		ts::Tensor<float> result_ = model.compute(input);

		Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> result =
		result_.getValue();

		model.wList.reset();


		// Display result

		std::cout << "*******************************************" << std::endl;

		std::cout << "Test " << i << ":" << std::endl;

		// TODO Display ASCII ? Possile to get a good result ?
		// asciiDigit(testingData[i].input);

		// Display label (expected output)
		unsigned label = 0;
		for(unsigned j=0; j<10; j++) {
			if(testingData[i].expected(j, 0) == 1.0f) {
				std::cout << "Label :" << j << std::endl;
				label = j;
			}
		}


		// Display prediction

		// Get prediction (maximum of the result vector)
		unsigned prediction = 0;
		float maxProbability = result(0,0);
		for(unsigned j=1; j<10; j++) {
			if(result(j, 0) > maxProbability) {
				prediction = j;
				maxProbability = result(j, 0);
			}
		}

		std::cout << "Prediction (" << prediction << "):" << std::endl;

		for(unsigned j=0; j<10; j++) {
			std::cout << j << ": " << result(j) << std::endl;
		}

		if(prediction == label) {
			std::cout << std::endl << "SUCCESS =)" << std::endl;
			nSuccesses++;
		}
		else {
			std::cout << std::endl << "ERROR =/" << std::endl;
			nErrors++;
		}

		std::cout << "*******************************************" << std::endl
		<< std::endl;
	}

	std::cout << "Number of successes: " << nSuccesses << std::endl;
	std::cout << "Number of failures: " << nErrors << std::endl;

	std::cout << "Accuracy: " << 100 * (float) nSuccesses / (float) nTests << "%" << std::endl;


	batch1.close();
	batch2.close();
	batch3.close();
	batch4.close();
	batch5.close();
	batch6.close();

	return 0;
}
