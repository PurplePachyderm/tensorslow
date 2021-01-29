/*
* Example usage of the library on the MNIST database.
* It is solved using a multilayer perceptron.
*
* NOTE: Run the examples/get-mnist.sh script before this example to download the
* MNIST dataset. Both the script and this example must be run from the root of
* the repository.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>

#include "../include/tensorslow.h"

#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define EXPECTED_IMAGE_SIZE 784
#define EXPECTED_ROW_COL 28



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



// Function to generate a 2D vector of ts::TrainingData from MNIST file
// descriptors (one for images, one for labels)

std::vector<std::vector< ts::TrainingData<float> >> readMnist(
	std::ifstream &imageFile, std::ifstream &labelFile,
	unsigned nBatches, unsigned batchSize
) {

	// Infos about the file format of MNIST files can be found at :
	// http://yann.lecun.com/exdb/mnist/
	// (see last two sections of the page)

	std::vector<std::vector<ts::TrainingData<float>>> data = {};

	// Read imageFile
	int magicNumber = 0;
	int nImages = 0;
	int nRows = 0;
	int nCols = 0;
	int imageSize = 0;

	imageFile.read((char *)&magicNumber, sizeof(magicNumber));
	magicNumber = bitReversal(magicNumber);
	if(magicNumber != IMAGE_MAGIC_NUMBER) {
		std::cout << "ERROR: image file seems invalid" << std::endl;
		exit(-1);
	}

	imageFile.read((char *)&nImages, sizeof(nImages));
	nImages = bitReversal(nImages);
	if((unsigned) nImages < nBatches * batchSize) {
		std::cout << "ERROR: too few images for training data" << std::endl;
		exit(-1);
	}

	imageFile.read((char *)&nRows, sizeof(nRows));
	nRows = bitReversal(nRows);

	imageFile.read((char *)&nCols, sizeof(nCols));
	nCols = bitReversal(nCols);

	imageSize = nRows * nCols;
	if(imageSize != EXPECTED_IMAGE_SIZE) {
		std::cout << "ERROR: image size is different than expected" << std::endl;
		exit(-1);
	}

	// Arranging raw dataset by grouping images as a single vector
	u_char ** rawImages = new u_char*[nImages];
	for(int i=0; i<nImages; i++) {
		rawImages[i] = new u_char[imageSize];
		imageFile.read((char *)rawImages[i], imageSize);
	}



	// Read labelFile
	magicNumber = 0;
	int nLabels = 0;

	labelFile.read((char *)&magicNumber, sizeof(magicNumber));
	magicNumber = bitReversal(magicNumber);
	if(magicNumber != LABEL_MAGIC_NUMBER) {
		std::cout << "ERROR: label file seems invalid" << std::endl;
		exit(-1);
	}

	labelFile.read((char *)&nLabels, sizeof(nLabels));
	nLabels = bitReversal(nLabels);

	u_char* rawLabels = new u_char[nLabels];
	for(int i = 0; i < nLabels; i++) {
		labelFile.read((char*)&rawLabels[i], 1);
	}



	if(nImages != nLabels) {
		std::cout << "ERROR: numbers of images and labels are different" << std::endl;
		exit(-1);
	}



	// Generate the ts::TrainingData
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> image;
	image.resize(imageSize, 1);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> label;
	label.resize(10, 1);

	for(unsigned i=0; i<nBatches; i++) {
		data.push_back({});
		for(unsigned j=0; j<batchSize; j++) {

			// Initialize Eigen Array for image
			// This loop seems inefficient but it seems to be the only way ?
			for(int k=0; k<imageSize; k++) {
				image(k, 0) = (float) rawImages[i * batchSize + j][k] / 255.0f;
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
	for(int i = 0; i<nImages; i++) {
		free(rawImages[i]);
	}
	free(rawImages);
	free(rawLabels);


	return data;
}



// Small function to display a digit in terminal given its Eigen array
void asciiDigit(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> digit) {

	std::cout << "┌";
	for(unsigned i=0; i<EXPECTED_ROW_COL; i++) {
		std::cout << "─";
	}
	std::cout << "┐" << std::endl;

	for(unsigned i=0; i<EXPECTED_ROW_COL; i++) {
		std::cout << "│";

		for(unsigned j=0; j<EXPECTED_ROW_COL; j++) {

			if(digit(i * EXPECTED_ROW_COL + j, 0) == 0.0f) {
				std::cout << " ";
			} else if(digit(i * EXPECTED_ROW_COL + j, 0) < 0.25) {
				std::cout << "░";
			} else if(digit(i * EXPECTED_ROW_COL + j, 0) < 0.5f) {
				std::cout << "▒";
			} else if(digit(i * EXPECTED_ROW_COL + j, 0) < 0.75f) {
				std::cout << "▓";
			} else {
				std::cout << "█";
			}

		}

		std::cout << "│" << std::endl;
	}

	std::cout << "└";
	for(unsigned i=0; i<EXPECTED_ROW_COL; i++) {
		std::cout << "─";
	}
	std::cout << "┘" << std::endl;

}



int main(void) {

	std::cout << std::setprecision(3);
	srand(time(NULL));
	omp_set_num_threads(4);

		// Define size of dataset, network, ...
		// (everything you might want to change should be here)

	unsigned batchSize = 5;
	unsigned nBatches = 1000;
	unsigned nEpochs = 10;

	// If you're using the SGD optimizer
	// float learningRate = 0.085f;

	unsigned nTests = 100;

	// WARNING You must include the output layer (size 10)
	std::vector<unsigned> layers = {512, 128, 10};



		// Open data files

	// Train images
	std::ifstream trainImageFile(
		"examples/mnist/train-images-idx3-ubyte", std::ios::binary
	);

	// Train labels
	std::ifstream trainLabelFile(
		"examples/mnist/train-labels-idx1-ubyte", std::ios::binary
	);

	// Test images
	std::ifstream testImageFile(
		"examples/mnist/t10k-images-idx3-ubyte", std::ios::binary
	);

	// Test labels
	std::ifstream testLabelFile(
		"examples/mnist/t10k-labels-idx1-ubyte", std::ios::binary
	);

	if(
		!trainImageFile.is_open() || !trainLabelFile.is_open() ||
		!testImageFile.is_open() || !testLabelFile.is_open()
	) {
		std::cout << "Error: MNIST dataset not found" << std::endl;
		std::cout << "Make sure you ran the examples/get-mnist.sh script" << std::endl;
		return -1;
	}



		// Read and generate training / testing data

	std::cout << "Preparing dataset..." << std::endl;

	std::vector<std::vector<ts::TrainingData<float>>> trainingData =
	readMnist(trainImageFile, trainLabelFile, nBatches, batchSize);

	trainImageFile.close();
	trainLabelFile.close();


	std::vector<ts::TrainingData<float>> testingData =
	readMnist(testImageFile, testLabelFile, 1, nTests)[0];

	testImageFile.close();
	testLabelFile.close();



		// Create and optimize the MultiLayerPerceptron (training phase)

	std::cout << "Creating model..." << std::endl;

	ts::MultiLayerPerceptron<float> model(EXPECTED_IMAGE_SIZE, layers);
	model.toggleGlobalOptimize(true);



	// Adam optimizer is now the default one
	ts::AdamOptimizer<float> optimizer;

	// You can use the SGD optimizer instead
	// (it is recommended to adjust parameters. For instance, you can increase
	// the number of batches)
	// ts::GradientDescentOptimizer<float> optimizer(learningRate);

	optimizer.epochs = nEpochs;

	std::cout << "Training model..." << std::endl;
	std::vector<std::vector<std::vector< float >>> losses =
	optimizer.run(model, trainingData);


	std::cout << "Training phase complete !" << std::endl << std::endl;



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

		// Display ASCII
		asciiDigit(testingData[i].input);

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

	return 0;
}
