/*
* This example trains a neural network recognizing cars/trucks (it will have
* a scalar output) and that will be reused in a R-CNN.
* This is only for the learning phase of the CNN, and the
* RCNN/prediction phase will be in the examples/rcnn_prediction.cpp file.
* There is still a prediction phase in this phase, but it's used to assert
* the efficacy of the base CNN.
*
* NOTE: Run the examples/get-cifar.sh script before this example to download the
* CIFAR dataset. Both the script and this example must be run from the root of
* the repository.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>

#include "../include/tensorslow.h"

#define FILE_SIZE 10000
#define IMAGE_SIZE 3072
#define IMAGE_WIDTH 32
// Images are stored in row-major order with each color component separated
#define IMAGE_HEIGHT 96
#define N_CLASSES 10

#define CARS_LABEL 1
#define TRUCKS_LABEL 9



// Function to generate a 2D vector of ts::TrainingData from CIFAR file
// descriptor. We will generate training data containing

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
	u_char * rawLabels = new u_char[FILE_SIZE];
	u_char ** rawImages = new u_char*[FILE_SIZE];

	for(unsigned i=0; i<FILE_SIZE; i++) {
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

	unsigned currentPos = 0;

	for(unsigned i=0; i<nBatches; i++) {
		data.push_back({});
		for(unsigned j=0; j<batchSize; j++) {

			// Read the next image anyway
			currentPos++;

			if(
				rawLabels[currentPos] != CARS_LABEL &&
				rawLabels[currentPos] != TRUCKS_LABEL
			) {
				// If image is not a car or truck, we have a 1/4 chance
				// of picking it for our training data (this will converge to a
				// 50/50 proportions of vehicles in the end)


				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				if(r > 0.25) {
					// We won't pick that image, decrement i and skip
					j--;
					continue;
				}
			}

			// Initialize Eigen Array for image
			// This loop seems inefficient but it seems to be the only way ?
			for(int k=0; k<IMAGE_HEIGHT; k++) {
				for(int l=0; l<IMAGE_WIDTH; l++) {
					// std::cout << "		(k, l) = " << k << "," << l << std::endl;
					image(k, l) = (float)
					rawImages[currentPos][k * IMAGE_WIDTH + l] / 255.0f;
				}
			}


			// Initialize Eigen Array for label
			label.setZero(1, 1);
			if(
				rawLabels[currentPos] == CARS_LABEL ||
				rawLabels[currentPos] == TRUCKS_LABEL
			) {
				// Label is one for car / truck
				label(0, 0) = 1.0f;
			}


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



std::vector<std::string> readClassNames(std::ifstream &file) {
	std::vector<std::string> classes = {};
	std::string tmp = "";

	for(unsigned i=0; i<N_CLASSES; i++) {
		std::getline(file, tmp);
		classes.push_back(tmp);
	}

	return classes;
}



// Small function to display an image in terminal given its Eigen array
void asciiCifar(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> img) {

	float pixelValue;

	std::cout << "┌";
	for(unsigned i=0; i<IMAGE_WIDTH; i++) {
		std::cout << "─";
	}
	std::cout << "┐" << std::endl;

	for(unsigned i=0; i<IMAGE_WIDTH; i++) {
		std::cout << "│";

		for(unsigned j=0; j<IMAGE_WIDTH; j++) {

			// Compute avg value for all 3 colors
			pixelValue = img(i, j) + img(i + IMAGE_WIDTH, j) + img(i + 2 * IMAGE_WIDTH, j);
			pixelValue = pixelValue / 3;

			if(img(i, j) == 0.0f) {
				std::cout << " ";
			} else if(img(i, j) < 0.25) {
				std::cout << "░";
			} else if(img(i, j) < 0.5f) {
				std::cout << "▒";
			} else if(img(i, j) < 0.75f) {
				std::cout << "▓";
			} else {
				std::cout << "█";
			}

		}

		std::cout << "│" << std::endl;
	}

	std::cout << "└";
	for(unsigned i=0; i<IMAGE_WIDTH; i++) {
		std::cout << "─";
	}
	std::cout << "┘" << std::endl;

}



int main(void) {

	std::cout << std::setprecision(3);
	srand(time(NULL));

	unsigned batchSize = 5;
	unsigned nBatches = 750;
	unsigned nEpochs = 15;

	unsigned nTests = 300;


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

	if(
		!batch1.is_open()
	) {
		std::cout << "Error: CIFAR dataset not found" << std::endl;
		std::cout << "Make sure you ran the examples/get-cifar.sh script" << std::endl;
		return -1;
	}


		// You can change batches for training / prediction phases

	std::vector<std::vector<ts::TrainingData<float>>> trainingData =
	readCifar(batch3, nBatches, batchSize);

	std::vector<ts::TrainingData<float>> testingData =
	readCifar(batch2, 1, nTests)[0];


		// Training phase

	std::cout << "Creating model..." << std::endl;
	ts::ConvolutionalNetwork<float> model(
		// Input
		{IMAGE_HEIGHT, IMAGE_WIDTH},

		// Number of channels for input (3 for RGB)
		ts::ChannelSplit::SPLIT_HOR, 3,

		// Convolution / pooling
		{{3, 3, 32}, {5, 5, 16}},
		{{0,0}, {2, 2}},

		// Dense layers (with output vector & not including first layer)
		{128, 64, N_CLASSES, 1}
	);
	model.toggleGlobalOptimize(true);

	ts::AdamOptimizer<float> optimizer;
	optimizer.epochs = nEpochs;

	std::cout << "Training model..." << std::endl;
	std::vector<std::vector<std::vector< float >>> losses =
	optimizer.run(model, trainingData);
	std::cout << "Training phase complete !" << std::endl << std::endl;


		// Run tests (prediction phase)

	// Display result
	std::ifstream classFile(
		"examples/cifar/batches.meta.txt", std::ios::binary
	);
	std::vector<std::string> classes = readClassNames(classFile);

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

		std::cout << "*******************************************" << std::endl;

		std::cout << "Test " << i << ":" << std::endl;

		asciiCifar(testingData[i].input);


		// Display result
		std::cout << "Expected :" << testingData[i].expected(0, 0) << std::endl;
		std::cout << "Prediction : " << result(0,0) << std::endl;

		if(
			// Predicted vehicle successfully
			(result(0,0) > 0.5 && testingData[i].expected(0, 0) == 1) ||
			// Predicted no vehicle successfully
			(result(0,0) < 0.5 && testingData[i].expected(0, 0) == 0)
		) {
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


		// Clean and save data

	batch1.close();
	batch2.close();
	batch3.close();
	batch4.close();
	batch5.close();
	batch6.close();

	std::cout << std::endl << "Saving model, this might take a few minutes..." <<
	std::endl;
	model.save("examples/rcnn.ts");

	// Generate datafile for gnuplot
	// freopen("rcnn.dat", "w", stdout);
	//
	// // Epoches
	// for(unsigned i=0; i<losses.size(); i++) {
	//
	// 	// Batches
	// 	for(unsigned j=0; j<losses[i].size(); j++) {
	// 		float avg = 0;
	//
	// 		// Data instances
	// 		for(unsigned k=0; k<losses[i][j].size(); k++) {
	// 			avg += losses[i][j][k];
	// 		}
	//
	// 		std::cout << avg /  (float) losses[i][j].size() << std::endl;
	// 	}
	// }

	return 0;
}
