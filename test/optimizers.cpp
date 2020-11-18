/*
* Test suite for different optimizers.
* This test suite will contain tests on simple models using different
* optimizers to ensure the convergence of the loss function.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "../include/tensorslow.h"



TEST(GradientDescent, Polynom) {

	int rows = 1;
	int cols = 1;

	float learningRate = 0.0001;

	float valSpan = 5.0f;
	float inputSpan = 10.0f;

	unsigned nEpochs = 3;
	unsigned nBatches = 40;
	unsigned nElements = 5;


		// Create model and wList

	ts::Polynom<float> model(2, {rows, cols});

	ts::GradientDescentOptimizer<float> optimizer(learningRate);


		// Create coefficients for expected output

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> a =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * valSpan;

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> b =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * valSpan;

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> c =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * valSpan;


		// Generate training data set

	std::vector<std::vector< ts::TrainingData<float> >> trainingData = {};

	// Batches
	for(unsigned i=0; i<nBatches; i++) {
		trainingData.push_back({});

		// Elements
		for(unsigned j=0; j<nElements; j++) {
			Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input =
			Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
			.setRandom(model.rows(), model.cols()) * inputSpan;

			Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> output =
			a * input.pow(2) + b * input+ c;

			trainingData[i].push_back(ts::TrainingData<float>(input, output));
		}
	}


		// Train model

	optimizer.epochs = nEpochs;
	optimizer.run(model, trainingData);

	// TODO Finish test (by comparing costs after each epochs ?)
	// WARNING Gradient descent seems to perform bady at polynomial regression.
	// Try with lineat or other ?

}



int main(int argc, char **argv) {
	std::cout << "*** OPTIMIZERS TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
