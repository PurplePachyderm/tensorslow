/*
* Test suite for the ts::Model class and its instantiations.
* Pre defined models will be instanciated and tested to ensure the compute
* function is accurate.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "../include/tensorslow.h"



TEST(MultiLayerPerceptron, ForwardPass) {
	// Same example as the autodiff test, pre computed with PyTorch
	ts::MultiLayerPerceptron<float> model(2, {3});


	Eigen::Array<float, 2, 1> inputTensor_;
	inputTensor_ <<
	0.6,
	0.4;
	ts::Tensor<float> inputTensor = ts::Tensor<float>(inputTensor_, &(model.wList));


	Eigen::Array<float, 3, 2> weights_;
	weights_ <<
	0.5, 0.5,
	2.0, 3.0,
	0.0, 6.0;
	ts::Tensor<float> weights = ts::Tensor<float>(weights_, &(model.wList));
	model.weights = {weights};

	Eigen::Array<float, 3, 1> biases_;
	biases_ <<
	-0.2,
	0.2,
	0.3;
	ts::Tensor<float> biases = ts::Tensor<float>(biases_, &(model.wList));
	model.biases = {biases};

	Eigen::Array<float, 3,1> expectedOutput;
	expectedOutput <<
	0.5744,
	0.9309,
	0.9370;

	ts::Tensor<float> actualOutput = model.compute(inputTensor);

	for(unsigned i=0; i<3; i++) {
		EXPECT_NEAR(expectedOutput(i,0), actualOutput.getValue()(i,0), 0.0001);
	}
}



int main(int argc, char **argv) {
	std::cout << "*** MODELS TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
