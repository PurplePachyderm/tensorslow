/*
* Test suite for the ts::Model class and its instantiations.
* Pre defined models will be instanciated and tested to ensure the compute
* function is accurate.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "../include/tensorslow.h"



TEST(Polynom, FullTest) {

	// Test the results of a polynom model and its derivative

		// Create model
	ts::Polynom<float> model(3, {3, 3});


		// Gen coefs
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> a =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * 5.0f;
	model.coefficients[3] = ts::Tensor<float>(a, &(model.wList));

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> b =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * 5.0f;
	model.coefficients[2] = ts::Tensor<float>(b, &(model.wList));

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> c =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * 5.0f;
	model.coefficients[1] = ts::Tensor<float>(c, &(model.wList));

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> d =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * 5.0f;
	model.coefficients[0] = ts::Tensor<float>(d, &(model.wList));


		// Get input tensor, actual and expected output
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> input_ =
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>()
	.setRandom(model.rows(), model.cols()) * 10.0f;
	ts::Tensor<float> input = ts::Tensor<float>(input_, &(model.wList));

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> expectedOutput =
	a * input_.pow(3) + b * input_.pow(2) + c * input_ + d;

	ts::Tensor<float> actualOutput = model.compute(input);


		// Get expected and actual gradient
	ts::Gradient<float> grad = actualOutput.grad();


		// Assert results
	for(unsigned i=0; i<model.rows(); i++) {
		for(unsigned j=0; j<model.cols(); j++) {
			EXPECT_NEAR(expectedOutput(i,j), actualOutput.getValue()(i,j), 0.001);

			EXPECT_NEAR(
				grad.getValue(input)(i, j),

				3.0f * a(i, j) * input_(i, j) * input_(i, j) +
				2.0f * b(i, j) * input_(i, j) +
				c(i, j),

				0.001
			);
		}
	}
}



TEST(MultiLayerPerceptron, ForwardPass) {
	// Same example as the autodiff test, pre computed with PyTorch
	// Assert result only (not its derivative as it's not easy to manually differentiate)
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
