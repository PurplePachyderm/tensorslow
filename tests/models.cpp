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
	model.activationFunction = &(ts::sigmoid);


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



TEST(Convolution, FullCNN) {

	// Test a full CNN model (without fully connected layers) on a pre computed
	// example

	ts::ConvolutionalNetwork<float> model(
		// Input
		{10, 10},

		// Input channels
		ts::ChannelSplit::NOSPLIT, 1,

		// Convolution / pooling (we'll manually add it later)
		{{3, 3, 3}},
		{{2, 2}},

		// No dense layer
		{}
	);


	// Re arrange
	Eigen::Array<float, 3, 9> ker;
	ker <<
	0.0818, -0.0473, -0.0813,
	0.0582, -0.2351, 0.0225,
	0.1489, -0.1572, -0.2979,

	-0.1656, -0.3325, 0.0392,
	0.2441, -0.0628, -0.0139,
	0.2471, -0.066, -0.1781,

	0.1205, -0.0070, 0.2955,
	-0.1671, -0.1583, -0.0712,
	0.3304, 0.2241, -0.2202;

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> biases;
	biases.setZero(3, 64);

	std::vector< ts::Tensor<float> > convKernels = {
		ts::Tensor<float>(ker, &(model.wList))
	};

	std::vector<ts::Tensor<float>> convBiases = {
		ts::Tensor<float>(biases, &(model.wList)),
	};


	model.convKernels = convKernels;
	model.convBiases = convBiases;
	model.convActivation = &(ts::sigmoid);
	model.denseActivation = &(ts::sigmoid);
	model.denseActivation = &(ts::sigmoid);

	model.toggleGlobalOptimize(true);

	Eigen::Array<float, 10, 10> x_;
	x_ <<
	0.5, 0.5, 0.8, 0.7, 0.6, 0.7, 0.9, 0.4, 0.8, 0.2,
	0.2, 0.3, 0.8, 0.7, 0.6, 0.0, 0.2, 0.7, 0.6, 0.1,
	0.0, 0.6, 0.8, 0.7, 0.6, 0.7, 0.1, 0.4, 0.8, 0.3,
	0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1,
	0.5, 0.5, 0.4, 0.1, 0.6, 0.7, 0.9, 0.4, 0.8, 0.2,
	0.0, 0.3, 0.8, 0.2, 0.3, 0.4, 0.2, 0.1, 0.6, 0.1,
	0.1, 0.4, 0.8, 0.6, 0.6, 0.8, 0.1, 0.1, 0.8, 0.9,
	0.9, 0.3, 0.3, 0.5, 0.4, 0.3, 0.7, 0.9, 0.7, 0.1,
	0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.2, 0.2, 0.1,
	0.5, 0.5, 0.8, 0.7, 0.6, 0.7, 0.9, 0.4, 0.8, 0.2;
	ts::Tensor<float> x = ts::Tensor<float>(x_, &(model.wList));

	ts::Tensor<float> output = model.compute(x);


	// Make sure output is correct
	Eigen::Array<float, 48, 1> expectedOutput_;
	expectedOutput_ <<
	0.4573, 0.4428, 0.5253, 0.4627, 0.5099, 0.4610, 0.4686, 0.4944, 0.4394,
	0.4417, 0.5059, 0.4612, 0.4833, 0.4834, 0.4877, 0.4957, 0.5275, 0.4570,
	0.5246, 0.5360, 0.5661, 0.4956, 0.4565, 0.5216, 0.5343, 0.5160, 0.5375,
	0.5049, 0.5183, 0.4966, 0.5404, 0.5179, 0.5674, 0.5622, 0.5984, 0.5551,
	0.5749, 0.5934, 0.5660, 0.5702, 0.6243, 0.5796, 0.5819, 0.5883, 0.5428,
	0.5575, 0.5701, 0.5671;

	ts::Tensor<float> expectedOutput = ts::Tensor<float>(expectedOutput_, &(model.wList));

	for(unsigned i=0; i<48; i++) {
		EXPECT_NEAR(output.getValue()(i, 0), expectedOutput.getValue()(i, 0), 0.001);
	}

	// Get gradient
	ts::Tensor<float> norm = ts::squaredNorm(output);
	ts::Gradient<float> gradient = norm.grad();
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> dker = gradient.getValue(model.convKernels[0]);

	// Make sure gradient is correct
	Eigen::Array<float, 1, 27> expectedDker;
	expectedDker <<
	1.9103, 1.3021, 1.7233,
	2.0699, 1.0967, 1.8750,
	2.5369, 1.3025, 1.2483,

	1.6342, 0.8726, 2.1986,
	2.5579, 1.4272, 1.9388,
	2.6091, 1.7386, 1.5762,

	1.5965, 1.7308, 2.4573,
	1.9599, 1.4796, 2.1940,
	3.1977, 2.0512, 1.5432;


	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<9; j++) {
			ASSERT_NEAR(dker(i, j), expectedDker(0, 9 * i + j), 0.001);
		}
	}

}



int main(int argc, char **argv) {
	std::cout << "*** MODELS TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
