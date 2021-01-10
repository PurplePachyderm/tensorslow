/*
* Test suite similar to autodiff, but dedicated to convolution-related
* operations. The functions will be tested in a similar way
*/

#include <gtest/gtest.h>
#include <math.h>

#include "../include/tensorslow.h"



TEST(Convolution, Convolution) {
	// Tests a convolution operation, as well as its differentiation (TODO)
	// In this example, we try to detect a diagonal line

	ts::WengertList<float> wList;


	Eigen::Array<float, 9, 9> mat_;
	mat_ <<
	-1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1,  1, -1, -1, -1, -1, -1,  1, -1,
	-1, -1,  1, -1, -1, -1,  1, -1, -1,
	-1, -1, -1,  1, -1,  1, -1, -1, -1,
	-1, -1, -1, -1,  1, -1, -1, -1, -1,
	-1, -1, -1,  1, -1,  1, -1, -1, -1,
	-1, -1,  1, -1, -1, -1,  1, -1, -1,
	-1,  1, -1, -1, -1, -1, -1,  1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1;

	Eigen::Array<float, 3, 3> ker_;
	ker_ <<
	 1, -1, -1,
	-1,  1, -1,
	-1, -1,  1;


	Eigen::Array<float, 7, 7> res_;
	res_ <<
	 7, -1,  1,  3,  5, -1,  3,
	-1,  9, -1,  3, -1,  1, -1,
	 1, -1,  9, -3,  1, -1,  5,
	 3,  3, -3,  5, -3,  3,  3,
	 5, -1,  1, -3,  9, -1,  1,
	-1,  1, -1,  3, -1,  9, -1,
	 3, -1,  5,  3,  1, -1,  7;


	ts::Tensor<float> mat = ts::Tensor<float>(mat_, &wList);
	ts::Tensor<float> ker = ts::Tensor<float>(ker_, &wList);

	ts::Tensor<float> res = ts::convolution(mat, ker);


	// Grad should be empty because res is not a scalar
	ts::Gradient<float> grad = res.grad();
	EXPECT_EQ(grad.isEmpty(), true);

	EXPECT_EQ(res.getValue().rows(), 7);
	EXPECT_EQ(res.getValue().cols(), 7);

	for(unsigned i=0; i<7; i++) {
		for(unsigned j=0; j<7; j++) {
			EXPECT_EQ(res_(i, j), res.getValue()(i,j));
		}
	}

	// Get correct gradient
	res = ts::squaredNorm(res);
	grad = res.grad();
	EXPECT_EQ(grad.isEmpty(), false);
}



TEST(Convolution, MaxPooling) {
	// We'll create a simple 6*9 matrix, and try a max pooling with a 3*3
	// pool size.

	ts::WengertList<float> wList;

	Eigen::Array<float, 6, 9> x_;
	x_ <<
	 0, 42, -1,  42,  0, -1,   1,  4,  4,
	-2,  1, -8,   0,  0,  1,  42,  4,  7,
	 1,  6,  1,   2,  6,  1,   1, -5,  9,

	-1, -1, -1,   1, -1,  1,  10,  1, 41,
	-1, 42, -1,   1,  1,  1,   1, 42, 11,
	-1, -1, -1,   1,  1, 42,   9,  8,  5;

	ts::Tensor<float> x = ts::Tensor<float>(x_, &wList);


	// First, test with a pool size that doesn't match
	ts::Tensor<float> res = ts::maxPooling(x, {3, 4});
	ASSERT_EQ(res.getValue().rows(), 0);
	ASSERT_EQ(res.getValue().cols(), 0);


	// Get actual result and gradient
	res = ts::maxPooling(x, {3, 3});

	// Check size / values of res
	// (expected matrix is full 42)
	ASSERT_EQ(res.getValue().rows(), 2);
	ASSERT_EQ(res.getValue().cols(), 3);

	for(unsigned i=0; i<2; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(res.getValue()(i, j), 42.0f);
		}
	}

	// Get gradient

	Eigen::Array<float, 6, 9> expectedGrad;
	expectedGrad <<
	0, 84.0f, 0, 84.0f, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 84.0f, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 84.0f, 0, 0, 0, 0, 0, 84.0f, 0,
	0, 0, 0, 0, 0, 84.0f, 0, 0, 0;

	res = ts::squaredNorm(res);
	ts::Gradient<float> grad = res.grad();
	EXPECT_EQ(grad.isEmpty(), false);

	for(unsigned i=0; i<2; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(grad.getValue(x)(i, j), expectedGrad(i, j));
		}
	}
}



TEST(Convolution, VerticalConcatenation) {
	// We'll create a simple 6*9 matrix, and try a max pooling with a 3*3
	// pool size.

	ts::WengertList<float> wList;

	Eigen::Array<float, 3, 3> x_;
	x_ <<
	1, 2, 3,
	4, 5, 6,
	7, 8, 9;
	ts::Tensor<float> x = ts::Tensor<float>(x_, &wList);

	Eigen::Array<float, 4, 3> y_;
	y_ <<
	1, 2, 3,
	4, 5, 6,
	7, 8, 9,
	10, 11, 12;
	ts::Tensor<float> y = ts::Tensor<float>(y_, &wList);

	std::vector<ts::Tensor<float>> vec = {x, y};

	ts::Tensor<float> res = ts::vertCat(vec);


	// Check size / values of res
	ASSERT_EQ(res.getValue().rows(), 7);
	ASSERT_EQ(res.getValue().cols(), 3);

	Eigen::Array<float, 7, 3> expected;
	expected <<
	1, 2, 3,
	4, 5, 6,
	7, 8, 9,
	1, 2, 3,
	4, 5, 6,
	7, 8, 9,
	10, 11, 12;

	for(unsigned i=0; i<7; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(res.getValue()(i, j), expected(i, j));
		}
	}
}



TEST(Convolution, Flattening) {
	// We'll test the flattening of a matrix to a vector

	// Init matrix
	unsigned x = 12;
	unsigned y= 5;

	ts::WengertList<float> wList;

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> mat_;
	mat_.setRandom(x, y);

	ts::Tensor<float> mat = ts::Tensor<float>(mat_, &wList);


	// Flatten and test values
	ts::Tensor<float> res = ts::flattening(mat);

	ASSERT_EQ(res.getValue().rows(), x * y);
	ASSERT_EQ(res.getValue().cols(), 1);

	for(unsigned i=0; i<x; i++) {
		for(unsigned j=0; j<y; j++) {
			EXPECT_EQ(res.getValue()(i * y + j, 0), mat_(i, j));
		}
	}

	// Get gradient
	res = ts::squaredNorm(res);
	ts::Gradient<float> grad = res.grad();
	EXPECT_EQ(grad.isEmpty(), false);

}


TEST(Convolution, FullCNN) {

	// Test a full CNN mode l(without fully connected layers) on a pre computed
	// example

	ts::ConvolutionalNetwork<float> model(
		// Input
		{10, 10},

		// Input channels
		ts::NOSPLIT, 1,

		// Convolution / pooling (we'll manually add it later)
		{},
		{},

		// No dense layer
		{}
	);


	Eigen::Array<float, 3, 3> ker1;
	ker1 <<
	0.0818,  0.0582,  0.1489,
	-0.0473, -0.2351, -0.1572,
	-0.0813,  0.0225, -0.2979;

	Eigen::Array<float, 3, 3> ker2;
	ker2 <<
	-0.1656,  0.2441,  0.2471,
	-0.3325, -0.0628, -0.066,
	0.0392, -0.0139, -0.1781;

	Eigen::Array<float, 3, 3> ker3;
	ker3 <<
	0.1205, -0.1671,  0.3304,
	-0.0070, -0.1583,  0.2241,
	0.2955, -0.0712, -0.2202;

	std::vector< std::vector< ts::Tensor<float> >> kernels = {
		{ts::Tensor<float>(ker1, &(model.wList))},
		{ts::Tensor<float>(ker2, &(model.wList))},
		{ts::Tensor<float>(ker3, &(model.wList))}
	};

	model.convKernels = {kernels};
	model.pooling = {{2, 2}};
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
	Eigen::Array<float, 48, 1> expectedOutput;
	expectedOutput <<
	0.4573, 0.4428, 0.5253, 0.4627, 0.5099, 0.4610, 0.4686, 0.4944, 0.4394,
	0.4417, 0.5059, 0.4612, 0.4833, 0.4834, 0.4877, 0.4957, 0.5275, 0.4570,
	0.5246, 0.5360, 0.5661, 0.4956, 0.4565, 0.5216, 0.5343, 0.5160, 0.5375,
	0.5049, 0.5183, 0.4966, 0.5404, 0.5179, 0.5674, 0.5622, 0.5984, 0.5551,
	0.5749, 0.5934, 0.5660, 0.5702, 0.6243, 0.5796, 0.5819, 0.5883, 0.5428,
	0.5575, 0.5701, 0.5671;

	for(unsigned i=0; i<48; i++) {
		ASSERT_NEAR(output.getValue()(i, 0), expectedOutput(i, 0), 0.001);
	}

	// Get gradient
	ts::Tensor<float> norm = ts::squaredNorm(output);


	ts::Gradient<float> gradient = norm.grad();

	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> dker1 = gradient.getValue(model.convKernels[0][0][0]);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> dker2 = gradient.getValue(model.convKernels[0][1][0]);
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> dker3 = gradient.getValue(model.convKernels[0][2][0]);


	// Make sure gradient is correct

	Eigen::Array<float, 3, 3> expectedDker1;
	expectedDker1 <<
	1.9103, 2.0699, 2.5369,
	1.3021, 1.0967, 1.3025,
	1.7233, 1.8750, 1.2483;

	Eigen::Array<float, 3, 3> expectedDker2;
	expectedDker2 <<
	1.6342, 2.5579, 2.6091,
	0.8726, 1.4272, 1.7386,
	2.1986, 1.9388, 1.5762;

	Eigen::Array<float, 3, 3> expectedDker3;
	expectedDker3 <<
	1.5965, 1.9599, 3.1977,
	1.7308, 1.4796, 2.0512,
	2.4573, 2.1940, 1.5432;

	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<3; j++) {
			ASSERT_NEAR(dker1(i, j), expectedDker1(i, j), 0.001);
			ASSERT_NEAR(dker2(i, j), expectedDker2(i, j), 0.001);
			ASSERT_NEAR(dker3(i, j), expectedDker3(i, j), 0.001);
		}
	}

}



int main(int argc, char **argv) {
	std::cout << "*** CONVOLUTION TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
