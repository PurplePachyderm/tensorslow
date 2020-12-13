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


	// Grad shoul be empty because res is not a scalar
	ts::Gradient<float> grad = res.grad();
	EXPECT_EQ(grad.isEmpty(), true);

	EXPECT_EQ(res.getValue().rows(), 7);
	EXPECT_EQ(res.getValue().cols(), 7);

	for(unsigned i=0; i<7; i++) {
		for(unsigned j=0; j<7; j++) {
			EXPECT_EQ(res_(i, j), res.getValue()(i,j));
		}
	}
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

}



int main(int argc, char **argv) {
	std::cout << "*** CONVOLUTION TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
