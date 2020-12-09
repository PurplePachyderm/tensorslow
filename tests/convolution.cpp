/*
* Test suite similar to autodiff, but dedicated to convolution-related
* operations. The functions will be tested in a simialr way
*/

#include <gtest/gtest.h>
#include <iostream>
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



int main(int argc, char **argv) {
	std::cout << "*** CONVOLUTION TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
