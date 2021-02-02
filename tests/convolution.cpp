/*
* Test suite similar to autodiff, but dedicated to convolution-related
* operations. The functions will be tested in a similar way
*/

#include <gtest/gtest.h>
#include <math.h>

#include "../include/tensorslow.h"



TEST(Convolution, Convolution) {
	// Tests a convolution operation
	// In this example, we try to detect a diagonal line
	// NOTE This is a test for legacy code, since im2col approach is now used

	ts::WengertList<float> wList;


	Eigen::Array<float, 9, 10> mat_;
	mat_ <<
	-1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
	-1,  1, -1, -1, -1, -1, -1,  1, -1, 2,
	-1, -1,  1, -1, -1, -1,  1, -1, -1, 3,
	-1, -1, -1,  1, -1,  1, -1, -1, -1, 4,
	-1, -1, -1, -1,  1, -1, -1, -1, -1, 5,
	-1, -1, -1,  1, -1,  1, -1, -1, -1, 6,
	-1, -1,  1, -1, -1, -1,  1, -1, -1, 7,
	-1,  1, -1, -1, -1, -1, -1,  1, -1, 8,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, 9;

	Eigen::Array<float, 3, 3> ker_;
	ker_ <<
	 1, -1, -1,
	-1,  1,  2,
	-1, -1,  1;


	Eigen::Array<float, 7, 8> res_;
	res_ <<
	 4, -4, -2,  0,  2,  2,  0,  6,
     2,  6, -4,  0,  2, -2, -4, 12,
    -2,  2,  6,  0, -2, -4,  2, 12,
     0,  0,  0,  2, -6,  0,  0, 14,
     2,  2, -2,  0,  6, -4, -2, 16,
     2, -2, -4,  0,  2,  6, -4, 16,
     0, -4,  2,  0, -2,  2,  4, 18;


	ts::Tensor<float> mat = ts::Tensor<float>(mat_, &wList);
	ts::Tensor<float> ker = ts::Tensor<float>(ker_, &wList);

	ts::Tensor<float> res = ts::convolution(mat, ker);

	EXPECT_EQ(res.getValue().rows(), 7);
	EXPECT_EQ(res.getValue().cols(), 8);

	for(unsigned i=0; i<7; i++) {
		for(unsigned j=0; j<7; j++) {
			EXPECT_EQ(res_(i, j), res.getValue()(i,j));
		}
	}

	// // Get correct gradient
	res = ts::squaredNorm(res);
	ts::Gradient<float> grad = res.grad();
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



TEST(Convolution, Split) {

	ts::WengertList<float> wList;

	Eigen::Array<float, 6, 3> x_;
	x_ <<
	1, 2, 3,
	4, 5, 6,
	7, 8, 9,
	10, 11, 12,
	13, 14, 15,
	16, 17, 18;
	ts::Tensor<float> x = ts::Tensor<float>(x_, &wList);

	std::vector<ts::Tensor<float>> resVec = ts::split(x, ts::ChannelSplit::SPLIT_HOR, 2);
	ts::Tensor<float> res = resVec[0] + resVec[1];

	// Make sure that base result is correct

	Eigen::Array<float, 3, 3> expectedRes;
	expectedRes <<
	11, 13, 15,
	17, 19, 21,
	23, 25, 27;

	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(expectedRes(i, j), res.getValue()(i, j));
		}
	}


	// Get norm & its derivative

	res = ts::squaredNorm(res);
	ts::Gradient<float> gradient = res.grad();
	Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> dx = gradient.getValue(x);

	Eigen::Array<float, 6, 3> expectedDx;
	expectedDx <<
	22, 26, 30,
	34, 38, 42,
	46, 50, 54,
	22, 26, 30,
	34, 38, 42,
	46, 50, 54;

	for(unsigned i=0; i<6; i++) {
		for(unsigned j=0; j<3; j++) {
			EXPECT_EQ(expectedDx(i, j), dx(i, j));
		}
	}

}



TEST(Convolution, VerticalConcatenation) {

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



TEST(Convolution, Im2Col) {

	ts::WengertList<float> wList;

	Eigen::Array<float, 3, 3> x1_;
	x1_ <<
	1, 2, 3,
	4, 5, 6,
	7, 8, 9;

	Eigen::Array<float, 3, 3> x2_;
	x2_ <<
	11, 12, 13,
	14, 15, 16,
	17, 18, 19;

	Eigen::Array<float, 3, 3> x3_;
	x3_ <<
	21, 22, 23,
	24, 25, 26,
	27, 28, 29;

	std::vector<ts::Tensor<float>> x = {
		ts::Tensor<float>(x1_, &wList),
		ts::Tensor<float>(x2_, &wList),
		ts::Tensor<float>(x3_, &wList)
	};


	ts::Tensor<float> mat = ts::im2col(x, {2, 2});

	ts::Tensor<float> norm = ts::squaredNorm(mat);

	ts::Gradient<float> grad = norm.grad();

	// Test im2col result
	Eigen::Array<float, 12, 4> expected;
	expected <<
	1, 2, 4, 5,
	4, 5, 7, 8,
	2, 3, 5, 6,
	5, 6, 8, 9,

	11, 12, 14, 15,
	14, 15, 17, 18,
	12, 13, 15, 16,
	15, 16, 18, 19,

	21, 22, 24, 25,
	24, 25, 27, 28,
	22, 23, 25, 26,
	25, 26, 28, 29;

	for(unsigned i=0; i<12; i++) {
		for(unsigned j=0; j<4; j++) {
			ASSERT_EQ(mat.getValue()(i,j), expected(i,j));
		}
	}

	// Test partial derivatives
	Eigen::Array<float, 3, 3> expectedDx1;
	expectedDx1 <<
	 2, 12, 10,
	12, 40, 28,
	10, 28, 18;

	Eigen::Array<float, 3, 3> expectedDx2;
	expectedDx2 <<
	22,  52,  30,
	52, 120,  68,
	30,  68,  38;

	Eigen::Array<float, 3, 3> expectedDx3;
	expectedDx3 <<
	42,  92,  50,
	92, 200, 108,
	50, 108,  58;

	Eigen::Array<float, 3, 3> dx1 = grad.getValue(x[0]);
	Eigen::Array<float, 3, 3> dx2 = grad.getValue(x[1]);
	Eigen::Array<float, 3, 3> dx3 = grad.getValue(x[2]);


	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<3; j++) {
			ASSERT_EQ(dx1(i,j), expectedDx1(i,j));
			ASSERT_EQ(dx2(i,j), expectedDx2(i,j));
			ASSERT_EQ(dx3(i,j), expectedDx3(i,j));
		}
	}
}



TEST(Convolution, Col2Im) {

	ts::WengertList<float> wList;

	Eigen::Array<float, 3, 4> x_;
	x_ <<
	1,  2,  3,  4,
	5,  6,  7,  8,
	9, 10, 11, 12;

	ts::Tensor<float> x = ts::Tensor<float>(x_, &wList);
	std::vector<ts::Tensor<float>> res = ts::col2im(x, {2, 2});
	ts::Tensor<float> norm = ts::squaredNorm(ts::vertCat(res));

	ts::Gradient<float> grad = norm.grad();
	Eigen::Array<float, 3, 4> dx = grad.getValue(x);


	// Check result
	Eigen::Array<float, 2, 2> expectedX1;
	expectedX1 <<
	1,  2,
	3,  4;


	Eigen::Array<float, 2, 2> expectedX2;
	expectedX2 <<
	5,  6,
	7,  8;

	Eigen::Array<float, 2, 2> expectedX3;
	expectedX3 <<
	 9,  10,
	11,  12;

	for(unsigned i=0; i<2; i++) {
		for(unsigned j=0; j<2; j++) {
			ASSERT_EQ(res[0].getValue()(i,j), expectedX1(i,j));
			ASSERT_EQ(res[1].getValue()(i,j), expectedX2(i,j));
			ASSERT_EQ(res[2].getValue()(i,j), expectedX3(i,j));
		}
	}


	// Check derivatives

	Eigen::Array<float, 3, 4> expectedDx;
	expectedDx <<
	 2,  4,  6,  8,
	10, 12, 14, 16,
	18, 20, 22, 24;

	for(unsigned i=0; i<3; i++) {
		for(unsigned j=0; j<4; j++) {
			ASSERT_EQ(dx(i,j), expectedDx(i,j));
		}
	}
}



int main(int argc, char **argv) {
	std::cout << "*** CONVOLUTION TEST SUITE ***" << std::endl;

	srand(time(NULL));

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
